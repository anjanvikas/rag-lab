"""
Qdrant hybrid search — Phase B hardened.

Search Strategy:
  - BM25 sparse + Dense vector, fused via Qdrant RRF internally.
  - Returns per-source counts so the UI can display real Dense vs BM25 split.
  - Alpha classifier determines the weighting:
      EXACT    → alpha 0.1  (mostly BM25, user wants exact keyword match)
      HYBRID   → alpha 0.5  (balanced RRF)
      SEMANTIC → alpha 0.9  (mostly dense, conceptual question)
"""
from __future__ import annotations
import asyncio
import os
from typing import Any

QDRANT_URL        = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "papers-v1")

ALPHA_MAP = {"EXACT": 0.1, "SEMANTIC": 0.9, "HYBRID": 0.5}

_client = None


def _get_client():
    global _client
    if _client is None:
        try:
            from qdrant_client import QdrantClient
            _client = QdrantClient(url=QDRANT_URL, timeout=20)
        except Exception:
            return None
    return _client


def _build_filter(filters: dict | None):
    if not filters:
        return None
    from qdrant_client.models import Filter, FieldCondition, MatchAny
    conditions = []
    if filters.get("tier"):
        conditions.append(FieldCondition(
            key="tier", match=MatchAny(any=[int(t) for t in filters["tier"]])
        ))
    if filters.get("chunk_type"):
        from qdrant_client.models import MatchValue
        conditions.append(FieldCondition(
            key="chunk_type", match=MatchValue(value=filters["chunk_type"])
        ))
    if not conditions:
        return None
    return Filter(must=conditions)


def _point_to_chunk(m: Any, source: str = "fused") -> dict:
    """Convert a Qdrant ScoredPoint into a flat chunk dict."""
    p = m.payload or {}
    return {
        "chunk_id":     str(m.id),
        "paper_id":     p.get("paper_id", f"arxiv:{p.get('arxiv_id', '')}"),
        "title":        p.get("title", ""),
        "authors":      p.get("authors", ""),
        "year":         p.get("year", ""),
        "tier":         p.get("tier", 0),
        "chunk_type":   p.get("chunk_type", "body"),
        "section":      p.get("section", ""),
        "text":         p.get("text", ""),
        "excerpt":      p.get("text", "")[:300],
        "hybrid_score": round(getattr(m, "score", 0) or 0, 4),
        "rerank_score": None,
        "source":       source,
        "arxiv_id":     p.get("arxiv_id", ""),
    }


async def hybrid_search(
    query:      str,
    top_k:      int = 10,
    alpha:      float | None = None,
    filters:    dict | None = None,
    rerank:     bool = True,
    use_hyde:   bool = False,
    namespace:  str | None = None,
) -> dict:
    """
    Qdrant hybrid search (dense + BM25 sparse) using native RRF fusion.

    Returns a dict with keys:
      results        - final list of chunk dicts
      dense_count    - how many unique papers matched via dense path
      bm25_count     - how many matched via BM25 sparse path
      fused_count    - total after RRF merge
      query_type     - EXACT | HYBRID | SEMANTIC
      alpha_used     - effective alpha value
    """
    from rag.ingestion.embed import embed_query, bm25_encode_query
    from rag.cache import cached_search, set_search_cache
    from rag.search.intelligence import classify_query, generate_hyde_document

    # ── 1. Intent classification ──────────────────────────────────────────────
    query_type = await classify_query(query)
    alpha_used = alpha if alpha is not None else ALPHA_MAP.get(query_type, 0.5)

    client = _get_client()
    if client is None:
        return {"results": [], "error": "Qdrant not available",
                "dense_count": 0, "bm25_count": 0, "fused_count": 0,
                "alpha_used": alpha_used, "query_type": query_type}

    # ── 2. Cache check ────────────────────────────────────────────────────────
    cache_params = {"top_k": top_k, "alpha": alpha_used, "filters": str(filters), "hyde": use_hyde}
    cached = await cached_search(query, cache_params)
    if cached:
        cached["from_cache"] = True
        return cached

    # ── 3. HyDE for dense embedding ───────────────────────────────────────────
    dense_query_text = query
    if use_hyde and query_type in ("SEMANTIC", "HYBRID"):
        hyde_doc = await generate_hyde_document(query)
        dense_query_text = hyde_doc

    # ── 4. Encode ─────────────────────────────────────────────────────────────
    dense_vec  = await embed_query(dense_query_text)
    sparse_raw = bm25_encode_query(query)  # keep original query for keyword recall

    sp_indices = [int(i) for i in sparse_raw.get("indices", [])]
    sp_values  = [float(v) for v in sparse_raw.get("values", [])]

    # Scale vectors per alpha
    dense_vec_scaled = [v * alpha_used for v in dense_vec]
    sparse_values_scaled = [v * (1 - alpha_used) for v in sp_values]

    qdrant_filter = _build_filter(filters)
    fetch_k = min(top_k * 4, 80)

    # ── 5. Parallel dense-only + sparse-only + fused RRF ─────────────────────
    try:
        from qdrant_client.models import (
            NamedVector, NamedSparseVector, SparseVector as QSparseVec,
            Prefetch, FusionQuery, Fusion, Filter,
        )

        # Run fused search AND separate dense/sparse searches in parallel
        # so we can report real per-source counts.
        async def run_fused():
            return client.query_points(
                collection_name=QDRANT_COLLECTION,
                prefetch=[
                    Prefetch(
                        query=dense_vec_scaled,
                        using="dense",
                        limit=fetch_k,
                        filter=qdrant_filter,
                    ),
                    Prefetch(
                        query=QSparseVec(indices=sp_indices, values=sparse_values_scaled),
                        using="sparse",
                        limit=fetch_k,
                        filter=qdrant_filter,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=fetch_k,
                with_payload=True,
            )

        async def run_dense_only():
            return client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=dense_vec,
                using="dense",
                limit=fetch_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )

        async def run_sparse_only():
            if not sp_indices:
                return None
            return client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=QSparseVec(indices=sp_indices, values=sp_values),
                using="sparse",
                limit=fetch_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )

        loop = asyncio.get_event_loop()
        fused_res, dense_res, sparse_res = await asyncio.gather(
            loop.run_in_executor(None, lambda: client.query_points(
                collection_name=QDRANT_COLLECTION,
                prefetch=[
                    Prefetch(query=dense_vec_scaled, using="dense", limit=fetch_k, filter=qdrant_filter),
                    Prefetch(query=QSparseVec(indices=sp_indices, values=sparse_values_scaled), using="sparse", limit=fetch_k, filter=qdrant_filter),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=fetch_k,
                with_payload=True,
            )),
            loop.run_in_executor(None, lambda: client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=dense_vec,
                using="dense",
                limit=fetch_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )),
            loop.run_in_executor(None, lambda: client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=QSparseVec(indices=sp_indices, values=sp_values) if sp_indices else dense_vec,
                using="sparse" if sp_indices else "dense",
                limit=fetch_k,
                query_filter=qdrant_filter,
                with_payload=True,
            ) if sp_indices else asyncio.coroutine(lambda: None)()),
            return_exceptions=True,
        )

        fused_points  = fused_res.points  if not isinstance(fused_res, Exception) else []
        dense_points  = dense_res.points  if not isinstance(dense_res, Exception) else []
        sparse_points = sparse_res.points if (sparse_res and not isinstance(sparse_res, Exception)) else []

    except Exception as e:
        # Fallback: dense-only
        try:
            loop = asyncio.get_event_loop()
            fallback = await loop.run_in_executor(None, lambda: client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=("dense", dense_vec),
                limit=fetch_k,
                query_filter=qdrant_filter,
                with_payload=True,
            ))
            fused_points  = fallback
            dense_points  = fallback
            sparse_points = []
        except Exception as e2:
            return {
                "results": [], "error": str(e2),
                "dense_count": 0, "bm25_count": 0, "fused_count": 0,
                "alpha_used": alpha_used, "query_type": query_type,
            }

    # ── 6. Build result list ──────────────────────────────────────────────────
    dense_ids  = {str(p.id) for p in dense_points}
    sparse_ids = {str(p.id) for p in sparse_points}

    results = []
    for m in fused_points:
        chunk = _point_to_chunk(m, source="fused")
        cid = str(m.id)
        if cid in dense_ids and cid in sparse_ids:
            chunk["source"] = "both"
        elif cid in dense_ids:
            chunk["source"] = "dense"
        elif cid in sparse_ids:
            chunk["source"] = "bm25"
        results.append(chunk)

    # ── 7. Optional local rerank (off by default — we use a separate call) ────
    if rerank and results:
        results = await _local_rerank(query, results, top_k=top_k)
    else:
        results = results[:top_k]

    out = {
        "query_type":   query_type,
        "alpha_used":   alpha_used,
        "results":      results,
        "total":        len(results),
        "from_cache":   False,
        # Per-source counts that the UI shows
        "fused_count":  len(fused_points),
        "dense_count":  len(dense_ids),
        "bm25_count":   len(sparse_ids),
        "bm25_active":  bool(sp_indices),   # tells UI whether BM25 produced any terms
    }
    await set_search_cache(query, cache_params, out)
    return out


async def _local_rerank(query: str, results: list[dict], top_k: int) -> list[dict]:
    try:
        from rag.pipeline import _rerank
        chunks = [{"excerpt": r["text"][:500], **r} for r in results]
        return await _rerank(query, chunks, top_k=top_k)
    except Exception:
        return results[:top_k]
