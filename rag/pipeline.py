"""
RAG Pipeline: Hybrid retrieval (Dense + BM25 → RRF → Cross-encoder rerank) + Claude streaming.
"""
import os
import json
import re
import asyncio
from typing import AsyncGenerator, Optional
from datetime import datetime

import anthropic
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "arxiv_papers")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Lazy globals
_chroma_client = None
_collection = None
_embed_fn = None
_rerank_tokenizer = None
_rerank_model = None


def _get_collection():
    global _chroma_client, _collection, _embed_fn
    if _collection is None:
        _embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_embed_fn,
        )
    return _collection


def _get_reranker():
    global _rerank_tokenizer, _rerank_model
    if _rerank_model is None:
        _rerank_tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL)
        _rerank_model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL)
        _rerank_model.eval()
    return _rerank_tokenizer, _rerank_model


def _rerank(query: str, chunks: list[dict], top_k: int = 7) -> list[dict]:
    """Cross-encoder reranking."""
    if not chunks:
        return []
    tokenizer, model = _get_reranker()
    pairs = [[query, c.get("excerpt", c.get("text", ""))] for c in chunks]
    inputs = tokenizer(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    with torch.no_grad():
        scores = model(**inputs).logits.squeeze(-1).tolist()

    for chunk, score in zip(chunks, scores):
        chunk["score"] = round(float(score), 4)
    ranked = sorted(chunks, key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


def _rrf_fusion(dense_results: list, bm25_results: list, k: int = 60) -> list:
    """Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    all_ids: dict[str, dict] = {}

    for rank, item in enumerate(dense_results):
        doc_id = item["id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        all_ids[doc_id] = item

    for rank, item in enumerate(bm25_results):
        doc_id = item["id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        all_ids[doc_id] = item

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [all_ids[i] for i in sorted_ids]


def _detect_temporal(query: str) -> dict:
    """Detect year references in query."""
    year_match = re.search(r"\b(20\d{2})\b", query)
    if year_match:
        return {"temporal": True, "year_hint": int(year_match.group(1))}
    recent_words = ["recent", "latest", "new", "2024", "2023"]
    if any(w in query.lower() for w in recent_words):
        return {"temporal": True, "year_hint": datetime.now().year}
    return {"temporal": False, "year_hint": None}


async def rewrite_query(query: str, api_key: Optional[str] = None) -> dict:
    """Use Claude Haiku to rewrite the query for better retrieval."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    client = anthropic.Anthropic(api_key=key)
    temporal = _detect_temporal(query)

    system = (
        "You are a query rewriting assistant for a scientific paper search engine. "
        "Rewrite the user query to be more specific and suitable for semantic search over research papers. "
        "Return ONLY the rewritten query, nothing else."
    )
    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=128,
        system=system,
        messages=[{"role": "user", "content": query}],
    )
    rewritten = message.content[0].text.strip()
    return {
        "original": query,
        "rewritten": rewritten,
        "temporal": temporal["temporal"],
        "year_hint": temporal["year_hint"],
    }


def hybrid_retrieve(query: str, selected_ids: list[str] | None = None, top_k: int = 20) -> dict:
    """Dense + BM25 retrieval with RRF fusion."""
    collection = _get_collection()

    where_filter = None
    if selected_ids:
        if len(selected_ids) == 1:
            where_filter = {"arxiv_id": selected_ids[0]}
        else:
            where_filter = {"arxiv_id": {"$in": selected_ids}}

    # Dense retrieval
    dense_res = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count() or 1),
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    dense_chunks = []
    docs = dense_res["documents"][0] if dense_res["documents"] else []
    metas = dense_res["metadatas"][0] if dense_res["metadatas"] else []
    ids = dense_res["ids"][0] if dense_res["ids"] else []

    for doc_id, text, meta in zip(ids, docs, metas):
        dense_chunks.append({
            "id": doc_id,
            "text": text,
            "excerpt": text[:200].strip(),
            "title": meta.get("title", "Unknown"),
            "authors": meta.get("authors", ""),
            "year": meta.get("year"),
            "arxiv_id": meta.get("arxiv_id", ""),
            "chunk_type": meta.get("chunk_type", "body"),
        })

    # BM25 retrieval (over dense corpus for simplicity)
    tokenized = [d["text"].lower().split() for d in dense_chunks]
    if tokenized:
        bm25 = BM25Okapi(tokenized)
        bm25_scores = bm25.get_scores(query.lower().split())
        bm25_ranked = [dense_chunks[i] for i in sorted(range(len(bm25_scores)), key=lambda x: bm25_scores[x], reverse=True)]
    else:
        bm25_ranked = []

    fused = _rrf_fusion(dense_chunks, bm25_ranked)

    return {
        "dense_top": dense_chunks[:5],
        "bm25_top": bm25_ranked[:5],
        "fused_top": fused[:5],
        "dense_count": len(dense_chunks),
        "bm25_count": len(bm25_ranked),
        "fused_count": len(fused),
        "all_fused": fused,
    }


async def stream_answer(
    query: str,
    rewritten_query: str,
    chunks: list[dict],
    history: list[dict] | None = None,
    api_key: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Stream the final answer from Claude."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    client = anthropic.Anthropic(api_key=key)

    context_parts = []
    for i, chunk in enumerate(chunks[:8]):
        context_parts.append(
            f"[{i+1}] {chunk['title']} ({chunk.get('year', 'N/A')})\n"
            f"Authors: {chunk.get('authors', 'Unknown')}\n"
            f"{chunk['text'][:600]}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system = (
        "You are an expert AI research assistant. Answer questions using ONLY the provided research paper excerpts. "
        "Be precise, cite papers by number [1], [2], etc., and highlight key insights. "
        "If the answer isn't in the context, say so clearly."
    )

    messages = []
    if history:
        for msg in history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({
        "role": "user",
        "content": f"Context from papers:\n\n{context}\n\nQuestion: {query}",
    })

    # Decide model based on complexity
    word_count = len(query.split())
    model = "claude-sonnet-4-5" if word_count > 20 else "claude-haiku-4-5"

    with client.messages.stream(
        model=model,
        max_tokens=1500,
        system=system,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text

    # Yield model info as final SSE
    yield f"\n\n[MODEL_INFO]{json.dumps({'model': 'sonnet' if 'sonnet' in model else 'haiku', 'reason': f'query length={word_count} words'})}"
