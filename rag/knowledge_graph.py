"""
Knowledge Graph Pipeline — Graph-enhanced RAG retrieval.

Architecture:
  ┌─────────────┐    ┌──────────────────────────────────────┐
  │  ChromaDB   │───▶│  Knowledge Graph (networkx DiGraph)  │
  │  (vectors)  │    │  Nodes: papers (arxiv_id, title...)   │
  └─────────────┘    │  Edges: shared topic keywords          │
                     └──────────────┬───────────────────────┘
                                    │
                     ┌──────────────▼───────────────────────┐
                     │  KG Retrieval:                        │
                     │  1. Dense seed (ChromaDB top-k)       │
                     │  2. Expand via graph neighbours       │
                     │  3. Cross-encoder rerank expanded set │
                     └──────────────────────────────────────┘
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ── Lazy globals ──────────────────────────────────────────────────────────────
_graph = None             # networkx DiGraph
_paper_index: dict = {}   # arxiv_id → paper metadata dict
_chunk_index: dict = {}   # arxiv_id → list[chunk dict]

# Stopwords to exclude from topic-keyword extraction
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "to", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "that", "this", "these", "those", "with", "for", "on", "at",
    "by", "from", "as", "it", "its", "we", "our", "their", "which", "who", "what", "how",
    "when", "where", "not", "no", "but", "if", "then", "than", "so", "yet", "after",
    "before", "can", "all", "also", "into", "over", "such", "more", "other", "between",
    "using", "used", "based", "results", "paper", "model", "models", "method", "methods",
    "approach", "proposed", "show", "shows", "shown", "novel", "present", "presents",
    "two", "three", "one", "new", "large", "high", "low", "use", "uses", "data", "task",
    "tasks", "performance", "training", "learning", "learned", "trained", "work",
    "works", "provide", "provides", "system", "systems", "research", "neural", "network",
    "networks", "deep", "study", "analysis", "evaluation", "experiment", "experiments",
}


# ── Graph construction ────────────────────────────────────────────────────────

def _extract_keywords(text: str, top_n: int = 20) -> set[str]:
    """Extract meaningful keywords from paper text via frequency + filtering."""
    words = re.findall(r'\b[a-zA-Z][a-zA-Z\-]{3,}\b', text.lower())
    freq: dict[str, int] = {}
    for w in words:
        if w not in _STOPWORDS and len(w) >= 4:
            freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq, key=lambda x: freq[x], reverse=True)
    return set(sorted_words[:top_n])


def _build_graph():
    """Build the knowledge graph from the ChromaDB collection."""
    global _graph, _paper_index, _chunk_index

    try:
        import networkx as nx
        from rag.pipeline import _get_collection
    except ImportError as e:
        logger.error(f"KG graph build failed — missing dependency: {e}")
        return None

    logger.info("Building knowledge graph from ChromaDB…")

    col = _get_collection()
    all_data = col.get(include=["documents", "metadatas"])

    docs   = all_data.get("documents", [])
    metas  = all_data.get("metadatas", [])
    ids_   = all_data.get("ids", [])

    # ── Index all papers and their chunks ────────────────────────────────────
    paper_texts: dict[str, str] = {}  # arxiv_id → concatenated chunk text

    for doc_id, text, meta in zip(ids_, docs, metas):
        arxiv_id = meta.get("arxiv_id", doc_id)

        if arxiv_id not in _paper_index:
            _paper_index[arxiv_id] = {
                "arxiv_id": arxiv_id,
                "title": meta.get("title", "Unknown"),
                "authors": meta.get("authors", ""),
                "year": meta.get("year", ""),
                "category": meta.get("category", ""),
            }
            _chunk_index[arxiv_id] = []
            paper_texts[arxiv_id] = ""

        _chunk_index[arxiv_id].append({
            "id": doc_id,
            "text": text,
            "excerpt": text[:200].strip(),
            "title": meta.get("title", "Unknown"),
            "authors": meta.get("authors", ""),
            "year": meta.get("year"),
            "arxiv_id": arxiv_id,
            "chunk_type": meta.get("chunk_type", "body"),
        })
        paper_texts[arxiv_id] = paper_texts.get(arxiv_id, "") + " " + text

    if not _paper_index:
        logger.warning("No papers found in ChromaDB for KG construction")
        return None

    # ── Extract keywords per paper ────────────────────────────────────────────
    paper_keywords: dict[str, set[str]] = {}
    for arxiv_id, text in paper_texts.items():
        paper_keywords[arxiv_id] = _extract_keywords(text, top_n=25)

    # ── Build graph ──────────────────────────────────────────────────────────
    G = nx.Graph()  # undirected — relationship is symmetric

    for arxiv_id, meta in _paper_index.items():
        G.add_node(
            arxiv_id,
            title=meta["title"],
            authors=meta["authors"],
            year=meta["year"],
            category=meta["category"],
        )

    paper_ids = list(_paper_index.keys())
    edge_threshold = 3  # minimum shared keywords to draw an edge

    for i in range(len(paper_ids)):
        for j in range(i + 1, len(paper_ids)):
            a, b = paper_ids[i], paper_ids[j]
            shared = paper_keywords.get(a, set()) & paper_keywords.get(b, set())
            if len(shared) >= edge_threshold:
                weight = len(shared)
                G.add_edge(a, b, weight=weight, shared_topics=list(shared)[:8])

    logger.info(
        f"KG built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )
    _graph = G
    return G


def _get_graph():
    """Return the KG singleton, building it if needed."""
    global _graph
    if _graph is None:
        _build_graph()
    return _graph


# ── KG Retrieval ──────────────────────────────────────────────────────────────

def kg_retrieve(
    query: str,
    selected_ids: list[str] | None = None,
    top_k: int = 20,
) -> dict:
    """
    Graph-enhanced retrieval:
      1. Dense vector seed   → top-k papers from ChromaDB
      2. Graph expansion     → add 1-hop neighbours of seed papers
      3. Return all chunks from expanded paper set for reranking
    """
    from rag.pipeline import _get_collection, _rrf_fusion
    from rag.pipeline import _rerank

    G = _get_graph()
    col = _get_collection()

    if col is None:
        return _empty_result()

    where_filter = None
    if selected_ids:
        if len(selected_ids) == 1:
            where_filter = {"arxiv_id": selected_ids[0]}
        else:
            where_filter = {"arxiv_id": {"$in": selected_ids}}

    # ── Step 1: Dense seed retrieval ─────────────────────────────────────────
    try:
        n_results = min(top_k, col.count() or 1)
        dense_res = col.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.error(f"KG dense retrieval failed: {e}")
        return _empty_result()

    docs  = dense_res["documents"][0] if dense_res["documents"] else []
    metas = dense_res["metadatas"][0] if dense_res["metadatas"] else []
    ids_  = dense_res["ids"][0]       if dense_res["ids"]       else []

    seed_chunks: list[dict] = []
    seed_paper_ids: set[str] = set()

    for doc_id, text, meta in zip(ids_, docs, metas):
        arxiv_id = meta.get("arxiv_id", doc_id)
        seed_paper_ids.add(arxiv_id)
        seed_chunks.append({
            "id": doc_id,
            "text": text,
            "excerpt": text[:200].strip(),
            "title": meta.get("title", "Unknown"),
            "authors": meta.get("authors", ""),
            "year": meta.get("year"),
            "arxiv_id": arxiv_id,
            "chunk_type": meta.get("chunk_type", "body"),
            "source": "dense",
        })

    # ── Step 2: Graph neighbour expansion ────────────────────────────────────
    expanded_paper_ids: set[str] = set(seed_paper_ids)
    graph_edges_used: list[dict] = []
    neighbour_chunks: list[dict] = []

    if G is not None:
        for paper_id in list(seed_paper_ids):
            if paper_id not in G:
                continue
            neighbors = sorted(
                G[paper_id].items(),
                key=lambda kv: kv[1].get("weight", 0),
                reverse=True,
            )[:3]  # top-3 neighbours per seed

            for neighbour_id, edge_data in neighbors:
                if neighbour_id in expanded_paper_ids:
                    continue
                # Skip if selected_ids filter is active and neighbour is not in it
                if selected_ids and neighbour_id not in selected_ids:
                    continue

                expanded_paper_ids.add(neighbour_id)
                graph_edges_used.append({
                    "from": paper_id,
                    "to": neighbour_id,
                    "weight": edge_data.get("weight", 1),
                    "shared_topics": edge_data.get("shared_topics", []),
                })

                # Add chunks from this neighbour paper
                for chunk in _chunk_index.get(neighbour_id, []):
                    ch = dict(chunk)
                    ch["source"] = "graph_expand"
                    neighbour_chunks.append(ch)

    # ── Combine + deduplicate ────────────────────────────────────────────────
    seen_ids: set[str] = set()
    all_chunks: list[dict] = []
    for ch in seed_chunks + neighbour_chunks:
        if ch["id"] not in seen_ids:
            seen_ids.add(ch["id"])
            all_chunks.append(ch)

    # ── Build graph topology for UI ──────────────────────────────────────────
    graph_nodes = []
    graph_edges = []

    if G is not None:
        for pid in expanded_paper_ids:
            meta = _paper_index.get(pid, {})
            graph_nodes.append({
                "id": pid,
                "title": meta.get("title", pid),
                "year": meta.get("year", ""),
                "category": meta.get("category", ""),
                "is_seed": pid in seed_paper_ids,
            })
        for edge in graph_edges_used:
            graph_edges.append(edge)

    return {
        "chunks": all_chunks,
        "seed_count": len(seed_paper_ids),
        "expand_count": len(expanded_paper_ids) - len(seed_paper_ids),
        "total_chunks": len(all_chunks),
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        # Also expose as hybrid-style keys for reuse in main.py
        "all_fused": all_chunks,
        "dense_count": len(seed_chunks),
        "bm25_count": 0,
        "fused_count": len(all_chunks),
        "dense_top": seed_chunks[:5],
        "bm25_top": [],
        "fused_top": all_chunks[:5],
    }


def _empty_result() -> dict:
    return {
        "chunks": [],
        "seed_count": 0,
        "expand_count": 0,
        "total_chunks": 0,
        "graph_nodes": [],
        "graph_edges": [],
        "all_fused": [],
        "dense_count": 0,
        "bm25_count": 0,
        "fused_count": 0,
        "dense_top": [],
        "bm25_top": [],
        "fused_top": [],
    }


# ── Graph topology API ────────────────────────────────────────────────────────

def get_graph_topology(limit_nodes: int = 50) -> dict:
    """
    Return a serialisable graph topology snapshot for the UI.
    Returns top `limit_nodes` most-connected papers.
    """
    G = _get_graph()
    if G is None or G.number_of_nodes() == 0:
        return {"nodes": [], "edges": [], "total_papers": 0, "total_edges": 0}

    # Pick top nodes by degree
    degrees = dict(G.degree())
    top_nodes = sorted(degrees, key=lambda n: degrees[n], reverse=True)[:limit_nodes]
    top_set = set(top_nodes)

    nodes = []
    for nid in top_nodes:
        meta = _paper_index.get(nid, {})
        nodes.append({
            "id": nid,
            "title": meta.get("title", nid)[:80],
            "year": meta.get("year", ""),
            "category": meta.get("category", ""),
            "degree": degrees.get(nid, 0),
        })

    edges = []
    for a, b, data in G.edges(data=True):
        if a in top_set and b in top_set:
            edges.append({
                "source": a,
                "target": b,
                "weight": data.get("weight", 1),
                "shared_topics": data.get("shared_topics", [])[:5],
            })

    return {
        "nodes": nodes,
        "edges": edges,
        "total_papers": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
    }


def reset_graph():
    """Force rebuild of the KG on next request."""
    global _graph, _paper_index, _chunk_index
    _graph = None
    _paper_index = {}
    _chunk_index = {}
    logger.info("Knowledge graph cache cleared")
