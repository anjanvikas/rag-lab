"""
Knowledge Graph RAG Pipeline — Phase B Entity-Centric Architecture.

Old Architecture (Removed):
  - Built on ChromaDB (legacy). Expanded via shared keyword edges. Fragile.

New Architecture:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  User Query                                                          │
  │       │                                                              │
  │  [1] Entity Extraction (Claude Haiku)                               │
  │       │  → ["Transformer", "LoRA", "RLHF", "fine-tuning"]          │
  │       │                                                              │
  │  [2] Entity → Paper Lookup (Neo4j)                                  │
  │       │  MATCH (e:Entity)-[:MENTIONS]-(p:Paper)                     │
  │       │  → anchor paper set via structured knowledge                │
  │       │                                                              │
  │  [3] Dense Seed (Qdrant)                                            │
  │       │  → top-k papers by embedding similarity                     │
  │       │                                                              │
  │  [4] Multi-hop Expansion (Neo4j)                                    │
  │       │  MATCH (seed)-[:RELATES_TO|USES|EVALUATED_ON]-(neighbor)    │
  │       │  → traverse 1-2 hops to discover connected concepts         │
  │       │                                                              │
  │  [5] Chunk Hydration (Qdrant)                                       │
  │       │  → fetch actual text chunks for all gathered papers          │
  │       │                                                              │
  │  [6] Cross-Encoder Rerank                                           │
  │       └  → select top-7 most relevant chunks for context            │
  └─────────────────────────────────────────────────────────────────────┘

WHY THIS APPROACH:
  - Entity-centric: connects papers via WHAT they talk about, not keyword overlap.
  - Multi-hop: finds "hidden" connections (e.g. paper A uses Method B, paper C evaluates B).
  - Grounded: all connections come from LLM-extracted, confidence-filtered facts in Neo4j.
  - Graceful degradation: falls back to Qdrant-only if Neo4j is unavailable.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "rag_password")

_driver = None


def _get_neo4j_driver():
    global _driver
    if _driver is None:
        try:
            from neo4j import AsyncGraphDatabase
            _driver = AsyncGraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
        except Exception as e:
            logger.warning(f"Neo4j driver init failed: {e}")
            return None
    return _driver


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Entity Extraction
# ══════════════════════════════════════════════════════════════════════════════

async def extract_query_entities(query: str, api_key: str) -> list[str]:
    """
    Use Claude Haiku to extract key AI/ML concepts from the user's query.
    Returns a list of entity names (normalized to lowercase for Neo4j lookup).

    Example:
      Query:   "How does LoRA reduce memory usage compared to full fine-tuning?"
      Returns: ["lora", "fine-tuning", "memory efficiency", "parameter-efficient"]
    """
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=api_key)
    prompt = (
        f"Extract the key AI/ML technical concepts, methods, models, and tasks from this query.\n"
        f"Return ONLY a JSON array of lowercase strings, e.g. [\"lora\", \"fine-tuning\"].\n"
        f"Query: {query}"
    )
    try:
        response = await client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=256,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        entities = json.loads(text)
        return [str(e).lower().strip() for e in entities if e][:10]
    except Exception as e:
        logger.warning(f"Entity extraction failed: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 & 4: Neo4j Graph Traversal
# ══════════════════════════════════════════════════════════════════════════════

async def entity_to_paper_ids_neo4j(entities: list[str]) -> list[str]:
    """
    Step 2: Find papers that MENTION any of the extracted entities.
    Queries: (Entity)-[:MENTIONS]-(Paper)

    Returns a list of arxiv_ids.
    """
    driver = _get_neo4j_driver()
    if not driver or not entities:
        return []

    try:
        async with driver.session() as session:
            # Match entities by partial name (case insensitive) and find their papers
            entity_pattern = "|".join(f"(?i).*{e}.*" for e in entities[:5])
            result = await session.run(
                """
                MATCH (e:Entity)
                WHERE any(pattern IN $patterns WHERE e.name =~ pattern OR e.id =~ pattern)
                MATCH (p:Paper)-[:MENTIONS]->(e)
                RETURN DISTINCT p.arxiv_id AS arxiv_id, count(e) AS entity_hits
                ORDER BY entity_hits DESC
                LIMIT 15
                """,
                patterns=[f"(?i).*{e}.*" for e in entities[:5]],
            )
            records = await result.data()
            return [r["arxiv_id"] for r in records if r.get("arxiv_id")]
    except Exception as e:
        logger.warning(f"Neo4j entity lookup failed: {e}")
        return []


async def expand_papers_via_graph(seed_arxiv_ids: list[str], hops: int = 2) -> tuple[list[str], list[dict]]:
    """
    Step 4: Multi-hop expansion from seed papers via entity relationships.
    Traverses: (Paper)-[:MENTIONS]->(Entity)<-[:MENTIONS]-(Neighbor)
    and:        (Entity)-[*1..2]->(RelatedEntity)<-[:MENTIONS]-(Neighbor)

    Returns:
      - expanded arxiv_ids (including seeds)
      - edge list for UI visualization
    """
    driver = _get_neo4j_driver()
    if not driver or not seed_arxiv_ids:
        return seed_arxiv_ids, []

    all_ids = list(seed_arxiv_ids)
    edges = []

    try:
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (seed:Paper)
                WHERE seed.arxiv_id IN $seed_ids
                // Expand via shared entity mentions
                MATCH (seed)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(neighbor:Paper)
                WHERE neighbor.arxiv_id <> seed.arxiv_id
                WITH seed, neighbor, e, count(e) AS shared_entities
                WHERE shared_entities >= 1
                RETURN
                    seed.arxiv_id    AS from_id,
                    neighbor.arxiv_id AS to_id,
                    collect(e.name)[0..3] AS shared_concepts,
                    shared_entities
                ORDER BY shared_entities DESC
                LIMIT 30
                """,
                seed_ids=seed_arxiv_ids,
            )
            records = await result.data()

            for r in records:
                to_id = r.get("to_id")
                if to_id and to_id not in all_ids:
                    all_ids.append(to_id)
                if r.get("from_id") and to_id:
                    edges.append({
                        "source": r["from_id"],
                        "target": to_id,
                        "shared_concepts": r.get("shared_concepts", []),
                        "weight": r.get("shared_entities", 1),
                        "type": "entity_shared",
                    })

            # Second-hop: entity relationships (USES, EVALUATES, etc.)
            if hops >= 2 and seed_arxiv_ids:
                result2 = await session.run(
                    """
                    MATCH (seed:Paper)-[:MENTIONS]->(e1:Entity)-[rel]->(e2:Entity)<-[:MENTIONS]-(neighbor:Paper)
                    WHERE seed.arxiv_id IN $seed_ids
                    AND neighbor.arxiv_id NOT IN $seed_ids
                    AND type(rel) IN ['RELATES_TO','USES','EVALUATED_ON','IMPROVES']
                    WITH neighbor, count(DISTINCT e1) AS hop2_score
                    RETURN neighbor.arxiv_id AS to_id, hop2_score
                    ORDER BY hop2_score DESC
                    LIMIT 10
                    """,
                    seed_ids=seed_arxiv_ids,
                )
                records2 = await result2.data()
                for r in records2:
                    to_id = r.get("to_id")
                    if to_id and to_id not in all_ids:
                        all_ids.append(to_id)
                        edges.append({
                            "source": seed_arxiv_ids[0],
                            "target": to_id,
                            "shared_concepts": ["multi-hop"],
                            "weight": r.get("hop2_score", 1),
                            "type": "hop2",
                        })

    except Exception as e:
        logger.warning(f"Neo4j graph expansion failed: {e}")

    return all_ids, edges


# ══════════════════════════════════════════════════════════════════════════════
# Step 5: Chunk Hydration from Qdrant
# ══════════════════════════════════════════════════════════════════════════════

async def hydrate_chunks_from_qdrant(arxiv_ids: list[str], query: str, chunks_per_paper: int = 3) -> list[dict]:
    """
    Step 5: Given a list of arxiv_ids, fetch the most relevant text chunks
    from Qdrant using dense-only search filtered to those papers.

    This is the 'chunk hydration' step — we have identified the right papers
    via the graph, now we get the actual text to pass to the LLM.
    """
    if not arxiv_ids:
        return []

    from rag.ingestion.embed import embed_query
    from rag.search.hybrid import _get_client, QDRANT_COLLECTION

    try:
        from qdrant_client.models import Filter, FieldCondition, MatchAny

        client = _get_client()
        if client is None:
            return []

        dense_vec = await embed_query(query)
        arxiv_filter = Filter(must=[
            FieldCondition(key="arxiv_id", match=MatchAny(any=arxiv_ids[:20]))
        ])

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, lambda: client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=dense_vec,
            using="dense",
            limit=len(arxiv_ids) * chunks_per_paper,
            query_filter=arxiv_filter,
            with_payload=True,
        ))

        chunks = []
        seen_papers: dict[str, int] = {}
        for point in results.points:
            p = point.payload or {}
            arxiv_id = p.get("arxiv_id", "")
            if seen_papers.get(arxiv_id, 0) >= chunks_per_paper:
                continue
            seen_papers[arxiv_id] = seen_papers.get(arxiv_id, 0) + 1
            chunks.append({
                "chunk_id":     str(point.id),
                "paper_id":     p.get("paper_id", f"arxiv:{arxiv_id}"),
                "arxiv_id":     arxiv_id,
                "title":        p.get("title", ""),
                "authors":      p.get("authors", ""),
                "year":         p.get("year", ""),
                "tier":         p.get("tier", 0),
                "text":         p.get("text", ""),
                "excerpt":      p.get("text", "")[:300],
                "hybrid_score": round(getattr(point, "score", 0) or 0, 4),
                "rerank_score": None,
                "source":       "kg_hydrated",
            })

        return chunks

    except Exception as e:
        logger.error(f"Chunk hydration failed: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Main KG Retrieval Entry Point
# ══════════════════════════════════════════════════════════════════════════════

async def kg_retrieve_v2(
    query:        str,
    api_key:      str,
    top_k:        int = 20,
    selected_ids: list[str] | None = None,
) -> dict:
    """
    Entity-centric Knowledge Graph RAG retrieval.

    Pipeline:
      1. Extract query entities (LLM)
      2. Find anchor papers via Neo4j entity lookup
      3. Dense seed retrieval (Qdrant)
      4. Multi-hop expansion via entity graph (Neo4j)
      5. Chunk hydration for all gathered papers (Qdrant)
      6. Dedup + return for reranking

    Falls back to dense-only Qdrant search if Neo4j is unavailable.
    """
    # ── Step 1: Extract entities ──────────────────────────────────────────────
    entities = await extract_query_entities(query, api_key)
    logger.info(f"KG entities extracted: {entities}")

    # ── Step 2: Entity → Paper via Neo4j ─────────────────────────────────────
    entity_paper_ids = await entity_to_paper_ids_neo4j(entities)
    logger.info(f"KG entity papers: {entity_paper_ids}")

    # ── Step 3: Dense seed via Qdrant ─────────────────────────────────────────
    from rag.search.hybrid import hybrid_search
    seed_result = await hybrid_search(query, top_k=top_k, rerank=False)
    dense_chunks  = seed_result.get("results", [])
    seed_arxiv_ids = list({c["arxiv_id"] for c in dense_chunks if c.get("arxiv_id")})

    # ── Step 4: Multi-hop expansion via graph ─────────────────────────────────
    # Merge entity-anchor papers with dense seeds for richer expansion base
    anchor_ids = list(dict.fromkeys(entity_paper_ids + seed_arxiv_ids))[:15]
    expanded_ids, graph_edges = await expand_papers_via_graph(anchor_ids, hops=2)

    if selected_ids:
        expanded_ids = [i for i in expanded_ids if i in selected_ids] or expanded_ids

    # ── Step 5: Chunk hydration ───────────────────────────────────────────────
    kg_chunks = await hydrate_chunks_from_qdrant(expanded_ids, query, chunks_per_paper=3)

    # Merge: dense chunks + kg-hydrated chunks (deduplicated by chunk_id)
    seen_chunk_ids: set[str] = set()
    all_chunks: list[dict] = []
    for chunk in dense_chunks + kg_chunks:
        if chunk["chunk_id"] not in seen_chunk_ids:
            seen_chunk_ids.add(chunk["chunk_id"])
            all_chunks.append(chunk)

    # ── Build graph topology for UI ───────────────────────────────────────────
    graph_nodes = []
    seen_arxiv: set[str] = set()
    for chunk in all_chunks:
        aid = chunk.get("arxiv_id", "")
        if aid and aid not in seen_arxiv:
            seen_arxiv.add(aid)
            graph_nodes.append({
                "id":      aid,
                "label":   chunk.get("title", aid)[:60],
                "size":    12 if aid in seed_arxiv_ids else 8,
                "color":   "#63b3ed" if aid in seed_arxiv_ids else "#9f7aea",
                "is_seed": aid in seed_arxiv_ids,
                "is_entity_anchor": aid in entity_paper_ids,
            })

    logger.info(
        f"KG retrieve: {len(seed_arxiv_ids)} seed papers, "
        f"{len(expanded_ids)} expanded, {len(all_chunks)} total chunks"
    )

    return {
        "chunks":           all_chunks,
        "seed_count":       len(seed_arxiv_ids),
        "expand_count":     len(expanded_ids) - len(seed_arxiv_ids),
        "total_chunks":     len(all_chunks),
        "entities":         entities,
        "entity_papers":    entity_paper_ids,
        "graph_nodes":      graph_nodes,
        "graph_edges":      graph_edges,
        # Alias keys for reuse in traces
        "results":          all_chunks,
        "dense_count":      len(seed_arxiv_ids),
        "bm25_count":       0,
        "fused_count":      len(all_chunks),
        "fused_top":        all_chunks[:5],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Legacy-compatible wrappers (for backward compat with main.py)
# ══════════════════════════════════════════════════════════════════════════════

def kg_retrieve(query: str, selected_ids: list[str] | None = None, top_k: int = 20) -> dict:
    """Sync wrapper for legacy compatibility — prefer kg_retrieve_v2 where possible."""
    return {
        "chunks": [], "seed_count": 0, "expand_count": 0, "total_chunks": 0,
        "graph_nodes": [], "graph_edges": [], "results": [],
        "dense_count": 0, "bm25_count": 0, "fused_count": 0, "fused_top": [],
        "error": "Use kg_retrieve_v2 (async) instead."
    }


def get_graph_topology(limit_nodes: int = 50) -> dict:
    """Return graph topology from Neo4j for the UI Explorer."""
    import asyncio

    async def _fetch():
        driver = _get_neo4j_driver()
        if not driver:
            return {"nodes": [], "edges": [], "total_papers": 0, "total_edges": 0}
        try:
            async with driver.session() as session:
                result = await session.run(
                    """
                    MATCH (p:Paper)
                    OPTIONAL MATCH (p)-[:MENTIONS]->(e:Entity)
                    WITH p, count(e) AS entity_count
                    ORDER BY entity_count DESC
                    LIMIT $limit
                    RETURN p.arxiv_id AS id, p.title AS title, entity_count
                    """,
                    limit=limit_nodes,
                )
                papers = await result.data()

                edge_result = await session.run(
                    """
                    MATCH (p1:Paper)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(p2:Paper)
                    WHERE id(p1) < id(p2)
                    WITH p1, p2, count(e) AS shared, collect(e.name)[0..3] AS concepts
                    WHERE shared >= 2
                    RETURN p1.arxiv_id AS source, p2.arxiv_id AS target, shared, concepts
                    LIMIT 100
                    """
                )
                edges = await edge_result.data()

            nodes = [
                {
                    "id":     p["id"],
                    "label":  (p.get("title") or p["id"])[:60],
                    "size":   min(20, 6 + p.get("entity_count", 0)),
                    "color":  "#63b3ed",
                }
                for p in papers
            ]
            edge_list = [
                {"source": e["source"], "target": e["target"],
                 "weight": e["shared"], "shared_topics": e.get("concepts", [])}
                for e in edges
            ]
            return {
                "nodes": nodes,
                "edges": edge_list,
                "total_papers": len(nodes),
                "total_edges": len(edge_list),
            }
        except Exception as ex:
            logger.error(f"get_graph_topology failed: {ex}")
            return {"nodes": [], "edges": [], "total_papers": 0, "total_edges": 0}

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Can't nest event loops — return a blank for now; caller should use async
            return {"nodes": [], "edges": [], "total_papers": 0, "total_edges": 0}
        return loop.run_until_complete(_fetch())
    except Exception:
        return {"nodes": [], "edges": [], "total_papers": 0, "total_edges": 0}


def reset_graph():
    """No-op for compat — Neo4j is persistent, nothing to reset in memory."""
    global _driver
    _driver = None
    logger.info("Neo4j driver cleared — will reconnect on next request")
