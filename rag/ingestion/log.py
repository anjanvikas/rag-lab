"""
Redis-backed ingestion log.
Tracks per-paper status for the admin panel and /api/ingestion/status endpoint.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone

from rag.cache import get_redis


async def update_ingestion_log(
    arxiv_id: str,
    status: str,           # queued | in_progress | done | failed
    title: str = "",
    tier: int = 0,
    chunks: int = 0,
    kg_nodes: int = 0,
    kg_edges: int = 0,
    error: str = "",
) -> None:
    r = await get_redis()
    now = datetime.now(timezone.utc).isoformat()

    # Preserve existing title/tier if not provided
    existing_raw = await r.get(f"ingestion:paper:{arxiv_id}")
    existing: dict = json.loads(existing_raw) if existing_raw else {}

    data = {
        "arxiv_id": arxiv_id,
        "title": title or existing.get("title", ""),
        "tier": tier or existing.get("tier", 0),
        "status": status,
        "chunks": chunks or existing.get("chunks", 0),
        "kg_nodes": kg_nodes or existing.get("kg_nodes", 0),
        "kg_edges": kg_edges or existing.get("kg_edges", 0),
        "error": error,
        "updated_at": now,
        "created_at": existing.get("created_at", now),
    }

    await r.set(f"ingestion:paper:{arxiv_id}", json.dumps(data))
    await r.sadd("ingestion:papers", arxiv_id)


async def get_paper_status(arxiv_id: str) -> dict | None:
    r = await get_redis()
    raw = await r.get(f"ingestion:paper:{arxiv_id}")
    return json.loads(raw) if raw else None


async def get_all_statuses() -> list[dict]:
    r = await get_redis()
    ids = await r.smembers("ingestion:papers")
    statuses: list[dict] = []
    for aid in sorted(ids):
        raw = await r.get(f"ingestion:paper:{aid}")
        if raw:
            statuses.append(json.loads(raw))
    return statuses


async def get_summary() -> dict:
    statuses = await get_all_statuses()
    counts: dict[str, int] = {"queued": 0, "in_progress": 0, "done": 0, "failed": 0}
    for s in statuses:
        counts[s.get("status", "queued")] = counts.get(s.get("status", "queued"), 0) + 1
    return {
        "total": len(statuses),
        "done": counts["done"],
        "in_progress": counts["in_progress"],
        "failed": counts["failed"],
        "queued": counts["queued"],
        "jobs": statuses,
    }
