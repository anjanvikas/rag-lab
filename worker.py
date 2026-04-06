"""
ARQ async worker for paper ingestion.
Start with: arq worker.WorkerSettings
Trigger all: arq enqueue ingest_all_papers  (or POST /api/ingestion/start)
"""
from __future__ import annotations
import os
from arq.connections import RedisSettings

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# ── Canonical paper list (36 arXiv IDs from spec §1.1) ───────────────────────
CANONICAL_PAPERS = [
    # Tier 1 — Transformative Foundations
    {"arxiv_id": "1706.03762", "title": "Attention Is All You Need", "tier": 1},
    {"arxiv_id": "1810.04805", "title": "BERT: Pre-training of Deep Bidirectional Transformers", "tier": 1},
    {"arxiv_id": "2005.14165", "title": "GPT-3: Language Models are Few-Shot Learners", "tier": 1},
    {"arxiv_id": "2001.08361", "title": "Scaling Laws for Neural Language Models", "tier": 1},
    {"arxiv_id": "2201.11903", "title": "Chain-of-Thought Prompting Elicits Reasoning in LLMs", "tier": 1},
    {"arxiv_id": "2212.08073", "title": "Constitutional AI: Harmlessness from AI Feedback", "tier": 1},
    {"arxiv_id": "2303.08774", "title": "GPT-4 Technical Report", "tier": 1},
    {"arxiv_id": "2307.09288", "title": "Llama 2: Open Foundation and Fine-Tuned Chat Models", "tier": 1},
    {"arxiv_id": "2401.04088", "title": "Mixtral of Experts", "tier": 1},
    {"arxiv_id": "2412.19437", "title": "DeepSeek-V3 Technical Report", "tier": 1},
    # Tier 2 — Retrieval, RAG & Memory
    {"arxiv_id": "2004.04906", "title": "Dense Passage Retrieval for Open-Domain QA", "tier": 2},
    {"arxiv_id": "2005.11401", "title": "RAG: Retrieval-Augmented Generation for NLP Tasks", "tier": 2},
    {"arxiv_id": "1702.08734", "title": "Billion-Scale Similarity Search with GPUs (FAISS)", "tier": 2},
    {"arxiv_id": "2004.12832", "title": "ColBERT: Efficient and Effective Passage Search", "tier": 2},
    {"arxiv_id": "2212.10496", "title": "HyDE: Precise Zero-Shot Dense Retrieval", "tier": 2},
    {"arxiv_id": "2401.18059", "title": "RAPTOR: Recursive Abstractive Processing", "tier": 2},
    {"arxiv_id": "2404.16130", "title": "From Local to Global: A GraphRAG Approach", "tier": 2},
    # Tier 3 — Knowledge Graphs & Reasoning
    # Note: TransE (DOI 10.5555/2999792.2999923) has no arXiv ID — manual ingestion required
    {"arxiv_id": "1902.10197", "title": "RotatE: Knowledge Graph Embedding by Relational Rotation", "tier": 3},
    {"arxiv_id": "2110.15256", "title": "REBEL: Relation Extraction By End-to-end Language Generation", "tier": 3},
    {"arxiv_id": "2002.00388", "title": "A Survey on Knowledge Graphs", "tier": 3},
    {"arxiv_id": "2306.08302", "title": "Unifying Large Language Models and Knowledge Graphs: A Roadmap", "tier": 3},
    {"arxiv_id": "2402.07630", "title": "G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding", "tier": 3},
    # Tier 4 — Agents, RLHF & Alignment
    {"arxiv_id": "2203.02155", "title": "Training Language Models to Follow Instructions with RLHF", "tier": 4},
    {"arxiv_id": "2210.03629", "title": "ReAct: Synergizing Reasoning and Acting in LLMs", "tier": 4},
    {"arxiv_id": "2302.04761", "title": "Toolformer: Language Models Can Teach Themselves to Use Tools", "tier": 4},
    {"arxiv_id": "2308.03688", "title": "AgentBench: Evaluating LLMs as Agents", "tier": 4},
    {"arxiv_id": "2310.03714", "title": "DSPY: Compiling Declarative Language Model Calls", "tier": 4},
    {"arxiv_id": "2305.18290", "title": "Direct Preference Optimization (DPO)", "tier": 4},
    {"arxiv_id": "2406.04692", "title": "Mixture-of-Agents Enhances Large Language Model Capabilities", "tier": 4},
    # Tier 5 — Multimodal & Vision (text-only ingestion per spec §1.1)
    {"arxiv_id": "2103.00020", "title": "CLIP: Learning Transferable Visual Models From Natural Language Supervision", "tier": 5},
    {"arxiv_id": "2112.10752", "title": "High-Resolution Image Synthesis with Latent Diffusion Models", "tier": 5},
    {"arxiv_id": "2204.14198", "title": "Flamingo: a Visual Language Model for Few-Shot Learning", "tier": 5},
    {"arxiv_id": "2309.17421", "title": "GPT-4V(ision) System Card", "tier": 5},
]


# ── Job functions ─────────────────────────────────────────────────────────────

async def ingest_paper(ctx: dict, arxiv_id: str, title: str = "", tier: int = 0, version: str = "v1") -> dict:
    """
    Single paper ingestion job.
    Both Pinecone upsert and status log are updated atomically.
    ARQ will retry up to max_tries on failure.
    """
    from rag.ingestion.log import update_ingestion_log
    from rag.ingestion.fetch import fetch_paper, chunk_paper
    from rag.ingestion.qdrant_store import embed_and_upsert
    from rag.ingestion.neo4j_store import setup_schema, extract_and_write_kg

    await setup_schema()
    await update_ingestion_log(arxiv_id, status="in_progress", title=title, tier=tier)

    try:
        paper   = await fetch_paper(arxiv_id)
        chunks  = chunk_paper(paper)
        n_chunks = await embed_and_upsert(chunks, version=version)

        # Extract and write Knowledge Graph
        nodes_added, edges_added = await extract_and_write_kg(paper, chunks)

        # Invalidate search cache so new paper is immediately searchable
        from rag.cache import invalidate_search_cache
        await invalidate_search_cache()

        await update_ingestion_log(
            arxiv_id,
            status="done",
            title=paper.get("title", title),
            tier=paper.get("tier", tier),
            chunks=n_chunks,
            kg_nodes=nodes_added,
            kg_edges=edges_added,
        )
        return {"arxiv_id": arxiv_id, "chunks": n_chunks}

    except Exception as exc:
        await update_ingestion_log(arxiv_id, status="failed", title=title, tier=tier, error=str(exc))
        raise   # ARQ will retry


async def ingest_all_papers(ctx: dict, version: str = "v1") -> dict:
    """Enqueue one ingest_paper job per canonical paper."""
    from rag.ingestion.log import update_ingestion_log

    redis = ctx["redis"]
    enqueued = 0
    for paper in CANONICAL_PAPERS:
        # Pre-register as "queued" so admin panel shows all papers immediately
        await update_ingestion_log(
            paper["arxiv_id"],
            status="queued",
            title=paper["title"],
            tier=paper["tier"],
        )
        await redis.enqueue_job(
            "ingest_paper",
            paper["arxiv_id"],
            paper["title"],
            paper["tier"],
            version,
        )
        enqueued += 1

    return {"enqueued": enqueued}


async def retry_paper(ctx: dict, arxiv_id: str, version: str = "v1") -> dict:
    """Re-enqueue a single failed paper."""
    redis = ctx["redis"]
    await redis.enqueue_job("ingest_paper", arxiv_id, "", 0, version)
    return {"arxiv_id": arxiv_id, "status": "requeued"}


# ── Worker settings ───────────────────────────────────────────────────────────

class WorkerSettings:
    from rag.evals.runner import run_faithfulness_eval
    functions   = [ingest_paper, ingest_all_papers, retry_paper, run_faithfulness_eval]
    redis_settings = RedisSettings.from_dsn(REDIS_URL)
    max_jobs    = 5          # cap concurrent arXiv + Pinecone calls
    job_timeout = 300        # 5 min max per paper
    max_tries   = 3          # retry failed jobs
    keep_result = 86400      # keep job results 24h
