"""
Qdrant vector store for Phase A.
Dense: BAAI/bge-large-en-v1.5 (1024d, cosine)
Sparse: BM25 (pinecone-text BM25Encoder)
Hybrid: Qdrant native RRF fusion
"""
from __future__ import annotations
import hashlib, os, uuid
from typing import Any

QDRANT_URL        = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "papers-v1")
EMBED_DIM         = int(os.getenv("EMBED_DIM", "384"))  # bge-small=384, bge-large=1024

_client = None


def _get_client():
    global _client
    if _client is None:
        from qdrant_client import QdrantClient
        _client = QdrantClient(url=QDRANT_URL, timeout=30)
    return _client


def setup_collection() -> None:
    """Create Qdrant collection if it doesn't exist. Run via scripts/setup_qdrant.py."""
    from qdrant_client.models import (
        VectorParams, Distance, SparseVectorParams, SparseIndexParams,
    )
    client = _get_client()
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={"dense": VectorParams(size=EMBED_DIM, distance=Distance.COSINE)},
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            },
        )
        print(f"✅ Created Qdrant collection '{QDRANT_COLLECTION}' (dim={EMBED_DIM})")
    else:
        print(f"✅ Qdrant collection '{QDRANT_COLLECTION}' already exists")


def _chunk_uuid(paper_id: str, chunk_index: int) -> str:
    raw = f"{paper_id}:{chunk_index}"
    hex_str = hashlib.sha256(raw.encode()).hexdigest()[:32]
    return str(uuid.UUID(hex_str))


async def embed_and_upsert(chunks: list[dict], version: str = "v1") -> int:
    """Embed chunks and upsert to Qdrant with dense + sparse vectors."""
    if not chunks:
        return 0

    from qdrant_client.models import PointStruct, SparseVector as QSparseVec
    from rag.ingestion.embed import embed_texts, bm25_encode_documents, extract_keywords

    texts      = [c["text"] for c in chunks]
    dense_vecs = await embed_texts(texts)
    sparse_raw = bm25_encode_documents(texts)

    client = _get_client()
    points = []
    for chunk, dense, sparse in zip(chunks, dense_vecs, sparse_raw):
        cid = _chunk_uuid(chunk["paper_id"], chunk["chunk_index"])

        # Normalise sparse vector from pinecone-text output
        if hasattr(sparse, "indices"):
            sp_indices = [int(i) for i in sparse.indices]
            sp_values  = [float(v) for v in sparse.values]
        elif isinstance(sparse, dict):
            sp_indices = [int(i) for i in sparse["indices"]]
            sp_values  = [float(v) for v in sparse["values"]]
        else:
            sp_indices, sp_values = [], []

        # Extract local keywords for metadata-based filtering/boost
        kw = extract_keywords(chunk["text"])

        payload = {
            "paper_id":    chunk.get("paper_id", ""),
            "arxiv_id":    chunk.get("arxiv_id", ""),
            "title":       chunk.get("title", "")[:512],
            "authors":     chunk.get("authors", "")[:256],
            "year":        str(chunk.get("year", "")),
            "tier":        int(chunk.get("tier", 0)),
            "section":     chunk.get("section", ""),
            "chunk_type":  chunk.get("chunk_type", "body"),
            "chunk_index": int(chunk.get("chunk_index", 0)),
            "venue":       chunk.get("venue", ""),
            "text":        chunk["text"][:3000], # Increased for richer context
            "keywords":    kw,
            "is_seminal":  chunk.get("is_seminal", False),
            "version":     version,
        }

        points.append(PointStruct(
            id=cid,
            vector={
                "dense":  dense,
                "sparse": QSparseVec(indices=sp_indices, values=sp_values),
            },
            payload=payload,
        ))

    # Upsert in batches of 50 (safer for large payloads)
    BATCH = 50
    for i in range(0, len(points), BATCH):
        client.upsert(collection_name=QDRANT_COLLECTION, points=points[i:i+BATCH])

    return len(points)
