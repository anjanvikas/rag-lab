#!/usr/bin/env python3
"""
ArXiv Paper Ingestion Script
Downloads and indexes AI/ML research papers from ArXiv into ChromaDB.

Uses fastembed (ONNX Runtime) instead of PyTorch-based sentence-transformers
to keep memory usage low (~150MB vs 1.5GB).

Usage:
    python scripts/ingest_arxiv.py              # 30 papers (default)
    python scripts/ingest_arxiv.py --count 50   # custom count
    python scripts/ingest_arxiv.py --reset      # reset collection first
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ArXiv categories to fetch
QUERIES = [
    "cat:cs.AI",
    "cat:cs.CL",   # NLP
    "cat:cs.LG",   # Machine Learning
    "cat:cs.CV",   # Computer Vision
    "cat:cs.IR",   # Information Retrieval
]

ARXIV_API = "https://export.arxiv.org/api/query"


def fetch_arxiv_papers(query: str, max_results: int = 10) -> list[dict]:
    """Fetch papers from ArXiv API."""
    url = (
        f"{ARXIV_API}?search_query={query}"
        f"&start=0&max_results={max_results}"
        f"&sortBy=submittedDate&sortOrder=descending"
    )
    try:
        req = urllib.request.urlopen(url, timeout=30)
        xml_data = req.read().decode("utf-8")
    except Exception as e:
        print(f"  ⚠️  ArXiv fetch error for {query}: {e}")
        return []

    root = ET.fromstring(xml_data)
    papers = []

    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        try:
            arxiv_id_url = (entry.find("{http://www.w3.org/2005/Atom}id").text or "")
            arxiv_id = arxiv_id_url.split("/abs/")[-1].strip()
            title = (entry.find("{http://www.w3.org/2005/Atom}title").text or "").replace("\n", " ").strip()
            abstract = (entry.find("{http://www.w3.org/2005/Atom}summary").text or "").replace("\n", " ").strip()

            authors = []
            for a in entry.findall("{http://www.w3.org/2005/Atom}author"):
                name = a.find("{http://www.w3.org/2005/Atom}name")
                if name is not None and name.text:
                    authors.append(name.text.strip())

            published = (entry.find("{http://www.w3.org/2005/Atom}published").text or "")[:10]
            year = published[:4] if published else ""

            categories = []
            for cat in entry.findall("{http://arxiv.org/schemas/atom}primary_category"):
                categories.append(cat.get("term", ""))
            if not categories:
                for cat in entry.findall("{http://www.w3.org/2005/Atom}category"):
                    categories.append(cat.get("term", ""))
            category = categories[0] if categories else query.replace("cat:", "")

            if title and abstract and arxiv_id:
                papers.append({
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "abstract": abstract,
                    "authors": ", ".join(authors[:5]),
                    "year": year,
                    "category": category,
                })
        except Exception:
            continue

    return papers


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 80) -> list[str]:
    """Simple chunking with overlap."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            last_period = chunk.rfind(". ")
            if last_period > chunk_size // 2:
                chunk = chunk[:last_period + 1]
        chunks.append(chunk.strip())
        start += len(chunk) - overlap
        if start >= len(text):
            break
    return [c for c in chunks if c.strip()]


def ingest(count: int = 30, reset: bool = False):
    from rag.config import CHROMA_PATH, COLLECTION_NAME
    import chromadb
    from fastembed import TextEmbedding

    # ONNX model — ~150MB RAM (vs 1.5GB for PyTorch sentence-transformers)
    FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"

    print(f"\n{'='*60}")
    print(f"  ArXiv RAG Ingestion Script")
    print(f"  Target  : {count} papers")
    print(f"  Embedder: {FASTEMBED_MODEL} (ONNX, low memory)")
    print(f"{'='*60}\n")

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    if reset:
        print("  🗑  Resetting collection...")
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        print("  ✅ Reset done\n")

    # Raw collection — we supply pre-computed embeddings
    col = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"  📦 Existing chunks: {col.count()}")

    # --- Step 1: Fetch paper metadata (network only, no model) ---
    papers_per_query = max(4, count // len(QUERIES))
    all_papers: dict = {}

    print(f"  📡 Fetching from ArXiv...")
    for q in QUERIES:
        print(f"    {q} (max {papers_per_query})...", end=" ", flush=True)
        fetched = fetch_arxiv_papers(q, max_results=papers_per_query)
        for p in fetched:
            all_papers[p["arxiv_id"]] = p
        print(f"{len(fetched)} papers")
        time.sleep(2)

    papers = list(all_papers.values())[:count]
    print(f"\n  📄 Unique papers fetched: {len(papers)}")

    if not papers:
        print("  ❌ No papers fetched. Check internet connection.")
        return

    # Filter already indexed
    existing = col.get(include=["metadatas"])
    existing_ids = {m.get("arxiv_id") for m in existing.get("metadatas", [])}
    new_papers = [p for p in papers if p["arxiv_id"] not in existing_ids]
    print(f"  🆕 New to index: {len(new_papers)}\n")

    if not new_papers:
        print("  ✅ All papers already indexed —", col.count(), "chunks total")
        return

    # --- Step 2: Load ONNX embedder ---
    print(f"  🔧 Loading ONNX embedding model…")
    embedder = TextEmbedding(model_name=FASTEMBED_MODEL)
    print("  ✅ Embedder ready\n")

    total_chunks = 0

    for idx, paper in enumerate(new_papers):
        try:
            text = (
                f"{paper['title']}\n\n"
                f"Authors: {paper['authors']}\n\n"
                f"{paper['abstract']}"
            )
            chunks = chunk_text(text)
            if not chunks:
                continue

            # fastembed.embed() → generator of numpy float32 arrays
            embeddings = [e.tolist() for e in embedder.embed(chunks)]

            docs, metas, ids = [], [], []
            for j, chunk in enumerate(chunks):
                docs.append(chunk)
                metas.append({
                    "arxiv_id": paper["arxiv_id"],
                    "title": paper["title"],
                    "authors": paper["authors"],
                    "year": paper["year"],
                    "category": paper["category"],
                    "chunk_index": j,
                    "total_chunks": len(chunks),
                    "excerpt": chunk[:150],
                })
                ids.append(f"{paper['arxiv_id']}_chunk_{j}")

            col.upsert(documents=docs, embeddings=embeddings, metadatas=metas, ids=ids)
            total_chunks += len(chunks)

            done = idx + 1
            pct = done * 30 // len(new_papers)
            bar = "█" * pct + "░" * (30 - pct)
            print(f"  [{bar}] {done}/{len(new_papers)} ✅ {paper['title'][:55]}")

        except Exception as e:
            print(f"  ⚠️  Error on {paper.get('arxiv_id', '?')}: {e}")

        gc.collect()

    print(f"\n{'='*60}")
    print(f"  ✅ Done!")
    print(f"     Papers indexed : {len(new_papers)}")
    print(f"     Chunks created : {total_chunks}")
    print(f"     DB total chunks: {col.count()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest ArXiv papers into ChromaDB")
    parser.add_argument("--count", type=int, default=30, help="Number of papers to fetch")
    parser.add_argument("--reset", action="store_true", help="Reset ChromaDB collection before indexing")
    args = parser.parse_args()
    ingest(count=args.count, reset=args.reset)
