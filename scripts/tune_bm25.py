#!/usr/bin/env python3
"""
scripts/tune_bm25.py — Fit BM25Encoder to the ArXiv corpus stored in Qdrant.

WHY THIS MATTERS:
  The default BM25Encoder.default() is pre-trained on MSMARCO (web queries).
  ArXiv papers use domain-specific vocabulary: "LoRA", "RLHF", "cross-attention", etc.
  A corpus-fitted BM25 dramatically improves sparse recall for these terms.

USAGE:
  python scripts/tune_bm25.py              # fits and saves bm25_params.json
  python scripts/tune_bm25.py --limit 500  # fit on first 500 papers
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "papers-v1")
BM25_PARAMS_PATH = ROOT / "bm25_params.json"


def fetch_all_texts(limit: int = 1000) -> list[str]:
    """Scroll all chunk texts from Qdrant."""
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        print("❌ qdrant-client not installed. Run: pip install qdrant-client")
        sys.exit(1)

    client = QdrantClient(url=QDRANT_URL, timeout=30)
    texts = []
    offset = None

    print(f"Fetching chunks from Qdrant collection '{QDRANT_COLLECTION}'...")

    while True:
        try:
            results, offset = client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=250,
                offset=offset,
                with_vectors=False,
                with_payload=True,
            )
        except Exception as e:
            print(f"❌ Qdrant scroll error: {e}")
            break

        for point in results:
            text = (point.payload or {}).get("text", "")
            if text:
                texts.append(text)

        print(f"  ...fetched {len(texts)} chunks so far", end="\r")

        if offset is None or len(texts) >= limit:
            break

    print(f"\n✅ Loaded {len(texts)} chunks for BM25 fitting.")
    return texts


def fit_and_save(texts: list[str]) -> None:
    """Fit BM25Encoder on the corpus and save params to disk."""
    try:
        from pinecone_text.sparse import BM25Encoder
    except ImportError:
        print("❌ pinecone-text not installed. Run: pip install pinecone-text")
        sys.exit(1)

    import nltk
    for pkg in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except Exception:
            nltk.download(pkg, quiet=True)

    print(f"Fitting BM25Encoder on {len(texts)} texts...")
    bm25 = BM25Encoder()
    bm25.fit(texts)

    # Save params to disk
    params = bm25.get_params()
    with open(BM25_PARAMS_PATH, "w") as f:
        json.dump(params, f)

    print(f"✅ BM25 params saved to: {BM25_PARAMS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit BM25Encoder to ArXiv corpus")
    parser.add_argument("--limit", type=int, default=5000, help="Max chunks to use for fitting")
    args = parser.parse_args()

    texts = fetch_all_texts(limit=args.limit)
    if not texts:
        print("❌ No texts found. Make sure Qdrant is running and has indexed papers.")
        sys.exit(1)

    fit_and_save(texts)
    print("\n▶ Now restart the FastAPI server so it picks up the new BM25 params.")
