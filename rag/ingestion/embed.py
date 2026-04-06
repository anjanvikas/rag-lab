"""
Embedding helpers — Phase B hardened.

Dense:  fastembed BAAI/bge-small-en-v1.5 (384d, cosine, no API key)
Sparse: BM25Encoder — fitted to ArXiv corpus via scripts/tune_bm25.py
        Falls back to the generic MSMARCO model if corpus params not found.
"""
from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path
from typing import Any

EMBED_MODEL       = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
BM25_PARAMS_PATH  = Path(__file__).parent.parent.parent / "bm25_params.json"

_dense_model   = None
_bm25_encoder  = None


def _ensure_nltk():
    import nltk
    for pkg in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except Exception:
            nltk.download(pkg, quiet=True)


def _get_dense_model():
    global _dense_model
    if _dense_model is None:
        from fastembed import TextEmbedding
        _dense_model = TextEmbedding(model_name=EMBED_MODEL)
    return _dense_model


def _get_bm25():
    """
    Return BM25Encoder fitted to the ArXiv corpus (preferred) or the generic
    MSMARCO default.  The corpus-fitted encoder lives at bm25_params.json at
    the project root and is produced by running:

        python scripts/tune_bm25.py
    """
    global _bm25_encoder
    if _bm25_encoder is not None:
        return _bm25_encoder

    _ensure_nltk()
    from pinecone_text.sparse import BM25Encoder

    if BM25_PARAMS_PATH.exists():
        try:
            with open(BM25_PARAMS_PATH) as fh:
                params = json.load(fh)
            enc = BM25Encoder()
            enc.set_params(**params)
            _bm25_encoder = enc
            print(f"✅ BM25: loaded corpus-fitted params from {BM25_PARAMS_PATH}")
            return _bm25_encoder
        except Exception as ex:
            print(f"⚠️  BM25 params load failed ({ex}), falling back to MSMARCO default")

    _bm25_encoder = BM25Encoder.default()
    print("⚠️  BM25: using generic MSMARCO model.  Run scripts/tune_bm25.py for better recall.")
    return _bm25_encoder


def embed_texts_sync(texts: list[str]) -> list[list[float]]:
    model = _get_dense_model()
    return [v.tolist() for v in model.embed(texts)]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embed_texts_sync, texts)


async def embed_query(text: str) -> list[float]:
    from rag.cache import cached_embed, set_embed_cache
    cached = await cached_embed(text)
    if cached:
        return cached
    vecs = await embed_texts([text])
    vec = vecs[0]
    await set_embed_cache(text, vec)
    return vec


def extract_keywords(text: str, limit: int = 15) -> list[str]:
    """
    Extract technical keywords and acronyms from text for BM25 metadata.
    Uses regex to find capitalized technical terms, camelCase, and 
    common AI/ML terminology.
    """
    import re
    
    # 1. Capture acronyms (2-6 uppercase letters) — e.g. BERT, LoRA, RLHF
    acronyms = set(re.findall(r'\b[A-Z]{2,6}[s]?\b', text))
    
    # 2. Capture technical Proper Case words (e.g. Transformer, Attention)
    # Skipping common sentence starters by checking if they are in the mid-sentence
    technical_terms = set(re.findall(r'(?<!\. )\b[A-Z][a-z]{3,15}\b', text))
    
    # Clean up and filter
    STOP_WORDS = {"The", "This", "That", "They", "These", "There", "Here", "What", "When", "Where", "Why"}
    keywords = [k for k in acronyms | technical_terms if k not in STOP_WORDS]
    
    # Sort by length (desc) as a proxy for 'uniqueness' then take top N
    keywords.sort(key=len, reverse=True)
    return list(dict.fromkeys(keywords))[:limit]


def bm25_encode_documents(texts: list[str]) -> list[dict]:
    enc = _get_bm25()
    # Pinecone-text returns a generator or list of sparse vectors
    return enc.encode_documents(texts)


def bm25_encode_query(text: str) -> dict:
    enc = _get_bm25()
    result = enc.encode_queries(text)
    if hasattr(result, "indices"):
        return {"indices": list(result.indices), "values": list(result.values)}
    return result  # already a dict
