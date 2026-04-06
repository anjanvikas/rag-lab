"""
RAG Pipeline — Phase A refactor.
ChromaDB removed. Retrieval now delegates to rag.search.hybrid (Pinecone).
Keeps: rewrite_query, stream_answer, _rerank (local cross-encoder).
"""
from __future__ import annotations
import os, json, asyncio
from typing import AsyncGenerator, Optional
from datetime import datetime
import re

import anthropic
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

_rerank_tokenizer = None
_rerank_model     = None


def _get_reranker():
    global _rerank_tokenizer, _rerank_model
    if _rerank_model is None:
        _rerank_tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL)
        _rerank_model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL)
        _rerank_model.eval()
    return _rerank_tokenizer, _rerank_model

def _local_rerank_sync(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    tokenizer, model = _get_reranker()
    texts = [c.get("excerpt", c.get("text", ""))[:512] for c in chunks]
    inputs = tokenizer(
        [query] * len(texts), texts,
        padding=True, truncation=True, max_length=512, return_tensors="pt",
    )
    with torch.no_grad():
        scores = model(**inputs).logits.squeeze(-1).tolist()
    if isinstance(scores, float):
        scores = [scores]
    for chunk, score in zip(chunks, scores):
        chunk["score"] = round(float(score), 4)
    return sorted(chunks, key=lambda x: x["score"], reverse=True)[:top_k]

async def _rerank(query: str, chunks: list[dict], top_k: int = 7) -> list[dict]:
    """Reranker: Uses Cohere v3 if API key present, otherwise falls back to local cross-encoder."""
    if not chunks:
        return []
        
    cohere_key = os.environ.get("COHERE_API_KEY")
    if cohere_key:
        try:
            import cohere
            co = cohere.AsyncClient(cohere_key)
            docs = [c.get("excerpt", c.get("text", ""))[:1024] for c in chunks]
            response = await co.rerank(
                query=query, documents=docs, top_n=top_k, model="rerank-english-v3.0"
            )
            ranked_chunks = []
            for r in response.results:
                chunk = dict(chunks[r.index])
                chunk["score"] = round(float(r.relevance_score), 4)
                ranked_chunks.append(chunk)
            return ranked_chunks
        except Exception as e:
            print(f"Cohere rerank failed: {e}. Falling back to local.")
            
    # Local fallback
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _local_rerank_sync, query, chunks, top_k)


def _detect_temporal(query: str) -> dict:
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
        model="claude-haiku-4-5", max_tokens=128,
        system=system, messages=[{"role": "user", "content": query}],
    )
    rewritten = message.content[0].text.strip()
    return {
        "original": query, "rewritten": rewritten,
        "temporal": temporal["temporal"], "year_hint": temporal["year_hint"],
    }


async def compress_history(history: list[dict], max_tokens: int = 6000, api_key: str = "") -> list[dict]:
    """
    Compress old conversation history when it exceeds budget.
    Always keeps last 4 messages (2 turns) verbatim.
    """
    if not history or len(history) <= 4:
        return history

    def approx_tokens(msgs: list[dict]) -> int:
        return sum(len(m.get("content", "")) // 4 for m in msgs)

    recent = history[-4:]
    older  = history[:-4]

    if approx_tokens(older) + approx_tokens(recent) <= max_tokens:
        return history

    # Summarise older turns
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    client = anthropic.Anthropic(api_key=key)
    older_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in older)
    summary_msg = client.messages.create(
        model="claude-haiku-4-5", max_tokens=300,
        system=(
            "Summarize this conversation in 3-5 sentences. "
            "Preserve all paper names, technical terms, and conclusions reached."
        ),
        messages=[{"role": "user", "content": older_text}],
    )
    summary = summary_msg.content[0].text.strip()
    return [
        {"role": "user",      "content": f"[Earlier conversation summary]: {summary}"},
        {"role": "assistant", "content": "Understood."},
        *recent,
    ]


async def stream_answer(
    query: str,
    rewritten_query: str,
    chunks: list[dict],
    history: list[dict] | None = None,
    api_key: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Stream final answer from Claude using retrieved chunks as context."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    client = anthropic.Anthropic(api_key=key)

    context_parts = []
    for i, chunk in enumerate(chunks[:8]):
        context_parts.append(
            f"[{i+1}] {chunk.get('title','Unknown')} ({chunk.get('year','N/A')})\n"
            f"Authors: {chunk.get('authors','Unknown')}\n"
            f"{chunk.get('text', chunk.get('excerpt',''))[:600]}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system = (
        "You are an expert AI research assistant specializing in ML/AI papers. "
        "Your goal is to provide **digestible, easy-to-understand** answers based ONLY on the provided excerpts. "
        "Structure your response with clear sections (e.g., # Overview, ## Key Findings). "
        "IMPORTANT: You MUST use double-newlines (\\n\\n) between every heading, paragraph, and list item to ensure proper layout. "
        "Use bullet points and bolding for technical terms to improve readability. "
        "Cite using inline notation [1], [2], etc. "
        "If the answer isn't in the context, say so clearly. "
        "Keep responses under 800 tokens unless a longer answer is clearly necessary."
    )

    messages: list[dict] = []
    if history:
        # Apply compression if needed
        compressed = await compress_history(history[-10:], api_key=key)
        for msg in compressed:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({
        "role": "user",
        "content": f"<retrieved_context>\n{context}\n</retrieved_context>\n\nQuestion: {query}",
    })

    word_count = len(query.split())
    model = "claude-sonnet-4-5" if word_count > 20 else "claude-haiku-4-5"

    with client.messages.stream(
        model=model, max_tokens=1500,
        system=system, messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text

    yield f"\n\n[MODEL_INFO]{json.dumps({'model': 'sonnet' if 'sonnet' in model else 'haiku', 'reason': f'query_words={word_count}'})}"
