#!/usr/bin/env python3
"""
One-time Langfuse setup script.
Uploads managed prompts and eval dataset to Langfuse.

Usage:
    python scripts/setup_langfuse.py
"""
from __future__ import annotations
import os, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

def main():
    from langfuse import Langfuse
    lf = Langfuse()

    # ── Auth check ───────────────────────────────────────────────────────
    ok = lf.auth_check()
    print(f"{'✅' if ok else '❌'} Auth check: {ok}")
    if not ok:
        return

    # ── Feature 4: Upload managed prompts ────────────────────────────────
    SYSTEM_PROMPT = """\
You are an expert AI research assistant with deep knowledge of machine learning \
and artificial intelligence. Answer the user's question using ONLY the provided \
context from research papers. Be precise and technical where appropriate.

Current date: {today}  ← use this when the user asks about "latest", "recent", or "new" work.
If the answer isn't in the context, say so clearly — do NOT hallucinate.
Always cite the paper title and ArXiv ID when referencing specific claims.

Context:
{context}"""

    REWRITE_PROMPT = """\
You are a query rewriting assistant for a scientific paper search engine. \
Rewrite the user query to be more specific and suitable for semantic search over \
research papers. Return ONLY the rewritten query, nothing else."""

    for name, text in [("rag_system_prompt", SYSTEM_PROMPT), ("rag_rewrite_prompt", REWRITE_PROMPT)]:
        try:
            lf.create_prompt(name=name, prompt=text, labels=["production"], type="text")
            print(f"✅ {name} uploaded")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"ℹ️  {name} already exists (skipped)")
            else:
                print(f"⚠️  {name}: {e}")

    # ── Feature 6: Upload eval dataset ───────────────────────────────────
    DATASET_NAME = "arxiv_rag_eval_v1"
    ITEMS = [
        {"input": {"question": "What is the Transformer architecture?", "category": "architecture"},
         "expected_output": "The Transformer is a neural network architecture based on self-attention."},
        {"input": {"question": "Explain self-attention in transformers", "category": "mechanism"},
         "expected_output": "Self-attention computes pairwise relationships between all positions in a sequence."},
        {"input": {"question": "What is LoRA and how does it work?", "category": "fine-tuning"},
         "expected_output": "LoRA (Low-Rank Adaptation) adds trainable low-rank matrices to frozen pretrained weights."},
        {"input": {"question": "How does attention work in transformers?", "category": "mechanism"},
         "expected_output": "Attention uses Q, K, V projections with scaled dot-product to compute weighted sums."},
        {"input": {"question": "What are the key contributions of GPT-3?", "category": "model"},
         "expected_output": "GPT-3 demonstrated that scaling language models improves few-shot learning."},
        {"input": {"question": "Explain RLHF in language models", "category": "training"},
         "expected_output": "RLHF trains a reward model from human preferences, then fine-tunes with PPO."},
        {"input": {"question": "What is retrieval-augmented generation?", "category": "architecture"},
         "expected_output": "RAG combines a retriever with a generator, grounding responses in external knowledge."},
        {"input": {"question": "How does BM25 compare to dense retrieval?", "category": "retrieval"},
         "expected_output": "BM25 uses exact term matching; dense retrieval uses learned embeddings for semantic similarity."},
        {"input": {"question": "What are mixture-of-experts models?", "category": "architecture"},
         "expected_output": "MoE models route inputs to specialized sub-networks, enabling sparse computation."},
        {"input": {"question": "Explain chain-of-thought prompting", "category": "prompting"},
         "expected_output": "Chain-of-thought prompting elicits step-by-step reasoning from LLMs."},
        {"input": {"question": "What is DPO and how does it differ from RLHF?", "category": "training"},
         "expected_output": "DPO directly optimizes a preference objective without a separate reward model."},
        {"input": {"question": "What are the latest advances in AI agent frameworks?", "category": "agents"},
         "expected_output": "Recent agent frameworks combine LLMs with tool use, planning loops, and memory."},
    ]

    try:
        lf.create_dataset(name=DATASET_NAME, description="Gold-standard QA pairs for ArXiv RAG evaluation")
        print(f"✅ Dataset '{DATASET_NAME}' created")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"ℹ️  Dataset '{DATASET_NAME}' already exists")
        else:
            print(f"⚠️  Dataset: {e}")

    count = 0
    for item in ITEMS:
        try:
            lf.create_dataset_item(
                dataset_name=DATASET_NAME,
                input=item["input"],
                expected_output=item.get("expected_output", ""),
            )
            count += 1
        except Exception:
            pass
    print(f"✅ Dataset: {count}/{len(ITEMS)} items uploaded")

    # ── Verify ───────────────────────────────────────────────────────────
    try:
        p = lf.get_prompt("rag_system_prompt")
        text = p.prompt if hasattr(p, "prompt") else str(p.compile())
        print(f"✅ System prompt loaded ({len(text)} chars)")
    except Exception as e:
        print(f"⚠️  Prompt verify: {e}")

    lf.flush()
    print("\n🎉 Langfuse setup complete!")


if __name__ == "__main__":
    main()
