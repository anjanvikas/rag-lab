#!/usr/bin/env python3
"""
Langfuse Experiment Runner — Run eval dataset against the RAG pipeline
with automated LLM-as-a-Judge scoring.

Usage:
    python scripts/run_experiment.py
    python scripts/run_experiment.py --sample 3            # Quick test with 3 items
    python scripts/run_experiment.py --skip-judge           # Skip LLM scoring
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")


# ── LLM-as-a-Judge Prompts ─────────────────────────────────────────────────

FAITHFULNESS_PROMPT = """\
You are evaluating the faithfulness of an AI assistant's answer.
Question: {question}
Context: {context}
Answer: {answer}

Rate faithfulness 0-1 (1=all claims supported by context, 0=fabricated).
Return ONLY JSON: {{"score": <float>, "reasoning": "<explanation>"}}"""

RELEVANCY_PROMPT = """\
You are evaluating answer relevancy.
Question: {question}
Answer: {answer}

Rate 0-1 (1=directly addresses question, 0=off-topic).
Return ONLY JSON: {{"score": <float>, "reasoning": "<explanation>"}}"""

HELPFULNESS_PROMPT = """\
You are evaluating helpfulness.
Question: {question}
Answer: {answer}
Expected: {expected}

Rate 0-1 (1=comprehensive and matches expected, 0=unhelpful).
Return ONLY JSON: {{"score": <float>, "reasoning": "<explanation>"}}"""


def judge_with_llm(prompt: str) -> dict:
    """Call Claude Haiku to score a response."""
    import anthropic
    client = anthropic.Anthropic()
    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        if "{" in text and "}" in text:
            return json.loads(text[text.index("{"):text.rindex("}") + 1])
    except Exception as e:
        print(f"    ⚠️  Judge error: {e}")
    return {"score": 0.0, "reasoning": "Judge call failed"}


def run_rag_query(question: str) -> tuple[str, str]:
    """Run the pipeline for a single question. Returns (answer, context)."""
    import asyncio
    from rag.pipeline import rewrite_query, hybrid_retrieve, stream_answer, _rerank

    loop = asyncio.new_event_loop()
    rewrite_result = loop.run_until_complete(rewrite_query(question))
    rewritten = rewrite_result["rewritten"]

    retrieval = hybrid_retrieve(rewritten, top_k=20)
    fused = retrieval["all_fused"]
    reranked = _rerank(rewritten, fused, top_k=7)

    context = "\n".join(c.get("text", "")[:300] for c in reranked[:5])

    tokens = []
    async def collect():
        async for tok in stream_answer(question, rewritten, reranked):
            if not tok.startswith("\n\n[MODEL_INFO]"):
                tokens.append(tok)
    loop.run_until_complete(collect())
    loop.close()

    return "".join(tokens), context


def main():
    parser = argparse.ArgumentParser(description="Langfuse Experiment Runner")
    parser.add_argument("--sample", type=int, default=0, help="Only run N items")
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM-as-a-Judge scoring")
    parser.add_argument("--name", default=None, help="Custom experiment name")
    args = parser.parse_args()

    from langfuse import Langfuse
    lf = Langfuse()

    if not lf.auth_check():
        print("❌ Langfuse auth failed. Check your .env file.")
        return

    print("═" * 60)
    print("  Langfuse Experiment Runner")
    print("═" * 60)

    try:
        dataset = lf.get_dataset("arxiv_rag_eval_v1")
    except Exception as e:
        print(f"❌ Dataset not found: {e}")
        print("   Run: python scripts/setup_langfuse.py  first")
        return

    items = dataset.items
    if args.sample > 0:
        items = items[:args.sample]

    run_name = args.name or f"rag-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"\n📊 Dataset: arxiv_rag_eval_v1 ({len(items)} items)")
    print(f"🏷️  Run: {run_name}")
    print(f"🤖 Judge: {'disabled' if args.skip_judge else 'enabled'}\n")

    results = []

    for i, item in enumerate(items):
        question = item.input.get("question", str(item.input))
        expected = item.expected_output or ""

        print(f"  [{i+1}/{len(items)}] {question[:80]}...")

        try:
            with item.run(run_name=run_name, run_metadata={"experiment": run_name}) as span:
                start = time.time()
                answer, context = run_rag_query(question)
                latency = time.time() - start

                span.update_trace(input={"question": question}, output={"answer": answer[:500]})

                scores = {}
                if not args.skip_judge and answer.strip():
                    faith = judge_with_llm(FAITHFULNESS_PROMPT.format(
                        question=question, context=context[:3000], answer=answer[:2000]))
                    scores["faithfulness"] = faith["score"]
                    span.score_trace(name="faithfulness", value=faith["score"], comment=faith.get("reasoning", ""))

                    rel = judge_with_llm(RELEVANCY_PROMPT.format(question=question, answer=answer[:2000]))
                    scores["relevancy"] = rel["score"]
                    span.score_trace(name="answer_relevancy", value=rel["score"], comment=rel.get("reasoning", ""))

                    if expected:
                        helpf = judge_with_llm(HELPFULNESS_PROMPT.format(
                            question=question, answer=answer[:2000], expected=expected[:1000]))
                        scores["helpfulness"] = helpf["score"]
                        span.score_trace(name="helpfulness", value=helpf["score"], comment=helpf.get("reasoning", ""))

                span.score_trace(name="latency_s", value=round(latency, 2))

                result = {"question": question, "latency_s": round(latency, 2), "answer_len": len(answer), **scores}
                results.append(result)

                score_str = " | ".join(f"{k}={v:.2f}" for k, v in scores.items())
                print(f"    ✅ {latency:.1f}s | {len(answer)} chars | {score_str}")

        except Exception as e:
            print(f"    ❌ Error: {e}")
            results.append({"question": question, "error": str(e)})

    lf.flush()
    total_time = sum(r.get("latency_s", 0) for r in results)

    print(f"\n{'═' * 60}")
    print(f"  Experiment Complete: {run_name}")
    print(f"{'═' * 60}")
    print(f"  Items: {len(results)}  |  Total time: {total_time:.1f}s")

    for key in ["faithfulness", "relevancy", "helpfulness"]:
        vals = [r[key] for r in results if key in r]
        if vals:
            print(f"  Avg {key}: {sum(vals)/len(vals):.3f}")

    output_path = ROOT / "eval" / f"experiment_{run_name}.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"run_name": run_name, "results": results}, f, indent=2)
    print(f"\n  📄 Results: {output_path}")
    print(f"  🔗 Langfuse: Datasets → arxiv_rag_eval_v1 → {run_name}\n")


if __name__ == "__main__":
    main()
