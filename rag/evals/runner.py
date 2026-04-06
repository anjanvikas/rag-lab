"""
Evaluation runner (Phase B).
Uses ARQ to run async evaluations on traces (faithfulness, citation accuracy).
"""
import asyncio
from arq.connections import RedisSettings
from rag.cache import get_redis

async def enqueue_eval(trace_id: str, question: str, answer: str, context: list[dict]):
    """Enqueue an async evaluation job in ARQ."""
    try:
        from arq import create_pool
        redis_pool = await create_pool(RedisSettings.from_dsn("redis://localhost:6379"))
        await redis_pool.enqueue_job("run_faithfulness_eval", trace_id, question, answer, context)
        await redis_pool.aclose()
    except Exception as e:
        print(f"Eval enqueue failed: {e}")

async def run_faithfulness_eval(ctx: dict, trace_id: str, question: str, answer: str, context: list[dict]) -> dict:
    """ARQ job to run LLM-as-a-judge faithfulness and citation accuracy eval."""
    import anthropic
    import os
    import json
    
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return {"status": "skipped - no key"}
        
    client = anthropic.AsyncAnthropic(api_key=key)
    context_text = "\n\n".join([f"[{i+1}] {c.get('text', '')}" for i, c in enumerate(context)])
    
    prompt = f"""
You are an expert evaluator assessing the faithfulness and citation accuracy of an AI response.
Question: {question}
Context:
{context_text}

Answer to evaluate:
{answer}

Assess the following:
1. Faithfulness (0.0 to 1.0): Does the answer solely rely on the context provided without hallucination?
2. Citation Accuracy (0.0 to 1.0): Are the citations (e.g. [1], [2]) placed correctly and do they support the statement?

Return ONLY a JSON object: {{"faithfulness": 0.9, "citation_accuracy": 0.8}}
"""
    try:
        response = await client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            temperature=0.0,
            system="You are an evaluation bot that strictly outputs JSON.",
            messages=[{"role": "user", "content": prompt}]
        )
        out = response.content[0].text.strip()
        if out.startswith("```json"):
            out = out[7:-3]
        scores = json.loads(out)
        
        # Save to Redis so the frontend can poll it
        r = await get_redis()
        await r.setex(f"eval:{trace_id}", 3600, json.dumps(scores))
        
        return scores
    except Exception as e:
        print(f"Faithfulness eval failed: {e}")
        return {"error": str(e)}
