"""Redis async cache helpers."""
import hashlib, json, os
import redis.asyncio as aioredis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
_redis: aioredis.Redis | None = None

async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis

def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

async def cached_embed(text: str) -> list[float] | None:
    r = await get_redis()
    raw = await r.get(f"embed:{_sha(text)}")
    return json.loads(raw) if raw else None

async def set_embed_cache(text: str, vector: list[float]) -> None:
    r = await get_redis()
    await r.setex(f"embed:{_sha(text)}", 86400, json.dumps(vector))

async def cached_search(query: str, params: dict) -> dict | None:
    r = await get_redis()
    key = f"search:{_sha(query + json.dumps(params, sort_keys=True))}"
    raw = await r.get(key)
    return json.loads(raw) if raw else None

async def set_search_cache(query: str, params: dict, result: dict) -> None:
    r = await get_redis()
    key = f"search:{_sha(query + json.dumps(params, sort_keys=True))}"
    await r.setex(key, 3600, json.dumps(result))

async def invalidate_search_cache() -> None:
    r = await get_redis()
    for pattern in ("search:*", "kg:*"):
        keys = await r.keys(pattern)
        if keys:
            await r.delete(*keys)

async def get_eval_result(trace_id: str) -> dict | None:
    r = await get_redis()
    raw = await r.get(f"eval:{trace_id}")
    return json.loads(raw) if raw else None

async def set_eval_result(trace_id: str, data: dict) -> None:
    r = await get_redis()
    await r.setex(f"eval:{trace_id}", 172800, json.dumps(data))

async def redis_ping() -> bool:
    try:
        r = await get_redis()
        return bool(await r.ping())
    except Exception:
        return False
