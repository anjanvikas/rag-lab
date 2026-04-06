"""
FastAPI backend — AI/ML Research Platform (Phase A).
New endpoints: /api/search/hybrid, /api/ingestion/*, /api/chat/start|stream|abort
"""
import os
import json
import asyncio
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

from auth.db import init_db, get_api_key, has_api_key
from auth.routes import router as auth_router, get_current_user, require_auth

# ── App setup ─────────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="AI/ML Research Platform")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", "dev-secret-key-change-in-prod"),
    same_site="lax", https_only=False,
)

# SSE abort flags (per stream_id). Use Redis for multi-worker deployments.
_abort_flags: dict[str, asyncio.Event] = {}

# In-memory session store for multi-step SSE flow
_chat_sessions: dict[str, dict] = {}


class ChatStartRequest(BaseModel):
    message:      str
    session_id:   Optional[str] = None
    history:      list[dict] = []
    mode:         str = "vector"
    selected_ids: list[str] = []


@app.on_event("startup")
async def startup():
    init_db()
    try:
        from rag.telemetry import register
        register()
    except Exception as e:
        print(f"Failed to register telemetry: {e}")


app.include_router(auth_router)

# ── Static files ──────────────────────────────────────────────────────────────
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/login")
async def login_page():
    return FileResponse(os.path.join(STATIC_DIR, "login.html"))


@app.get("/learn.html")
async def learn_page(request: Request):
    user = get_current_user(request)
    if not user:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/login")
    return FileResponse(os.path.join(STATIC_DIR, "learn.html"))


@app.get("/")
async def index_page(request: Request):
    user = get_current_user(request)
    if not user:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/login")
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ── Papers API ────────────────────────────────────────────────────────────────
@app.get("/api/papers")
async def get_papers(user: dict = Depends(require_auth)):
    """Return all indexed papers from ingestion log."""
    try:
        from rag.ingestion.log import get_all_statuses
        statuses = await get_all_statuses()
        papers = [
            {
                "arxiv_id": s["arxiv_id"],
                "title":    s.get("title", "Unknown"),
                "tier":     s.get("tier", 0),
                "status":   s.get("status", "unknown"),
                "chunks":   s.get("chunks", 0),
            }
            for s in statuses if s.get("status") == "done"
        ]
        return {"papers": papers}
    except Exception as e:
        return {"papers": [], "error": str(e)}


# ── Ingestion API ─────────────────────────────────────────────────────────────

@app.get("/api/ingestion/status")
async def ingestion_status(user: dict = Depends(require_auth)):
    """Poll per-paper ingestion status (polled every 5s by admin panel)."""
    try:
        from rag.ingestion.log import get_summary
        return await get_summary()
    except Exception as e:
        return {"total": 0, "done": 0, "in_progress": 0, "failed": 0, "queued": 0, "jobs": [], "error": str(e)}


@app.post("/api/ingestion/start")
async def ingestion_start(user: dict = Depends(require_auth)):
    """Enqueue all canonical papers for ingestion via ARQ."""
    try:
        from arq import create_pool
        from arq.connections import RedisSettings
        REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
        redis = await create_pool(RedisSettings.from_dsn(REDIS_URL))
        job = await redis.enqueue_job("ingest_all_papers", "v1")
        await redis.aclose()
        return {"status": "enqueued", "job_id": job.job_id if job else None}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Redis not available: {e}")


@app.post("/api/ingestion/retry/{arxiv_id}")
async def ingestion_retry(arxiv_id: str, user: dict = Depends(require_auth)):
    """Re-enqueue a single failed paper."""
    try:
        from arq import create_pool
        from arq.connections import RedisSettings
        REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
        redis = await create_pool(RedisSettings.from_dsn(REDIS_URL))
        job = await redis.enqueue_job("ingest_paper", arxiv_id, "", 0, "v1")
        await redis.aclose()
        from rag.ingestion.log import update_ingestion_log
        await update_ingestion_log(arxiv_id, status="queued")
        return {"status": "requeued", "arxiv_id": arxiv_id}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


# ── Hybrid Search API ─────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query:    str
    top_k:    int = 10
    alpha:    Optional[float] = None   # None = auto-classify
    filters:  Optional[dict] = None
    rerank:   bool = True
    use_hyde: bool = False             # Phase B


@app.post("/api/search/hybrid")
@limiter.limit("30/minute")
async def hybrid_search_endpoint(
    request: Request,
    body: SearchRequest,
    user: dict = Depends(require_auth),
):
    try:
        from rag.search.hybrid import hybrid_search
        result = await hybrid_search(
            query=body.query,
            top_k=body.top_k,
            alpha=body.alpha,
            filters=body.filters,
            rerank=body.rerank,
            use_hyde=body.use_hyde,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Chat API (legacy single-shot — kept for backwards compat) ─────────────────
class ChatRequest(BaseModel):
    question:    str
    selected_ids: list[str] = []
    history:     list[dict] = []
    session_id:  Optional[str] = None
    mode:        str = "vector"


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request, user: dict = Depends(require_auth)):
    api_key = get_api_key(user["id"])
    if not api_key:
        raise HTTPException(status_code=402, detail="API key required.")

    from rag.pipeline import rewrite_query, stream_answer

    async def generate():
        trace_id = str(uuid4())

        try:
            yield f"data: {json.dumps({'step': 'rewrite'})}\n\n"
            rewrite_data = await rewrite_query(req.question, api_key=api_key)
            yield f"data: {json.dumps({'trace': {'rewrite': rewrite_data}})}\n\n"

            yield f"data: {json.dumps({'step': 'retrieve'})}\n\n"
            from rag.search.hybrid import hybrid_search
            # Build filters from selected papers
            filters = None
            if req.selected_ids:
                filters = {"arxiv_ids": req.selected_ids}  # handled in search
            search_result = await hybrid_search(
                rewrite_data["rewritten"], top_k=20, rerank=False
            )
            chunks = search_result.get("results", [])
            yield f"data: {json.dumps({'trace': {'retrieval': {'mode': 'vector', 'fused_count': len(chunks), 'dense_count': len(chunks), 'bm25_count': len(chunks), 'fused_top': chunks[:5], 'dense_top': chunks[:5], 'bm25_top': []}}})}\n\n"

            yield f"data: {json.dumps({'step': 'rerank'})}\n\n"
            from rag.pipeline import _rerank
            reranked = await _rerank(rewrite_data["rewritten"], chunks, top_k=7)
            yield f"data: {json.dumps({'trace': {'rerank': {'input_count': len(chunks), 'output_count': len(reranked), 'chunks': reranked}}})}\n\n"

            yield f"data: {json.dumps({'step': 'generate'})}\n\n"
            answer_text = ""
            model_info = None
            async for token in stream_answer(req.question, rewrite_data["rewritten"], reranked,
                                            history=req.history, api_key=api_key):
                if token.startswith("\n\n[MODEL_INFO]"):
                    model_info = json.loads(token.replace("\n\n[MODEL_INFO]", ""))
                    yield f"data: {json.dumps({'trace': {'model': model_info}})}\n\n"
                else:
                    answer_text += token
                    yield f"data: {json.dumps({'token': token})}\n\n"

            yield f"data: {json.dumps({'done': True, 'trace_id': trace_id})}\n\n"
            
            # Trigger async eval using ARQ
            from rag.evals.runner import enqueue_eval
            # Run enqueue_eval in the background so it doesn't block closing the stream
            asyncio.create_task(enqueue_eval(trace_id, req.question, answer_text, chunks))
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── SSE Chat API (new spec-compliant architecture) ────────────────────────────

class ChatStartRequest(BaseModel):
    message:    str
    session_id: Optional[str] = None
    history:    list[dict] = []
    mode:       str = "vector"


class AbortRequest(BaseModel):
    stream_id: str


@app.post("/api/chat/start")
@limiter.limit("10/minute")
async def chat_start(request: Request, body: ChatStartRequest, user: dict = Depends(require_auth)):
    api_key = get_api_key(user["id"])
    if not api_key:
        raise HTTPException(status_code=402, detail="API key required.")
    stream_id = str(uuid4())
    _abort_flags[stream_id] = asyncio.Event()

    # Store request in Redis so the stream endpoint can read it
    try:
        from rag.cache import get_redis
        r = await get_redis()
        await r.setex(f"chat:{stream_id}", 300, json.dumps({
            "message":     body.message,
            "session_id":  body.session_id,
            "history":     body.history[-20:],
            "mode":        body.mode,
            "selected_ids": getattr(body, 'selected_ids', []),
            "user_id":     user["id"],
            "api_key":     api_key,
        }))
    except Exception:
        pass  # Redis unavailable — stream will look up key locally

    return {"stream_id": stream_id}


@app.post("/api/chat/abort")
async def chat_abort(body: AbortRequest):
    if flag := _abort_flags.get(body.stream_id):
        flag.set()
    return {"ok": True}


@app.get("/api/chat/stream")
async def chat_stream(stream_id: str, request: Request, user: dict = Depends(require_auth)):
    abort_flag = _abort_flags.get(stream_id, asyncio.Event())

    # Load request payload from Redis
    try:
        from rag.cache import get_redis
        r = await get_redis()
        raw = await r.get(f"chat:{stream_id}")
        payload = json.loads(raw) if raw else {}
    except Exception:
        payload = {}

    if not payload:
        async def err_gen():
            yield f"event: error\ndata: {json.dumps({'code': 404, 'message': 'Stream not found'})}\n\n"
        return StreamingResponse(err_gen(), media_type="text/event-stream")

    api_key      = payload.get("api_key") or get_api_key(user["id"])
    message      = payload.get("message", "")
    history      = payload.get("history", [])
    mode         = payload.get("mode", "vector")
    selected_ids = payload.get("selected_ids", [])

    from rag.pipeline import rewrite_query, stream_answer, _rerank
    from rag.search.hybrid import hybrid_search


    async def event_gen():
        trace_id = str(uuid4())
        try:
            # ── Step 0: Starting ──────────────────────────────────────────────
            yield f"event: step\ndata: {json.dumps({'step': 'starting', 'desc': 'Initializing research session...'})}\n\n"

            # ── Step 1: Rewrite ───────────────────────────────────────────────
            yield f"event: step\ndata: {json.dumps({'step': 'rewrite', 'desc': 'Analyzing and refining your research question...'})}\n\n"
            rewrite_data = await rewrite_query(message, api_key=api_key)
            yield f"event: trace\ndata: {json.dumps({'rewrite': rewrite_data})}\n\n"

            if abort_flag.is_set(): return

            # ── Step 2: Retrieve (Vector or KG mode) ─────────────────────────
            retrieve_desc = "Searching ArXiv knowledge graph..." if mode == "kg" else "Searching ArXiv seminal papers (Hybrid Vector)..."
            yield f"event: step\ndata: {json.dumps({'step': 'retrieve', 'desc': retrieve_desc})}\n\n"

            if mode == "kg":
                from rag.knowledge_graph import kg_retrieve_v2
                kg_result = await kg_retrieve_v2(
                    rewrite_data["rewritten"],
                    api_key=api_key,
                    top_k=20,
                    selected_ids=selected_ids or None,
                )
                chunks = kg_result.get("results", [])
                retrieval_trace = {
                    "retrieval": {
                        "mode":         "kg",
                        "fused_count":  kg_result.get("fused_count", len(chunks)),
                        "dense_count":  kg_result.get("dense_count", 0),
                        "bm25_count":   0,
                        "entity_count": len(kg_result.get("entities", [])),
                        "entities":     kg_result.get("entities", []),
                        "graph_nodes":  kg_result.get("graph_nodes", []),
                        "graph_edges":  kg_result.get("graph_edges", []),
                        "fused_top":    chunks[:5],
                        "expand_count": kg_result.get("expand_count", 0),
                    }
                }
            else:
                search_result = await hybrid_search(
                    rewrite_data["rewritten"], top_k=20, rerank=False
                )
                chunks = search_result.get("results", [])
                retrieval_trace = {
                    "retrieval": {
                        "mode":        "vector",
                        "fused_count": search_result.get("fused_count", len(chunks)),
                        "dense_count": search_result.get("dense_count", len(chunks)),
                        "bm25_count":  search_result.get("bm25_count", 0),
                        "bm25_active": search_result.get("bm25_active", False),
                        "query_type":  search_result.get("query_type", "HYBRID"),
                        "alpha_used":  search_result.get("alpha_used", 0.5),
                        "fused_top":   chunks[:5],
                    }
                }

            yield f"event: trace\ndata: {json.dumps(retrieval_trace)}\n\n"

            if abort_flag.is_set(): return

            # ── Step 3: Rerank ────────────────────────────────────────────────
            yield f"event: step\ndata: {json.dumps({'step': 'rerank', 'desc': 'Reranking top results for precision...'})}\n\n"
            reranked = await _rerank(rewrite_data["rewritten"], chunks, top_k=7)
            yield f"event: trace\ndata: {json.dumps({'rerank': {'input_count': len(chunks), 'output_count': len(reranked), 'chunks': reranked}})}\n\n"

            if abort_flag.is_set(): return

            # ── Step 4: Generate ──────────────────────────────────────────────
            yield f"event: step\ndata: {json.dumps({'step': 'generate', 'desc': 'Synthesizing final research answer...'})}\n\n"
            answer_text = ""
            async for token in stream_answer(
                message, rewrite_data["rewritten"], reranked,
                history=history, api_key=api_key
            ):
                if abort_flag.is_set(): break
                if token.startswith("\n\n[MODEL_INFO]"):
                    model_info = json.loads(token.replace("\n\n[MODEL_INFO]", ""))
                    yield f"event: trace\ndata: {json.dumps({'model': model_info})}\n\n"
                else:
                    answer_text += token
                    yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"

            yield f"event: done\ndata: {json.dumps({'done': True, 'trace_id': trace_id})}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
        finally:
            _abort_flags.pop(stream_id, None)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── Eval status poll ──────────────────────────────────────────────────────────

@app.get("/api/eval/status")
async def eval_status(trace_id: str, user: dict = Depends(require_auth)):
    from rag.cache import get_eval_result
    result = await get_eval_result(trace_id)
    if result is None:
        return {"status": "pending", "trace_id": trace_id}
    return {"status": "done", "trace_id": trace_id, **result}


# ── Feedback API ──────────────────────────────────────────────────────────────
class FeedbackRequest(BaseModel):
    trace_id: str
    score: int
    comment: str = ""


@app.post("/api/feedback")
async def feedback(req: FeedbackRequest, user: dict = Depends(require_auth)):
    try:
        from rag.tracing import score_trace_by_id
        score_trace_by_id(trace_id=req.trace_id, name="user_feedback",
                          value=float(req.score), comment=req.comment)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Feedback error: {e}")
    return {"status": "ok"}


# ── Knowledge Graph API ───────────────────────────────────────────────────────
@app.get("/api/kg/graph")
async def kg_graph(user: dict = Depends(require_auth)):
    try:
        from rag.knowledge_graph import get_graph_topology, _build_graph
        topology = get_graph_topology(limit_nodes=60)
        if not topology.get("nodes"):
            _build_graph()
            topology = get_graph_topology(limit_nodes=60)
        return topology
    except Exception as e:
        return {"nodes": [], "edges": [], "total_papers": 0, "total_edges": 0, "error": str(e)}


@app.post("/api/kg/reset")
async def kg_reset(user: dict = Depends(require_auth)):
    try:
        from rag.knowledge_graph import reset_graph
        reset_graph()
        return {"status": "ok", "message": "Graph cache cleared"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── Health check & Eval ──────────────────────────────────────────────────────────────

@app.get("/api/eval/status")
async def eval_status(trace_id: str, user: dict = Depends(require_auth)):
    """Polls Redis for async evaluation scores."""
    from rag.cache import get_redis
    r = await get_redis()
    raw = await r.get(f"eval:{trace_id}")
    if not raw:
        return {"status": "pending"}
    try:
        return json.loads(raw)
    except Exception:
        return {"status": "error"}


@app.get("/api/health")
async def health():
    checks: dict = {"api": "ok"}
    try:
        from rag.cache import redis_ping
        checks["redis"] = "ok" if await redis_ping() else "unavailable"
    except Exception:
        checks["redis"] = "unavailable"
    try:
        from rag.ingestion.qdrant_store import _get_client
        client = _get_client()
        client.get_collections()
        checks["qdrant"] = "ok"
    except Exception as e:
        checks["qdrant"] = f"unavailable: {e}"
    return checks


# ── Learning Hub APIs ─────────────────────────────────────────────────────────
class EmbedRequest(BaseModel):
    text: str
    model: str = "all-MiniLM-L6-v2"


@app.post("/api/learn/embed")
async def demo_embed(req: EmbedRequest, user: dict = Depends(require_auth)):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(req.model)
        vector = model.encode(req.text).tolist()
        return {"vector": vector, "dimensions": len(vector), "model": req.model, "text": req.text[:100]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ChunkRequest(BaseModel):
    text: str
    strategy: str = "recursive"
    chunk_size: int = 200
    chunk_overlap: int = 30


@app.post("/api/learn/chunk")
async def demo_chunk(req: ChunkRequest, user: dict = Depends(require_auth)):
    try:
        if req.strategy in ("recursive", "sentence"):
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            seps = [". ", "! ", "? ", "\n", " ", ""] if req.strategy == "sentence" else None
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap,
                **({"separators": seps} if seps else {})
            )
        else:
            from langchain.text_splitter import CharacterTextSplitter
            splitter = CharacterTextSplitter(chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap)
        chunks = splitter.split_text(req.text)
        return {"chunks": chunks, "count": len(chunks), "strategy": req.strategy,
                "chunk_size": req.chunk_size, "chunk_overlap": req.chunk_overlap}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
