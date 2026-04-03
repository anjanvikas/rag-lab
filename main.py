"""
FastAPI backend for the RAG Learning Platform.
Serves static files, auth routes, and API endpoints.
"""
import os
import json
import asyncio
from typing import Optional
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from auth.db import init_db, get_api_key, has_api_key
from auth.routes import router as auth_router, get_current_user, require_auth

# ── Lazy RAG imports (avoid loading models at startup if not needed) ─────────
_pipeline_loaded = False

def _load_pipeline():
    global _pipeline_loaded, rewrite_query, hybrid_retrieve, stream_answer, _get_collection
    if not _pipeline_loaded:
        from rag.pipeline import rewrite_query, hybrid_retrieve, stream_answer, _get_collection
        _pipeline_loaded = True

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Learning Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB on startup
@app.on_event("startup")
async def startup():
    init_db()

# Include auth router
app.include_router(auth_router)

# ── Static files ─────────────────────────────────────────────────────────────
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

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
    """Return all indexed papers."""
    try:
        _load_pipeline()
        col = _get_collection()
        results = col.get(include=["metadatas"])
        seen = {}
        for meta in results.get("metadatas", []):
            aid = meta.get("arxiv_id", "")
            if aid and aid not in seen:
                seen[aid] = {
                    "arxiv_id": aid,
                    "title": meta.get("title", "Unknown"),
                    "authors": meta.get("authors", ""),
                    "year": meta.get("year"),
                    "category": meta.get("category", ""),
                }
        papers = sorted(seen.values(), key=lambda p: p.get("year") or 0, reverse=True)
        return {"papers": papers}
    except Exception as e:
        return {"papers": [], "error": str(e)}


# ── Chat API ──────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    selected_ids: list[str] = []
    history: list[dict] = []
    session_id: Optional[str] = None


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request, user: dict = Depends(require_auth)):
    """Stream chat response using RAG pipeline."""
    # Get user's API key
    api_key = get_api_key(user["id"])
    if not api_key:
        raise HTTPException(status_code=402, detail="API key required. Please add your Anthropic API key.")

    _load_pipeline()

    async def generate():
        trace = {}
        trace_id = 'noop'
        try:
            from rag.tracing import create_trace as _ct, flush as _flush_traces
            langfuse_trace = _ct(
                name=f"chat_pipeline",
                session_id=req.session_id,
                metadata={"question": req.question[:200], "paper_filter": bool(req.selected_ids)},
                tags=["chat"],
            )
            trace_id = getattr(langfuse_trace, 'trace_id', 'noop')
        except Exception:
            pass
        try:
            # Step 1: Rewrite query
            yield f"data: {json.dumps({'step': 'rewrite'})}\n\n"
            rewrite_data = await rewrite_query(req.question, api_key=api_key)
            trace["rewrite"] = rewrite_data
            yield f"data: {json.dumps({'trace': {'rewrite': rewrite_data}})}\n\n"

            # Step 2: Hybrid retrieval
            yield f"data: {json.dumps({'step': 'retrieve'})}\n\n"
            retrieval = hybrid_retrieve(
                rewrite_data["rewritten"],
                selected_ids=req.selected_ids or None,
            )
            trace["retrieval"] = {
                "dense_count": retrieval["dense_count"],
                "bm25_count": retrieval["bm25_count"],
                "fused_count": retrieval["fused_count"],
                "dense_top": retrieval["dense_top"],
                "bm25_top": retrieval["bm25_top"],
                "fused_top": retrieval["fused_top"],
            }
            yield f"data: {json.dumps({'trace': {'retrieval': trace['retrieval']}})}\n\n"

            # Step 3: Rerank
            yield f"data: {json.dumps({'step': 'rerank'})}\n\n"
            from rag.pipeline import _rerank
            reranked = _rerank(rewrite_data["rewritten"], retrieval["all_fused"])
            trace["rerank"] = {
                "input_count": len(retrieval["all_fused"]),
                "output_count": len(reranked),
                "chunks": reranked,
            }
            yield f"data: {json.dumps({'trace': {'rerank': trace['rerank']}})}\n\n"

            # Step 4: Generate answer
            yield f"data: {json.dumps({'step': 'generate'})}\n\n"
            answer_text = ""
            model_info = None

            async for token in stream_answer(
                req.question,
                rewrite_data["rewritten"],
                reranked,
                history=req.history,
                api_key=api_key,
            ):
                if token.startswith("\n\n[MODEL_INFO]"):
                    model_info = json.loads(token.replace("\n\n[MODEL_INFO]", ""))
                    trace["model"] = model_info
                    yield f"data: {json.dumps({'trace': {'model': model_info}})}\n\n"
                else:
                    answer_text += token
                    yield f"data: {json.dumps({'token': token})}\n\n"

            try:
                _flush_traces()
            except Exception:
                pass
            yield f"data: {json.dumps({'done': True, 'trace_id': trace_id})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Feedback API ──────────────────────────────────────────────────────────────
class FeedbackRequest(BaseModel):
    trace_id: str
    score: int
    comment: str = ""

@app.post("/api/feedback")
async def feedback(req: FeedbackRequest, user: dict = Depends(require_auth)):
    """Score a trace in Langfuse with user feedback."""
    try:
        from rag.tracing import score_trace_by_id
        score_trace_by_id(
            trace_id=req.trace_id,
            name="user_feedback",
            value=float(req.score),
            comment=req.comment,
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Feedback error: {e}")
    return {"status": "ok"}


# ── Learning Hub APIs ────────────────────────────────────────────────────────
class EmbedRequest(BaseModel):
    text: str
    model: str = "all-MiniLM-L6-v2"

@app.post("/api/learn/embed")
async def demo_embed(req: EmbedRequest, user: dict = Depends(require_auth)):
    """Generate a real embedding vector for learning demo."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(req.model)
        vector = model.encode(req.text).tolist()
        return {
            "vector": vector,
            "dimensions": len(vector),
            "model": req.model,
            "text": req.text[:100],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ChunkRequest(BaseModel):
    text: str
    strategy: str = "recursive"
    chunk_size: int = 200
    chunk_overlap: int = 30

@app.post("/api/learn/chunk")
async def demo_chunk(req: ChunkRequest, user: dict = Depends(require_auth)):
    """Chunk text using specified strategy for learning demo."""
    try:
        if req.strategy == "recursive":
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=req.chunk_size,
                chunk_overlap=req.chunk_overlap,
            )
        elif req.strategy == "sentence":
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=req.chunk_size,
                chunk_overlap=req.chunk_overlap,
                separators=[". ", "! ", "? ", "\n", " ", ""],
            )
        else:
            from langchain.text_splitter import CharacterTextSplitter
            splitter = CharacterTextSplitter(
                chunk_size=req.chunk_size,
                chunk_overlap=req.chunk_overlap,
            )

        chunks = splitter.split_text(req.text)
        return {
            "chunks": chunks,
            "count": len(chunks),
            "strategy": req.strategy,
            "chunk_size": req.chunk_size,
            "chunk_overlap": req.chunk_overlap,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
