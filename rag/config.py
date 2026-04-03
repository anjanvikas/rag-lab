"""
Centralised configuration for the RAG application.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Embedding / Retrieval ────────────────────────────────────────────────────
CHROMA_PATH      = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME  = os.getenv("COLLECTION_NAME", "arxiv_papers")
EMBED_MODEL      = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
RERANK_MODEL     = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── LLM ──────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Langfuse Observability ───────────────────────────────────────────────────
LANGFUSE_ENABLED    = os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST       = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# ── App Metadata (used in Langfuse tags) ─────────────────────────────────────
APP_VERSION = "4.1.0"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
