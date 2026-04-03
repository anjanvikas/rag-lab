"""
Langfuse tracing wrapper for the RAG application.

Features:
  1. Tracing          – nested spans for every pipeline step
  2. Sessions         – session_id grouping for multi-turn conversations
  3. User Feedback    – score_trace for thumbs up/down from frontend
  4. Prompt Mgmt      – get_prompt / upload_prompt with cache
  5. Datasets         – create_dataset / upload_dataset_items
  6. Cost Tracking    – usage_details in generations
  7. Release/Env Tags – auto-tagged from config
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

# ── Lazy singleton ───────────────────────────────────────────────────────────
_client = None


def _get_client():
    """Return the Langfuse client singleton (or None if disabled)."""
    global _client
    if _client is not None:
        return _client

    from rag.config import LANGFUSE_ENABLED, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST

    if not LANGFUSE_ENABLED:
        logger.info("Langfuse disabled via LANGFUSE_ENABLED=false")
        return None
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        logger.warning("Langfuse keys missing — tracing disabled")
        return None

    try:
        from langfuse import Langfuse
        _client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
        logger.info("Langfuse client initialised")
        return _client
    except Exception as e:
        logger.error(f"Langfuse init failed: {e}")
        return None


# ── No-op fallback ───────────────────────────────────────────────────────────
class _NoOpSpan:
    """Silent no-op so callers never crash when Langfuse is off."""
    trace_id = "noop"
    def start_span(self, **kw):       return _NOOP
    def start_observation(self, **kw): return _NOOP
    def update(self, **kw):           return self
    def update_trace(self, **kw):     return self
    def end(self, **kw):              return self
    def score_trace(self, **kw):      return self
    def __enter__(self):              return self
    def __exit__(self, *a):           pass

_NOOP = _NoOpSpan()


# ═════════════════════════════════════════════════════════════════════════════
#  FEATURE 1 + 2 + 8: Tracing + Sessions + Release/Env Tags
# ═════════════════════════════════════════════════════════════════════════════

def create_trace(
    name: str,
    user_id: str = "anonymous",
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
):
    """
    Create a trace for a user request.

    Features integrated:
      - Sessions:  pass session_id to group conversation turns
      - Release:   automatically tagged with APP_VERSION
      - Env:       automatically tagged with ENVIRONMENT
    """
    from rag.config import APP_VERSION, ENVIRONMENT

    client = _get_client()
    if client is None:
        return _NOOP

    try:
        all_tags = list(tags or [])
        all_tags.append(ENVIRONMENT)

        full_metadata = dict(metadata or {})
        full_metadata["app_version"] = APP_VERSION
        full_metadata["environment"] = ENVIRONMENT

        root = client.start_span(
            name=name,
            input=full_metadata,
        )

        # Set trace-level properties (session, user, release, tags)
        trace_update = {
            "name": name,
            "metadata": full_metadata,
            "tags": all_tags,
        }
        if session_id:
            trace_update["session_id"] = session_id
        if user_id:
            trace_update["user_id"] = user_id

        root.update_trace(**trace_update)
        return root

    except Exception as e:
        logger.error(f"create_trace failed: {e}")
        return _NOOP


# ═════════════════════════════════════════════════════════════════════════════
#  Spans & Generations
# ═════════════════════════════════════════════════════════════════════════════

def create_span(parent, name: str, input: dict | None = None, metadata: dict[str, Any] | None = None):
    """Create a child span under parent."""
    if isinstance(parent, _NoOpSpan) or parent is None:
        return _NOOP
    try:
        return parent.start_span(
            name=name,
            input=input or {},
            metadata=metadata or {},
        )
    except Exception as e:
        logger.error(f"create_span '{name}' failed: {e}")
        return _NOOP


def end_span(span, output: dict | None = None):
    """Update span output and end it."""
    if isinstance(span, _NoOpSpan) or span is None:
        return
    try:
        if output:
            span.update(output=output)
        span.end()
    except Exception as e:
        logger.error(f"end_span failed: {e}")


def create_generation(
    parent, name: str, model: str,
    input: dict | None = None,
    output: str | None = None,
    usage: dict[str, int] | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Log an LLM generation as a child observation."""
    if isinstance(parent, _NoOpSpan) or parent is None:
        return _NOOP
    try:
        gen = parent.start_observation(
            as_type="generation",
            name=name,
            model=model,
            input=input or {},
            metadata=metadata or {},
        )
        update_kwargs = {}
        if output:
            update_kwargs["output"] = output
        if usage:
            update_kwargs["usage_details"] = usage
        if update_kwargs:
            gen.update(**update_kwargs)
        gen.end()
        return gen
    except Exception as e:
        logger.error(f"Failed to create generation '{name}': {e}")
        return _NOOP


# ═════════════════════════════════════════════════════════════════════════════
#  SpanTimer context manager
# ═════════════════════════════════════════════════════════════════════════════

class SpanTimer:
    """Context manager that creates a timed child span."""

    def __init__(self, parent, name: str, input: dict | None = None, metadata: dict | None = None):
        self.parent = parent if parent is not None else _NOOP
        self.name = name
        self.input = input
        self.metadata = metadata
        self._start = 0.0
        self._child = _NOOP
        self._output = None

    def set_output(self, output):
        self._output = output

    def __enter__(self):
        self._start = time.perf_counter()
        if not isinstance(self.parent, _NoOpSpan):
            try:
                self._child = self.parent.start_span(
                    name=self.name,
                    input=self.input or {},
                    metadata=self.metadata or {},
                )
            except Exception as e:
                logger.error(f"SpanTimer: failed to start span '{self.name}': {e}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self._start

        if not isinstance(self._child, _NoOpSpan):
            try:
                output = self._output or {}
                if isinstance(output, dict):
                    output["latency_ms"] = round(elapsed * 1000, 1)
                self._child.update(output=output)
                self._child.end()
            except Exception as e:
                logger.error(f"SpanTimer: failed to end span '{self.name}': {e}")
        return False


# ═════════════════════════════════════════════════════════════════════════════
#  FEATURE 3: User Feedback
# ═════════════════════════════════════════════════════════════════════════════

def score_trace_by_id(trace_id: str, name: str = "user_feedback", value: float = 1.0, comment: str = ""):
    """Score a trace (used for user thumbs-up/thumbs-down feedback)."""
    client = _get_client()
    if client is None:
        return
    try:
        client.create_score(
            trace_id=trace_id,
            name=name,
            value=value,
            comment=comment,
        )
        client.flush()
        logger.info(f"Scored trace {trace_id}: {name}={value}")
    except Exception as e:
        logger.error(f"score_trace_by_id failed: {e}")


# ═════════════════════════════════════════════════════════════════════════════
#  FEATURE 4: Prompt Management
# ═════════════════════════════════════════════════════════════════════════════

def get_prompt(name: str, fallback: str = "") -> str:
    """Fetch a managed prompt from Langfuse with local fallback."""
    client = _get_client()
    if client is None:
        return fallback
    try:
        prompt = client.get_prompt(name)
        return prompt.prompt if hasattr(prompt, "prompt") else str(prompt.compile())
    except Exception as e:
        logger.warning(f"get_prompt('{name}') failed, using fallback: {e}")
        return fallback


def upload_prompt(name: str, prompt_text: str, labels: list[str] | None = None):
    """Create or update a prompt in Langfuse."""
    client = _get_client()
    if client is None:
        return
    try:
        client.create_prompt(
            name=name,
            prompt=prompt_text,
            labels=labels or ["production"],
            type="text",
        )
        client.flush()
    except Exception as e:
        logger.error(f"upload_prompt('{name}') failed: {e}")


# ═════════════════════════════════════════════════════════════════════════════
#  FEATURE 6: Datasets
# ═════════════════════════════════════════════════════════════════════════════

def create_dataset(name: str, description: str = ""):
    """Create a dataset in Langfuse."""
    client = _get_client()
    if client is None:
        return None
    try:
        ds = client.create_dataset(name=name, description=description)
        client.flush()
        return ds
    except Exception as e:
        logger.error(f"create_dataset('{name}') failed: {e}")
        return None


def upload_dataset_items(dataset_name: str, items: list[dict]):
    """Upload items to a Langfuse dataset."""
    client = _get_client()
    if client is None:
        return 0
    count = 0
    for item in items:
        try:
            client.create_dataset_item(
                dataset_name=dataset_name,
                input=item.get("input", {}),
                expected_output=item.get("expected_output", ""),
                metadata=item.get("metadata", {}),
            )
            count += 1
        except Exception as e:
            logger.error(f"upload_dataset_item failed: {e}")
    client.flush()
    return count


# ═════════════════════════════════════════════════════════════════════════════
#  Flush
# ═════════════════════════════════════════════════════════════════════════════

def flush():
    """Flush pending Langfuse events."""
    client = _get_client()
    if client:
        try:
            client.flush()
        except Exception:
            pass
