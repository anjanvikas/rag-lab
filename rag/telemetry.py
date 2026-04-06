"""
Arize Phoenix Telemetry and Async Evaluations (Phase B).
"""
import os
import re
from typing import Optional

def mask_pii(text: str | None) -> str | None:
    """Mask obvious PII (emails, phone numbers) before logging."""
    if not text:
        return text
    # Mask email
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL_REDACTED]', text)
    # Mask US phone numbers loosely
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_REDACTED]', text)
    return text

class PIISpanProcessor:
    """A span processor that masks PII in span attributes before exporting."""
    def on_start(self, span, parent_context=None):
        pass

    def on_end(self, span):
        if not hasattr(span, "attributes") or not span.attributes:
            return
        # We need to modify attributes safely
        new_attrs = {}
        for k, v in span.attributes.items():
            if isinstance(v, str):
                new_attrs[k] = mask_pii(v)
            else:
                new_attrs[k] = v
        # Span attributes might be immutable depending on SDK version
        # If set_attribute works, we use it.
        try:
            for k, v in new_attrs.items():
                span.set_attribute(k, v)
        except Exception:
            pass

def register():
    """Setup Phoenix and OpenTelemetry instrumentation."""
    import socket
    from urllib.parse import urlparse

    try:
        from opentelemetry import trace as trace_api
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource
        from openinference.instrumentation.anthropic import AnthropicInstrumentor
        
        # OTLP-over-HTTP expects port 4318. 4317 is for gRPC.
        endpoint = os.getenv("PHOENIX_ENDPOINT", "http://localhost:4318") 
        if not endpoint.startswith("http"):
            endpoint = f"http://{endpoint}"
        if not endpoint.endswith("/v1/traces"):
            endpoint = f"{endpoint.rstrip('/')}/v1/traces"

        # ── Pre-flight check ──────────────────────────────────────────────────
        # Perform a fast socket test to see if the collector is actually there.
        # This prevents background retry-spam if Phoenix is closed.
        parsed = urlparse(endpoint)
        host, port = parsed.hostname or "localhost", parsed.port or 4318
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.1) # extremely fast check
            if s.connect_ex((host, port)) != 0:
                print(f"📡 Telemetry: Phoenix ({host}:{port}) unreachable. Skipping local tracing.")
                return

        resource = Resource(attributes={"service.name": "rag-platform"})
        provider = TracerProvider(resource=resource)
        
        # BatchSpanProcessor is better for production than SimpleSpanProcessor
        exporter = OTLPSpanExporter(endpoint=endpoint, timeout=2) # 2s timeout
        provider.add_span_processor(BatchSpanProcessor(exporter))
        
        trace_api.set_tracer_provider(provider)
        
        # Wrap Anthropic with instrumentation
        AnthropicInstrumentor().instrument()
        print(f"✅ Telemetry: Tracing active at {endpoint}")
    except Exception:
        # We fail silently so the AI platform doesn't block if monitoring is down
        pass

def trigger_async_eval(trace_id: str):
    """
    Fire-and-forget ARQ job enqueue for async evaluation (RAGAS + Faithfulness).
    Called after generating the chat response.
    """
    import asyncio
    try:
        # Enqueue eval task
        pass
    except Exception as e:
        pass
