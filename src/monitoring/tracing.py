"""OpenTelemetry setup and safe manual-span helpers.

The application exports OTLP/HTTP spans when ``OTEL_TRACES_EXPORTER=otlp``.
Docker Compose configures Jaeger as the local backend. Without OpenTelemetry
packages or when tracing is disabled, these helpers degrade to no-ops so core
request processing is never coupled to the observability backend.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator

from src.version import VERSION

logger = logging.getLogger(__name__)
_configured = False

REQUIRED_TRACE_STAGES = (
    "query",
    "entity_extraction",
    "vector_retrieval",
    "keyword_retrieval",
    "rank_fusion",
    "prompt_construction",
    "llm_call",
    "grounding_check",
    "final_response",
)


class _NoOpSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        del key, value

    def record_exception(self, exception: BaseException) -> None:
        del exception

    def set_status(self, status: Any) -> None:
        del status


def _attribute_value(value: Any) -> str | bool | int | float:
    if isinstance(value, (str, bool, int, float)):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def content_attributes(query: str) -> dict[str, str | int]:
    """Return trace-safe query attributes, with an opt-out for content capture."""
    attributes: dict[str, str | int] = {
        "genai.query.length": len(query),
        "genai.query.sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
    }
    if os.getenv("OTEL_CAPTURE_CONTENT", "false").lower() == "true":
        attributes["genai.query"] = query
    return attributes


def current_trace_id() -> str | None:
    """Return the current W3C trace ID as 32 lowercase hexadecimal characters."""
    try:
        from opentelemetry import trace

        span_context = trace.get_current_span().get_span_context()
        if span_context.is_valid:
            return f"{span_context.trace_id:032x}"
    except ImportError:
        pass
    return None


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
    """Start a current span and attach primitive/JSON-encoded attributes."""
    try:
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode

        tracer = trace.get_tracer("telecom-genai-platform")
        with tracer.start_as_current_span(name) as span:
            for key, value in (attributes or {}).items():
                if value is not None:
                    span.set_attribute(key, _attribute_value(value))
            try:
                yield span
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise
    except ImportError:
        yield _NoOpSpan()


def configure_tracing(app: Any | None = None) -> bool:
    """Configure one global tracer provider and optional FastAPI instrumentation."""
    global _configured
    if _configured:
        return True
    if os.getenv("OTEL_TRACES_ENABLED", "true").lower() != "true":
        return False

    exporter_name = os.getenv("OTEL_TRACES_EXPORTER", "none").lower()
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )

        provider = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": os.getenv("OTEL_SERVICE_NAME", "telecom-genai-api"),
                    "service.version": VERSION,
                    "deployment.environment.name": os.getenv("DEPLOYMENT_ENVIRONMENT", "local"),
                }
            )
        )
        if exporter_name == "otlp":
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            endpoint = os.getenv(
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
                "http://localhost:4318/v1/traces",
            )
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        elif exporter_name == "console":
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        elif exporter_name != "none":
            raise ValueError(f"Unsupported OTEL_TRACES_EXPORTER: {exporter_name}")

        trace.set_tracer_provider(provider)
        if app is not None and exporter_name != "none":
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app)
        _configured = True
        logger.info("OpenTelemetry tracing configured", extra={"exporter": exporter_name})
        return True
    except ImportError as exc:
        logger.warning("OpenTelemetry packages are not installed: %s", exc)
        return False
    except Exception:
        logger.exception("OpenTelemetry setup failed; continuing without exported traces")
        return False
