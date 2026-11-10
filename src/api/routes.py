"""HTTP adapters for the managed runtime."""

import asyncio
import logging
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from src.api.models import ChatRequest, ChatResponse, IngestRequest, IngestResponse, SourceReference
from src.guardrails.input_filter import InputFilter
from src.guardrails.output_filter import OutputFilter
from src.monitoring.metrics import (
    ANSWER_OUTCOMES,
    HALLUCINATION_SCORE,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    track_token_usage,
)
from src.monitoring.tracing import current_trace_id

router = APIRouter()
logger = logging.getLogger(__name__)
input_filter = InputFilter()
output_filter = OutputFilter()


@router.delete("/sessions/{session_id}", status_code=204)
async def forget_session(session_id: str, request: Request):
    runtime = request.app.state.runtime
    if runtime.agent is None:
        raise HTTPException(503, "Live runtime is not ready")
    await runtime.agent.memory.forget(session_id)


def validate_query(message):
    result = input_filter.validate(message)
    if not result.is_safe:
        raise HTTPException(400, "Input rejected by safety checks")


@router.post("/demo/chat", response_model=ChatResponse)
async def demo_chat(payload: ChatRequest, request: Request):
    validate_query(payload.message)
    runtime = request.app.state.runtime
    if runtime.demo is None:
        raise HTTPException(503, "Demo is starting; try again shortly")
    async with asyncio.timeout(runtime.settings.request_timeout_seconds):
        async with runtime.capacity:
            result = await asyncio.to_thread(runtime.demo.answer, payload.message)
    sources = [SourceReference(
        document=hit.source, section=hit.ticket_id, relevance_score=result.confidence,
    ) for hit in result.hits[:1]]
    filtered = output_filter.apply(result.text, sources)
    return ChatResponse(
        response=filtered.text, sources=filtered.sources, confidence=result.confidence,
        tools_used=["offline_hybrid_retrieval", "weighted_rank_fusion", "field_reranker"],
        session_id=payload.session_id, trace_id=result.trace_id,
        outcome="answered" if sources else "abstained", mode="demo",
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request):
    validate_query(payload.message)
    runtime = request.app.state.runtime
    if runtime.agent is None:
        raise HTTPException(503, "Live runtime is not ready")
    session_id = payload.session_id or str(uuid.uuid4())
    start = time.monotonic()
    REQUEST_COUNT.labels(endpoint="/chat", method="POST").inc()
    try:
        async with asyncio.timeout(runtime.settings.request_timeout_seconds):
            async with runtime.capacity:
                result = await runtime.agent.process(payload.message, session_id, payload.context)
        filtered = output_filter.apply(result.response, result.sources)
        HALLUCINATION_SCORE.observe(result.confidence)
        ANSWER_OUTCOMES.labels(outcome="answered" if filtered.sources else "abstained").inc()
        track_token_usage(result.tokens_used)
        return ChatResponse(
            response=filtered.text, sources=filtered.sources, confidence=result.confidence,
            tools_used=[tool.name for tool in result.tools_called], session_id=session_id,
            trace_id=current_trace_id(), outcome="answered" if filtered.sources else "abstained", mode="live",
        )
    except TimeoutError:
        ANSWER_OUTCOMES.labels(outcome="timeout").inc()
        raise HTTPException(504, "Request deadline exceeded") from None
    except Exception:
        ANSWER_OUTCOMES.labels(outcome="dependency_error").inc()
        logger.exception("Chat request failed")
        REQUEST_COUNT.labels(endpoint="/chat", method="POST_ERROR").inc()
        raise HTTPException(503, "A required dependency is unavailable") from None
    finally:
        REQUEST_LATENCY.labels(endpoint="/chat").observe(time.monotonic() - start)


def allowed_source(source: str, root: Path) -> Path:
    path = Path(source).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise HTTPException(400, "Source must be a file inside INGESTION_ROOT")
    return path


@router.post("/ingest", response_model=IngestResponse)
async def ingest(payload: IngestRequest, request: Request):
    allowed_source(payload.source_path, request.app.state.settings.ingestion_root)
    raise HTTPException(409, "Use the ingestion CLI and validate the collection before restarting live mode")


@router.post("/ingest/upload", response_model=IngestResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    await file.close()
    raise HTTPException(409, "HTTP ingestion is disabled; use the operator ingestion CLI")
