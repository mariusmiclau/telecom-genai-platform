"""HTTP adapters for the managed runtime."""

import asyncio
import logging
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from src.answer_records import FeedbackDecision, FeedbackSubmission, safe
from src.api.models import ChatRequest, ChatResponse, IngestRequest, IngestResponse, SourceReference
from src.budgets import BudgetExceeded, consumption_context, prompt_record_context, token_bound
from src.evidence import public_evidence
from src.guardrails.input_filter import InputFilter
from src.guardrails.output_filter import OutputFilter
from src.investigation import troubleshooting
from src.monitoring.metrics import (
    ANSWER_OUTCOMES,
    HALLUCINATION_SCORE,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    track_token_usage,
)
from src.monitoring.tracing import current_trace_id
from src.security import current_principal

router = APIRouter()
logger = logging.getLogger(__name__)
input_filter = InputFilter()
output_filter = OutputFilter()


async def record_answer(runtime, response, payload):
    budget = consumption_context.get()
    response.consumption = budget.report() if budget else {}
    dataset = {"version": "tickets-v2.0-ticket-aware-chunking"}
    retrieval = {"configuration": "C", "top_k": 5, "implementation": "offline-deterministic"}
    if response.mode == "live":
        dataset = {"version": runtime.settings.knowledge_version}
        rows = await asyncio.to_thread(runtime.agent.memory._transaction,
            "SELECT source_sha256,embedding_model,dimensions,configuration FROM knowledge_versions WHERE name=%s",
            (runtime.settings.knowledge_version,), True)
        if rows:
            dataset.update(source_sha256=rows[0][0], embedding_model=rows[0][1], embedding_dimensions=rows[0][2])
        retrieval = {"configuration": runtime.settings.retrieval_configuration, "top_k": 5,
            "implementation": "postgres-exact-vector-bm25-rrf", "vector_weight": .65, "keyword_weight": .35, "rrf_k": 60}
    else:
        import hashlib
        source = runtime.demo.repository_root / "datasets" / dataset["version"] / "tickets.jsonl"
        dataset["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    retrieval.update(source_review_max_age_days=runtime.settings.source_review_max_age_days,
        passage_character_limit=1500, source_policy_version="evidence-v1.4",
        chunking="recursive-240-overlap-40" if retrieval["configuration"] == "A" else "ticket-identity-incident-resolution-v1")
    prompt = prompt_record_context.get() or {"model": "none", "implementation": "offline-extractive-v1.4"}
    response.answer_id = await runtime.answer_records.capture(response, payload.message,
        {key: value for key, value in (payload.context or {}).items() if key in ("element_id", "interface", "region", "software_version", "equipment_family")},
        prompt, dataset, retrieval, response.session_id)
    return response


@router.delete("/sessions/{session_id}", status_code=204)
async def forget_session(session_id: str, request: Request):
    runtime = request.app.state.runtime
    if runtime.agent is None:
        raise HTTPException(503, "Live runtime is not ready")
    await runtime.agent.memory.forget(session_id)
    principal = current_principal()
    await asyncio.to_thread(runtime.agent.memory._transaction,
        "DELETE FROM answer_records WHERE session_id=%s AND tenant_id=%s AND owner_id=%s",
        (session_id, principal.tenant, principal.subject))


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
            budget = consumption_context.get()
            if budget:
                budget.reserve(token_bound(payload.message))
            result = await asyncio.to_thread(runtime.demo.answer, payload.message, payload.context)
            sources = [SourceReference(
                document=hit.source, section=hit.ticket_id, relevance_score=result.confidence,
                excerpt=result.evidence[0].passage if result.evidence else "", citation_id="source-1",
            ) for hit in result.hits[:1]]
            filtered = output_filter.apply(result.text, sources)
            response = ChatResponse(
                response=filtered.text, sources=filtered.sources, confidence=result.confidence,
                tools_used=["offline_hybrid_retrieval", "weighted_rank_fusion", "field_reranker"],
                session_id=payload.session_id, trace_id=result.trace_id,
                outcome=result.outcome, mode="demo",
                answer_kind=result.answer_kind, **public_evidence(result),
                clarification=result.clarification,
                troubleshooting=safe(troubleshooting(payload.message, result.evidence).model_dump()) if result.outcome == "answered" else None,
            )
            return await record_answer(runtime, response, payload)


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
                outcome = result.outcome
                if outcome == "answered" and not filtered.sources:
                    outcome = "abstained"
                ANSWER_OUTCOMES.labels(outcome=outcome).inc()
                track_token_usage(result.tokens_used)
                response = ChatResponse(
                    response=filtered.text, sources=filtered.sources, confidence=result.confidence,
                    tools_used=[tool.name for tool in result.tools_called], session_id=session_id,
                    trace_id=current_trace_id(), outcome=outcome, mode="live",
                    answer_kind=result.answer_kind, **public_evidence(result),
                    clarification=result.clarification,
                    troubleshooting=safe(result.troubleshooting.model_dump()) if result.troubleshooting else None,
                )
                return await record_answer(runtime, response, payload)
    except BudgetExceeded:
        raise
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


async def records_dependency(request: Request):
    runtime = request.app.state.runtime
    try:
        async with asyncio.timeout(runtime.settings.request_timeout_seconds):
            async with runtime.capacity:
                yield runtime.answer_records
    except TimeoutError:
        raise HTTPException(504, "Request deadline exceeded") from None


@router.get("/answers/{answer_id}")
async def answer_record(answer_id: str, records=Depends(records_dependency)):
    return await records.get(answer_id)


@router.post("/answers/{answer_id}/replay")
async def replay_answer(answer_id: str, records=Depends(records_dependency)):
    return await records.replay(answer_id)


@router.post("/answers/{answer_id}/feedback", status_code=201)
async def feedback(answer_id: str, payload: FeedbackSubmission, records=Depends(records_dependency)):
    return await records.submit(answer_id, payload)


@router.get("/feedback/{feedback_id}")
async def feedback_status(feedback_id: str, records=Depends(records_dependency)):
    return await records.feedback_item(feedback_id)


@router.post("/feedback/{feedback_id}/review")
async def review_feedback(feedback_id: str, payload: FeedbackDecision, records=Depends(records_dependency)):
    return await records.review(feedback_id, payload)


@router.get("/feedback/{feedback_id}/regression")
async def export_regression(feedback_id: str, records=Depends(records_dependency)):
    return await records.regression(feedback_id)


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
