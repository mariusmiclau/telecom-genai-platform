"""API routes for the Telecom GenAI Platform."""

import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse

from src.api.models import ChatRequest, ChatResponse, IngestRequest, IngestResponse
from src.agents.orchestrator import AgentOrchestrator
from src.rag_pipeline.ingest import DocumentIngester
from src.guardrails.input_filter import InputFilter
from src.guardrails.output_filter import OutputFilter
from src.monitoring.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    HALLUCINATION_SCORE,
    track_token_usage,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize components
orchestrator = AgentOrchestrator()
ingester = DocumentIngester()
input_filter = InputFilter()
output_filter = OutputFilter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message through the AI agent pipeline.

    Flow:
    1. Input guardrails (injection detection, validation)
    2. Agent orchestration (RAG + tool-calling + reasoning)
    3. Output guardrails (PII filter, hallucination check)
    4. Response with sources and confidence score
    """
    start_time = time.time()
    REQUEST_COUNT.labels(endpoint="/chat", method="POST").inc()

    try:
        # Step 1: Input guardrails
        input_check = input_filter.validate(request.message)
        if not input_check.is_safe:
            logger.warning(
                "Input blocked by guardrails",
                extra={"reason": input_check.reason, "session": request.session_id},
            )
            raise HTTPException(
                status_code=400,
                detail=f"Input validation failed: {input_check.reason}",
            )

        # Step 2: Agent orchestration
        result = await orchestrator.process(
            message=request.message,
            session_id=request.session_id,
            context=request.context,
        )

        # Step 3: Output guardrails
        filtered_response = output_filter.apply(
            response=result.response,
            sources=result.sources,
        )

        # Step 4: Track metrics
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="/chat").observe(latency)
        HALLUCINATION_SCORE.observe(result.confidence)
        track_token_usage(result.tokens_used)

        logger.info(
            "Chat processed",
            extra={
                "session": request.session_id,
                "latency_ms": round(latency * 1000),
                "confidence": result.confidence,
                "tools_called": [t.name for t in result.tools_called],
                "sources_count": len(result.sources),
            },
        )

        return ChatResponse(
            response=filtered_response.text,
            sources=filtered_response.sources,
            confidence=result.confidence,
            tools_used=[t.name for t in result.tools_called],
            session_id=request.session_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat processing error: {e}", exc_info=True)
        REQUEST_COUNT.labels(endpoint="/chat", method="POST_ERROR").inc()
        raise HTTPException(status_code=500, detail="Internal processing error")


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """Ingest documents into the RAG knowledge base.

    Supports: PDF, DOCX, TXT, CSV, JSON, XML
    Applies telecom-optimized chunking strategies.
    """
    REQUEST_COUNT.labels(endpoint="/ingest", method="POST").inc()

    try:
        result = await ingester.ingest(
            source_path=request.source_path,
            doc_type=request.doc_type,
            metadata=request.metadata,
        )

        logger.info(
            "Documents ingested",
            extra={
                "source": request.source_path,
                "chunks_created": result.chunks_created,
                "doc_type": request.doc_type,
            },
        )

        return IngestResponse(
            status="success",
            chunks_created=result.chunks_created,
            documents_processed=result.documents_processed,
        )

    except Exception as e:
        logger.error(f"Ingestion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/ingest/upload")
async def upload_and_ingest(file: UploadFile = File(...)):
    """Upload a file and ingest it into the knowledge base."""
    REQUEST_COUNT.labels(endpoint="/ingest/upload", method="POST").inc()

    allowed_types = {
        "application/pdf",
        "text/plain",
        "text/csv",
        "application/json",
        "application/xml",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}",
        )

    try:
        result = await ingester.ingest_upload(file)
        return IngestResponse(
            status="success",
            chunks_created=result.chunks_created,
            documents_processed=1,
        )
    except Exception as e:
        logger.error(f"Upload ingestion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
