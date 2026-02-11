"""Pydantic models for API request/response validation."""

from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Chat message request."""

    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    context: Optional[dict] = Field(None, description="Additional context for the agent")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "What is the current network status for Region A?",
                    "session_id": "sess_abc123",
                    "context": {"region": "A", "priority": "high"},
                }
            ]
        }
    }


class SourceReference(BaseModel):
    """Reference to a source document used in the response."""

    document: str = Field(..., description="Source document name")
    section: Optional[str] = Field(None, description="Specific section reference")
    relevance_score: float = Field(..., ge=0, le=1, description="Retrieval relevance score")


class ChatResponse(BaseModel):
    """Chat response with sources and confidence."""

    response: str = Field(..., description="AI-generated response")
    sources: list[SourceReference] = Field(default_factory=list, description="Source attributions")
    confidence: float = Field(..., ge=0, le=1, description="Response confidence score")
    tools_used: list[str] = Field(default_factory=list, description="Tools called during processing")
    session_id: Optional[str] = Field(None, description="Session ID for follow-up")


class IngestRequest(BaseModel):
    """Document ingestion request."""

    source_path: str = Field(..., description="Path or URL to document source")
    doc_type: str = Field(
        "auto",
        description="Document type: pdf, csv, json, xml, txt, or auto-detect",
    )
    metadata: Optional[dict] = Field(None, description="Additional metadata tags")


class IngestResponse(BaseModel):
    """Document ingestion result."""

    status: str
    chunks_created: int
    documents_processed: int
