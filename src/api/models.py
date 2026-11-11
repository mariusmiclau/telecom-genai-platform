"""Pydantic models for API request/response validation."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.evidence import ClaimLink, EvidenceConflict, EvidenceItem, SafetyNotice, suspicious
from src.investigation import Clarification, Troubleshooting


class ChatRequest(BaseModel):
    """Chat message request."""

    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    session_id: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9_.:-]{1,128}$", description="Session ID for conversation continuity")
    context: Optional[dict] = Field(None, description="Additional context for the agent")

    @field_validator("context")
    @classmethod
    def bounded_context(cls, value):
        if value is not None and (len(value) > 16 or any(not isinstance(key, str) or len(key) > 64 or
                not isinstance(item, (str, int, float, bool)) or len(str(item).encode("utf-8")) > 256 for key, item in value.items())):
            raise ValueError("Context must contain at most 16 bounded scalar fields")
        if value is not None and any(not isinstance(item, str) or suspicious(item)
                for key, item in value.items() if key in ("element_id", "interface", "region", "software_version", "equipment_family")):
            raise ValueError("Scope fields must be safe text identifiers")
        return value

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

    model_config = ConfigDict(from_attributes=True)

    document: str = Field(..., description="Source document name")
    section: Optional[str] = Field(None, description="Specific section reference")
    relevance_score: float = Field(..., ge=0, le=1, description="Retrieval relevance score")
    excerpt: str = ""
    citation_id: str = ""


class ChatResponse(BaseModel):
    """Chat response with sources and confidence."""

    response: str = Field(..., description="AI-generated response")
    sources: list[SourceReference] = Field(default_factory=list, description="Source attributions")
    confidence: float = Field(..., ge=0, le=1, description="Response confidence score")
    tools_used: list[str] = Field(default_factory=list, description="Tools called during processing")
    session_id: Optional[str] = Field(None, description="Session ID for follow-up")
    trace_id: Optional[str] = Field(None, description="OpenTelemetry trace ID")
    outcome: Literal["answered", "abstained", "escalated"] = "answered"
    mode: Literal["demo", "live"] = "live"
    evidence: list[EvidenceItem] = Field(default_factory=list)
    claims: list[ClaimLink] = Field(default_factory=list)
    warnings: list[SafetyNotice] = Field(default_factory=list)
    conflicts: list[EvidenceConflict] = Field(default_factory=list)
    answer_kind: Literal["historical_resolution", "advisory", "insufficient_evidence", "conflict"] = "advisory"
    clarification: Clarification = Field(default_factory=Clarification)
    troubleshooting: Troubleshooting | None = None
    answer_id: str | None = None
    consumption: dict = Field(default_factory=dict)


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
