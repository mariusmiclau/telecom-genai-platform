"""Tests for Pydantic API models validation."""

import pytest
from pydantic import ValidationError

from src.api.models import (
    ChatRequest,
    ChatResponse,
    IngestRequest,
    IngestResponse,
    SourceReference,
)


class TestChatRequest:
    """Tests for ChatRequest model."""

    def test_valid_request(self):
        """Test valid chat request."""
        request = ChatRequest(
            message="What is the network status?",
            session_id="sess_123",
            context={"region": "A"},
        )
        assert request.message == "What is the network status?"
        assert request.session_id == "sess_123"
        assert request.context == {"region": "A"}

    def test_minimal_request(self):
        """Test request with only required fields."""
        request = ChatRequest(message="Hello")
        assert request.message == "Hello"
        assert request.session_id is None
        assert request.context is None

    def test_empty_message_rejected(self):
        """Test that empty message is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(message="")
        assert "min_length" in str(exc_info.value).lower() or "at least 1" in str(exc_info.value).lower()

    def test_message_too_long_rejected(self):
        """Test that message exceeding max length is rejected."""
        long_message = "x" * 4001
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(message=long_message)
        assert "max_length" in str(exc_info.value).lower() or "at most 4000" in str(exc_info.value).lower()

    def test_message_at_max_length(self):
        """Test message at exactly max length is accepted."""
        max_message = "x" * 4000
        request = ChatRequest(message=max_message)
        assert len(request.message) == 4000

    def test_missing_message_rejected(self):
        """Test that missing message field is rejected."""
        with pytest.raises(ValidationError):
            ChatRequest()


class TestChatResponse:
    """Tests for ChatResponse model."""

    def test_valid_response(self):
        """Test valid chat response."""
        response = ChatResponse(
            response="The network is healthy.",
            sources=[
                SourceReference(
                    document="network_report.pdf",
                    section="Summary",
                    relevance_score=0.95,
                )
            ],
            confidence=0.92,
            tools_used=["network_health"],
            session_id="sess_123",
        )
        assert response.response == "The network is healthy."
        assert len(response.sources) == 1
        assert response.confidence == 0.92
        assert response.tools_used == ["network_health"]

    def test_minimal_response(self):
        """Test response with only required fields."""
        response = ChatResponse(
            response="Hello",
            confidence=0.5,
        )
        assert response.response == "Hello"
        assert response.sources == []
        assert response.tools_used == []

    def test_confidence_out_of_range(self):
        """Test that confidence outside 0-1 range is rejected."""
        with pytest.raises(ValidationError):
            ChatResponse(response="Test", confidence=1.5)

        with pytest.raises(ValidationError):
            ChatResponse(response="Test", confidence=-0.1)

    def test_confidence_boundaries(self):
        """Test confidence at boundary values."""
        response_min = ChatResponse(response="Test", confidence=0.0)
        assert response_min.confidence == 0.0

        response_max = ChatResponse(response="Test", confidence=1.0)
        assert response_max.confidence == 1.0


class TestSourceReference:
    """Tests for SourceReference model."""

    def test_valid_source(self):
        """Test valid source reference."""
        source = SourceReference(
            document="manual.pdf",
            section="Chapter 3",
            relevance_score=0.88,
        )
        assert source.document == "manual.pdf"
        assert source.section == "Chapter 3"
        assert source.relevance_score == 0.88

    def test_source_without_section(self):
        """Test source reference without section."""
        source = SourceReference(
            document="config.json",
            relevance_score=0.75,
        )
        assert source.section is None

    def test_relevance_score_validation(self):
        """Test relevance score must be 0-1."""
        with pytest.raises(ValidationError):
            SourceReference(document="test.pdf", relevance_score=1.5)

        with pytest.raises(ValidationError):
            SourceReference(document="test.pdf", relevance_score=-0.1)


class TestIngestRequest:
    """Tests for IngestRequest model."""

    def test_valid_ingest_request(self):
        """Test valid ingestion request."""
        request = IngestRequest(
            source_path="/data/documents/manual.pdf",
            doc_type="pdf",
            metadata={"category": "network", "version": "2.0"},
        )
        assert request.source_path == "/data/documents/manual.pdf"
        assert request.doc_type == "pdf"
        assert request.metadata["category"] == "network"

    def test_auto_doc_type_default(self):
        """Test doc_type defaults to auto."""
        request = IngestRequest(source_path="/data/file.txt")
        assert request.doc_type == "auto"

    def test_missing_source_path_rejected(self):
        """Test that missing source_path is rejected."""
        with pytest.raises(ValidationError):
            IngestRequest()


class TestIngestResponse:
    """Tests for IngestResponse model."""

    def test_valid_ingest_response(self):
        """Test valid ingestion response."""
        response = IngestResponse(
            status="success",
            chunks_created=150,
            documents_processed=5,
        )
        assert response.status == "success"
        assert response.chunks_created == 150
        assert response.documents_processed == 5

    def test_missing_fields_rejected(self):
        """Test that missing required fields are rejected."""
        with pytest.raises(ValidationError):
            IngestResponse(status="success")
