"""Tests for the stable trace-stage and prompt-capture contract."""

from src.monitoring.tracing import REQUIRED_TRACE_STAGES, content_attributes


def test_required_trace_stages_cover_full_rag_request():
    assert REQUIRED_TRACE_STAGES == (
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


def test_query_content_can_be_disabled(monkeypatch):
    monkeypatch.setenv("OTEL_CAPTURE_CONTENT", "false")

    attributes = content_attributes("status of PCRF-LON-01")

    assert "genai.query" not in attributes
    assert attributes["genai.query.length"] == len("status of PCRF-LON-01")
    assert len(attributes["genai.query.sha256"]) == 64
