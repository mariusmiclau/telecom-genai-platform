"""Regression checks for evidence verification in the local acceptance harness."""

import json

import pytest

from scripts.accept_deployment import verify_metrics_export, verify_trace_export
from src.monitoring.tracing import REQUIRED_TRACE_STAGES

TRACE_ID = "a" * 32


def trace_log(stages=REQUIRED_TRACE_STAGES, attributes=None):
    return "container startup\n" + "\n".join(json.dumps({
        "name": stage, "context": {"trace_id": "0x" + TRACE_ID},
        "attributes": attributes or {},
    }, indent=2) for stage in stages)


def test_trace_verification_requires_all_exported_stages_for_the_same_request():
    logs = trace_log()
    assert verify_trace_export(logs, TRACE_ID) == sorted(REQUIRED_TRACE_STAGES)
    with pytest.raises(AssertionError, match="Missing exported"):
        verify_trace_export(logs, "b" * 32)
    with pytest.raises(AssertionError, match="llm_call"):
        verify_trace_export(trace_log([stage for stage in REQUIRED_TRACE_STAGES if stage != "llm_call"]), TRACE_ID)


@pytest.mark.parametrize("key", ["genai.query", "telecom.request.context", "telecom.entities"])
def test_trace_verification_rejects_private_content(key):
    with pytest.raises(AssertionError, match="Raw"):
        verify_trace_export(trace_log(attributes={key: "private"}), TRACE_ID)


def test_metrics_verification_checks_actual_outcomes():
    metrics = '\n'.join([
        'genai_answer_outcomes_total{outcome="answered"} 2',
        'genai_answer_outcomes_total{outcome="abstained"} 1',
        'genai_answer_outcomes_total{outcome="dependency_error"} 1',
        'genai_answer_outcomes_total{outcome="escalated"} 1',
        'genai_auth_failures_total 1',
    ])
    verify_metrics_export(metrics)
    with pytest.raises(AssertionError, match="dependency_error"):
        verify_metrics_export(metrics.replace('outcome="dependency_error"', 'outcome="other"'))
