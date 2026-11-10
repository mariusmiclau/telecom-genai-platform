"""Regression tests for dataset contracts and measured retrieval quality."""

from pathlib import Path

from src.evaluation.dataset import load_jsonl, validate_repository_data
from src.evaluation.runner import EvaluationRunner

ROOT = Path(__file__).resolve().parents[1]


def test_versioned_datasets_and_golden_set_validate():
    result = validate_repository_data(ROOT)

    assert result["golden_questions"] == 50
    assert [version["version"] for version in result["versions"]] == [
        "tickets-v1.0",
        "tickets-v1.1-cleaned",
        "tickets-v1.2-pii-redacted",
        "tickets-v2.0-ticket-aware-chunking",
    ]
    assert all(version["documents"] == 50 for version in result["versions"])


def test_retrieval_baseline_is_reproducible():
    tickets = load_jsonl(
        ROOT / "datasets" / "tickets-v2.0-ticket-aware-chunking" / "tickets.jsonl"
    )
    questions = load_jsonl(ROOT / "evaluation" / "golden_questions.jsonl")

    report = EvaluationRunner(tickets, questions).run()
    configurations = report["configurations"]

    assert configurations["A_recursive_vector"]["metrics"]["recall_at_5"] == 0.78
    assert configurations["B_ticket_aware_hybrid"]["metrics"]["recall_at_5"] == 0.86
    assert configurations["C_ticket_aware_hybrid_reranker"]["metrics"]["recall_at_5"] == 1.0


def test_ticket_aware_release_contains_no_raw_contact_pii():
    records = load_jsonl(
        ROOT / "datasets" / "tickets-v2.0-ticket-aware-chunking" / "tickets.jsonl"
    )

    assert all(record["reporter_email"] == "[EMAIL_REDACTED]" for record in records)
    assert all(record["contact_phone"] == "[PHONE_REDACTED]" for record in records)
    assert all(record["management_ip"] == "[IP_REDACTED]" for record in records)
