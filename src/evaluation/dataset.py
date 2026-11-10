"""Dataset loading and validation with explicit release-manifest checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED_TICKET_FIELDS = {
    "ticket_id",
    "status",
    "severity",
    "region",
    "element_id",
    "interface",
    "service",
    "summary",
    "symptoms",
    "root_cause",
    "resolution",
    "runbook_id",
    "source",
}
REQUIRED_GOLDEN_FIELDS = {
    "question_id",
    "question",
    "expected_ticket_ids",
    "expected_sources",
    "reference_answer",
    "tags",
}
REQUIRED_MANIFEST_FIELDS = {
    "dataset_version",
    "source_snapshot",
    "cleaning_rules",
    "chunking_configuration",
    "embedding_model",
    "embedding_dimensions",
    "number_of_documents",
    "number_of_rejected_records",
    "creation_timestamp",
    "evaluation_results",
    "snapshot_sha256",
}


class DatasetValidationError(ValueError):
    """Raised when a versioned dataset violates its release contract."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise DatasetValidationError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _require_fields(value: dict[str, Any], fields: set[str], label: str) -> None:
    missing = sorted(fields - value.keys())
    if missing:
        raise DatasetValidationError(f"{label}: missing required fields: {', '.join(missing)}")


def validate_dataset_version(version_dir: Path) -> dict[str, Any]:
    manifest_path = version_dir / "manifest.json"
    snapshot_path = version_dir / "tickets.jsonl"
    if not manifest_path.exists() or not snapshot_path.exists():
        raise DatasetValidationError(f"{version_dir}: manifest.json and tickets.jsonl are required")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require_fields(manifest, REQUIRED_MANIFEST_FIELDS, str(manifest_path))
    records = load_jsonl(snapshot_path)

    seen: set[str] = set()
    for index, record in enumerate(records, 1):
        _require_fields(record, REQUIRED_TICKET_FIELDS, f"{snapshot_path}:{index}")
        ticket_id = str(record["ticket_id"])
        if ticket_id in seen:
            raise DatasetValidationError(f"{snapshot_path}:{index}: duplicate ticket_id {ticket_id}")
        seen.add(ticket_id)
        if not ticket_id.startswith("INC"):
            raise DatasetValidationError(f"{snapshot_path}:{index}: invalid ticket_id {ticket_id}")

    if manifest["number_of_documents"] != len(records):
        raise DatasetValidationError(
            f"{manifest_path}: document count {manifest['number_of_documents']} != {len(records)}"
        )
    actual_hash = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    if manifest["snapshot_sha256"] != actual_hash:
        raise DatasetValidationError(f"{manifest_path}: snapshot_sha256 does not match tickets.jsonl")
    if not isinstance(manifest["embedding_dimensions"], int):
        raise DatasetValidationError(f"{manifest_path}: embedding_dimensions must be an integer")
    return {"version": manifest["dataset_version"], "documents": len(records)}


def validate_golden_set(path: Path, known_tickets: set[str], known_sources: set[str]) -> int:
    questions = load_jsonl(path)
    seen: set[str] = set()
    for index, question in enumerate(questions, 1):
        _require_fields(question, REQUIRED_GOLDEN_FIELDS, f"{path}:{index}")
        question_id = str(question["question_id"])
        if question_id in seen:
            raise DatasetValidationError(f"{path}:{index}: duplicate question_id {question_id}")
        seen.add(question_id)
        if not question["expected_ticket_ids"]:
            raise DatasetValidationError(f"{path}:{index}: expected_ticket_ids cannot be empty")
        unknown_tickets = set(question["expected_ticket_ids"]) - known_tickets
        unknown_sources = set(question["expected_sources"]) - known_sources
        if unknown_tickets:
            raise DatasetValidationError(f"{path}:{index}: unknown tickets {sorted(unknown_tickets)}")
        if unknown_sources:
            raise DatasetValidationError(f"{path}:{index}: unknown sources {sorted(unknown_sources)}")
    if len(questions) != 50:
        raise DatasetValidationError(f"{path}: expected exactly 50 questions, found {len(questions)}")
    return len(questions)


def validate_repository_data(root: Path) -> dict[str, Any]:
    dataset_root = root / "datasets"
    versions = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    validated = [validate_dataset_version(path) for path in versions]
    latest_records = load_jsonl(dataset_root / "tickets-v2.0-ticket-aware-chunking" / "tickets.jsonl")
    question_count = validate_golden_set(
        root / "evaluation" / "golden_questions.jsonl",
        {row["ticket_id"] for row in latest_records},
        {row["source"] for row in latest_records},
    )
    return {"versions": validated, "golden_questions": question_count}
