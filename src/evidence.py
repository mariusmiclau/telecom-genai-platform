"""Inspectable evidence and conservative source policy, applied AFTER access control.

These are deterministic checks, not an entailment model or a complete prompt-
injection defense. No source metadata can authorize an action or override policy.
"""

import copy
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

from pydantic import BaseModel, Field

from src.guardrails.hallucination import HallucinationDetector
from src.guardrails.output_filter import OutputFilter

CITATIONS = re.compile(r"\[(?:Source\s*)?(\d+)\]", re.IGNORECASE)
UNSAFE = re.compile(
    r"ignore\s+(?:all\s+)?(?:previous|prior|system)\s+(?:instructions|prompts)|"
    r"(?:override|disregard)\s+(?:the\s+)?(?:system|safety|previous)\s+(?:prompt|rules|instructions)|"
    r"<\|(?:im_start|start_header_id)\|>|\[/?INST\]|"
    r"\[Source\s+\d+|###\s*(?:User Query|System)|"
    r"(?:^|\n)\s*(?:system|assistant)\s*:|"
    r"(?:reveal|print|expose|send|export)\b.{0,60}\b(?:system prompt|api[_ -]?keys?|credentials|access[_ -]?tokens?)\b",
    re.IGNORECASE,
)
CREDENTIAL = re.compile(r"\b(?:api[_-]?key|password|access[_-]?token|secret[_-]?key)\s*[\"']?\s*[:=]\s*[\"']?([^\s\"',;]+)", re.IGNORECASE)
REPORTED_INSTRUCTION = re.compile(r"(\b(?:note|payload|attack|ticket)\s+(?:contained|included|quoted)\s+)(?:ignore\s+(?:all\s+)?(?:previous|prior|system)\s+(?:instructions|prompts))", re.IGNORECASE)


class SafetyNotice(BaseModel):
    code: str
    message: str
    source_ids: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    citation_id: str
    document: str
    document_id: str
    passage: str
    document_kind: str
    freshness: Literal["historical", "current", "unknown"]
    reviewed_at: str | None = None
    valid_from: str | None = None
    valid_until: str | None = None
    software_version: str | None = None
    equipment_family: str | None = None
    redacted: bool = False
    retrieved_content_sha256: str = ""


class ClaimLink(BaseModel):
    claim: str
    citation_ids: list[str]
    support: Literal["lexical_match"] = "lexical_match"


class ConflictValue(BaseModel):
    value: str
    source_ids: list[str]


class EvidenceConflict(BaseModel):
    scope: str
    fact: str
    values: list[ConflictValue]


@dataclass
class EvidenceSelection:
    documents: list = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    warnings: list[SafetyNotice] = field(default_factory=list)
    conflicts: list[EvidenceConflict] = field(default_factory=list)


def normalized(text):
    return "".join(char for char in unicodedata.normalize("NFKC", text) if unicodedata.category(char) != "Cf")


def suspicious(text):
    text = normalized(text)
    if any(match[1].casefold().rstrip(".") not in {"[redacted]", "<redacted>", "<value>", "<placeholder>"} for match in CREDENTIAL.finditer(text)):
        return True
    if re.search(r"\bbearer\s+[A-Za-z0-9._-]{16,}", text, re.IGNORECASE):
        return True
    text = REPORTED_INSTRUCTION.sub(r"\1[REPORTED INJECTION TEXT REMOVED]", text)
    # Do not mistake a defensive notice for a command to disclose a secret.
    text = re.sub(r"\b(?:never|do not|don't)\s+(?:reveal|print|expose|send|export)\b[^\n.!?]*", "", text, flags=re.IGNORECASE)
    return bool(UNSAFE.search(text))


def timestamp(value):
    if value is None or value == "":
        return None
    if not isinstance(value, str) or len(value) > 40:
        raise ValueError("Invalid timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def validate_metadata(metadata):
    if not isinstance(metadata, dict) or metadata.get("document_type", "ticket") not in ("ticket", "runbook", "sla", "configuration"):
        raise ValueError("invalid_evidence_metadata")
    for key in ("reviewed_at", "valid_from", "valid_until"):
        timestamp(metadata.get(key))
    for key in ("software_version", "equipment_family", "superseded_by", "fact_scope"):
        if metadata.get(key) in (None, ""):
            continue
        if not isinstance(metadata[key], str) or not 0 < len(metadata[key].strip()) <= 128:
            raise ValueError("invalid_evidence_metadata")
    if metadata.get("element_id") is not None and (not isinstance(metadata["element_id"], str) or not 0 < len(metadata["element_id"]) <= 128):
        raise ValueError("invalid_evidence_metadata")
    declared = metadata.get("facts", {})
    if not isinstance(declared, dict) or len(declared) > 20 or any(
        not isinstance(key, str) or not 0 < len(key) <= 80 or not isinstance(value, str) or not 0 < len(value) <= 256
        for key, value in declared.items()
    ):
        raise ValueError("invalid_evidence_metadata")


def record_rejection(record):
    """Reject poisoned records before external embedding calls, not just at query time."""
    from src.evaluation.retrieval import render_ticket
    try:
        validate_metadata(record)
        if not all(isinstance(record.get(key), str) and 0 < len(record[key]) <= limit for key, limit in (("ticket_id", 128), ("source", 1024))):
            return "invalid_evidence_metadata"
        policy_data = {key: record[key] for key in ("facts", "fact_scope", "software_version", "equipment_family", "superseded_by") if key in record}
        if suspicious(render_ticket(record) + "\n" + str(record.get("source", "")) + "\n" + json.dumps(policy_data, ensure_ascii=False)):
            return "suspicious_source"
    except (ValueError, TypeError, OverflowError):
        return "invalid_evidence_metadata"
    return None


def select_evidence(documents, context=None, now=None, max_review_age_days=365):
    now = now or datetime.now(timezone.utc)
    context = context or {}
    selection = EvidenceSelection()
    # Structured fact keys are scoped to the same asset and compatibility version.
    facts: dict[tuple, dict[str, list[str]]] = {}
    compatibility: dict[tuple, dict[str, list[str]]] = {}
    for doc in documents:
        metadata = doc.metadata or {}
        doc_id = doc.document_id
        def notice(code, message):
            selection.warnings.append(SafetyNotice(code=code, message=message, source_ids=[doc_id]))
        try:
            validate_metadata(metadata)
        except (ValueError, TypeError, OverflowError):
            notice("invalid_metadata", "A source has invalid evidence metadata and was excluded.")
            continue
        policy_data = {key: metadata[key] for key in ("facts", "fact_scope", "software_version", "equipment_family", "superseded_by") if key in metadata}
        if suspicious(doc.content + "\n" + doc.source_path + "\n" + doc_id + "\n" + json.dumps(policy_data, ensure_ascii=False)):
            notice("suspicious_source", "A source was excluded because it contains suspicious instructions or credential material.")
            continue
        kind = metadata.get("document_type", "ticket")
        if kind not in ("ticket", "runbook", "sla", "configuration"):
            notice("invalid_metadata", "A source has an unsupported document type.")
            continue
        try:
            reviewed, valid_from, valid_until = (timestamp(metadata.get(key)) for key in ("reviewed_at", "valid_from", "valid_until"))
            if (reviewed and reviewed > now) or (valid_from and valid_until and valid_until <= valid_from):
                raise ValueError("Invalid date range")
        except (ValueError, TypeError, OverflowError):
            notice("invalid_metadata", "A source has invalid temporal metadata; it was excluded.")
            continue
        if metadata.get("superseded_by"):
            notice("superseded_source", "A superseded source was excluded; verify its replacement is available.")
            continue
        if valid_from and valid_from > now:
            notice("not_yet_valid", "A source is not yet effective and was excluded.")
            continue
        if valid_until and valid_until <= now:
            notice("expired_source", "An expired source was excluded.")
            continue
        if kind != "ticket" and reviewed and now - reviewed > timedelta(days=max_review_age_days):
            notice("stale_source", "A source is overdue for review and was excluded.")
            continue
        incompatible = False
        for key in ("software_version", "equipment_family"):
            value = metadata.get(key)
            if value is not None and (not isinstance(value, str) or not 0 < len(value) <= 128):
                notice("invalid_metadata", "A source has invalid compatibility metadata.")
                incompatible = True
                break
            requested = context.get(key)
            if requested is not None and value and requested != value:
                notice("compatibility_mismatch", "A source does not match the requested software version or equipment family.")
                incompatible = True
                break
            if requested is not None and not value:
                notice("compatibility_unknown", "A source does not declare the requested compatibility information.")
        if incompatible:
            continue
        original_content = doc.content
        cleaned_content = REPORTED_INSTRUCTION.sub(r"\1[REPORTED INJECTION TEXT REMOVED]", normalized(original_content))
        if cleaned_content != original_content:
            reported = bool(REPORTED_INSTRUCTION.search(normalized(original_content)))
            notice("reported_instruction_removed" if reported else "source_normalized",
                "An excerpt was safety-redacted; a reported attack instruction was omitted." if reported else "An excerpt was Unicode-normalized before source matching.")
            doc = copy.copy(doc)
            doc.content = cleaned_content
        freshness = "historical" if kind == "ticket" else "current" if reviewed else "unknown"
        if freshness == "historical":
            notice("historical_resolution", "This is a historical incident resolution, not a verified current operating instruction.")
        elif freshness == "unknown":
            notice("freshness_unknown", "No review timestamp is available; source freshness is unknown.")
        selection.documents.append(doc)
        selection.evidence.append(EvidenceItem(citation_id=f"source-{len(selection.documents)}",
            document=doc.source_path, document_id=doc_id, passage=doc.content[:1500], document_kind=kind,
            freshness=freshness, reviewed_at=reviewed.isoformat() if reviewed else None,
            valid_from=valid_from.isoformat() if valid_from else None, valid_until=valid_until.isoformat() if valid_until else None,
            software_version=metadata.get("software_version"), equipment_family=metadata.get("equipment_family")))
        selection.evidence[-1].redacted = cleaned_content != original_content
        selection.evidence[-1].retrieved_content_sha256 = hashlib.sha256(original_content.encode()).hexdigest()
        declared = metadata.get("facts", {})
        scope = metadata.get("fact_scope") or metadata.get("element_id")
        if kind == "ticket":
            # Different historical incidents are not simultaneous current facts.
            continue
        if scope:
            for key in ("software_version", "equipment_family"):
                value = metadata.get(key)
                if value and context.get(key) is None:
                    compatibility.setdefault((scope, key), {}).setdefault(value, []).append(doc_id)
        parsed = [("availability_target", match) for match in re.findall(r"\b(?:target availability|availability(?: target| threshold)?)\s*[:=]\s*(\d+(?:\.\d+)?\s*%)", doc.content, re.IGNORECASE)]
        parsed += [("maximum_latency", match) for match in re.findall(r"\b(?:max(?:imum)? latency)\s*[:=]\s*(\d+(?:\.\d+)?\s*ms)", doc.content, re.IGNORECASE)]
        if (declared or parsed) and not scope:
            notice("conflict_scope_unknown", "Facts cannot be compared without an asset/fact scope.")
        if isinstance(declared, dict) and isinstance(scope, str):
            for key, value in [*declared.items(), *parsed]:
                if not isinstance(key, str) or not 0 < len(key) <= 80 or not isinstance(value, str) or not 0 < len(value) <= 256:
                    notice("facts_invalid", "A structured fact was ignored because its schema is invalid.")
                    continue
                fact_key = (scope, metadata.get("software_version") or "", metadata.get("equipment_family") or "", key)
                canonical = re.sub(r"\s+", " ", value.strip().casefold())
                quantity = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(%|ms)", canonical)
                if quantity:
                    integer, _, fraction = quantity[1].partition(".")
                    fraction = fraction.rstrip("0")
                    canonical = str(int(integer)) + ("." + fraction if fraction else "") + quantity[2]
                ids = facts.setdefault(fact_key, {}).setdefault(canonical, [])
                if doc_id not in ids:
                    ids.append(doc_id)
    for (scope, _, _, key), values in facts.items():
        if len(values) > 1:
            selection.conflicts.append(EvidenceConflict(scope=scope, fact=key,
                values=[ConflictValue(value=value, source_ids=ids) for value, ids in sorted(values.items())]))
    if selection.conflicts:
        selection.warnings.append(SafetyNotice(code="conflicting_sources", message="Current sources disagree on structured facts. Human review is required."))
    for (_, key), values in compatibility.items():
        if len(values) > 1:
            selection.warnings.append(SafetyNotice(code="compatibility_ambiguous", message=f"Sources apply to different {key} values. Specify the intended version/family before using a procedure.", source_ids=[doc_id for ids in values.values() for doc_id in ids]))
    return selection


def link_claims(text, evidence):
    """Require every answer sentence to match the evidence it actually cites.

    Lexical/number matching is deliberately labelled as such, never 'verified'.
    Sources are the exact 1,500-character passages supplied to the prompt.
    """
    text = normalized(text)
    checker = HallucinationDetector()
    if suspicious(CITATIONS.sub("", text)):
        return [], False
    claims, complete = [], True
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        sentence = sentence.strip()
        if not sentence:
            continue
        indices = [int(match) - 1 for match in CITATIONS.findall(sentence)]
        plain = CITATIONS.sub("", sentence).strip(" -*\t")
        if not plain:
            continue
        valid = bool(indices) and all(0 <= index < len(evidence) for index in indices)
        if valid:
            def supported(index):
                passage = evidence[index].passage
                if not checker._is_claim_supported(sentence, [passage]):
                    return False
                tokens = set(re.findall(r"\w+", plain.lower()))
                lines = [line for line in re.split(r"(?<=[.!?])\s+|\n+", passage) if line.strip()]
                closest = max(lines, key=lambda line: len(tokens & set(re.findall(r"\w+", line.lower()))), default=passage)
                def negated(value):
                    return bool(re.search(r"\b(?:not|never|no|without)\b", value, re.IGNORECASE))
                if negated(plain) != negated(closest):
                    return False
                for positive, negative in (("enabled", "disabled"), ("compliant", "noncompliant")):
                    if positive in tokens and re.search(r"\b" + negative + r"\b", closest, re.IGNORECASE):
                        return False
                    if negative in tokens and re.search(r"\b" + positive + r"\b", closest, re.IGNORECASE):
                        return False
                return True
            valid = all(supported(index) for index in indices)
        if not valid:
            complete = False
            continue
        claims.append(ClaimLink(claim=plain, citation_ids=list(dict.fromkeys(evidence[index].citation_id for index in indices))))
    return claims, complete and bool(claims)


def public_evidence(result):
    """Redact every nested public field, including excerpts and conflict values."""
    output_filter = OutputFilter()
    def redact(value):
        if isinstance(value, BaseModel):
            filtered = redact(value.model_dump())
            if isinstance(value, EvidenceItem) and filtered["passage"] != value.passage:
                filtered["redacted"] = True
            return filtered
        if isinstance(value, str):
            return output_filter.apply(value).text
        if isinstance(value, list):
            return [redact(item) for item in value]
        if isinstance(value, dict):
            return {key: redact(item) for key, item in value.items()}
        return value
    return {key: redact(getattr(result, key, [])) for key in ("evidence", "claims", "warnings", "conflicts")}
