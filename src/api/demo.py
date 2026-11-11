"""Dependency-free interview demo backed by the committed evaluated dataset."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from src.evaluation.dataset import load_jsonl
from src.evaluation.retrieval import SearchHit, TicketAwareRerankedRetriever, render_ticket
from src.evidence import (
    ClaimLink,
    EvidenceConflict,
    EvidenceItem,
    SafetyNotice,
    link_claims,
    select_evidence,
)
from src.investigation import Clarification, clarify, query_scope, retrieval_query, scoped_documents
from src.monitoring.tracing import content_attributes, current_trace_id, trace_span
from src.rag_pipeline.retriever import RetrievedDocument


@dataclass(frozen=True)
class DemoAnswer:
    """Grounded extractive answer returned by the offline demo pipeline."""

    text: str
    hits: list[SearchHit]
    confidence: float
    trace_id: str | None
    evidence: list[EvidenceItem] = field(default_factory=list)
    claims: list[ClaimLink] = field(default_factory=list)
    warnings: list[SafetyNotice] = field(default_factory=list)
    conflicts: list[EvidenceConflict] = field(default_factory=list)
    answer_kind: Literal["historical_resolution", "advisory", "insufficient_evidence", "conflict"] = "historical_resolution"
    outcome: Literal["answered", "abstained", "escalated"] = "answered"
    clarification: Clarification = field(default_factory=Clarification)


class OfflineDemoService:
    """Search the versioned ticket corpus without external infrastructure.

    This deliberately does not impersonate the production LLM path. It runs
    the same deterministic configuration-C retriever used by the committed
    benchmark and extracts facts from the retrieved ticket. Golden answers are
    reserved for evaluation and are never read by the answer pipeline.
    """

    def __init__(self, repository_root: Path | None = None, evidence_clock=None, source_review_max_age_days=365):
        self.evidence_clock = evidence_clock
        self.source_review_max_age_days = source_review_max_age_days
        self.repository_root = repository_root or Path(__file__).resolve().parents[2]
        if repository_root is None and not (self.repository_root / "datasets").is_dir():
            self.repository_root = Path(sys.prefix) / "share" / "telecom-genai"
        tickets = load_jsonl(
            self.repository_root
            / "datasets"
            / "tickets-v2.0-ticket-aware-chunking"
            / "tickets.jsonl"
        )
        self.retriever = TicketAwareRerankedRetriever(tickets)

    @staticmethod
    def _confidence(hits: list[SearchHit]) -> float:
        """Calculate a transparent retrieval confidence from overlap and margin."""
        if not hits:
            return 0.0
        top = hits[0]
        second_score = hits[1].score if len(hits) > 1 else 0.0
        margin = max(0.0, (top.score - second_score) / max(top.score, 1e-9))
        overlap = top.component_scores.get("token_overlap", 0.0)
        entity_match = top.component_scores.get("entity_match", 0.0)
        return round(min(1.0, 0.55 * overlap + 0.30 * entity_match + 0.15 * margin), 3)

    def answer(self, query: str, context=None) -> DemoAnswer:
        """Run the offline pipeline with the same trace-stage names as production."""
        with trace_span("query", {**content_attributes(query), "demo.offline": True}):
            clarification = clarify(query, context)
            if clarification.required_fields:
                return DemoAnswer(clarification.question, [], 0.0, current_trace_id(),
                    outcome="abstained", answer_kind="insufficient_evidence", clarification=clarification)
            with trace_span("entity_extraction") as span:
                original_query = query
                query = retrieval_query(query, context)
                entities = self.retriever._entity_terms(query)
                span.set_attribute("telecom.entity.count", len(entities))

            pool = min(80, len(self.retriever.chunks))
            with trace_span("vector_retrieval") as span:
                vector = self.retriever.vector.search(query, pool)
                span.set_attribute("retrieval.vector.result_count", len(vector))

            with trace_span("keyword_retrieval") as span:
                keyword = self.retriever.keyword.search(query, pool)
                span.set_attribute("retrieval.keyword.result_count", len(keyword))

            with trace_span("rank_fusion") as span:
                fused = self.retriever.fuse_rankings(vector, keyword)
                span.set_attribute("retrieval.fused.result_count", len(fused))

            with trace_span("reranker") as span:
                hits = self.retriever.rerank_chunks(query, fused, limit=5)
                span.set_attribute("retrieval.reranked.result_count", len(hits))

            with trace_span("prompt_construction") as span:
                top_ticket = hits[0].ticket_id if hits else None
                ticket = self.retriever.tickets.get(top_ticket or "")
                confidence = self._confidence(hits)
                supported = bool(ticket and confidence >= 0.25)
                candidates = []
                if supported and ticket is not None:
                    for hit in hits:
                        candidate = self.retriever.tickets[hit.ticket_id]
                        if candidate.get("element_id") == ticket.get("element_id"):
                            candidates.append(RetrievedDocument(hit.ticket_id, render_ticket(candidate),
                                confidence, hit.source, candidate))
                candidates = scoped_documents(original_query, context, candidates)
                clarification = clarify(original_query, context, candidates)
                if clarification.required_fields:
                    return DemoAnswer(clarification.question, [], 0.0, current_trace_id(),
                        outcome="abstained", answer_kind="insufficient_evidence", clarification=clarification)
                selection = select_evidence(candidates, {**(context or {}), **query_scope(query, context)}, now=self.evidence_clock() if self.evidence_clock else None,
                    max_review_age_days=self.source_review_max_age_days)
                ambiguous = any(warning.code == "compatibility_ambiguous" for warning in selection.warnings)
                eligible_primary = bool(selection.documents and selection.documents[0].document_id == top_ticket)
                supported = supported and eligible_primary and not selection.conflicts and not ambiguous
                span.set_attribute("demo.evidence.found", supported)

            with trace_span(
                "llm_call",
                {
                    "genai.request.model": "none",
                    "genai.system": "offline-extractive-answer",
                    "demo.llm.skipped": True,
                },
            ):
                if supported and ticket is not None:
                    text = (
                        f"{ticket['ticket_id']}: {ticket['summary']} [Source 1]\n\n"
                        f"Root cause: {ticket['root_cause']} [Source 1]\n"
                        f"Resolution: {ticket['resolution']} [Source 1]\n"
                        f"Runbook: {ticket['runbook_id']} [Source 1]"
                    )
                else:
                    text = (
                        "No grounded answer was found in the committed 50-ticket demo dataset. "
                        "Try one of the supplied example questions."
                    )

            with trace_span("grounding_check") as span:
                claims, fully_linked = link_claims(text, selection.evidence) if supported else ([], False)
                supported = supported and fully_linked
                grounded = supported
                span.set_attribute("genai.grounding.is_grounded", grounded)

            if not supported:
                hits = []
                confidence = 0.0
                text = "Sources disagree on current facts. Review the conflicting evidence with a network engineer." if selection.conflicts else "Sources apply to different versions or equipment families. Specify the intended compatibility context." if ambiguous else "No grounded answer was found in the committed 50-ticket demo dataset. Try one of the supplied example questions."
            trace_id = current_trace_id()
            with trace_span(
                "final_response",
                {
                    "genai.response.confidence": confidence,
                    "genai.response.source_count": int(bool(hits)),
                },
            ):
                return DemoAnswer(
                    text=text,
                    hits=hits,
                    confidence=confidence,
                    trace_id=trace_id,
                    evidence=selection.evidence, claims=claims if supported else [], warnings=selection.warnings,
                    conflicts=selection.conflicts,
                    answer_kind="conflict" if selection.conflicts else "historical_resolution" if supported and selection.evidence[0].freshness == "historical" else "advisory" if supported else "insufficient_evidence",
                    outcome="escalated" if selection.conflicts or ambiguous else "answered" if supported else "abstained",
                    clarification=Clarification(required_fields=["software_version", "equipment_family"],
                        question="Which software version and equipment family apply to this incident?") if ambiguous else Clarification(),
                )
