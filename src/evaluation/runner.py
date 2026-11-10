"""Metric calculation and report generation for the three retrieval configurations."""

from __future__ import annotations

import math
import statistics
import time
from collections import defaultdict
from typing import Any, Protocol

from src.evaluation.retrieval import (
    RecursiveVectorRetriever,
    SearchHit,
    TicketAwareHybridRetriever,
    TicketAwareRerankedRetriever,
    render_ticket,
    tokenize,
)


class RetrieverProtocol(Protocol):
    name: str

    def search(self, query: str, limit: int = 10) -> list[SearchHit]: ...


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(math.ceil(percentile * len(ordered)) - 1, len(ordered) - 1)
    return ordered[max(index, 0)]


def _round(value: float) -> float:
    return round(value, 4)


class EvaluationRunner:
    """Evaluate identical questions against A, B, and C configurations."""

    def __init__(self, tickets: list[dict[str, Any]], questions: list[dict[str, Any]], retrievers: list[RetrieverProtocol] | None = None):
        self.tickets = tickets
        self.questions = questions
        self.ticket_text = {ticket["ticket_id"]: render_ticket(ticket) for ticket in tickets}
        self.retrievers: list[RetrieverProtocol] = retrievers if retrievers is not None else [
            RecursiveVectorRetriever(tickets),
            TicketAwareHybridRetriever(tickets),
            TicketAwareRerankedRetriever(tickets),
        ]

    def _evaluate_question(
        self, question: dict[str, Any], hits: list[SearchHit],
    ) -> dict[str, float | int | None]:
        expected_tickets = set(question["expected_ticket_ids"])
        expected_sources = set(question["expected_sources"])
        ranked_tickets = [hit.ticket_id for hit in hits]
        ranked_sources = [hit.source for hit in hits]
        rank = next(
            (index for index, ticket_id in enumerate(ranked_tickets, 1) if ticket_id in expected_tickets),
            None,
        )
        reference_tokens = set(tokenize(question["reference_answer"], semantic=True))
        evidence_tokens = set(
            tokenize(" ".join(self.ticket_text[hit.ticket_id] for hit in hits[:5]), semantic=True)
        )
        evidence_coverage = len(reference_tokens & evidence_tokens) / max(len(reference_tokens), 1)
        return {
            "rank": rank,
            "recall_at_1": float(rank is not None and rank <= 1),
            "recall_at_5": float(rank is not None and rank <= 5),
            "recall_at_10": float(rank is not None and rank <= 10),
            "source_recall_at_5": float(bool(expected_sources & set(ranked_sources[:5]))),
            "reciprocal_rank": 1 / rank if rank else 0.0,
            "ndcg_at_5": 1 / math.log2(rank + 1) if rank and rank <= 5 else 0.0,
            "evidence_coverage_at_5": evidence_coverage,
        }

    def _aggregate(self, rows: list[dict[str, Any]], latencies: list[float]) -> dict[str, float | int]:
        keys = [
            "recall_at_1", "recall_at_5", "recall_at_10", "source_recall_at_5",
            "reciprocal_rank", "ndcg_at_5", "evidence_coverage_at_5",
        ]
        metrics: dict[str, float | int] = {
            key: _round(statistics.fmean(float(row[key]) for row in rows)) for key in keys
        }
        metrics.update(
            {
                "questions": len(rows),
                "latency_p50_ms": _round(_percentile(latencies, 0.50)),
                "latency_p95_ms": _round(_percentile(latencies, 0.95)),
            }
        )
        return metrics

    def run(self) -> dict[str, Any]:
        configurations: dict[str, Any] = {}
        question_results: dict[str, list[dict[str, Any]]] = {}

        for retriever in self.retrievers:
            rows: list[dict[str, Any]] = []
            latencies: list[float] = []
            by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for question in self.questions:
                start = time.perf_counter()
                hits = retriever.search(question["question"], limit=10)
                latencies.append((time.perf_counter() - start) * 1000)
                values = self._evaluate_question(question, hits)
                row = {
                    "question_id": question["question_id"],
                    "expected_ticket_ids": question["expected_ticket_ids"],
                    "retrieved_ticket_ids": [hit.ticket_id for hit in hits],
                    **values,
                }
                rows.append(row)
                for tag in question["tags"]:
                    by_tag[tag].append(row)

            configurations[retriever.name] = {
                "metrics": self._aggregate(rows, latencies),
                "breakdown": {
                    tag: {
                        "questions": len(tag_rows),
                        "recall_at_5": _round(
                            statistics.fmean(float(row["recall_at_5"]) for row in tag_rows)
                        ),
                        "reciprocal_rank": _round(
                            statistics.fmean(float(row["reciprocal_rank"]) for row in tag_rows)
                        ),
                    }
                    for tag, tag_rows in sorted(by_tag.items())
                },
            }
            question_results[retriever.name] = rows

        return {
            "evaluation_name": "telecom-ticket-retrieval-golden-v1",
            "dataset_version": "tickets-v2.0-ticket-aware-chunking",
            "golden_questions": len(self.questions),
            "embedding": {"model": "telecom-hashing-embedding-v1", "dimensions": 384},
            "configurations": configurations,
            "question_results": question_results,
        }
