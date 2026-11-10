"""Dependency-free retrieval baselines used by the reproducible benchmark.

The benchmark intentionally runs without an API key. Semantic retrieval uses a
stable 384-dimensional feature-hashing embedding; lexical retrieval is BM25;
rank fusion is weighted RRF; and the final reranker is deterministic and
field-aware. This makes CI results reproducible while keeping the production
retriever free to use OpenAI or SentenceTransformers embeddings.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-_/.:][a-z0-9]+)*", re.IGNORECASE)
TECH_ENTITY_PATTERN = re.compile(
    r"\b(?:INC\d+|[A-Z]{2,}[A-Z0-9]*-[A-Z0-9-]+|TAC\s*\d+|"
    r"S-NSSAI\s*[0-9-]+|[A-Z]+\d+(?:/\d+)+|\d+(?:\.\d+){3}|"
    r"[A-Z]{2,}(?:-[A-Z0-9]+)?)\b",
)
STOP_WORDS = {
    "a", "after", "all", "an", "and", "at", "be", "being", "by", "did",
    "for", "from", "had", "has", "in", "is", "it", "its", "of", "on",
    "or", "the", "their", "this", "to", "was", "were", "what", "when",
    "where", "which", "with",
}
SEMANTIC_NORMALIZATION = {
    "addressed": "fix", "corrected": "fix", "fixed": "fix", "repair": "fix",
    "repaired": "fix", "resolved": "fix", "restored": "fix", "recover": "fix",
    "recovered": "fix", "stopped": "fix", "failed": "failure", "failures": "failure",
    "failing": "failure", "rejected": "reject", "rejects": "reject",
    "timeouts": "timeout", "timed": "timeout", "errors": "error", "calls": "call",
    "sessions": "session", "registrations": "registration", "responses": "response",
    "messages": "message", "removed": "remove", "added": "add", "missing": "absent",
    "outage": "failure", "down": "failure", "unavailable": "failure",
}


def tokenize(text: str, *, semantic: bool = False) -> list[str]:
    tokens: list[str] = []
    for original in TOKEN_PATTERN.findall(text):
        raw = original.lower()
        candidates = [raw]
        if any(separator in raw for separator in "-_/.:"):
            candidates.extend(part for part in re.split(r"[-_/.:]+", raw) if part)
            candidates.append(re.sub(r"[-_/.:]+", "", raw))
        for token in candidates:
            if token in STOP_WORDS or len(token) < 2:
                continue
            # Opaque identifiers and short acronyms are intentionally excluded
            # from the semantic baseline. Real sentence embeddings commonly
            # underweight these tokens; BM25 in configurations B/C retains them.
            if semantic and (
                any(char.isdigit() for char in token)
                or len(token) <= 3
                or (original.isupper() and len(original) <= 6)
            ):
                continue
            tokens.append(SEMANTIC_NORMALIZATION.get(token, token) if semantic else token)
    return tokens


@dataclass
class Chunk:
    chunk_id: str
    ticket_id: str
    source: str
    content: str
    fields: dict[str, str] = field(default_factory=dict)


@dataclass
class SearchHit:
    ticket_id: str
    source: str
    score: float
    content: str
    component_scores: dict[str, float] = field(default_factory=dict)


def render_ticket(ticket: dict[str, Any]) -> str:
    fields = [
        "ticket_id", "status", "severity", "region", "element_id", "interface",
        "service", "summary", "description", "symptoms", "root_cause", "resolution",
        "runbook_id",
    ]
    return "\n".join(f"{field}: {ticket.get(field, '')}" for field in fields)


def field_rerank_score(query: str, ticket_id: str, text: str, fields: dict, fused_score: float):
    """Shared ranking policy used by offline evaluation and live retrieval."""
    query_tokens = set(tokenize(query))
    entities = {match.lower().replace(" ", "") for match in TECH_ENTITY_PATTERN.findall(query)}
    entities.update(token for token in TOKEN_PATTERN.findall(query.lower())
                    if any(char.isdigit() for char in token) or (token.isalpha() and 2 <= len(token) <= 6))
    entities -= STOP_WORDS
    full_text = text.lower()
    overlap = len(query_tokens & set(tokenize(full_text))) / max(len(query_tokens), 1)
    entity_match = sum(entity in full_text.replace(" ", "") for entity in entities) / max(len(entities), 1)
    exact_ticket = float(ticket_id.lower() in query.lower())
    field_bonus = min(sum(0.25 for key in ("element_id", "interface", "runbook_id", "service")
                          if str(fields.get(key, "")).lower() and str(fields[key]).lower() in query.lower()), 1.0)
    score = 0.35 * fused_score + 0.25 * overlap + 0.25 * entity_match + 0.10 * exact_ticket + 0.05 * field_bonus
    return score, {"token_overlap": overlap, "entity_match": entity_match, "field_bonus": field_bonus}


def recursive_chunks(
    tickets: list[dict[str, Any]], chunk_size: int = 240, overlap: int = 40,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    step = chunk_size - overlap
    for ticket in tickets:
        text = render_ticket(ticket)
        starts = list(range(0, len(text), step)) or [0]
        for index, start in enumerate(starts):
            content = text[start : start + chunk_size]
            if content:
                chunks.append(
                    Chunk(
                        chunk_id=f"{ticket['ticket_id']}:recursive:{index}",
                        ticket_id=ticket["ticket_id"],
                        source=ticket["source"],
                        content=content,
                    )
                )
            if start + chunk_size >= len(text):
                break
    return chunks


def ticket_aware_chunks(tickets: list[dict[str, Any]]) -> list[Chunk]:
    chunks: list[Chunk] = []
    field_groups = {
        "identity": ["ticket_id", "severity", "region", "element_id", "interface", "service", "runbook_id"],
        "incident": ["ticket_id", "element_id", "interface", "summary", "description", "symptoms"],
        "resolution": ["ticket_id", "element_id", "root_cause", "resolution", "runbook_id"],
    }
    for ticket in tickets:
        for group, fields in field_groups.items():
            selected = {field: str(ticket.get(field, "")) for field in fields}
            content = "\n".join(f"{field}: {value}" for field, value in selected.items())
            chunks.append(
                Chunk(
                    chunk_id=f"{ticket['ticket_id']}:{group}",
                    ticket_id=ticket["ticket_id"],
                    source=ticket["source"],
                    content=content,
                    fields=selected,
                )
            )
    return chunks


class HashingVectorIndex:
    """TF-IDF weighted dense feature-hashing index with cosine search."""

    def __init__(self, chunks: list[Chunk], dimensions: int = 384):
        self.chunks = chunks
        self.dimensions = dimensions
        tokenized = [set(tokenize(chunk.content, semantic=True)) for chunk in chunks]
        frequencies = Counter(token for tokens in tokenized for token in tokens)
        self.idf = {
            token: math.log((len(chunks) + 1) / (frequency + 1)) + 1
            for token, frequency in frequencies.items()
        }
        self.vectors = [self._embed(chunk.content) for chunk in chunks]

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dimensions, 1.0 if value & 1 else -1.0

    def _embed(self, text: str) -> list[float]:
        terms = tokenize(text, semantic=True)
        counts = Counter(terms)
        vector = [0.0] * self.dimensions
        for term, count in counts.items():
            bucket, sign = self._bucket(term)
            vector[bucket] += sign * (1 + math.log(count)) * self.idf.get(term, 1.0)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def search(self, query: str, limit: int) -> list[tuple[Chunk, float]]:
        query_vector = self._embed(query)
        scored = [
            (chunk, sum(left * right for left, right in zip(query_vector, vector)))
            for chunk, vector in zip(self.chunks, self.vectors)
        ]
        scored = [item for item in scored if item[1] > 0]
        return sorted(scored, key=lambda item: (-item[1], item[0].chunk_id))[:limit]


class BM25Index:
    """Small Okapi BM25 implementation for exact telecom terminology."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.documents = [tokenize(chunk.content) for chunk in chunks]
        self.lengths = [len(tokens) for tokens in self.documents]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        self.term_frequencies = [Counter(tokens) for tokens in self.documents]
        document_frequency = Counter(token for tokens in self.documents for token in set(tokens))
        count = len(chunks)
        self.idf = {
            token: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def search(self, query: str, limit: int) -> list[tuple[Chunk, float]]:
        query_terms = tokenize(query)
        scored: list[tuple[Chunk, float]] = []
        for chunk, frequencies, length in zip(self.chunks, self.term_frequencies, self.lengths):
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / max(self.average_length, 1)
                )
                score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1) / denominator
            if score > 0:
                scored.append((chunk, score))
        return sorted(scored, key=lambda item: (-item[1], item[0].chunk_id))[:limit]


def _roll_up(
    ranked_chunks: Iterable[tuple[Chunk, float]], limit: int, component: str,
) -> list[SearchHit]:
    by_ticket: dict[str, SearchHit] = {}
    for chunk, score in ranked_chunks:
        previous = by_ticket.get(chunk.ticket_id)
        if previous is None or score > previous.score:
            by_ticket[chunk.ticket_id] = SearchHit(
                ticket_id=chunk.ticket_id,
                source=chunk.source,
                score=score,
                content=chunk.content,
                component_scores={component: score},
            )
    return sorted(by_ticket.values(), key=lambda hit: (-hit.score, hit.ticket_id))[:limit]


class RecursiveVectorRetriever:
    """Configuration A: generic recursive chunks and dense vector search."""

    name = "A_recursive_vector"

    def __init__(self, tickets: list[dict[str, Any]]):
        self.chunks = recursive_chunks(tickets)
        self.vector = HashingVectorIndex(self.chunks)

    def search(self, query: str, limit: int = 10) -> list[SearchHit]:
        return _roll_up(self.vector.search(query, limit=len(self.chunks)), limit, "vector")


class TicketAwareHybridRetriever:
    """Configuration B: field-preserving chunks, BM25, and weighted RRF."""

    name = "B_ticket_aware_hybrid"

    def __init__(self, tickets: list[dict[str, Any]]):
        self.tickets = {ticket["ticket_id"]: ticket for ticket in tickets}
        self.chunks = ticket_aware_chunks(tickets)
        self.vector = HashingVectorIndex(self.chunks)
        self.keyword = BM25Index(self.chunks)

    def fuse_rankings(
        self,
        vector: list[tuple[Chunk, float]],
        keyword: list[tuple[Chunk, float]],
    ) -> list[tuple[Chunk, float]]:
        """Fuse already-computed vector and keyword candidates with weighted RRF."""
        scores: dict[str, float] = defaultdict(float)
        chunks: dict[str, Chunk] = {}
        for weight, ranking in ((0.65, vector), (0.35, keyword)):
            for rank, (chunk, _) in enumerate(ranking, 1):
                scores[chunk.chunk_id] += weight / (60 + rank)
                chunks[chunk.chunk_id] = chunk
        return sorted(
            ((chunks[chunk_id], score) for chunk_id, score in scores.items()),
            key=lambda item: (-item[1], item[0].chunk_id),
        )

    def _fused_chunks(self, query: str) -> list[tuple[Chunk, float]]:
        pool = min(80, len(self.chunks))
        vector = self.vector.search(query, pool)
        keyword = self.keyword.search(query, pool)
        return self.fuse_rankings(vector, keyword)

    def search(self, query: str, limit: int = 10) -> list[SearchHit]:
        return _roll_up(self._fused_chunks(query), limit, "rrf")


class TicketAwareRerankedRetriever(TicketAwareHybridRetriever):
    """Configuration C: hybrid candidates with a deterministic field reranker."""

    name = "C_ticket_aware_hybrid_reranker"

    def _entity_terms(self, query: str) -> set[str]:
        entities = {match.lower().replace(" ", "") for match in TECH_ENTITY_PATTERN.findall(query)}
        entities.update(
            token for token in TOKEN_PATTERN.findall(query.lower())
            if any(char.isdigit() for char in token) or (token.isalpha() and 2 <= len(token) <= 6)
        )
        return entities - STOP_WORDS

    def rerank_chunks(
        self,
        query: str,
        fused_chunks: list[tuple[Chunk, float]],
        limit: int = 10,
    ) -> list[SearchHit]:
        """Rerank a fused chunk list while preserving the benchmark algorithm."""
        candidates = _roll_up(fused_chunks, 25, "rrf")
        reranked: list[SearchHit] = []

        for candidate in candidates:
            ticket = self.tickets[candidate.ticket_id]
            rerank_score, components = field_rerank_score(
                query, candidate.ticket_id, render_ticket(ticket), ticket, candidate.score,
            )
            reranked.append(
                SearchHit(
                    ticket_id=candidate.ticket_id,
                    source=candidate.source,
                    score=rerank_score,
                    content=render_ticket(ticket),
                    component_scores={
                        **candidate.component_scores,
                        **components,
                    },
                )
            )
        return sorted(reranked, key=lambda hit: (-hit.score, hit.ticket_id))[:limit]

    def search(self, query: str, limit: int = 10) -> list[SearchHit]:
        return self.rerank_chunks(query, self._fused_chunks(query), limit)
