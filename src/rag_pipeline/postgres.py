"""Immutable ticket snapshots and bounded exact-vector/BM25 retrieval in PostgreSQL."""

import asyncio
import hashlib
import json
import math
import time
from collections import defaultdict

from src.evaluation.retrieval import (
    BM25Index,
    Chunk,
    field_rerank_score,
    render_ticket,
    ticket_aware_chunks,
)
from src.monitoring.tracing import content_attributes, trace_span
from src.rag_pipeline.retriever import RetrievalResult, RetrievedDocument

SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_versions (
    name TEXT PRIMARY KEY, source_sha256 TEXT NOT NULL,
    embedding_model TEXT NOT NULL, dimensions INTEGER NOT NULL,
    documents INTEGER NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    version TEXT NOT NULL REFERENCES knowledge_versions(name),
    chunk_id TEXT NOT NULL, ticket_id TEXT NOT NULL, source TEXT NOT NULL,
    content TEXT NOT NULL, metadata JSONB NOT NULL, embedding DOUBLE PRECISION[] NOT NULL,
    PRIMARY KEY(version, chunk_id)
);
"""


class PostgresTicketRetriever:
    def __init__(self, store, embeddings, version: str, model: str):
        self.store, self.embeddings, self.version, self.model = store, embeddings, version, model

    async def start(self, require_data=True):
        await asyncio.to_thread(self.store._transaction, SCHEMA)
        if require_data:
            await self.ready()

    async def ready(self):
        rows = await asyncio.to_thread(self.store._transaction,
            "SELECT embedding_model FROM knowledge_versions WHERE name=%s", (self.version,), True)
        if not rows or rows[0][0] != self.model:
            raise RuntimeError("Publish a non-empty snapshot with the configured embedding model before live startup")

    async def publish(self, tickets: list[dict], source_bytes: bytes):
        if not tickets or len({row["ticket_id"] for row in tickets}) != len(tickets):
            raise ValueError("Tickets must be non-empty and have unique IDs")
        chunks = ticket_aware_chunks(tickets)
        if len(chunks) > 10000:
            raise ValueError("Release capacity is 10,000 chunks per snapshot")
        vectors = await self.embeddings.embed_documents([chunk.content for chunk in chunks])
        if len(vectors) != len(chunks) or not vectors or not vectors[0]:
            raise ValueError("Embedding provider returned invalid results")
        dimensions = len(vectors[0])
        if any(len(vector) != dimensions or not all(math.isfinite(value) for value in vector) for vector in vectors):
            raise ValueError("Embeddings must have consistent finite dimensions")
        await asyncio.to_thread(self._publish, tickets, chunks, vectors, dimensions, hashlib.sha256(source_bytes).hexdigest())
        return {"version": self.version, "documents": len(tickets), "chunks": len(chunks), "dimensions": dimensions}

    def _publish(self, tickets, chunks, vectors, dimensions, digest):
        pool = self.store.pool
        connection = pool.getconn()
        by_id = {row["ticket_id"]: row for row in tickets}
        try:
            with connection:
                with connection.cursor() as cursor:
                    # A duplicate version fails; publishing never overwrites an accepted index.
                    cursor.execute("INSERT INTO knowledge_versions(name,source_sha256,embedding_model,dimensions,documents) VALUES (%s,%s,%s,%s,%s)",
                                   (self.version, digest, self.model, dimensions, len(tickets)))
                    cursor.executemany("INSERT INTO knowledge_chunks(version,chunk_id,ticket_id,source,content,metadata,embedding) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                        [(self.version, chunk.chunk_id, chunk.ticket_id, chunk.source, chunk.content,
                          json.dumps(by_id[chunk.ticket_id]), vector) for chunk, vector in zip(chunks, vectors)])
        finally:
            pool.putconn(connection)

    async def search(self, query: str, top_k=5, filters=None):
        started = time.monotonic()
        rows = await asyncio.to_thread(self.store._transaction,
            "SELECT chunk_id,ticket_id,source,content,metadata,embedding FROM knowledge_chunks WHERE version=%s LIMIT 10001",
            (self.version,), True)
        if len(rows) > 10000:
            raise RuntimeError("Snapshot exceeds release capacity")
        rows = [row for row in rows if all(str(row[4].get(key, "")) == str(value)
                for key, value in (filters or {}).items() if key in ("region", "source"))]
        chunks = [Chunk(row[0], row[1], row[2], row[3]) for row in rows]
        with trace_span("vector_retrieval", content_attributes(query)) as span:
            vector = await self.embeddings.embed_query(query)
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            ranking = []
            for chunk, row in zip(chunks, rows):
                stored = row[5]
                if len(stored) != len(vector):
                    raise RuntimeError("Embedding dimension mismatch")
                denominator = norm * (math.sqrt(sum(value * value for value in stored)) or 1.0)
                score = sum(a * b for a, b in zip(vector, stored)) / denominator
                if score > 0:
                    ranking.append((chunk, score))
            ranking.sort(key=lambda item: (-item[1], item[0].chunk_id))
            span.set_attribute("retrieval.vector.result_count", len(ranking))
        with trace_span("keyword_retrieval"):
            keyword = BM25Index(chunks).search(query, 80)
        with trace_span("rank_fusion"):
            scores: dict[str, float] = defaultdict(float)
            for weight, results in ((0.65, ranking[:80]), (0.35, keyword)):
                for index, (chunk, _) in enumerate(results, 1):
                    scores[chunk.chunk_id] += weight / (60 + index)
        with trace_span("reranker"):
            metadata = {row[1]: row[4] for row in rows}
            ticket_scores: dict[str, float] = {}
            sources = {row[1]: row[2] for row in rows}
            for chunk in chunks:
                if chunk.chunk_id in scores:
                    ticket_scores[chunk.ticket_id] = max(ticket_scores.get(chunk.ticket_id, 0), scores[chunk.chunk_id])
            candidates = sorted(ticket_scores, key=lambda ticket: (-ticket_scores[ticket], ticket))[:25]
            documents = []
            for ticket in candidates:
                content = render_ticket(metadata[ticket])
                score, _ = field_rerank_score(query, ticket, content, metadata[ticket], ticket_scores[ticket])
                documents.append(RetrievedDocument(ticket, content, score, sources[ticket], metadata[ticket], "hybrid"))
            documents.sort(key=lambda doc: (-doc.relevance_score, doc.document_id))
        return RetrievalResult(documents[:top_k], query, len(documents), (time.monotonic() - started) * 1000)

    async def close(self):
        await self.embeddings.close()
