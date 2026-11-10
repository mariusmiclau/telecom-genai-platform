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
    recursive_chunks,
    render_ticket,
    ticket_aware_chunks,
)
from src.monitoring.tracing import content_attributes, trace_span
from src.rag_pipeline.retriever import RetrievalResult, RetrievedDocument
from src.security import current_principal


class PostgresTicketRetriever:
    def __init__(self, store, embeddings, version: str, model: str, dimensions: int | None = None, configuration="C", job_id=None, lease_token=None):
        self.store, self.embeddings, self.version, self.model = store, embeddings, version, model
        self.dimensions = dimensions
        if configuration not in ("A", "B", "C"):
            raise ValueError("Unknown retrieval configuration")
        self.configuration, self.job_id, self.lease_token = configuration, job_id, lease_token

    async def start(self, require_data=True):
        if require_data:
            await self.ready()

    async def ready(self):
        rows = await asyncio.to_thread(self.store._transaction,
            "SELECT embedding_model,dimensions,documents,configuration,(SELECT count(*) FROM knowledge_chunks WHERE version=name) FROM knowledge_versions WHERE name=%s", (self.version,), True)
        if not rows or rows[0][0] != self.model or (self.dimensions and rows[0][1] != self.dimensions) or rows[0][3] != self.configuration:
            raise RuntimeError("Publish a non-empty snapshot with the configured embedding model before live startup")
        if not rows[0][4] or rows[0][4] > 10000 or rows[0][1] * rows[0][4] > 2_000_000:
            raise RuntimeError("Snapshot exceeds bounded vector capacity")

    async def publish(self, tickets: list[dict], source_bytes: bytes):
        if not tickets or len({row["ticket_id"] for row in tickets}) != len(tickets):
            raise ValueError("Tickets must be non-empty and have unique IDs")
        chunks = recursive_chunks(tickets) if self.configuration == "A" else ticket_aware_chunks(tickets)
        if len(chunks) > 10000:
            raise ValueError("Release capacity is 10,000 chunks per snapshot")
        if self.dimensions and len(chunks) * self.dimensions > 2_000_000:
            raise ValueError("Snapshot exceeds bounded vector capacity; use smaller snapshots")
        vectors = await self.embeddings.embed_documents([chunk.content for chunk in chunks])
        if len(vectors) != len(chunks) or not vectors or not vectors[0]:
            raise ValueError("Embedding provider returned invalid results")
        dimensions = len(vectors[0])
        if self.dimensions and dimensions != self.dimensions:
            raise ValueError("Provider returned dimensions different from configured embedding contract")
        if len(chunks) * dimensions > 2_000_000:
            raise ValueError("Snapshot exceeds bounded vector capacity")
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
                    if self.job_id:
                        cursor.execute("SELECT id FROM ingestion_jobs WHERE id=%s AND lease_token=%s AND state='running' AND lease_until>now() FOR UPDATE", (self.job_id, self.lease_token))
                        if not cursor.fetchone():
                            raise RuntimeError("Publication lease was lost")
                    # A duplicate version fails; publishing never overwrites an accepted index.
                    cursor.execute("INSERT INTO knowledge_versions(name,source_sha256,embedding_model,dimensions,documents,configuration,publish_job_id) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                                   (self.version, digest, self.model, dimensions, len(tickets), self.configuration, self.job_id))
                    cursor.executemany("INSERT INTO knowledge_chunks(version,chunk_id,ticket_id,source,content,metadata,embedding) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                        [(self.version, chunk.chunk_id, chunk.ticket_id, chunk.source, chunk.content,
                          json.dumps(by_id[chunk.ticket_id]), vector) for chunk, vector in zip(chunks, vectors)])
        finally:
            pool.putconn(connection)

    async def search(self, query: str, top_k=5, filters=None):
        started = time.monotonic()
        principal = current_principal()
        restriction = ""
        parameters: list[object] = [self.version]
        if principal.enforce_acl:
            restriction = " AND metadata->>'tenant_id'=%s AND EXISTS (SELECT 1 FROM jsonb_array_elements_text(CASE WHEN jsonb_typeof(metadata->'acl_groups')='array' THEN metadata->'acl_groups' ELSE '[]'::jsonb END) AS acl(value) WHERE acl.value=ANY(%s))"
            parameters.extend([principal.tenant, list(principal.groups)])
        else:
            # Shared-token compatibility never grants access to another tenant's
            # explicitly labelled records, even after a configuration downgrade.
            restriction = " AND (metadata->>'tenant_id' IS NULL OR metadata->>'tenant_id'=%s)"
            parameters.append(principal.tenant)
        rows = await asyncio.to_thread(self.store._transaction,
            "SELECT chunk_id,ticket_id,source,content,metadata,embedding FROM knowledge_chunks WHERE version=%s" + restriction + " LIMIT 10001",
            tuple(parameters), True)
        if len(rows) > 10000:
            raise RuntimeError("Snapshot exceeds release capacity")
        rows = [row for row in rows if all(str(row[4].get(key, "")) == str(value)
                for key, value in (filters or {}).items() if key in ("region", "source"))]
        chunks = [Chunk(row[0], row[1], row[2], row[3]) for row in rows]
        if not chunks:
            return RetrievalResult([], query, 0, (time.monotonic() - started) * 1000)
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
            keyword = BM25Index(chunks).search(query, 80) if self.configuration != "A" else []
        with trace_span("rank_fusion"):
            scores: dict[str, float] = defaultdict(float)
            for weight, results in ((0.65, ranking[:80]), (0.35, keyword)):
                for index, (chunk, rank_score) in enumerate(results, 1):
                    scores[chunk.chunk_id] += rank_score if self.configuration == "A" else weight / (60 + index)
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
                score = ticket_scores[ticket]
                if self.configuration == "C":
                    score, _ = field_rerank_score(query, ticket, content, metadata[ticket], score)
                documents.append(RetrievedDocument(ticket, content, score, sources[ticket], metadata[ticket], "hybrid"))
            documents.sort(key=lambda doc: (-doc.relevance_score, doc.document_id))
        return RetrievalResult(documents[:top_k], query, len(documents), (time.monotonic() - started) * 1000)

    async def close(self):
        await self.embeddings.close()
