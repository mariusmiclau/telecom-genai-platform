"""Hybrid Retriever combining semantic and keyword search.

Implements a fusion retrieval strategy for telecom knowledge base:
1. Semantic search via ChromaDB embeddings
2. Keyword search for exact matches (node IDs, config params)
3. Reciprocal Rank Fusion to combine results
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from src.monitoring.tracing import content_attributes, trace_span
from src.rag_pipeline.embeddings import EmbeddingService, EmbeddingSettings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """A document retrieved from the knowledge base."""

    document_id: str
    content: str
    relevance_score: float
    source_path: str = ""
    metadata: dict = field(default_factory=dict)
    retrieval_method: str = "semantic"  # semantic, keyword, hybrid


@dataclass
class RetrievalResult:
    """Result from retrieval operation."""

    documents: list[RetrievedDocument]
    query: str
    total_found: int
    retrieval_time_ms: float


class HybridRetriever:
    """Hybrid retrieval combining semantic and keyword search.

    Strategy:
    1. Semantic Search (ChromaDB):
       - Embed query using same model as documents
       - Find nearest neighbors in vector space
       - Good for conceptual/semantic similarity

    2. Keyword Search (BM25-style):
       - Extract key terms from query
       - Match against document metadata and content
       - Good for exact matches (node IDs, config values)

    3. Reciprocal Rank Fusion:
       - Combine rankings from both methods
       - Score = sum(1 / (k + rank)) for each method
       - Balances semantic understanding with keyword precision

    Telecom-specific features:
    - Boosts documents matching extracted entities (node IDs, regions)
    - Filters by region/document type when specified
    - Handles config file formats specially
    """

    def __init__(
        self,
        chroma_host: Optional[str] = None,
        chroma_port: Optional[int] = None,
        collection_name: str = "telecom_kb",
        embedding_settings: Optional[EmbeddingSettings] = None,
    ):
        self.chroma_host = chroma_host or os.getenv("CHROMA_HOST", "localhost")
        self.chroma_port = chroma_port or int(os.getenv("CHROMA_PORT", "8100"))
        self.collection_name = collection_name
        self.embedding_settings = embedding_settings or EmbeddingSettings()

        self._chroma_client = None
        self._collection = None
        self._embedding_service: EmbeddingService | None = None

        # RRF parameter (typically 60)
        self.rrf_k = 60

        # Weights for combining methods
        self.semantic_weight = 0.65
        self.keyword_weight = 0.35

    def _get_chroma_client(self):
        """Get or create ChromaDB client."""
        if self._chroma_client is None:
            import chromadb
            from chromadb.config import Settings
            self._chroma_client = chromadb.HttpClient(
                host=self.chroma_host,
                port=self.chroma_port,
                settings=Settings(anonymized_telemetry=False),
            )
        return self._chroma_client

    def _get_collection(self):
        """Get or create ChromaDB collection."""
        if self._collection is None:
            client = self._get_chroma_client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    async def _get_embedding_service(self) -> EmbeddingService:
        """Get initialized embedding service."""
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService(self.embedding_settings)
            await self._embedding_service.initialize()
        return self._embedding_service

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
        semantic_only: bool = False,
        rerank: bool = True,
    ) -> RetrievalResult:
        """Perform hybrid search over the knowledge base.

        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Metadata filters (region, doc_type, etc.)
            semantic_only: Skip keyword search if True

        Returns:
            RetrievalResult with ranked documents
        """
        import time
        start_time = time.time()

        # Build ChromaDB where filter
        where_filter = self._build_where_filter(filters)

        # Get results from both methods. Each stage is a child span in the
        # end-to-end request trace.
        with trace_span("vector_retrieval", content_attributes(query)) as span:
            semantic_docs = await self._semantic_search(
                query=query,
                top_k=top_k * 4,  # Candidate pool for fusion/reranking
                where=where_filter,
            )
            span.set_attribute("retrieval.vector.result_count", len(semantic_docs))

        if semantic_only:
            keyword_docs = []
        else:
            with trace_span("keyword_retrieval") as span:
                keyword_docs = await self._keyword_search(
                    query=query,
                    top_k=top_k * 4,
                    where=where_filter,
                )
                span.set_attribute("retrieval.keyword.result_count", len(keyword_docs))

        # Fuse results using RRF
        with trace_span("rank_fusion") as span:
            fused_docs = self._reciprocal_rank_fusion(
                semantic_docs=semantic_docs,
                keyword_docs=keyword_docs,
                top_k=top_k * 4 if rerank and not semantic_only else top_k,
            )
            span.set_attribute("retrieval.fused.result_count", len(fused_docs))

        if rerank and not semantic_only:
            with trace_span("reranker") as span:
                fused_docs = self._rerank_documents(query, fused_docs, top_k)
                span.set_attribute("retrieval.reranked.result_count", len(fused_docs))

        retrieval_time = (time.time() - start_time) * 1000

        logger.info(
            "Hybrid search completed",
            extra={
                "query_length": len(query),
                "semantic_results": len(semantic_docs),
                "keyword_results": len(keyword_docs),
                "fused_results": len(fused_docs),
                "retrieval_time_ms": retrieval_time,
            },
        )

        return RetrievalResult(
            documents=fused_docs,
            query=query,
            total_found=len(fused_docs),
            retrieval_time_ms=retrieval_time,
        )

    async def _semantic_search(
        self,
        query: str,
        top_k: int,
        where: Optional[dict] = None,
    ) -> list[RetrievedDocument]:
        """Perform semantic search using embeddings."""
        try:
            # Get query embedding
            embedding_service = await self._get_embedding_service()
            query_embedding = await embedding_service.embed_query(query)

            # Search ChromaDB
            collection = self._get_collection()

            results = await asyncio.to_thread(collection.query,
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            documents: list[RetrievedDocument] = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    # Convert distance to similarity score (cosine)
                    distance = results["distances"][0][i] if results["distances"] else 0
                    similarity = 1 - distance  # Cosine distance to similarity

                    documents.append(RetrievedDocument(
                        document_id=doc_id,
                        content=results["documents"][0][i] if results["documents"] else "",
                        relevance_score=max(0, similarity),
                        source_path=results["metadatas"][0][i].get("source_path", "")
                            if results["metadatas"] else "",
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                        retrieval_method="semantic",
                    ))

            return documents

        except Exception as e:
            logger.error("Semantic search failed: %s", type(e).__name__)
            raise RuntimeError("Vector retrieval unavailable") from e

    async def _keyword_search(self, query: str, top_k: int, where=None) -> list[RetrievedDocument]:
        """BM25 over a bounded, permission-filtered collection snapshot."""
        from src.evaluation.retrieval import BM25Index, Chunk

        collection = self._get_collection()
        args = {"limit": 10001, "include": ["documents", "metadatas"]}
        if where:
            args["where"] = where
        results = await asyncio.to_thread(collection.get, **args)
        if len(results["ids"]) > 10000:
            raise RuntimeError("Collection exceeds local BM25 capacity; configure a search backend")
        chunks, metadata_by_id = [], {}
        for index, doc_id in enumerate(results["ids"]):
            metadata = (results.get("metadatas") or [])[index] or {}
            content = (results.get("documents") or [])[index] or ""
            metadata_by_id[doc_id] = metadata
            chunks.append(Chunk(doc_id, metadata.get("ticket_id", doc_id),
                                metadata.get("source_path", ""), content))
        ranked = BM25Index(chunks).search(query, top_k)
        return [RetrievedDocument(
            document_id=chunk.chunk_id, content=chunk.content, relevance_score=score,
            source_path=chunk.source, metadata=metadata_by_id[chunk.chunk_id],
            retrieval_method="keyword",
        ) for chunk, score in ranked]

    def _extract_keywords(self, query: str) -> list[str]:
        """Extract important keywords and entities from query."""
        keywords = []

        # Extract telecom entities
        entity_patterns = {
            "node_id": r"node[_\-]?\d+",
            "element_id": r"[ce]r[_\-][a-z][_\-]\d+",
            "interface": r"(eth|ge|xe)\d+",
            "region": r"region[_\s]?[a-z]",
            "ip": r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
        }

        for pattern in entity_patterns.values():
            matches = re.findall(pattern, query, re.IGNORECASE)
            keywords.extend(matches)

        # Preserve opaque IDs and acronyms exactly as written. These are the
        # terms semantic embeddings are least reliable at matching.
        keywords.extend(
            re.findall(
                r"\b(?:INC\d+|[A-Z]{2,}[A-Z0-9]*(?:[-_/.:][A-Z0-9]+)+|"
                r"[A-Z]{2,6}|[A-Za-z]+\.[A-Za-z0-9.]+)\b",
                query,
            )
        )

        # Extract important terms (nouns, technical terms)
        important_terms = [
            "latency", "throughput", "availability", "error", "failure",
            "config", "configuration", "interface", "routing", "ospf", "bgp",
            "vlan", "qos", "policy", "threshold", "sla", "incident",
            "outage", "degraded", "operational", "maintenance",
        ]

        query_lower = query.lower()
        for term in important_terms:
            if term in query_lower:
                keywords.append(term)

        return list(dict.fromkeys(keywords))

    def _build_where_filter(self, filters: Optional[dict]) -> Optional[dict]:
        """Build ChromaDB where filter from filters dict."""
        if not filters:
            return None

        conditions = []

        if "region" in filters:
            conditions.append({"region": {"$eq": filters["region"]}})

        if "doc_type" in filters:
            conditions.append({"document_type": {"$eq": filters["doc_type"]}})

        if "source_path" in filters:
            conditions.append({"source_path": {"$contains": filters["source_path"]}})

        if not conditions:
            return None

        if len(conditions) == 1:
            return conditions[0]

        return {"$and": conditions}

    def _rerank_documents(self, query: str, documents: list[RetrievedDocument], top_k: int):
        """Apply the same field ranking policy as the golden evaluator."""
        from src.evaluation.retrieval import field_rerank_score

        by_ticket: dict[str, RetrievedDocument] = {}
        for document in documents:
            ticket_id = document.metadata.get("ticket_id", document.document_id)
            text = document.metadata.get("ticket_text", document.content)
            fused_score = document.metadata.get("_rrf_score", document.relevance_score)
            score, _ = field_rerank_score(query, ticket_id, text, document.metadata, fused_score)
            document.content = text
            document.relevance_score = min(score, 1.0)
            previous = by_ticket.get(ticket_id)
            if previous is None or score > previous.relevance_score:
                by_ticket[ticket_id] = document
        return sorted(by_ticket.values(), key=lambda doc: (-doc.relevance_score, doc.document_id))[:top_k]

    def _reciprocal_rank_fusion(
        self,
        semantic_docs: list[RetrievedDocument] | list[tuple[str, float]],
        keyword_docs: list[RetrievedDocument] | list[tuple[str, float]],
        top_k: int | None = None,
        k: int | None = None,
    ) -> list[RetrievedDocument]:
        """Combine results using Reciprocal Rank Fusion.

        RRF Score = sum(1 / (k + rank)) for each ranking list
        """
        if k is not None:
            rrf_k = k
        else:
            rrf_k = self.rrf_k
        if top_k is None:
            top_k = len(semantic_docs) + len(keyword_docs)

        def normalize(
            values: list[RetrievedDocument] | list[tuple[str, float]], method: str
        ) -> list[RetrievedDocument]:
            return [
                value
                if isinstance(value, RetrievedDocument)
                else RetrievedDocument(
                    document_id=value[0],
                    content="",
                    relevance_score=value[1],
                    retrieval_method=method,
                )
                for value in values
            ]

        normalized_semantic = normalize(semantic_docs, "semantic")
        normalized_keyword = normalize(keyword_docs, "keyword")
        scores: dict[str, float] = {}
        doc_map: dict[str, RetrievedDocument] = {}

        # Score semantic results
        for rank, doc in enumerate(normalized_semantic):
            rrf_score = self.semantic_weight / (rrf_k + rank + 1)
            scores[doc.document_id] = scores.get(doc.document_id, 0) + rrf_score
            doc_map[doc.document_id] = doc

        # Score keyword results
        for rank, doc in enumerate(normalized_keyword):
            rrf_score = self.keyword_weight / (rrf_k + rank + 1)
            scores[doc.document_id] = scores.get(doc.document_id, 0) + rrf_score

            if doc.document_id not in doc_map:
                doc_map[doc.document_id] = doc
            else:
                # Mark as hybrid if found by both methods
                doc_map[doc.document_id].retrieval_method = "hybrid"

        # Sort by fused score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Build final result list
        result = []
        for doc_id in sorted_ids[:top_k]:
            doc = doc_map[doc_id]
            doc.metadata = {**doc.metadata, "_rrf_score": scores[doc_id]}
            # Normalize score to 0-1 range
            doc.relevance_score = min(scores[doc_id] * 10, 1.0)
            result.append(doc)

        return result

    async def close(self):
        """Clean up resources."""
        if self._embedding_service:
            await self._embedding_service.close()
            self._embedding_service = None
