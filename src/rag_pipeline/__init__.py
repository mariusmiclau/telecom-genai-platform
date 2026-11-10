"""RAG Pipeline module for document ingestion, retrieval, and generation."""

from src.rag_pipeline.chain import GeneratedResponse, RAGChain, SourceAttribution
from src.rag_pipeline.embeddings import EmbeddingError, EmbeddingService, EmbeddingSettings
from src.rag_pipeline.hallucination import Claim, HallucinationDetector, HallucinationResult
from src.rag_pipeline.ingest import TicketChunker
from src.rag_pipeline.retriever import HybridRetriever, RetrievalResult, RetrievedDocument

__all__ = [
    # Embeddings
    "EmbeddingService",
    "EmbeddingSettings",
    "EmbeddingError",
    # Retrieval
    "HybridRetriever",
    "RetrievedDocument",
    "RetrievalResult",
    # Generation
    "RAGChain",
    "GeneratedResponse",
    "SourceAttribution",
    # Hallucination
    "HallucinationDetector",
    "HallucinationResult",
    "Claim",
    "TicketChunker",
]
