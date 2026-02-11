"""
Embedding Service for Telecom GenAI Platform.

Provides async embedding generation with multi-provider support,
Redis caching, and automatic rate limit handling.
"""

import asyncio
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Self

import redis.asyncio as redis
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Custom exception for embedding-related errors."""

    def __init__(self, message: str, provider: str | None = None, cause: Exception | None = None):
        self.message = message
        self.provider = provider
        self.cause = cause
        super().__init__(self.message)

    def __str__(self) -> str:
        parts = [self.message]
        if self.provider:
            parts.append(f"[provider={self.provider}]")
        if self.cause:
            parts.append(f"[cause={self.cause}]")
        return " ".join(parts)


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""

    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class EmbeddingSettings(BaseSettings):
    """Configuration settings for embedding service."""

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        env_file=".env",
        extra="ignore",
    )

    # Provider settings
    provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.OPENAI,
        description="Embedding provider to use",
    )

    # Model settings
    model_name: str = Field(
        default="text-embedding-3-small",
        description="Model name for embeddings",
    )
    dimensions: int = Field(
        default=1536,
        description="Embedding vector dimensions",
    )
    batch_size: int = Field(
        default=100,
        ge=1,
        le=2048,
        description="Maximum batch size for embedding requests",
    )

    # OpenAI settings
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")

    # Azure OpenAI settings
    azure_openai_api_key: str | None = Field(default=None)
    azure_openai_endpoint: str | None = Field(default=None)
    azure_openai_api_version: str = Field(default="2024-02-01")
    azure_openai_deployment: str | None = Field(default=None)

    # Sentence Transformers settings
    sentence_transformers_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Local sentence-transformers model",
    )

    # Redis cache settings
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )
    cache_ttl: int = Field(
        default=86400 * 7,  # 7 days
        description="Cache TTL in seconds",
    )
    cache_enabled: bool = Field(
        default=True,
        description="Enable Redis caching",
    )

    # Rate limit settings
    max_retries: int = Field(default=5, description="Maximum retry attempts")
    base_delay: float = Field(default=1.0, description="Base delay for exponential backoff")
    max_delay: float = Field(default=60.0, description="Maximum delay between retries")


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        pass

    @abstractmethod
    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up provider resources."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(self, settings: EmbeddingSettings):
        self.settings = settings
        self._client: Any = None

    async def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise EmbeddingError(
                    "openai package required: pip install openai",
                    provider=self.provider_name,
                )

            if not self.settings.openai_api_key:
                raise EmbeddingError(
                    "OPENAI_API_KEY environment variable required",
                    provider=self.provider_name,
                )

            self._client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        return self._client

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        client = await self._get_client()
        response = await client.embeddings.create(
            model=self.settings.model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]

    async def embed_single(self, text: str) -> list[float]:
        result = await self.embed_texts([text])
        return result[0]

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    @property
    def provider_name(self) -> str:
        return "openai"


class AzureOpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Azure OpenAI embedding provider."""

    def __init__(self, settings: EmbeddingSettings):
        self.settings = settings
        self._client: Any = None

    async def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncAzureOpenAI
            except ImportError:
                raise EmbeddingError(
                    "openai package required: pip install openai",
                    provider=self.provider_name,
                )

            if not all([
                self.settings.azure_openai_api_key,
                self.settings.azure_openai_endpoint,
                self.settings.azure_openai_deployment,
            ]):
                raise EmbeddingError(
                    "Azure OpenAI requires EMBEDDING_AZURE_OPENAI_API_KEY, "
                    "EMBEDDING_AZURE_OPENAI_ENDPOINT, and EMBEDDING_AZURE_OPENAI_DEPLOYMENT",
                    provider=self.provider_name,
                )

            self._client = AsyncAzureOpenAI(
                api_key=self.settings.azure_openai_api_key,
                azure_endpoint=self.settings.azure_openai_endpoint,
                api_version=self.settings.azure_openai_api_version,
            )
        return self._client

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        client = await self._get_client()
        response = await client.embeddings.create(
            model=self.settings.azure_openai_deployment,
            input=texts,
        )
        return [item.embedding for item in response.data]

    async def embed_single(self, text: str) -> list[float]:
        result = await self.embed_texts([text])
        return result[0]

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    @property
    def provider_name(self) -> str:
        return "azure_openai"


class SentenceTransformersProvider(BaseEmbeddingProvider):
    """Local sentence-transformers embedding provider for development."""

    def __init__(self, settings: EmbeddingSettings):
        self.settings = settings
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise EmbeddingError(
                    "sentence-transformers package required: pip install sentence-transformers",
                    provider=self.provider_name,
                )

            self._model = SentenceTransformer(self.settings.sentence_transformers_model)
        return self._model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(texts, convert_to_numpy=True).tolist(),
        )
        return embeddings

    async def embed_single(self, text: str) -> list[float]:
        result = await self.embed_texts([text])
        return result[0]

    async def close(self) -> None:
        self._model = None

    @property
    def provider_name(self) -> str:
        return "sentence_transformers"


class EmbeddingCache:
    """Redis-based caching layer for embeddings."""

    CACHE_PREFIX = "emb:v1:"

    def __init__(self, settings: EmbeddingSettings):
        self.settings = settings
        self._redis: redis.Redis | None = None

    async def connect(self) -> None:
        """Initialize Redis connection."""
        if self._redis is None:
            self._redis = redis.from_url(
                self.settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            # Test connection
            try:
                await self._redis.ping()
                logger.info("Redis cache connected")
            except Exception as e:
                logger.warning(f"Redis connection failed, caching disabled: {e}")
                self._redis = None

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis is not None:
            await self._redis.close()
            self._redis = None

    def _cache_key(self, text: str, model: str) -> str:
        """Generate cache key from text hash."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:32]
        model_hash = hashlib.md5(model.encode()).hexdigest()[:8]
        return f"{self.CACHE_PREFIX}{model_hash}:{text_hash}"

    async def get(self, text: str, model: str) -> list[float] | None:
        """Retrieve cached embedding."""
        if self._redis is None:
            return None

        try:
            key = self._cache_key(text, model)
            cached = await self._redis.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
        return None

    async def get_many(
        self, texts: list[str], model: str
    ) -> dict[int, list[float]]:
        """Retrieve multiple cached embeddings."""
        if self._redis is None:
            return {}

        try:
            keys = [self._cache_key(text, model) for text in texts]
            values = await self._redis.mget(keys)

            results = {}
            for i, value in enumerate(values):
                if value:
                    results[i] = json.loads(value)
            return results
        except Exception as e:
            logger.warning(f"Cache get_many failed: {e}")
            return {}

    async def set(self, text: str, model: str, embedding: list[float]) -> None:
        """Cache an embedding."""
        if self._redis is None:
            return

        try:
            key = self._cache_key(text, model)
            await self._redis.setex(
                key,
                self.settings.cache_ttl,
                json.dumps(embedding),
            )
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")

    async def set_many(
        self, texts: list[str], model: str, embeddings: list[list[float]]
    ) -> None:
        """Cache multiple embeddings."""
        if self._redis is None:
            return

        try:
            pipe = self._redis.pipeline()
            for text, embedding in zip(texts, embeddings):
                key = self._cache_key(text, model)
                pipe.setex(key, self.settings.cache_ttl, json.dumps(embedding))
            await pipe.execute()
        except Exception as e:
            logger.warning(f"Cache set_many failed: {e}")


class EmbeddingService:
    """
    Main embedding service with multi-provider support and caching.

    Provides async embedding generation with automatic rate limit handling,
    Redis caching, and support for multiple embedding providers.

    Usage:
        async with EmbeddingService() as service:
            embedding = await service.embed_query("What is 5G?")
            embeddings = await service.embed_documents(["doc1", "doc2"])
    """

    def __init__(self, settings: EmbeddingSettings | None = None):
        """
        Initialize the embedding service.

        Args:
            settings: Optional EmbeddingSettings instance.
                     If not provided, settings are loaded from environment.
        """
        self.settings = settings or EmbeddingSettings()
        self._provider: BaseEmbeddingProvider | None = None
        self._cache: EmbeddingCache | None = None
        self._initialized = False

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit with resource cleanup."""
        await self.close()

    async def initialize(self) -> None:
        """Initialize provider and cache connections."""
        if self._initialized:
            return

        # Initialize provider
        self._provider = self._create_provider()
        logger.info(f"Initialized embedding provider: {self._provider.provider_name}")

        # Initialize cache
        if self.settings.cache_enabled:
            self._cache = EmbeddingCache(self.settings)
            await self._cache.connect()

        self._initialized = True

    def _create_provider(self) -> BaseEmbeddingProvider:
        """Create the appropriate embedding provider."""
        providers = {
            EmbeddingProvider.OPENAI: OpenAIEmbeddingProvider,
            EmbeddingProvider.AZURE_OPENAI: AzureOpenAIEmbeddingProvider,
            EmbeddingProvider.SENTENCE_TRANSFORMERS: SentenceTransformersProvider,
        }

        provider_class = providers.get(self.settings.provider)
        if not provider_class:
            raise EmbeddingError(f"Unknown provider: {self.settings.provider}")

        return provider_class(self.settings)

    async def close(self) -> None:
        """Clean up resources."""
        if self._provider:
            await self._provider.close()
            self._provider = None

        if self._cache:
            await self._cache.close()
            self._cache = None

        self._initialized = False
        logger.info("Embedding service closed")

    async def _embed_with_retry(
        self, texts: list[str]
    ) -> list[list[float]]:
        """Embed texts with exponential backoff retry."""
        if not self._provider:
            raise EmbeddingError("Service not initialized. Use 'async with' or call initialize()")

        last_error: Exception | None = None

        for attempt in range(self.settings.max_retries):
            try:
                return await self._provider.embed_texts(texts)

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Check if it's a rate limit error
                is_rate_limit = any(
                    term in error_str
                    for term in ["rate limit", "429", "too many requests", "quota"]
                )

                if not is_rate_limit and attempt == 0:
                    # Non-rate-limit error on first attempt, fail fast
                    raise EmbeddingError(
                        f"Embedding failed: {e}",
                        provider=self._provider.provider_name,
                        cause=e,
                    )

                # Calculate delay with exponential backoff
                delay = min(
                    self.settings.base_delay * (2 ** attempt),
                    self.settings.max_delay,
                )

                logger.warning(
                    f"Embedding attempt {attempt + 1} failed, retrying in {delay:.1f}s: {e}"
                )
                await asyncio.sleep(delay)

        raise EmbeddingError(
            f"Embedding failed after {self.settings.max_retries} attempts",
            provider=self._provider.provider_name,
            cause=last_error,
        )

    async def embed_query(self, text: str) -> list[float]:
        """
        Generate embedding for a single query text.

        Args:
            text: The query text to embed.

        Returns:
            Embedding vector as a list of floats.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        if not self._initialized:
            await self.initialize()

        # Check cache
        if self._cache:
            cached = await self._cache.get(text, self.settings.model_name)
            if cached:
                logger.debug("Cache hit for query embedding")
                return cached

        # Generate embedding
        embeddings = await self._embed_with_retry([text])
        embedding = embeddings[0]

        # Cache result
        if self._cache:
            await self._cache.set(text, self.settings.model_name, embedding)

        return embedding

    async def embed_documents(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple documents with batching.

        Args:
            texts: List of document texts to embed.
            show_progress: If True, log progress during embedding.

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        if not self._initialized:
            await self.initialize()

        if not texts:
            return []

        # Check cache for all texts
        cached_embeddings: dict[int, list[float]] = {}
        if self._cache:
            cached_embeddings = await self._cache.get_many(texts, self.settings.model_name)
            if cached_embeddings:
                logger.info(f"Cache hit for {len(cached_embeddings)}/{len(texts)} documents")

        # Identify texts that need embedding
        texts_to_embed: list[tuple[int, str]] = []
        for i, text in enumerate(texts):
            if i not in cached_embeddings:
                texts_to_embed.append((i, text))

        if not texts_to_embed:
            # All from cache
            return [cached_embeddings[i] for i in range(len(texts))]

        # Embed in batches
        new_embeddings: dict[int, list[float]] = {}
        batch_size = self.settings.batch_size
        total_batches = (len(texts_to_embed) + batch_size - 1) // batch_size

        for batch_num, batch_start in enumerate(range(0, len(texts_to_embed), batch_size)):
            batch = texts_to_embed[batch_start : batch_start + batch_size]
            batch_indices = [item[0] for item in batch]
            batch_texts = [item[1] for item in batch]

            if show_progress:
                logger.info(f"Embedding batch {batch_num + 1}/{total_batches}")

            embeddings = await self._embed_with_retry(batch_texts)

            # Store results
            for idx, embedding in zip(batch_indices, embeddings):
                new_embeddings[idx] = embedding

            # Cache new embeddings
            if self._cache:
                await self._cache.set_many(batch_texts, self.settings.model_name, embeddings)

        # Combine cached and new embeddings
        all_embeddings = {**cached_embeddings, **new_embeddings}
        return [all_embeddings[i] for i in range(len(texts))]

    @property
    def provider_name(self) -> str:
        """Return the current provider name."""
        if self._provider:
            return self._provider.provider_name
        return self.settings.provider.value

    @property
    def model_name(self) -> str:
        """Return the current model name."""
        return self.settings.model_name

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        return self.settings.dimensions


async def main() -> None:
    """Example usage of EmbeddingService."""
    import os

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Example 1: Basic usage with context manager
    print("=" * 60)
    print("Example 1: Basic embedding generation")
    print("=" * 60)

    # Use sentence-transformers for local dev if no OpenAI key
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["EMBEDDING_PROVIDER"] = "sentence_transformers"
        print("Using sentence-transformers (no OPENAI_API_KEY found)")

    async with EmbeddingService() as service:
        print(f"Provider: {service.provider_name}")
        print(f"Model: {service.model_name}")

        # Single query embedding
        query = "What is network latency in 5G?"
        embedding = await service.embed_query(query)
        print(f"\nQuery: '{query}'")
        print(f"Embedding dimensions: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")

        # Batch document embeddings
        documents = [
            "5G networks provide ultra-low latency communication.",
            "Network slicing enables dedicated virtual networks.",
            "Edge computing reduces round-trip time for data.",
            "QoS policies ensure service level agreements are met.",
        ]

        print(f"\nEmbedding {len(documents)} documents...")
        embeddings = await service.embed_documents(documents, show_progress=True)
        print(f"Generated {len(embeddings)} embeddings")

        # Demonstrate caching (embed same query again)
        print("\nRe-embedding same query (should hit cache)...")
        cached_embedding = await service.embed_query(query)
        print(f"Cache working: {embedding == cached_embedding}")

    # Example 2: Custom settings
    print("\n" + "=" * 60)
    print("Example 2: Custom settings")
    print("=" * 60)

    custom_settings = EmbeddingSettings(
        provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
        sentence_transformers_model="all-MiniLM-L6-v2",
        batch_size=50,
        cache_enabled=False,
    )

    async with EmbeddingService(settings=custom_settings) as service:
        embedding = await service.embed_query("Custom settings example")
        print(f"Provider: {service.provider_name}")
        print(f"Embedding dimensions: {len(embedding)}")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
