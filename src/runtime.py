"""Application-owned dependencies, startup checks, bounded work and shutdown."""

import asyncio
from typing import Any

from src.answer_records import AnswerRecords
from src.settings import AppSettings


class Runtime:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.demo: Any = None
        self.agent: Any = None
        self.ingester: Any = None
        self.capacity = asyncio.Semaphore(settings.max_concurrent_requests)
        self.started = False
        self.answer_records = AnswerRecords(retention_days=settings.answer_retention_days)

    async def start(self):
        if self.started:
            return
        if self.settings.app_mode == "demo":
            from src.api.demo import OfflineDemoService
            self.demo = await asyncio.to_thread(OfflineDemoService, source_review_max_age_days=self.settings.source_review_max_age_days)
        else:
            from src.agents.orchestrator import AgentOrchestrator
            from src.agents.store import PostgresConversationStore
            from src.rag_pipeline.chain import RAGChain
            from src.rag_pipeline.embeddings import EmbeddingService, EmbeddingSettings
            from src.rag_pipeline.postgres import PostgresTicketRetriever

            embeddings = EmbeddingSettings(
                OPENAI_API_KEY=self.settings.openai_api_key.get_secret_value(),
                model_name=self.settings.embedding_model_name,
                dimensions=self.settings.embedding_dimensions,
                base_url=self.settings.openai_base_url,
                cache_enabled=False,
                max_retries=1,
            )
            chain = RAGChain(model=self.settings.openai_model)
            chain.api_key = self.settings.openai_api_key.get_secret_value()
            chain.base_url = self.settings.openai_base_url
            chain.require_citations = True
            chain.max_tokens = self.settings.max_output_tokens
            memory = PostgresConversationStore(
                self.settings.database_url.get_secret_value(),
                self.settings.max_concurrent_requests + 1,
                retention_days=self.settings.conversation_retention_days,
            )
            retriever = PostgresTicketRetriever(memory, EmbeddingService(embeddings),
                                                self.settings.knowledge_version, embeddings.model_name, embeddings.dimensions,
                                                configuration=self.settings.retrieval_configuration)
            self.agent = AgentOrchestrator(
                retriever=retriever, rag_chain=chain, memory=memory, enable_tools=False,
                source_review_max_age_days=self.settings.source_review_max_age_days,
            )
            try:
                await memory.start()
                await retriever.start()
                self.answer_records = AnswerRecords(memory, self.settings.answer_retention_days)
            except Exception:
                await self.close()
                raise
        self.started = True

    async def readiness(self):
        if not self.started:
            return False
        if self.agent:
            try:
                async with asyncio.timeout(6):
                    await self.agent.memory.ready()
                    await self.agent.retriever.ready()
            except Exception:
                return False
        return True

    async def close(self):
        if self.agent:
            await self.agent.retriever.close()
            await self.agent.rag_chain.close()
            await self.agent.memory.close()
        self.started = False
