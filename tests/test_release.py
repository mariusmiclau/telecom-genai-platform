"""Release contracts: evidence isolation, failure policy, lifecycle, real persistence."""

import os
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agents.orchestrator import AgentOrchestrator
from src.api.demo import OfflineDemoService
from src.api.models import SourceReference
from src.guardrails.output_filter import OutputFilter
from src.rag_pipeline.chain import GeneratedResponse, RAGChain, SourceAttribution
from src.rag_pipeline.retriever import RetrievedDocument


def test_demo_does_not_read_golden_answers(tmp_path):
    import shutil
    destination = tmp_path / "datasets" / "tickets-v2.0-ticket-aware-chunking"
    destination.mkdir(parents=True)
    shutil.copyfile(Path("datasets/tickets-v2.0-ticket-aware-chunking/tickets.jsonl"), destination / "tickets.jsonl")
    result = OfflineDemoService(tmp_path).answer("PCRF-LON-01 DIAMETER_TOO_BUSY")
    assert "INC20260001" in result.text
    assert not (tmp_path / "evaluation").exists()


def test_structured_source_redaction():
    original = SourceReference(document="engineer@example.com", relevance_score=0.7)
    filtered = OutputFilter().apply("Evidence", [original])
    assert "@" not in filtered.sources[0].document
    assert original.document == "engineer@example.com"


def test_live_chain_does_not_fabricate_citations():
    chain = RAGChain()
    chain.require_citations = True
    assert chain._extract_attributions("An answer with no citation", [{"source": "ticket", "content": "evidence"}]) == []


def test_grounding_rejects_wrong_decimal_despite_word_overlap():
    from src.guardrails.hallucination import HallucinationDetector
    detector = HallucinationDetector()
    evidence = [{"content": "Availability restored to 99.8%."}]
    assert not detector.check_grounding("Availability restored to 99.9% [Source 1].", evidence, []).is_grounded
    assert detector.check_grounding("Availability restored to 99.8% [Source 1].", evidence, []).is_grounded


@pytest.mark.asyncio
@pytest.mark.parametrize("grounded,citations,finish", [(False, True, "stop"), (True, False, "stop"), (True, True, "length")])
async def test_real_orchestration_abstains_on_failed_publication_check(grounded, citations, finish):
    memory = SimpleNamespace(get_history=AsyncMock(return_value=None), save=AsyncMock())
    doc = RetrievedDocument("ticket", "Availability restored to 99.8%", 0.8, "source")
    retriever = SimpleNamespace(search=AsyncMock(return_value=SimpleNamespace(documents=[doc])))
    generated = GeneratedResponse(
        "Availability is 99.8% [Source 1]", [SourceAttribution("source")] if citations else [], 5, "test", finish,
    )
    chain = SimpleNamespace(generate=AsyncMock(return_value=generated))
    agent = AgentOrchestrator(retriever, chain, memory, enable_tools=False)
    agent.reasoning.classify_intent = AsyncMock(return_value=SimpleNamespace(category="query", requires_tools=False))
    agent.grounding_detector.check_grounding = lambda **kwargs: SimpleNamespace(
        is_grounded=grounded, grounding_score=float(grounded), ungrounded_claims=[],
    )
    result = await agent.process("historical availability")
    assert result.sources == []
    assert result.confidence == 0
    assert "could not be verified" in result.response
    assert memory.save.call_args.kwargs["assistant_response"] == result.response


@pytest.mark.asyncio
async def test_live_orchestration_does_not_execute_simulators():
    memory = SimpleNamespace(get_history=AsyncMock(return_value=None))
    agent = AgentOrchestrator(memory=memory, enable_tools=False)
    agent.reasoning.classify_intent = AsyncMock(return_value=SimpleNamespace(category="action", requires_tools=True))
    result = await agent.process("Create a ticket")
    assert result.tools_called == []
    assert result.confidence == 0
    assert "disabled" in result.response


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION"), reason="Requires Compose integration profile")
async def test_postgres_real_transactions_and_recent_history():
    from src.agents.store import PostgresConversationStore
    store = PostgresConversationStore("postgresql://genai:local-development-only@localhost:5432/telecom_genai")
    session = "test-" + uuid.uuid4().hex
    await store.start()
    try:
        for index in range(7):
            await store.save(session, f"question {index}", f"answer {index}")
        history = await store.get_history(session, 4)
        assert [message.content for message in history.messages] == ["question 5", "answer 5", "question 6", "answer 6"]
        await store.ready()
    finally:
        import asyncio
        await asyncio.to_thread(store._transaction, "DELETE FROM platform_messages WHERE session_id=%s", (session,))
        await store.close()


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION"), reason="Requires Compose integration profile")
async def test_postgres_real_hybrid_publication_and_rollback():
    import asyncio

    from src.agents.store import PostgresConversationStore
    from src.evaluation.dataset import load_jsonl
    from src.rag_pipeline.postgres import PostgresTicketRetriever

    store = PostgresConversationStore("postgresql://genai:local-development-only@localhost:5432/telecom_genai")
    version = "release-test-" + uuid.uuid4().hex
    tickets = load_jsonl(Path("datasets/tickets-v2.0-ticket-aware-chunking/tickets.jsonl"))[:2]
    embeddings = SimpleNamespace(
        embed_documents=AsyncMock(return_value=[[1.0, 0.0]] * 6),
        embed_query=AsyncMock(return_value=[1.0, 0.0]), close=AsyncMock(),
    )
    retriever = PostgresTicketRetriever(store, embeddings, version, "test-embedding")
    await store.start()
    try:
        await retriever.start(require_data=False)
        result = await retriever.publish(tickets, b"test snapshot")
        assert result["chunks"] == 6
        await retriever.ready()
        hits = await retriever.search("PCRF-LON-01 DIAMETER_TOO_BUSY", top_k=1)
        assert hits.documents[0].document_id == "INC20260001"
        assert "stale route" in hits.documents[0].content
        with pytest.raises(Exception):
            await retriever.publish(tickets, b"duplicate publication must fail")
        assert len((await retriever.search("PCRF-LON-01", top_k=5)).documents) == 2
        wrong_model = PostgresTicketRetriever(store, embeddings, version, "different-model")
        with pytest.raises(RuntimeError, match="embedding model"):
            await wrong_model.ready()
    finally:
        await asyncio.to_thread(store._transaction,
            "DELETE FROM knowledge_chunks WHERE version=%s; DELETE FROM knowledge_versions WHERE name=%s",
            (version, version))
        await store.close()
