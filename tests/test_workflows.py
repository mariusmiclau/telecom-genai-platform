"""Durable approval/ingestion behavior against isolated real PostgreSQL databases."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from src.actions import ActionService
from src.evaluation.dataset import load_jsonl
from src.jobs import IngestionQueue, IngestionSubmission
from src.security import Principal, principal_context
from src.settings import AppSettings


def actor(name, tenant="tenant-a", roles=frozenset({"admin", "reader"})):
    return Principal(name, tenant, roles, frozenset({"noc"}), True)


@pytest.mark.asyncio
async def test_two_person_approval_permissions_tampering_replay_and_audit(isolated_store):
    service = ActionService(isolated_store)
    scope = principal_context.set(actor("requester"))
    try:
        plan = await service.request({"node_id": "PCRF-LON-01", "change_summary": "Simulated rollback"})
        with pytest.raises(HTTPException, match="409"):
            await service.transition(plan["id"], "approve", plan["payload_sha256"])
        with pytest.raises(HTTPException, match="409"):
            await service.transition(plan["id"], "execute", plan["payload_sha256"])
        principal_context.set(Principal())
        with pytest.raises(HTTPException, match="409"):
            await service.transition(plan["id"], "approve", plan["payload_sha256"])
        principal_context.set(actor("approver"))
        with pytest.raises(HTTPException, match="409"):
            await service.transition(plan["id"], "approve", "0" * 64)
        assert (await service.transition(plan["id"], "approve", plan["payload_sha256"]))["state"] == "approved"
        principal_context.set(actor("executor", roles=frozenset({"reader"})))
        with pytest.raises(HTTPException, match="403"):
            await service.transition(plan["id"], "execute", plan["payload_sha256"])
        principal_context.set(actor("executor", tenant="another-tenant"))
        with pytest.raises(HTTPException, match="404"):
            await service.transition(plan["id"], "execute", plan["payload_sha256"])
        principal_context.set(actor("executor"))
        result = await service.transition(plan["id"], "execute", plan["payload_sha256"])
        assert result["mode"] == "local-simulation"
        with pytest.raises(HTTPException, match="409"):
            await service.transition(plan["id"], "execute", plan["payload_sha256"])
        assert [event["event"] for event in await service.audit(plan["id"])] == ["requested", "approve", "execute"]
        with pytest.raises(Exception, match="append-only"):
            await asyncio.to_thread(isolated_store._transaction, "DELETE FROM action_audit WHERE action_id=%s", (plan["id"],))
    finally:
        principal_context.reset(scope)


@pytest.mark.asyncio
async def test_expired_and_modified_plans_cannot_be_approved(isolated_store):
    service = ActionService(isolated_store)
    scope = principal_context.set(actor("requester"))
    try:
        plan = await service.request({"node_id": "R1", "change_summary": "Test only"})
        principal_context.set(actor("approver"))
        await asyncio.to_thread(isolated_store._transaction, "UPDATE operational_actions SET parameters='{}' WHERE id=%s", (plan["id"],))
        with pytest.raises(HTTPException, match="409"):
            await service.transition(plan["id"], "approve", plan["payload_sha256"])
        await asyncio.to_thread(isolated_store._transaction, "UPDATE operational_actions SET expires_at=now()-interval '1 minute' WHERE id=%s", (plan["id"],))
        with pytest.raises(HTTPException, match="409"):
            await service.transition(plan["id"], "approve", plan["payload_sha256"])
    finally:
        principal_context.reset(scope)


def queue_fixture(store):
    settings = AppSettings(_env_file=None, embedding_model_name="fixture", embedding_dimensions=2)
    queue = IngestionQueue(store, settings)
    ticket = load_jsonl(Path("datasets/tickets-v2.0-ticket-aware-chunking/tickets.jsonl"))[0]
    submission = IngestionSubmission(version="first", source_jsonl=json.dumps(ticket) + '\n{"invalid":true}', acl_groups=["noc"])
    embeddings = SimpleNamespace(embed_documents=AsyncMock(return_value=[[1.0, 0.0]] * 3))
    return queue, submission, embeddings


@pytest.mark.asyncio
async def test_durable_ingestion_quarantine_atomic_publication_and_no_overwrite(isolated_store):
    queue, submission, embeddings = queue_fixture(isolated_store)
    scope = principal_context.set(actor("ingester"))
    try:
        job = await queue.submit(submission)
        with pytest.raises(HTTPException, match="409"):
            await queue.submit(submission)
        assert await queue.work_once(embeddings)
        status = await queue.status(job["id"])
        assert status["state"] == "succeeded"
        assert status["result"]["accepted_records"] == 1
        assert status["result"]["rejected_records"] == 1
        assert status["quarantine"][0]["line_number"] == 2
        assert not await queue.work_once(embeddings)
        principal_context.set(actor("intruder", tenant="tenant-b"))
        with pytest.raises(HTTPException, match="404"):
            await queue.status(job["id"])
    finally:
        principal_context.reset(scope)


@pytest.mark.asyncio
async def test_ingestion_retry_crash_recovery_and_fenced_worker(isolated_store):
    from src.rag_pipeline.postgres import PostgresTicketRetriever
    queue, submission, embeddings = queue_fixture(isolated_store)
    scope = principal_context.set(actor("ingester"))
    try:
        job = await queue.submit(submission)
        embeddings.embed_documents.side_effect = RuntimeError("provider secret must not be persisted")
        await queue.work_once(embeddings)
        status = await queue.status(job["id"])
        assert status["state"] == "retry" and status["error_code"] == "dependency_failed"
        assert "secret" not in json.dumps(status)
        await asyncio.to_thread(isolated_store._transaction, "UPDATE ingestion_jobs SET available_at=now() WHERE id=%s", (job["id"],))
        old = await queue.claim()
        assert old
        # Simulate process death: recover the same durable job with a new fence.
        await asyncio.to_thread(isolated_store._transaction, "UPDATE ingestion_jobs SET lease_until=now()-interval '1 second' WHERE id=%s", (job["id"],))
        new = await queue.claim()
        assert new["lease"] != old["lease"]
        ticket = load_jsonl(Path("datasets/tickets-v2.0-ticket-aware-chunking/tickets.jsonl"))[:1]
        embeddings.embed_documents.side_effect = None
        retriever = PostgresTicketRetriever(isolated_store, embeddings, old["version"], "fixture", 2, job_id=old["id"], lease_token=old["lease"])
        with pytest.raises(RuntimeError, match="lease"):
            await retriever.publish(ticket, b"stale worker")
        rows = await asyncio.to_thread(isolated_store._transaction, "SELECT count(*) FROM knowledge_versions", None, True)
        assert rows[0][0] == 0
    finally:
        principal_context.reset(scope)


@pytest.mark.asyncio
async def test_queue_claim_is_exclusive_between_competing_workers(isolated_store):
    queue, submission, _ = queue_fixture(isolated_store)
    scope = principal_context.set(actor("ingester"))
    try:
        await queue.submit(submission)
        claims = await asyncio.gather(queue.claim(), queue.claim())
        assert sum(claim is not None for claim in claims) == 1
    finally:
        principal_context.reset(scope)


@pytest.mark.asyncio
async def test_migrations_are_idempotent_and_checksum_checked(isolated_store):
    from src.migrations import MIGRATIONS, migrate
    await isolated_store.run(lambda cursor: cursor.execute("SELECT 1"))
    connection = isolated_store.pool.getconn()
    try:
        await asyncio.to_thread(migrate, connection)
        rows = await asyncio.to_thread(isolated_store._transaction, "SELECT count(*) FROM platform_schema_migrations", None, True)
        assert rows[0][0] == len(MIGRATIONS)
        await asyncio.to_thread(isolated_store._transaction, "UPDATE platform_schema_migrations SET checksum='tampered' WHERE version=1")
        with pytest.raises(RuntimeError, match="checksum"):
            await asyncio.to_thread(migrate, connection)
    finally:
        isolated_store.pool.putconn(connection)


@pytest.mark.asyncio
async def test_ingestion_recovers_publication_even_after_last_attempt_process_death(isolated_store):
    from src.rag_pipeline.postgres import PostgresTicketRetriever
    queue, submission, embeddings = queue_fixture(isolated_store)
    scope = principal_context.set(actor("ingester"))
    try:
        job = await queue.submit(submission)
        claimed = await queue.claim()
        retriever = PostgresTicketRetriever(isolated_store, embeddings, claimed["version"], "fixture", 2,
            job_id=claimed["id"], lease_token=claimed["lease"])
        tickets = load_jsonl(Path("datasets/tickets-v2.0-ticket-aware-chunking/tickets.jsonl"))[:1]
        await retriever.publish(tickets, b"committed before process died")
        await asyncio.to_thread(isolated_store._transaction, "UPDATE ingestion_jobs SET attempts=max_attempts,lease_until=now()-interval '1 second' WHERE id=%s", (job["id"],))
        assert await queue.claim() is None
        status = await queue.status(job["id"])
        assert status["state"] == "succeeded"
        assert status["result"]["recovered_after_publication"]
    finally:
        principal_context.reset(scope)


@pytest.mark.asyncio
async def test_ingestion_retry_limit_is_terminal_and_all_invalid_input_is_quarantined(isolated_store):
    queue, submission, embeddings = queue_fixture(isolated_store)
    scope = principal_context.set(actor("ingester"))
    try:
        job = await queue.submit(submission)
        await asyncio.to_thread(isolated_store._transaction, "UPDATE ingestion_jobs SET max_attempts=1 WHERE id=%s", (job["id"],))
        embeddings.embed_documents.side_effect = RuntimeError("Unavailable")
        await queue.work_once(embeddings)
        assert (await queue.status(job["id"]))["state"] == "failed"
        assert not await queue.work_once(embeddings)
        invalid = await queue.submit(IngestionSubmission(version="invalid", source_jsonl="not json", acl_groups=["noc"]))
        await queue.work_once(embeddings)
        status = await queue.status(invalid["id"])
        assert status["state"] == "failed" and len(status["quarantine"]) == 1
    finally:
        principal_context.reset(scope)


@pytest.mark.asyncio
async def test_runtime_retrieval_evaluation_and_snapshot_rollback(isolated_store):
    from src.evaluation.production import evaluate
    from src.evaluation.retrieval import HashingVectorIndex, ticket_aware_chunks
    from src.rag_pipeline.postgres import PostgresTicketRetriever
    tickets = load_jsonl(Path("datasets/tickets-v2.0-ticket-aware-chunking/tickets.jsonl"))[:2]
    questions = load_jsonl(Path("evaluation/golden_questions.jsonl"))[:2]
    index = HashingVectorIndex(ticket_aware_chunks(tickets))
    embeddings = SimpleNamespace(embed_documents=AsyncMock(side_effect=lambda texts: [index._embed(text) for text in texts]),
                                 embed_query=AsyncMock(side_effect=index._embed))
    retrievers = {name: PostgresTicketRetriever(isolated_store, embeddings, "test-" + name, "fixture", 384, configuration=name) for name in ("A", "B", "C")}
    for retriever in retrievers.values():
        await retriever.publish(tickets, b"same source")
    report = await evaluate(retrievers, tickets, questions)
    assert report["backend"] == "runtime-postgresql"
    assert report["configurations"]["C"]["metrics"]["recall_at_5"] == 1.0
    before = await retrievers["C"].search("PCRF-LON-01", top_k=1)
    altered = [{**tickets[0], "ticket_id": "NEW-INCIDENT", "summary": "New index snapshot"}]
    embeddings.embed_documents.side_effect = lambda texts: [index._embed(text) for text in texts]
    newer = PostgresTicketRetriever(isolated_store, embeddings, "new-C", "fixture", 384)
    await newer.publish(altered, b"new source")
    assert (await newer.search("PCRF-LON-01", top_k=1)).documents[0].document_id == "NEW-INCIDENT"
    restored = PostgresTicketRetriever(isolated_store, embeddings, "test-C", "fixture", 384)
    await restored.ready()
    assert (await restored.search("PCRF-LON-01", top_k=1)).documents[0].document_id == before.documents[0].document_id


@pytest.mark.asyncio
async def test_queue_capacity_eventual_retry_success_payload_cleanup_and_alert_metrics(isolated_store):
    from prometheus_client import REGISTRY

    from src.monitoring.metrics import collect_ingestion_metrics
    queue, submission, embeddings = queue_fixture(isolated_store)
    queue.settings.max_pending_ingestion_jobs = 1
    scope = principal_context.set(actor("ingester"))
    try:
        job = await queue.submit(submission)
        with pytest.raises(HTTPException, match="429"):
            await queue.submit(submission.model_copy(update={"version": "another"}))
        await collect_ingestion_metrics(isolated_store)
        assert REGISTRY.get_sample_value("genai_ingestion_jobs", {"state": "pending"}) == 1
        embeddings.embed_documents.side_effect = RuntimeError("Temporary outage")
        await queue.work_once(embeddings)
        await asyncio.to_thread(isolated_store._transaction, "UPDATE ingestion_jobs SET available_at=now() WHERE id=%s", (job["id"],))
        embeddings.embed_documents.side_effect = None
        await queue.work_once(embeddings)
        result = await queue.status(job["id"])
        assert result["state"] == "succeeded" and result["attempts"] == 2
        rows = await asyncio.to_thread(isolated_store._transaction, "SELECT octet_length(source),length(source_sha256) FROM ingestion_jobs WHERE id=%s", (job["id"],), True)
        assert rows[0] == (0, 64)
        await queue.submit(IngestionSubmission(version="bad", source_jsonl="invalid", acl_groups=["noc"]))
        await queue.work_once(embeddings)
        await collect_ingestion_metrics(isolated_store)
        assert REGISTRY.get_sample_value("genai_ingestion_jobs", {"state": "failed"}) == 1
        assert REGISTRY.get_sample_value("genai_ingestion_jobs", {"state": "pending"}) == 0
    finally:
        principal_context.reset(scope)
