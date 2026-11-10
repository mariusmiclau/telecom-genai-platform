"""Configuration and retention contracts needed by deployment automation."""

import asyncio
import os
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.deployment import initialize_secrets, preflight
from src.settings import AppSettings


def test_mounted_secrets_provide_live_settings(tmp_path):
    directory = tmp_path / "secrets"
    initialize_secrets(directory, "test-model-key")
    settings = AppSettings(_env_file=None, _secrets_dir=directory, app_mode="live")
    assert len(settings.api_token.get_secret_value()) >= 32
    assert settings.api_token != settings.admin_token
    assert settings.openai_api_key.get_secret_value() == "test-model-key"
    preflight("registry.example/app@sha256:" + "a" * 64, "ops.example.com", directory)
    with pytest.raises(ValueError, match="never overwrites"):
        initialize_secrets(directory, "another-key")
    with pytest.raises(ValueError, match="digest"):
        preflight("app:latest", "ops.example.com", directory)


def test_missing_model_key_does_not_create_secrets(tmp_path):
    directory = tmp_path / "secrets"
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        initialize_secrets(directory, "")
    assert not directory.exists()


def test_insecure_provider_is_an_explicit_test_opt_in():
    with pytest.raises(ValueError, match="HTTPS"):
        AppSettings(_env_file=None, app_mode="live", api_token="a" * 32,
                    database_url="postgresql://test", openai_api_key="test", openai_base_url="http://fixture/v1")


@pytest.mark.asyncio
async def test_publication_rejects_provider_dimension_contract_violation():
    from src.evaluation.dataset import load_jsonl
    from src.rag_pipeline.postgres import PostgresTicketRetriever
    tickets = load_jsonl(Path("datasets/tickets-v2.0-ticket-aware-chunking/tickets.jsonl"))[:1]
    embeddings = SimpleNamespace(embed_documents=AsyncMock(return_value=[[1.0, 0.0]] * 3))
    retriever = PostgresTicketRetriever(None, embeddings, "test", "test-model", dimensions=384)
    with pytest.raises(ValueError, match="configured embedding contract"):
        await retriever.publish(tickets, b"fixture")


@pytest.mark.asyncio
async def test_publication_capacity_fails_before_provider_calls():
    from src.evaluation.dataset import load_jsonl
    from src.rag_pipeline.postgres import PostgresTicketRetriever
    tickets = load_jsonl(Path("datasets/tickets-v2.0-ticket-aware-chunking/tickets.jsonl"))
    embeddings = SimpleNamespace(embed_documents=AsyncMock())
    retriever = PostgresTicketRetriever(None, embeddings, "test", "test-model", dimensions=20_000)
    with pytest.raises(ValueError, match="capacity"):
        await retriever.publish(tickets, b"fixture")
    embeddings.embed_documents.assert_not_called()


def test_backup_never_overwrites_an_existing_file(tmp_path):
    from scripts.database_backup import manage
    backup = tmp_path / "existing.dump"
    backup.write_bytes(b"existing data")
    with pytest.raises(FileExistsError):
        manage("backup", backup, Path("unused.env"), "unused", "")
    assert backup.read_bytes() == b"existing data"


def test_restore_rejects_a_production_database_name(tmp_path):
    from scripts.database_backup import manage
    with pytest.raises(ValueError, match="new database"):
        manage("restore", tmp_path / "missing.dump", Path("unused.env"), "unused", "telecom_genai")


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION"), reason="Requires PostgreSQL integration profile")
async def test_retention_and_explicit_session_deletion():
    from src.agents.store import PostgresConversationStore
    store = PostgresConversationStore("postgresql://genai:local-development-only@localhost:5432/telecom_genai", retention_days=1)
    session = "retention-test-" + uuid.uuid4().hex
    await store.start()
    try:
        await store.save(session, "old", "old")
        await asyncio.to_thread(store._transaction,
            "UPDATE platform_messages SET created_at=now()-interval '2 days' WHERE session_id=%s", (session,))
        assert (await store.get_history(session)).messages == []
        await store.save(session, "new", "new")
        assert len((await store.get_history(session)).messages) == 2
        rows = await asyncio.to_thread(store._transaction,
            "SELECT count(*) FROM platform_messages WHERE session_id=%s", (session,), True)
        assert rows[0][0] == 2
        await store.forget(session)
        assert (await store.get_history(session)).messages == []
    finally:
        await store.forget(session)
        await store.close()
