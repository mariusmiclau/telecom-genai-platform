"""Real signed JWT validation and PostgreSQL isolation, including hostile filters."""

import asyncio
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from httpx import ASGITransport, AsyncClient

from src.api.main import create_app
from src.security import Authenticator, Principal, principal_context
from src.settings import AppSettings


@pytest.fixture
def identity():
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key()))
    jwk.update(kid="test", alg="RS256", use="sig")
    settings = AppSettings(_env_file=None, app_mode="live", auth_mode="oidc",
        oidc_issuer="https://identity.example", oidc_audience="telecom",
        oidc_jwks_url="https://identity.example/keys", database_url="postgresql://fixture",
        openai_api_key="fixture", requests_per_minute=100)
    auth = Authenticator(settings)
    auth.keys.fetch_data = lambda: {"keys": [jwk]}

    def token(**overrides):
        claims = {"iss": settings.oidc_issuer, "aud": settings.oidc_audience,
                  "sub": "alice", "tenant_id": "tenant-a", "roles": ["reader"],
                  "groups": ["noc"], "iat": int(time.time()), "exp": int(time.time()) + 120}
        claims.update(overrides)
        return jwt.encode(claims, private, algorithm="RS256", headers={"kid": "test"})
    return settings, auth, token


def test_signed_identity_requires_valid_issuer_audience_expiry_and_tenant(identity):
    _, auth, token = identity
    assert auth.decode(token()).subject == "alice"
    for overrides in ({"aud": "another-app"}, {"iss": "https://attacker.example"},
                      {"exp": int(time.time()) - 60}, {"tenant_id": None}, {"tenant_id": "local"}, {"groups": "noc"}):
        with pytest.raises(jwt.PyJWTError):
            auth.decode(token(**overrides))
    with pytest.raises(jwt.PyJWTError):
        auth.decode(jwt.encode({"sub": "attacker"}, "not-a-trusted-key" * 3, algorithm="HS256"))


def test_jwks_http_fetch_and_key_rotation(identity):
    _, auth, token = identity
    keys = auth.keys.fetch_data()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = json.dumps(keys).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # Zero cooldown only in this rotation test; production retains the
        # library's anti-refresh-storm cooldown and five-minute JWKS cache.
        auth.keys = jwt.PyJWKClient(f"http://127.0.0.1:{server.server_port}/keys", timeout=2, cooldown_duration=0)
        assert auth.decode(token()).subject == "alice"
        private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        new_key = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key()))
        new_key.update(kid="rotated", alg="RS256", use="sig")
        keys["keys"] = [new_key]
        claims = jwt.decode(token(), options={"verify_signature": False})
        rotated = jwt.encode(claims, private, algorithm="RS256", headers={"kid": "rotated"})
        assert auth.decode(rotated).tenant == "tenant-a"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.asyncio
async def test_signed_http_workflows_enforce_separation_and_job_ownership(identity, isolated_store):
    settings, auth, token = identity
    settings.enable_local_actions = True
    app = create_app(settings)
    app.state.authenticator.keys = auth.keys
    app.state.runtime.agent = SimpleNamespace(memory=isolated_store)
    def headers(subject, roles):
        return {"Authorization": "Bearer " + token(sub=subject, roles=roles)}
    before = principal_context.get()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        plan = {"node_id": "R1", "change_summary": "Local simulation"}
        assert (await client.post("/api/v1/actions", json=plan, headers=headers("reader", ["reader"])) ).status_code == 403
        response = await client.post("/api/v1/actions", json=plan, headers=headers("alice", ["action_request"]))
        assert response.status_code == 201, response.text
        action = response.json()
        path = "/api/v1/actions/" + action["id"]
        decision = {"payload_sha256": action["payload_sha256"]}
        visible = await client.get(path, headers=headers("bob", ["reader", "action_approve"]))
        assert visible.status_code == 200 and visible.json()["parameters"] == plan
        assert (await client.get(path, headers=headers("unrelated-reader", ["reader"]))).status_code == 403
        assert (await client.post(path + "/approve", json=decision, headers=headers("alice", ["action_approve"]))).status_code == 409
        assert (await client.post(path + "/approve", json=decision, headers=headers("bob", ["action_approve"]))).status_code == 200
        assert (await client.post(path + "/execute", json=decision, headers=headers("charlie", ["action_execute"]))).status_code == 200
        result = await client.get(path + "/audit", headers=headers("auditor", ["reader", "auditor"]))
        assert len(result.json()) == 3
        submission = {"version": "http-job", "source_jsonl": "invalid", "acl_groups": ["noc"]}
        result = await client.post("/api/v1/ingestion/jobs", json=submission, headers=headers("alice", ["ingest"]))
        assert result.status_code == 202, result.text
        job_id = result.json()["id"]
        assert (await client.get("/api/v1/ingestion/jobs/" + job_id, headers=headers("bob", ["ingest"]))).status_code == 404
        assert (await client.post("/api/v1/ingestion/jobs", json={**submission, "acl_groups": ["unauthorized"]}, headers=headers("alice", ["ingest"]))).status_code == 403
    assert principal_context.get() == before


@pytest.mark.asyncio
async def test_oidc_http_roles_are_enforced_and_invalid_tokens_rejected(identity):
    settings, auth, token = identity
    app = create_app(settings)
    app.state.authenticator.keys = auth.keys
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/metrics", headers={"Authorization": "Bearer " + token()})).status_code == 200
        assert (await client.get("/metrics", headers={"Authorization": "Bearer " + token(aud="wrong")})).status_code == 401
        assert (await client.post("/api/v1/ingest", json={}, headers={"Authorization": "Bearer " + token()})).status_code == 403
        assert (await client.get("/metrics", headers={"Authorization": "Bearer " + token(roles=[])})).status_code == 403


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION"), reason="Requires PostgreSQL")
async def test_postgres_acl_and_session_isolation_are_server_enforced():
    from src.agents.store import PostgresConversationStore
    from src.evaluation.dataset import load_jsonl
    from src.rag_pipeline.postgres import PostgresTicketRetriever
    store = PostgresConversationStore("postgresql://genai:local-development-only@localhost:5432/telecom_genai")
    version, session = "acl-" + uuid.uuid4().hex, "session-" + uuid.uuid4().hex
    tickets = load_jsonl(Path("datasets/tickets-v2.0-ticket-aware-chunking/tickets.jsonl"))[:2]
    tickets[0].update(tenant_id="tenant-a", acl_groups=["noc"])
    tickets[1].update(tenant_id="tenant-b", acl_groups=["noc"])
    embeddings = SimpleNamespace(embed_documents=AsyncMock(return_value=[[1.0, 0.0]] * 6),
        embed_query=AsyncMock(return_value=[1.0, 0.0]), close=AsyncMock())
    retriever = PostgresTicketRetriever(store, embeddings, version, "fixture", dimensions=2)
    await store.start()
    scope = principal_context.set(Principal("alice", "tenant-a", frozenset({"reader"}), frozenset({"noc"}), True))
    try:
        await retriever.publish(tickets, b"acl fixture")
        hits = await retriever.search("PCRF-LON-01", filters={"tenant_id": "tenant-b", "acl_groups": ["*"]})
        assert [doc.document_id for doc in hits.documents] == [tickets[0]["ticket_id"]]
        legacy_scope = principal_context.set(Principal())
        try:
            assert (await retriever.search("PCRF-LON-01")).documents == []
        finally:
            principal_context.reset(legacy_scope)
        await store.save(session, "alice's private query", "alice's answer")
        assert len((await store.get_history(session)).messages) == 2
        other_scope = principal_context.set(Principal("bob", "tenant-b", groups=frozenset({"noc"}), enforce_acl=True))
        try:
            assert (await store.get_history(session)).messages == []
            await store.forget(session)
            assert [doc.document_id for doc in (await retriever.search("PCRF-LON-01")).documents] == [tickets[1]["ticket_id"]]
        finally:
            principal_context.reset(other_scope)
        assert len((await store.get_history(session)).messages) == 2
        denied_scope = principal_context.set(Principal("eve", "tenant-a", enforce_acl=True))
        try:
            embeddings.embed_query.reset_mock()
            assert (await retriever.search("PCRF-LON-01")).documents == []
            embeddings.embed_query.assert_not_called()
        finally:
            principal_context.reset(denied_scope)
    finally:
        await store.forget(session)
        principal_context.reset(scope)
        await asyncio.to_thread(store._transaction, "DELETE FROM knowledge_chunks WHERE version=%s; DELETE FROM knowledge_versions WHERE name=%s", (version, version))
        await store.close()
