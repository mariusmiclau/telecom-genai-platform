"""Scope, budgets, review authority and snapshot replay regression contracts."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from src.agents.memory import ConversationHistory, Message
from src.agents.orchestrator import AgentOrchestrator
from src.answer_records import AnswerRecords, FeedbackDecision, FeedbackSubmission
from src.api.demo import OfflineDemoService
from src.api.main import create_app
from src.api.models import ChatResponse
from src.budgets import (
    BudgetExceeded,
    Consumption,
    consumption_context,
    messages_bound,
)
from src.evidence import EvidenceItem
from src.investigation import clarify, scoped_documents, troubleshooting
from src.rag_pipeline.chain import RAGChain
from src.rag_pipeline.retriever import RetrievedDocument
from src.security import Principal, principal_context
from src.settings import AppSettings


def principal(name="alice", tenant="tenant-a", review=False, groups=frozenset({"noc"})):
    return Principal(name, tenant, frozenset({"reader", "feedback_review"} if review else {"reader"}), groups, True)


def response_fixture():
    return ChatResponse(response="Root cause: A stale Diameter route [Source 1]", sources=[], confidence=.8,
        evidence=[EvidenceItem(citation_id="source-1", document="ticket.json", document_id="INC1",
            passage="Root cause: A stale Diameter route", document_kind="ticket", freshness="historical")])


@pytest.mark.parametrize("query,context,fields", [
    ("Why is the interface down?", {}, {"element_id", "interface"}),
    ("Diagnose packet loss in region London", {}, set()),
    ("Troubleshoot PCRF-LON-01 on software version", {}, {"software_version"}),
    ("Why is PCRF-LON-01 slow?", {}, set()),
    ("Why is pcrf-lon-01 slow?", {}, set()),
    ("Why is the interface down?", {"element_id": "PCRF-LON-01", "interface": "Gx"}, set()),
    ("Explain BGP", {}, set()),
    ("Explain network health checks", {}, set()),
    ("What is the node status?", {}, {"element_id"}),
    ("What caused INC20260001?", {}, set()),
    ("Why are CORE-A and CORE-B down?", {}, {"element_id"}),
])
def test_scope_clarification(query, context, fields):
    assert set(clarify(query, context).required_fields) == fields


@pytest.mark.asyncio
async def test_ambiguous_operation_never_retrieves_or_calls_model():
    retriever, chain, memory = SimpleNamespace(search=AsyncMock()), SimpleNamespace(generate=AsyncMock()), SimpleNamespace(get_history=AsyncMock())
    result = await AgentOrchestrator(retriever, chain, memory, enable_tools=False).process("Why is the interface down?")
    assert result.outcome == "abstained" and result.clarification.required_fields
    retriever.search.assert_not_called()
    chain.generate.assert_not_called()
    memory.get_history.assert_not_called()


def test_scope_never_uses_high_scoring_wrong_node_or_interface():
    doc = RetrievedDocument("wrong", "CORE-B has Gx issues", .99, "wrong.json", {"element_id": "CORE-B", "interface": "Gx"})
    assert scoped_documents("Why is CORE-A slow?", {}, [doc]) == []
    assert scoped_documents("Why is CORE-B slow?", {"interface": "Gy"}, [doc]) == []
    assert clarify("Diagnose region London", {}, [doc, RetrievedDocument("other", "", .5, "", {"element_id": "CORE-C"})]).required_fields == ["element_id"]


def test_structured_hypotheses_never_claim_live_diagnosis_or_execute_fixes():
    view = troubleshooting("Packets dropped", response_fixture().evidence)
    assert view.symptoms[0].status == "user_report_unverified"
    assert view.likely_causes[0].citation_ids == ["source-1"]
    assert view.likely_causes[0].status == "historical_hypothesis_not_current_diagnosis"
    assert len(view.suggested_read_only_checks) == 3 and view.missing_evidence
    assert not any("restart" in check.lower() for check in view.suggested_read_only_checks)


@pytest.mark.asyncio
async def test_prompt_budget_fails_before_paid_completion():
    chain = RAGChain()
    chain._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock())))
    token = consumption_context.set(Consumption(prompt_token_limit=100))
    try:
        with pytest.raises(BudgetExceeded):
            await chain.generate("x" * 500, [])
        chain._client.chat.completions.create.assert_not_called()
    finally:
        consumption_context.reset(token)


def test_unicode_history_is_bounded_and_system_roles_are_not_inherited():
    now = datetime.now(timezone.utc)
    history = ConversationHistory("session", [Message("user", "界" * 2000, now), Message("system", "untrusted", now), Message("user", "small", now)], now, now)
    token = consumption_context.set(Consumption(history_token_limit=100))
    try:
        messages = RAGChain()._build_messages("q", "context", "", history, None)
        assert "small" in [message["content"] for message in messages]
        assert "untrusted" not in [message["content"] for message in messages]
        assert not any("界" in message["content"] for message in messages)
        assert messages_bound(messages) >= len(json.dumps(messages).encode()) - 100
    finally:
        consumption_context.reset(token)


@pytest.mark.asyncio
async def test_retry_budget_has_no_hidden_extra_attempts_and_reserves_failed_calls():
    import httpx
    from openai import RateLimitError
    error = RateLimitError("fixture", response=httpx.Response(429, request=httpx.Request("POST", "http://fixture")), body=None)
    create = AsyncMock(side_effect=error)
    chain = RAGChain(max_tokens=64)
    chain._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    budget = Consumption(retry_limit=1)
    token = consumption_context.set(budget)
    try:
        result = await chain.generate("query", [])
        assert result.finish_reason == "error"
        assert create.call_count == 2 == budget.completion_attempts
        assert budget.reserved_token_upper_bound == 2 * (budget.input_token_upper_bound + 64)
    finally:
        consumption_context.reset(token)


@pytest.mark.asyncio
async def test_retry_cannot_overrun_total_request_reservation():
    import httpx
    from openai import RateLimitError
    error = RateLimitError("fixture", response=httpx.Response(429, request=httpx.Request("POST", "http://fixture")), body=None)
    create = AsyncMock(side_effect=error)
    chain = RAGChain(max_tokens=64)
    chain._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    budget = Consumption(retry_limit=2)
    bound = messages_bound(chain._build_messages("query", chain._format_context([]), "", None, None)) + 64
    budget.maximum_request_tokens = bound
    token = consumption_context.set(budget)
    try:
        with pytest.raises(BudgetExceeded):
            await chain.generate("query", [])
        assert create.call_count == 1
    finally:
        consumption_context.reset(token)


@pytest.mark.asyncio
async def test_demo_scope_feedback_replay_end_to_end():
    app = create_app(AppSettings(_env_file=None))
    app.state.runtime.demo = OfflineDemoService()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        ambiguous = (await client.post("/api/v1/demo/chat", json={"message": "Why is the interface down?"})).json()
        assert ambiguous["clarification"]["required_fields"] == ["element_id", "interface"]
        result = await client.post("/api/v1/demo/chat", json={"message": "Why is the interface down?", "context": {"element_id": "PCRF-LON-01", "interface": "Gx"}})
        # Scope must also influence retrieval, not only filter its output.
        assert result.status_code == 200 and result.json()["outcome"] == "answered", result.text
        answer = (await client.post("/api/v1/demo/chat", json={"message": "Which incident fixed DIAMETER_TOO_BUSY on PCRF-LON-01?"})).json()
        assert answer["outcome"] == "answered" and answer["troubleshooting"]["relevant_incidents"]
        assert answer["consumption"]["completion_attempts"] == 0
        identifier = answer["answer_id"]
        record = (await client.get(f"/api/v1/answers/{identifier}")).json()
        assert record["dataset"]["source_sha256"] and record["retrieval"]["configuration"] == "C"
        replay = (await client.post(f"/api/v1/answers/{identifier}/replay")).json()
        assert replay["model_called"] is False and replay["claim_gate_supported"] is True
        feedback = (await client.post(f"/api/v1/answers/{identifier}/feedback", json={"rating": "wrong", "incorrect_citation_ids": ["source-1"], "proposed_correction": "untrusted"})).json()
        assert feedback["state"] == "pending"
        assert (await client.get(f"/api/v1/feedback/{feedback['id']}/regression")).status_code == 409
        assert (await client.post(f"/api/v1/feedback/{feedback['id']}/review", json={"approve": True, "reviewed_answer": answer["response"], "rationale": "Reviewed in demo"})).status_code == 409


@pytest.mark.asyncio
async def test_snapshot_redaction_integrity_and_owner_isolation():
    records = AnswerRecords()
    scope = principal_context.set(principal())
    try:
        identifier = await records.capture(response_fixture(), "engineer@example.com", {},
            {"messages": [{"role": "user", "content": 'api_key="CANARY_SECRET"'}]}, {"version": "fixture"}, {})
        record = await records.get(identifier)
        assert record["prompt_redacted"] is True
        assert "CANARY_SECRET" not in json.dumps(record) and "engineer@example.com" not in json.dumps(record)
        principal_context.set(principal("bob"))
        with pytest.raises(HTTPException) as denied:
            await records.get(identifier)
        assert denied.value.status_code == 404
        principal_context.set(principal(tenant="tenant-b"))
        with pytest.raises(HTTPException):
            await records.get(identifier)
        principal_context.set(principal())
        records.records[identifier]["query"] = "tampered"
        with pytest.raises(HTTPException) as tampered:
            await records.replay(identifier)
        assert tampered.value.status_code == 409
    finally:
        principal_context.reset(scope)


@pytest.mark.asyncio
async def test_feedback_needs_independent_review_and_never_auto_trains():
    records = AnswerRecords()
    scope = principal_context.set(principal(review=True))
    try:
        identifier = await records.capture(response_fixture(), "query", {}, {}, {}, {})
        item = await records.submit(identifier, FeedbackSubmission(rating="wrong", incorrect_citation_ids=["source-1"], proposed_correction="Not trusted"))
        decision = FeedbackDecision(approve=True, reviewed_answer=response_fixture().response, rationale="Checked original source passage")
        with pytest.raises(HTTPException) as denied:
            await records.review(item["id"], decision)
        assert denied.value.status_code == 409
        principal_context.set(principal("bob"))
        with pytest.raises(HTTPException) as denied:
            await records.review(item["id"], decision)
        assert denied.value.status_code == 403
        principal_context.set(principal("bob", review=True))
        bad = decision.model_copy(update={"reviewed_answer": "Reboot all routers [Source 1]"})
        with pytest.raises(HTTPException) as denied:
            await records.review(item["id"], bad)
        assert denied.value.status_code == 422
        reviewed = await records.review(item["id"], decision)
        assert reviewed["state"] == "approved" and reviewed["reviewer"] == "bob"
        case = await records.regression(item["id"])
        assert case["auto_training"] is False and case["expected_document_ids"] == ["INC1"]
        assert case["expected_ticket_ids"] == ["INC1"] and case["question_id"].startswith("feedback-")
        with pytest.raises(HTTPException):
            await records.review(item["id"], decision)
    finally:
        principal_context.reset(scope)


@pytest.mark.asyncio
async def test_http_budget_denial_and_concurrent_request_accounting_are_isolated():
    import asyncio

    from src.agents.orchestrator import AgentResult
    settings = AppSettings(_env_file=None, app_mode="live", api_token="r" * 32,
        openai_api_key="fixture", database_url="postgresql://fixture/fixture")
    app = create_app(settings)
    async def process(message, *args):
        budget = consumption_context.get()
        budget.reserve(999999 if message == "exceed" else 100 if message == "small" else 200)
        await asyncio.sleep(.01)
        return AgentResult("No evidence", [], 0., outcome="abstained")
    app.state.runtime.agent = SimpleNamespace(process=process, memory=SimpleNamespace(_transaction=lambda *args: []))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", headers={"Authorization": "Bearer " + "r" * 32}) as client:
        responses = await asyncio.gather(*(client.post("/api/v1/chat", json={"message": query}) for query in ("small", "larger")))
        assert [item.json()["consumption"]["reserved_token_upper_bound"] for item in responses] == [100, 200]
        denied = await client.post("/api/v1/chat", json={"message": "exceed"})
        assert denied.status_code == 429 and "token budget" in denied.json()["detail"]


@pytest.mark.asyncio
async def test_admission_covers_generation_and_replay_record_writes(monkeypatch):
    import asyncio

    import src.api.routes as routes
    from src.agents.orchestrator import AgentResult
    active, peak = 0, 0
    app = create_app(AppSettings(_env_file=None, app_mode="live", api_token="r" * 32,
        openai_api_key="fixture", database_url="postgresql://fixture/fixture", max_concurrent_requests=2))
    async def process(*args):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(.01)
            return AgentResult("No evidence", [], 0., outcome="abstained")
        finally:
            active -= 1
    original_record = routes.record_answer
    async def delayed_record(*args):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(.05)
            return await original_record(*args)
        finally:
            active -= 1
    monkeypatch.setattr(routes, "record_answer", delayed_record)
    app.state.runtime.agent = SimpleNamespace(process=process, memory=SimpleNamespace(_transaction=lambda *args: []))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", headers={"Authorization": "Bearer " + "r" * 32}) as client:
        responses = await asyncio.gather(*(client.post("/api/v1/chat", json={"message": "Find incident"}) for _ in range(8)))
        assert all(response.status_code == 200 for response in responses)
    assert peak <= 2, f"Generation and capture exceeded admission capacity: {peak}"


@pytest.mark.asyncio
async def test_postgres_replay_survives_service_restart_and_checks_current_acls(isolated_store):
    from pathlib import Path

    from src.evaluation.dataset import load_jsonl
    from src.rag_pipeline.postgres import PostgresTicketRetriever
    tickets = load_jsonl(Path("datasets/tickets-v2.0-ticket-aware-chunking/tickets.jsonl"))[:1]
    tickets[0].update(tenant_id="tenant-a", acl_groups=["noc"])
    retriever = PostgresTicketRetriever(isolated_store, SimpleNamespace(embed_documents=AsyncMock(return_value=[[1., 0.]] * 3)), "feedback-fixture", "fixture", 2)
    await retriever.publish(tickets, b"fixture")
    response = response_fixture()
    response.evidence[0].document_id = tickets[0]["ticket_id"]
    records = AnswerRecords(isolated_store)
    scope = principal_context.set(principal())
    try:
        identifier = await records.capture(response, "query", {}, {}, {"version": "feedback-fixture"}, {})
        item = await records.submit(identifier, FeedbackSubmission(rating="wrong"))
        restarted = AnswerRecords(isolated_store)
        assert (await restarted.replay(identifier))["claim_gate_supported"] is True
        principal_context.set(principal(groups=frozenset({"other"})))
        with pytest.raises(HTTPException) as denied:
            await restarted.get(identifier)
        assert denied.value.status_code == 404
        principal_context.set(principal("bob", review=True))
        decision = FeedbackDecision(approve=True, reviewed_answer=response.response, rationale="Independent source review completed")
        assert (await restarted.review(item["id"], decision))["state"] == "approved"
        assert (await restarted.regression(item["id"]))["expected_document_ids"] == [tickets[0]["ticket_id"]]
        def audit(cursor):
            cursor.execute("SELECT event FROM feedback_events WHERE feedback_id=%s ORDER BY id", (item["id"],))
            return cursor.fetchall()
        assert await isolated_store.run(audit) == [("submitted",), ("approved",)]
    finally:
        principal_context.reset(scope)
