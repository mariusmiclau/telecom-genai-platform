"""Owner/tenant-scoped replay snapshots and separately reviewed corrections.

PostgreSQL is durable in live mode. The public local demo uses a bounded,
expiring in-memory store, never a pretend durable production queue.
"""

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.evidence import CREDENTIAL, EvidenceItem, link_claims, normalized
from src.guardrails.output_filter import OutputFilter
from src.security import current_principal
from src.version import VERSION


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def safe(value):
    if isinstance(value, str):
        return OutputFilter().apply(CREDENTIAL.sub("credential=[REDACTED]", normalized(value))).text
    if isinstance(value, list):
        return [safe(item) for item in value]
    if isinstance(value, dict):
        return {key: safe(item) for key, item in value.items()}
    return value


class FeedbackSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rating: Literal["helpful", "wrong"]
    incorrect_citation_ids: list[str] = Field(default_factory=list, max_length=20)
    comment: str = Field("", max_length=2000)
    proposed_correction: str = Field("", max_length=4000)


class FeedbackDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approve: bool
    reviewed_answer: str = Field("", max_length=4000)
    rationale: str = Field(min_length=10, max_length=2000)


class AnswerRecords:
    def __init__(self, store=None, retention_days=30):
        self.store = store
        self.retention_days = retention_days
        self.records: dict = {}
        self.feedback: dict = {}
        self.events: list = []

    async def capture(self, response, query, context, prompt, dataset, retrieval, session_id=None):
        principal = current_principal()
        raw_prompt_hash = digest(prompt)
        record = safe({"schema": "answer-replay-v1", "application_version": VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(), "query": query,
            "context": context or {}, "prompt": prompt, "dataset": dataset,
            "retrieval": retrieval, "response": response.model_dump(mode="json"),
            "replay_kind": "recorded_evidence_and_claim_gate_not_neural_regeneration"})
        record["scope"] = {"tenant": principal.tenant, "owner": principal.subject}
        record["prompt_redacted"] = digest(record["prompt"]) != raw_prompt_hash
        record["original_prompt_sha256"] = raw_prompt_hash
        record["record_sha256"] = digest(record)
        answer_id = str(uuid.uuid4())
        if self.store:
            def insert(cursor):
                cursor.execute("DELETE FROM answer_records WHERE created_at < now() - (%s * interval '1 day')", (self.retention_days,))
                cursor.execute("INSERT INTO answer_records(id,tenant_id,owner_id,session_id,record) VALUES (%s,%s,%s,%s,%s)",
                    (answer_id, principal.tenant, principal.subject, session_id, json.dumps(record)))
            await self.store.run(insert)
        else:
            cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
            self.records = {key: item for key, item in self.records.items() if datetime.fromisoformat(item["created_at"]) >= cutoff}
            if len(self.records) >= 500:
                self.records.pop(next(iter(self.records)))
            valid = set(self.records)
            self.feedback = {key: value for key, value in self.feedback.items() if value["answer_id"] in valid}
            self.events = self.events[-1000:]
            self.records[answer_id] = record
        return answer_id

    async def get(self, answer_id, review=False):
        principal = current_principal()
        if review:
            principal.require("feedback_review")
        if self.store:
            def read(cursor):
                cursor.execute("SELECT record FROM answer_records WHERE id=%s AND tenant_id=%s AND created_at >= now() - (%s * interval '1 day')", (answer_id, principal.tenant, self.retention_days))
                row = cursor.fetchone()
                return row[0] if row else None
            record = await self.store.run(read)
        else:
            record = self.records.get(answer_id)
            if record and datetime.fromisoformat(record["created_at"]) < datetime.now(timezone.utc) - timedelta(days=self.retention_days):
                record = None
        if not record or record["scope"]["tenant"] != principal.tenant or (not review and record["scope"]["owner"] != principal.subject):
            raise HTTPException(404, "Answer not found")
        if principal.enforce_acl and self.store:
            version = record["dataset"].get("version")
            for evidence in record["response"].get("evidence", []):
                def acl(cursor):
                    cursor.execute("SELECT 1 FROM knowledge_chunks WHERE version=%s AND ticket_id=%s AND metadata->>'tenant_id'=%s AND EXISTS (SELECT 1 FROM jsonb_array_elements_text(metadata->'acl_groups') AS g(value) WHERE g.value=ANY(%s)) LIMIT 1",
                        (version, evidence["document_id"], principal.tenant, list(principal.groups)))
                    return cursor.fetchone()
                if not await self.store.run(acl):
                    raise HTTPException(404, "Answer not found")
        return record

    async def replay(self, answer_id):
        record = await self.get(answer_id)
        expected = record["record_sha256"]
        if digest({key: value for key, value in record.items() if key != "record_sha256"}) != expected:
            raise HTTPException(409, "Replay integrity check failed")
        evidence = [EvidenceItem.model_validate(item) for item in record["response"]["evidence"]]
        claims, supported = link_claims(record["response"]["response"], evidence)
        return {"record_sha256": expected, "claim_gate_supported": supported,
            "claims": [claim.model_dump() for claim in claims], "model_called": False,
            "prompt_redacted": record["prompt_redacted"], "stored_outcome": record["response"]["outcome"],
            "limitations": "Rechecks saved, redacted evidence under current claim rules; does not repeat retrieval or reproduce stochastic model output."}

    async def submit(self, answer_id, submission):
        record = await self.get(answer_id)
        citations = {item["citation_id"] for item in record["response"]["evidence"]}
        if set(submission.incorrect_citation_ids) - citations:
            raise HTTPException(422, "Incorrect citation IDs must refer to this answer")
        principal = current_principal()
        feedback_id = str(uuid.uuid4())
        item = {"id": feedback_id, "answer_id": answer_id, "tenant_id": principal.tenant,
            "submitter": principal.subject, "payload": safe(submission.model_dump()), "state": "pending",
            "reviewer": None, "decision": None}
        if self.store:
            def insert(cursor):
                cursor.execute("INSERT INTO engineer_feedback(id,answer_id,tenant_id,submitter,payload) VALUES (%s,%s,%s,%s,%s)",
                    (feedback_id, answer_id, principal.tenant, principal.subject, json.dumps(item["payload"])))
                cursor.execute("INSERT INTO feedback_events(feedback_id,tenant_id,actor,event,payload_sha256) VALUES (%s,%s,%s,'submitted',%s)",
                    (feedback_id, principal.tenant, principal.subject, digest(item["payload"])))
            await self.store.run(insert)
        else:
            if len(self.feedback) >= 1000:
                raise HTTPException(429, "Demo feedback capacity reached")
            self.feedback[feedback_id] = item
        return item

    async def feedback_item(self, feedback_id):
        principal = current_principal()
        if self.store:
            def read(cursor):
                cursor.execute("SELECT answer_id,submitter,payload,state,reviewer,decision FROM engineer_feedback WHERE id=%s AND tenant_id=%s", (feedback_id, principal.tenant))
                row = cursor.fetchone()
                return dict(zip(("answer_id", "submitter", "payload", "state", "reviewer", "decision"), row)) if row else None
            item = await self.store.run(read)
        else:
            item = self.feedback.get(feedback_id)
            if item and item["tenant_id"] != principal.tenant:
                item = None
        if not item:
            raise HTTPException(404, "Feedback not found")
        await self.get(item["answer_id"], review=item["submitter"] != principal.subject)
        return item

    async def review(self, feedback_id, decision):
        principal = current_principal()
        principal.require("feedback_review")
        item = await self.feedback_item(feedback_id)
        record = await self.get(item["answer_id"], review=True)
        if not principal.enforce_acl or principal.subject == item["submitter"]:
            raise HTTPException(409, "A different verified OIDC reviewer is required")
        if item["state"] != "pending":
            raise HTTPException(409, "Feedback has already been reviewed")
        reviewed = safe(decision.model_dump())
        if decision.approve:
            evidence = [EvidenceItem.model_validate(value) for value in record["response"]["evidence"]]
            _, supported = link_claims(reviewed["reviewed_answer"], evidence)
            if not supported:
                raise HTTPException(422, "Reviewed answer needs source-supported citations; new evidence requires separate approved ingestion")
        state = "approved" if decision.approve else "rejected"
        if self.store:
            def update(cursor):
                cursor.execute("UPDATE engineer_feedback SET state=%s,reviewer=%s,decision=%s WHERE id=%s AND tenant_id=%s AND state='pending' RETURNING id",
                    (state, principal.subject, json.dumps(reviewed), feedback_id, principal.tenant))
                if not cursor.fetchone():
                    raise HTTPException(409, "Feedback has already been reviewed")
                cursor.execute("INSERT INTO feedback_events(feedback_id,tenant_id,actor,event,payload_sha256) VALUES (%s,%s,%s,%s,%s)",
                    (feedback_id, principal.tenant, principal.subject, state, digest(reviewed)))
            await self.store.run(update)
        else:
            self.feedback[feedback_id].update(state=state, reviewer=principal.subject, decision=reviewed)
        return await self.feedback_item(feedback_id)

    async def regression(self, feedback_id):
        current_principal().require("feedback_review")
        item = await self.feedback_item(feedback_id)
        if item["state"] != "approved":
            raise HTTPException(409, "Only approved corrections can be exported")
        record = await self.get(item["answer_id"], review=True)
        evidence = [EvidenceItem.model_validate(value) for value in record["response"]["evidence"]]
        claims, _ = link_claims(item["decision"]["reviewed_answer"], evidence)
        cited = {citation for claim in claims for citation in claim.citation_ids}
        return {"id": feedback_id, "query": record["query"], "reference_answer": item["decision"]["reviewed_answer"],
            "question_id": "feedback-" + feedback_id, "question": record["query"],
            "expected_ticket_ids": [item.document_id for item in evidence if item.citation_id in cited],
            "expected_sources": [item.document for item in evidence if item.citation_id in cited], "tags": ["reviewed_engineer_correction"],
            "expected_document_ids": [item.document_id for item in evidence if item.citation_id in cited],
            "sources": [item.model_dump() for item in evidence if item.citation_id in cited],
            "dataset": record["dataset"], "reviewer": item["reviewer"],
            "auto_training": False, "status": "reviewed_regression_candidate_requires_repository_review"}
