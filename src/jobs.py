"""PostgreSQL-backed ingestion queue with leases, fencing, quarantine and recovery."""

import argparse
import asyncio
import hashlib
import json
import uuid
from pathlib import Path
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.evaluation.dataset import REQUIRED_TICKET_FIELDS
from src.evidence import record_rejection
from src.security import current_principal


class IngestionSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
    source_jsonl: str = Field(min_length=1, max_length=5_000_000)
    acl_groups: list[str] = Field(min_length=1, max_length=128)
    configuration: Literal["A", "B", "C"] = "C"


class IngestionQueue:
    def __init__(self, store, settings):
        self.store, self.settings = store, settings

    async def submit(self, submission: IngestionSubmission):
        principal = current_principal()
        principal.require("ingest")
        groups = submission.acl_groups
        if not all(isinstance(group, str) and 0 < len(group) <= 128 for group in groups):
            raise HTTPException(422, "Invalid ACL groups")
        if principal.enforce_acl and not set(groups).issubset(principal.groups):
            raise HTTPException(403, "Cannot grant documents to groups outside your identity")
        source = submission.source_jsonl.encode()
        if len(source) > self.settings.max_upload_bytes:
            raise HTTPException(413, "Snapshot is too large")
        job_id = str(uuid.uuid4())
        version = principal.tenant + ":" + submission.version

        def operation(cursor):
            cursor.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,42))", ("ingestion:" + principal.tenant,))
            cursor.execute("SELECT count(*) FROM ingestion_jobs WHERE tenant_id=%s AND state IN ('pending','running','retry')", (principal.tenant,))
            if cursor.fetchone()[0] >= self.settings.max_pending_ingestion_jobs:
                raise HTTPException(429, "Tenant ingestion queue capacity reached")
            cursor.execute("INSERT INTO ingestion_jobs(id,version,tenant_id,owner_id,acl_groups,source,source_sha256,configuration,embedding_model,embedding_dimensions) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(version) DO NOTHING RETURNING id",
                (job_id, version, principal.tenant, principal.subject, json.dumps(groups), source,
                 hashlib.sha256(source).hexdigest(), submission.configuration,
                 self.settings.embedding_model_name, self.settings.embedding_dimensions))
            if not cursor.fetchone():
                raise HTTPException(409, "Version is not available; accepted versions are immutable")
            return {"id": job_id, "version": version, "state": "pending"}
        return await self.store.run(operation)

    async def status(self, job_id):
        principal = current_principal()
        principal.require("ingest")

        def operation(cursor):
            cursor.execute("SELECT id,version,state,attempts,error_code,result FROM ingestion_jobs WHERE id=%s AND tenant_id=%s AND owner_id=%s", (job_id, principal.tenant, principal.subject))
            row = cursor.fetchone()
            if not row:
                raise HTTPException(404, "Job not found")
            cursor.execute("SELECT line_number,record_sha256,error_code FROM ingestion_quarantine WHERE job_id=%s ORDER BY line_number", (job_id,))
            quarantine = [dict(zip(("line_number", "record_sha256", "error_code"), item)) for item in cursor.fetchall()]
            return {**dict(zip(("id", "version", "state", "attempts", "error_code", "result"), row)), "quarantine": quarantine}
        return await self.store.run(operation)

    async def claim(self):
        lease = str(uuid.uuid4())

        def operation(cursor):
            # Expired workers cannot retain ownership; each claim has a new fence token.
            cursor.execute("UPDATE ingestion_jobs j SET state='succeeded',source=''::bytea,lease_until=NULL,error_code=NULL,result=jsonb_build_object('recovered_after_publication',true,'version',j.version,'accepted_records',v.documents,'rejected_records',(SELECT count(*) FROM ingestion_quarantine q WHERE q.job_id=j.id)) FROM knowledge_versions v WHERE j.state='running' AND j.lease_until<now() AND v.publish_job_id=j.id AND v.name=j.version")
            cursor.execute("UPDATE ingestion_jobs SET state='failed',source=''::bytea,error_code='attempts_exhausted' WHERE state='running' AND lease_until<now() AND attempts>=max_attempts")
            cursor.execute("UPDATE ingestion_jobs SET state='running',attempts=attempts+1,lease_token=%s,lease_until=now()+interval '180 seconds' WHERE id=(SELECT id FROM ingestion_jobs WHERE attempts<max_attempts AND ((state IN ('pending','retry') AND available_at<=now()) OR (state='running' AND lease_until<now())) ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1) RETURNING id,version,tenant_id,acl_groups,source,configuration,embedding_model,embedding_dimensions,attempts,max_attempts",
                           (lease,))
            row = cursor.fetchone()
            return {**dict(zip(("id", "version", "tenant", "groups", "source", "configuration", "model", "dimensions", "attempts", "max_attempts"), row)), "lease": lease} if row else None
        return await self.store.run(operation)

    async def work_once(self, embeddings):
        from src.rag_pipeline.postgres import PostgresTicketRetriever
        job = await self.claim()
        if job is None:
            return False

        async def heartbeat():
            while True:
                await asyncio.sleep(30)
                await asyncio.to_thread(self.store._transaction, "UPDATE ingestion_jobs SET lease_until=now()+interval '180 seconds' WHERE id=%s AND lease_token=%s AND state='running'", (job["id"], job["lease"]))

        heartbeat_task = asyncio.create_task(heartbeat())
        try:
            if (job["model"], job["dimensions"]) != (self.settings.embedding_model_name, self.settings.embedding_dimensions):
                raise ValueError("Embedding contract mismatch")
            existing = await asyncio.to_thread(self.store._transaction,
                "SELECT publish_job_id FROM knowledge_versions WHERE name=%s", (job["version"],), True)
            if existing and existing[0][0] != job["id"]:
                raise ValueError("Version already belongs to another publication")
            tickets, rejected, seen = [], [], set()
            for number, line in enumerate(bytes(job["source"]).decode("utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict) or not REQUIRED_TICKET_FIELDS.issubset(record) or not all(isinstance(record[key], str) and record[key].strip() for key in REQUIRED_TICKET_FIELDS):
                        raise ValueError("Invalid ticket schema")
                    if record["ticket_id"] in seen:
                        raise ValueError("Duplicate ticket")
                    rejection = record_rejection(record)
                    if rejection:
                        raise ValueError(rejection)
                    seen.add(record["ticket_id"])
                    # ACLs are assigned from the verified job, never trusted from source text.
                    record.update(tenant_id=job["tenant"], acl_groups=job["groups"])
                    tickets.append(record)
                except (ValueError, TypeError) as error:
                    code = str(error) if str(error) in ("suspicious_source", "invalid_evidence_metadata") else "invalid_record"
                    rejected.append((job["id"], number, hashlib.sha256(line.encode()).hexdigest(), code))
            if rejected:
                def quarantine(cursor):
                    cursor.executemany("INSERT INTO ingestion_quarantine(job_id,line_number,record_sha256,error_code) VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING", rejected)
                await self.store.run(quarantine)
            if not tickets:
                raise ValueError("No valid tickets")
            if not existing:
                retriever = PostgresTicketRetriever(self.store, embeddings, job["version"], job["model"], job["dimensions"],
                    configuration=job["configuration"], job_id=job["id"], lease_token=job["lease"])
                result = await retriever.publish(tickets, bytes(job["source"]))
            else:
                result = {"version": job["version"], "recovered_after_publication": True}
            result.update(accepted_records=len(tickets), rejected_records=len(rejected))
            await asyncio.to_thread(self.store._transaction, "UPDATE ingestion_jobs SET state='succeeded',source=''::bytea,result=%s,error_code=NULL,lease_until=NULL WHERE id=%s AND lease_token=%s AND state='running'", (json.dumps(result), job["id"], job["lease"]))
        except Exception as error:
            permanent = isinstance(error, (ValueError, UnicodeError))
            state = "failed" if permanent or job["attempts"] >= job["max_attempts"] else "retry"
            await asyncio.to_thread(self.store._transaction, "UPDATE ingestion_jobs SET state=%s,source=CASE WHEN %s='failed' THEN ''::bytea ELSE source END,error_code=%s,available_at=now()+(%s * interval '1 second'),lease_until=NULL WHERE id=%s AND lease_token=%s AND state='running'",
                (state, state, "validation_failed" if permanent else "dependency_failed", min(300, 5 * 2 ** job["attempts"]), job["id"], job["lease"]))
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
        return True


async def worker(once=False):
    from src.agents.store import PostgresConversationStore
    from src.rag_pipeline.embeddings import EmbeddingService, EmbeddingSettings
    from src.settings import AppSettings
    settings = AppSettings.load()
    store = PostgresConversationStore(settings.database_url.get_secret_value())
    embeddings = EmbeddingService(EmbeddingSettings(OPENAI_API_KEY=settings.openai_api_key.get_secret_value(),
        model_name=settings.embedding_model_name, dimensions=settings.embedding_dimensions,
        base_url=settings.openai_base_url, cache_enabled=False))
    try:
        await store.start()
        queue = IngestionQueue(store, settings)
        while True:
            worked = await queue.work_once(embeddings)
            if once:
                return
            if not worked:
                await asyncio.sleep(1)
    finally:
        await embeddings.close()
        await store.close()


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    worker_command = commands.add_parser("worker")
    worker_command.add_argument("--once", action="store_true")
    submit_command = commands.add_parser("submit")
    submit_command.add_argument("--source", type=Path, required=True)
    submit_command.add_argument("--version", required=True)
    submit_command.add_argument("--group", action="append", required=True)
    submit_command.add_argument("--tenant", default="local")
    submit_command.add_argument("--owner", default="operator")
    status_command = commands.add_parser("status")
    status_command.add_argument("--job", required=True)
    status_command.add_argument("--tenant", default="local")
    status_command.add_argument("--owner", default="operator")
    args = parser.parse_args()
    if args.command == "worker":
        asyncio.run(worker(args.once))
    else:
        from src.agents.store import PostgresConversationStore
        from src.security import Principal, principal_context
        from src.settings import AppSettings

        async def submit():
            settings = AppSettings.load()
            store = PostgresConversationStore(settings.database_url.get_secret_value())
            # This CLI requires database credentials: it is an explicit trusted
            # bootstrap operator, not an alternative HTTP authentication path.
            scope = principal_context.set(Principal(subject=args.owner, tenant=args.tenant))
            try:
                await store.start()
                queue = IngestionQueue(store, settings)
                if args.command == "status":
                    print(json.dumps(await queue.status(args.job)))
                else:
                    print(json.dumps(await queue.submit(IngestionSubmission(
                        version=args.version, source_jsonl=args.source.read_text(encoding="utf-8"), acl_groups=args.group))))
            finally:
                await store.close()
                principal_context.reset(scope)
        asyncio.run(submit())


if __name__ == "__main__":
    main()
