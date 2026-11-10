"""Transactional, checksummed forward migrations, serialized across API/workers.

Existing 1.1 databases are adopted by idempotent initial migrations. Never edit
an applied migration; append a new version. Database rollback uses a backup.
"""

import hashlib

MIGRATIONS = (
    (1, """
CREATE TABLE IF NOT EXISTS platform_messages (
 id BIGSERIAL PRIMARY KEY, session_id TEXT NOT NULL,
 role TEXT NOT NULL CHECK (role IN ('user','assistant')), content TEXT NOT NULL,
 created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE INDEX IF NOT EXISTS platform_messages_session ON platform_messages(session_id,id DESC);
CREATE INDEX IF NOT EXISTS platform_messages_retention ON platform_messages(created_at);
"""),
    (2, """
CREATE TABLE IF NOT EXISTS knowledge_versions (
 name TEXT PRIMARY KEY, source_sha256 TEXT NOT NULL, embedding_model TEXT NOT NULL,
 dimensions INTEGER NOT NULL, documents INTEGER NOT NULL,
 created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS knowledge_chunks (
 version TEXT NOT NULL REFERENCES knowledge_versions(name), chunk_id TEXT NOT NULL,
 ticket_id TEXT NOT NULL, source TEXT NOT NULL, content TEXT NOT NULL,
 metadata JSONB NOT NULL, embedding DOUBLE PRECISION[] NOT NULL,
 PRIMARY KEY(version,chunk_id));
"""),
    (3, """
ALTER TABLE platform_messages ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'local';
ALTER TABLE platform_messages ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT 'operator';
CREATE INDEX IF NOT EXISTS platform_messages_owner ON platform_messages(tenant_id,user_id,session_id,id DESC);
ALTER TABLE knowledge_versions ADD COLUMN IF NOT EXISTS configuration TEXT NOT NULL DEFAULT 'C';
ALTER TABLE knowledge_versions ADD COLUMN IF NOT EXISTS publish_job_id TEXT;
"""),
    (4, """
CREATE TABLE ingestion_jobs (
 id TEXT PRIMARY KEY, version TEXT UNIQUE NOT NULL, tenant_id TEXT NOT NULL,
 owner_id TEXT NOT NULL, acl_groups JSONB NOT NULL, source BYTEA NOT NULL,
 source_sha256 TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending'
 CHECK(state IN ('pending','running','retry','succeeded','failed')),
 attempts INTEGER NOT NULL DEFAULT 0, max_attempts INTEGER NOT NULL DEFAULT 3,
 available_at TIMESTAMPTZ NOT NULL DEFAULT now(), lease_until TIMESTAMPTZ,
 lease_token TEXT, error_code TEXT, result JSONB,
 created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE INDEX ingestion_jobs_claim ON ingestion_jobs(state,available_at);
CREATE TABLE ingestion_quarantine (
 job_id TEXT REFERENCES ingestion_jobs(id), line_number INTEGER NOT NULL,
 record_sha256 TEXT NOT NULL, error_code TEXT NOT NULL,
 PRIMARY KEY(job_id,line_number));
CREATE TABLE operational_actions (
 id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, requester TEXT NOT NULL,
 operation TEXT NOT NULL, parameters JSONB NOT NULL, payload_sha256 TEXT NOT NULL,
 state TEXT NOT NULL DEFAULT 'pending' CHECK(state IN ('pending','approved','rejected','succeeded')),
 approver TEXT, executor TEXT, result JSONB,
 expires_at TIMESTAMPTZ NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE action_audit (
 id BIGSERIAL PRIMARY KEY, action_id TEXT NOT NULL REFERENCES operational_actions(id),
 tenant_id TEXT NOT NULL, actor TEXT NOT NULL, event TEXT NOT NULL,
 payload_sha256 TEXT NOT NULL, previous_hash TEXT NOT NULL, event_hash TEXT NOT NULL,
 created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE FUNCTION forbid_action_audit_mutation() RETURNS trigger LANGUAGE plpgsql AS $$
 BEGIN RAISE EXCEPTION 'Action audit is append-only'; END; $$;
CREATE TRIGGER action_audit_immutable BEFORE UPDATE OR DELETE ON action_audit
 FOR EACH ROW EXECUTE FUNCTION forbid_action_audit_mutation();
"""),
    (5, """
ALTER TABLE ingestion_jobs ADD COLUMN configuration TEXT NOT NULL DEFAULT 'C';
ALTER TABLE ingestion_jobs ADD COLUMN embedding_model TEXT NOT NULL DEFAULT 'text-embedding-3-small';
ALTER TABLE ingestion_jobs ADD COLUMN embedding_dimensions INTEGER NOT NULL DEFAULT 1536;
"""),
    (6, """
ALTER TABLE ingestion_jobs ADD COLUMN updated_at TIMESTAMPTZ NOT NULL DEFAULT now();
CREATE FUNCTION ingestion_touch_updated_at() RETURNS trigger LANGUAGE plpgsql AS $$
 BEGIN NEW.updated_at=now(); RETURN NEW; END; $$;
CREATE TRIGGER ingestion_job_updated BEFORE UPDATE ON ingestion_jobs
 FOR EACH ROW EXECUTE FUNCTION ingestion_touch_updated_at();
CREATE INDEX ingestion_jobs_updated ON ingestion_jobs(updated_at);
"""),
)


def migrate(connection):
    with connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(746352109)")
            cursor.execute("CREATE TABLE IF NOT EXISTS platform_schema_migrations (version INTEGER PRIMARY KEY, checksum TEXT NOT NULL, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())")
            cursor.execute("SELECT version,checksum FROM platform_schema_migrations ORDER BY version")
            applied = dict(cursor.fetchall())
            if set(applied) - {version for version, _ in MIGRATIONS}:
                raise RuntimeError("Database schema is newer than this application; rollback requires a compatible image")
            for version, sql in MIGRATIONS:
                checksum = hashlib.sha256(sql.encode()).hexdigest()
                if version in applied:
                    if applied[version] != checksum:
                        raise RuntimeError("Applied migration checksum mismatch")
                    continue
                cursor.execute(sql)
                cursor.execute("INSERT INTO platform_schema_migrations(version,checksum) VALUES (%s,%s)", (version, checksum))
