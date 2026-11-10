# Release operations

## Delivered scope

Version 1.0 delivers the local demo, protected single-operator live RAG runtime,
dataset/evaluation tooling, Python distributions and Docker image. It is not a
certification for use against a carrier network. Cloud deployment, enterprise
identity, tenant document ACLs, operational actions and real-ticket acceptance
require the receiving organization's accounts, policies and approved data.

## Live setup

Copy `.env.example` to `.env`. Generate separate random reader/admin tokens
with at least 32 characters, set OPENAI_API_KEY, and select OPENAI_MODEL. Keep
the file outside Git. All network bindings are localhost. For remote access,
use an authenticated TLS gateway; never expose the databases publicly.

Start dependencies before indexing:

```sh
docker compose --profile live up -d --wait postgres redis
```

From a uv environment with the key configured, ingest an approved snapshot:

```sh
uv run python scripts/publish_dataset.py --source datasets/tickets-v2.0-ticket-aware-chunking/tickets.jsonl --version tickets-v1
```

The included snapshot is synthetic. Set `KNOWLEDGE_VERSION=tickets-v1` in
`.env`, then start `docker compose --profile live up -d api`. The live endpoint
is `http://127.0.0.1:8002/api/v1/chat`; provide `Authorization: Bearer <API_TOKEN>`.
Startup fails for missing credentials or an empty/unreachable snapshot.
No model call is made by readiness; a valid configured key still requires an
operator smoke query to confirm account/model access.

The native process uses APP_MODE=live and DATABASE_URL from `.env`; Compose
overrides addresses for the container network. Redis is an optional embedding
cache. Credentials are never returned by health or readiness responses.

## Index publication and rollback

Ingestion is an operator CLI in this release. HTTP ingestion deliberately
returns 409 and performs no indexing. Build a new uniquely named snapshot,
evaluate it, and change KNOWLEDGE_VERSION only after acceptance. Restart the
runtime and verify `/ready` and a known cited query. Roll back by restoring the
previous snapshot setting and image version. Keep previous snapshots until
the rollback window closes. Never mix embedding models or dimensions in one
snapshot. Live BM25 supports up to 10,000 chunks and rejects larger snapshots.

## Persistence and schema

The v1 runtime installs an idempotent `platform_messages` schema at startup,
independent of legacy conversation tables. Complete user/assistant turns are
saved in one transaction. The pool and statement deadlines bound database work.
Schema changes after v1 require a reviewed forward migration and backup.
There is no automatic content retention deletion in this release: configure
retention and deletion with the receiving organization before storing personal
data. A shared operator token represents one trust domain, not multiple users.

## Failure handling

| Symptom | Check / recovery |
|---|---|
| `/health` 200, `/ready` 503 | Database, PostgreSQL knowledge connectivity, snapshot configuration |
| HTTP 401 | Exact Bearer header; reader and admin credentials differ |
| HTTP 429 | Process-wide request quota; Retry-After header |
| HTTP 504 | Request deadline; inspect dependency latency |
| HTTP 503 on chat | Dependency/provider failure; inspect server logs and traces |
| Answer abstained | No evidence, missing citation, incomplete output or failed grounding |
| Empty live snapshot | Index approved data before starting live runtime |

Grounding is a heuristic and cannot prove factual correctness. Review high
impact recommendations with a network engineer. Live operational actions are
disabled. Do not register the demonstration simulator tools as live adapters.

## Backup and recovery

Back up PostgreSQL with `pg_dump` using the deployment's secret management.
Restore into a separate database with `pg_restore`, verify recent conversation
turns, then switch DATABASE_URL and restart. The same backup contains knowledge
snapshots, or recreate a new snapshot from
the versioned approved source and pinned embedding configuration. Preserve
source manifests alongside each index release. Run a restore drill before an
external production pilot. This local delivery has no claimed RPO/RTO.

## Observability

`docker compose --profile tracing up -d jaeger` starts the local trace backend.
Configure OTEL_TRACES_EXPORTER=otlp and the exporter URL for a native process;
the live Compose profile points at Jaeger. Enable query capture only after
privacy review. Monitor error rate, response latency, abstentions, model usage,
snapshot freshness and database capacity. Attach organization-specific SLOs,
alert routes and on-call ownership before externally serving users.

## Acceptance

Run CI-equivalent commands in README. Tests include real PostgreSQL publication,
retrieval, conversation transactions and synthetic extractive HTTP responses. Credential-backed
model calls, OSS/BSS actions and Azure deployment are not exercised without
external accounts. All benchmark reports describe the synthetic experiment;
do not use its Recall@5 as a production accuracy claim.

Known retained scaffolding: `infrastructure/terraform/` provisions Azure AI
Search but this release runs PostgreSQL ticket retrieval. Worker and simulator Dockerfiles refer to
unimplemented modules. They are excluded from Compose and release CI.

## Legacy Chroma security exclusion

Chroma is excluded from the default dependencies and deployment. The audit
reported CVE-2026-45829, CVE-2026-45833, CVE-2026-45831 and CVE-2026-45830;
the advisory database listed no patched release at review time. Historical
adapter code remains available only through the optional `legacy-chroma` extra
for migration work. Do not expose its server or use that extra as the live
backend. References: https://github.com/advisories/GHSA-f4j7-r4q5-qw2c and
https://github.com/advisories/GHSA-36p7-vc44-83pf.
