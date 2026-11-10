# Local roadmap completion — release 1.2.0

This release completes the locally implementable development items identified
in the gap review. It does **not** claim a production cloud deployment, real
carrier integration, approved operational data or real-provider answer quality.
Infrastructure remains deferred by the owner.

Local verification: 210 tests with PostgreSQL integration enabled, lint and
types pass. Six Prometheus alert rules validate. Dependency audit has no known
default-runtime findings. Full-stack acceptance restores 752 chunks, verifies
the actual durable SDK worker and exercises eight concurrent HTTPS requests.

## Implementation and evidence

| Gap | Delivered local implementation | Verification |
|---|---|---|
| Runtime composition | Validated settings, owned lifespan, readiness and cleanup | Existing lifecycle, timeout and failure tests |
| Runtime-path evaluation | A/B/C use PostgresTicketRetriever, the actual embedding SDK, persisted chunks and runtime ranking | 50 questions through the local wire provider |
| Identity and document permissions | OIDC RS256, issuer/audience/time claims, trusted JWKS, roles, tenants and group ACLs; tenant/user session scoping | Signed JWTs, HTTP JWKS/key rotation, HTTP roles and real PostgreSQL cross-tenant denial tests |
| Operational control | Independent approver, executor permission, exact plan digest, expiry, replay prevention and durable hash-linked audit | HTTP and PostgreSQL approval, rejection/tampering/expiry/isolation tests; local simulation only |
| Explicit outcomes | answered, abstained and escalated carried through orchestration, HTTP and metrics | Source-less publication guard and distinct escalation tests |
| Persistence | Bounded connection pool, off-event-loop transactions and six checksummed forward migrations | Real PostgreSQL migrations, idempotency/checksum and isolation tests |
| Durable ingestion | Persisted jobs, exclusive claims, bounded retries/backoff, heartbeat leases, fencing, quarantine and post-publication crash recovery | Real database competing-worker, retry, terminal failure, recovery and publication tests; actual SDK worker acceptance |
| Recovery and load | Immutable snapshots, activation and rollback, bounded admission, backup/restore | Actual API-container recreation and snapshot rollback; concurrent HTTPS burst and admission tests |
| Alerts and gates | Six validated alert rules, queue age/failures/expired leases, CI lint/types/dependency audit and integration gates | Actual metrics collection and promtool rule validation |

Azure infrastructure was aligned and provider-mocked plans validated in 1.1.
This is not an Azure subscription-specific plan or successful cloud apply.

## Measured retrieval, not a substitute backend

The runtime-path benchmark uses recursive chunks/vector search (A), ticket-aware
chunks/vector plus BM25 (B), and B plus the runtime field reranker (C).
On 50 synthetic questions with the explicit local embedding fixture:

| Configuration | Recall@5 |
|---|---:|
| A | 76% |
| B | 86% |
| C | 100% |

The original in-memory baseline remains a separate reproducibility experiment;
its A result was 78%, not 76%. Do not substitute those numbers for one another.
The measured [runtime fixture summary](../reports/evaluation/runtime-fixture-1.2.json)
is committed separately from that original experiment.
The runtime-path fixture uses deterministic hashing embeddings, not a neural
model. It also generated 50 cited answers for review; expected-source citation
checks are not human correctness scores. Reports explicitly retain
`real_model_quality_measured: false` and unfilled human-review fields.

Reproduce using `uv run python scripts/accept_deployment.py` after building the
demo image. It runs only isolated synthetic containers, never a paid provider.
Reports are `.release-check/production-evaluation.json` and
`.release-check/deployment-acceptance.json`; CI preserves them as artifacts.

For an approved real-provider experiment, configure settings, then explicitly run:

```sh
uv run python -m src.evaluation.production --source APPROVED_TICKETS.jsonl \
  --questions APPROVED_QUESTIONS.jsonl --prefix UNIQUE_EXPERIMENT \
  --publish --answers --output .release-check/real-provider-review.json
```

This incurs account usage. Omit `--publish` to evaluate existing A/B/C snapshots
named PREFIX-A/B/C. Add `--tenant` and repeated `--group` flags to evaluate with
enforced ACLs. Do not use `--fixture` for real providers. Review generated answers
against references before claiming accuracy; never commit operational data or
real answer-review exports into the public repository.

## Identity contract

For native local live mode, set AUTH_MODE=oidc plus OIDC_ISSUER, OIDC_AUDIENCE,
OIDC_JWKS_URL and, if needed, OIDC_TENANT_CLAIM (default tenant_id). Both identity
URLs require HTTPS outside explicit test fixtures. The API validates exp, iat,
sub, issuer and audience using RS256 only. It never follows a token-supplied
key URL. JWKS caching/refresh cooldown is bounded; issuer key rotation must allow
for cached keys. Token issuance, login UI, provider provisioning and enterprise
revocation-policy acceptance belong to the receiving identity system.

Claims roles/groups are bounded string arrays. Roles are reader, ingest,
action_request, action_approve, action_execute, auditor and admin.
Admin implies endpoint permissions but does not bypass document
ACLs or the independent-approver rule. Audit reads additionally require auditor
(or admin). Job routes require ingest. Documents require matching tenant_id and
an intersecting acl_groups array. Missing ACLs deny access in OIDC mode. Client
context filters cannot change tenant or groups. The tenant name local is reserved
for shared-token compatibility and rejected in OIDC tokens. Shared bearer mode sees only
unlabelled/local-tenant documents and is not multi-user authentication.

One configured knowledge snapshot can contain multiple tenants' labelled
documents. An ingestion job produces a tenant-specific immutable snapshot;
promotion is an explicit configuration change, not an automatic worker side
effect. Combining tenant snapshots into a shared release is a trusted operator
publication step. No per-tenant self-service onboarding portal is claimed.

## Durable ingestion

HTTP: POST /api/v1/ingestion/jobs with version, source_jsonl, acl_groups and
configuration A/B/C; GET /api/v1/ingestion/jobs/{id} returns only that tenant/user's
job status and quarantine metadata. Assigned ACLs come from verified job ownership,
not source-record claims. Ingesters cannot grant groups outside their own claims.

Bootstrap before starting an empty live API using the trusted database-operator CLI:

```sh
uv run python -m src.jobs submit --source APPROVED_TICKETS.jsonl \
  --version UNIQUE_VERSION --tenant TENANT --owner OPERATOR --group noc
uv run python -m src.jobs worker --once
uv run python -m src.jobs status --job JOB_ID --tenant TENANT --owner OPERATOR
```

The CLI requires database/model credentials and deliberately is not an HTTP
authentication bypass. For continuous processing omit --once and supervise the
worker process. No worker must be publicly exposed. Use the returned namespaced
version in KNOWLEDGE_VERSION only after evaluation. A/B snapshots also require
matching RETRIEVAL_CONFIGURATION. The older atomic publish CLI remains an
explicit one-shot operator compatibility path, not a durable job queue.

Jobs have three attempts by default, exponential backoff, 180-second leases and
30-second heartbeats. Publication checks the current fence token transactionally.
A crash after publication is reconciled without repeating embedding calls. Queue
capacity defaults to 20 unfinished jobs per tenant. Terminal jobs clear raw source
payloads while retaining snapshot/record hashes, counts and safe error codes;
quarantine identifies original line numbers without returning raw rejected data.
Correct invalid input upstream and submit a new version; accepted versions are
never overwritten. Protect original approved snapshots and database backups.

## Approved local action simulation

Enable ENABLE_LOCAL_ACTIONS only for an intentional local test. A verified actor
POSTs /api/v1/actions with node_id and change_summary. An independent identity
GETs /actions/{id} to review the exact plan (reader plus an action-review role),
then POSTs /actions/{id}/approve or /reject with payload_sha256; an authorized executor
POSTs /execute with that same digest. Plans expire after 15 minutes. GET
/actions/{id}/audit verifies the ordered audit chain for an authorized auditor.

The sole adapter is simulate_config_change: it writes a local ledger result
atomically with execution state/audit. It never changes a network or contacts an
OSS/BSS. The LLM cannot approve or execute this workflow. Application-level
UPDATE/DELETE of audit events is blocked; this is not an externally anchored,
tamper-proof store against a privileged database administrator. A real adapter
would need separate target-level permissions, idempotency and carrier acceptance.

## Boundaries still requiring external acceptance

Real enterprise login/provider policy, OSS/BSS credentials/adapters, an approved
operational dataset and real-provider human-scored answers cannot be manufactured
by local fixtures. Neither can cloud capacity, cross-host scaling or on-call
delivery. The local burst is a functional concurrency check, not a production
throughput/SLO claim. Rollback verifies container/configuration/snapshot selection,
not arbitrary downgrade across incompatible database or authorization versions.
Never downgrade to an image that drops OIDC/ACL enforcement.

Default dependency audit is clean. The image still has unfixed OS high/critical
advisories; review the full scan and obtain production security acceptance later.
Registry publication previously failed at the registry; no accepted published
digest/provenance or cloud deployment is claimed. No Mercury repository is used.
