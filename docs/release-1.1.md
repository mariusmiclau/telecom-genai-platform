# Deployment release 1.1 — 10 November 2026

Supported scope: single-operator historical-ticket assistant. This release adds
deployable software and recovery tooling, not carrier-system integration or
multi-tenant authorization.

## Locally measured acceptance

- 188 tests passed with real PostgreSQL integration enabled; lint and type checks passed.
- Golden retrieval regression passed: Recall@5 A 78%, B 86%, C 100%, on the
  committed 50-question synthetic set. These are not real-model quality scores.
- The actual production Compose topology passed HTTPS certificate verification,
  anonymous-request rejection, SDK embedding and completion integration, cited
  response, follow-up, unsupported-number abstention, sanitized provider failure
  and session deletion. The provider was an explicit local wire fixture, not OpenAI.
- PostgreSQL custom-format backup restored all 150 chunks into a new database.
- Terraform provider validation and two mocked plans passed without cloud credentials.
- Locked default runtime dependency audit found no known advisories.
- Container scan: zero high/critical Python-package findings and zero fixable
  high/critical findings; 47 unfixed high/critical OS-package findings remain.
  Review the complete CI report before production risk acceptance.

## Delivered operations

TLS gateway, separate mounted operator/admin credentials, private PostgreSQL,
non-root/read-only API, bounded resource use, explicit embedding contracts,
conversation retention/deletion, safe new-database restore, immutable snapshot
publication, digest-pinned rollout/rollback preflight, manual image publication
with provenance, outcome/authentication metrics, alert rules and configurable
privacy-safe OTLP export.

The Azure module replaces stale scaffolding with a staged private PostgreSQL /
Container Apps deployment. Mocked tests do not establish subscription-specific
capacity, networking policy, permission propagation or successful cloud apply.

## Before real production traffic

Follow [the deployment runbook](deployment.md). Supply an approved dataset,
real model credentials, accepted image digest and target host/domain or Azure
account. Run credential-backed smoke and real-ticket quality acceptance; review
the unfixed security findings, monitoring destination and backup policy. No cloud
resources were provisioned, no real-provider quality was measured, and no Mercury
organization repository was modified.
