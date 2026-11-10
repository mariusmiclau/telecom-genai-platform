# Local release 1.0 — 10 November 2026

## Acceptance evidence

- Fresh Python 3.11 environment installed from `uv.lock`.
- 180 tests passed with RUN_INTEGRATION=1, including real PostgreSQL publication,
  immutable version protection, embedding-model checks and recent-history transactions.
- Ruff and mypy passed.
- Golden retrieval regression passed: Recall@5 A 78%, B 86%, C 100% on the
  explicitly synthetic hashing/BM25 experiment.
- Default runtime dependency audit: no known vulnerabilities reported.
- Python wheel and source distribution built; isolated wheel install can load
  the packaged dataset and answer the known PCRF ticket query.
- Docker demo build and HTTP acceptance smoke test; run `scripts/verify_release.py`
  to repeat readiness, static assets, evidence attribution and abstention checks.

## Delivered behavior

Default mode is a local extractive demo with no external account requirements.
Live mode owns startup/shutdown of a bounded PostgreSQL pool and model clients,
validates configuration, checks knowledge readiness, authenticates operator
requests, restricts operational actions, limits request work, and abstains when
answer publication checks fail. Ingestion publishes immutable PostgreSQL ticket
snapshots through an operator CLI. Switching KNOWLEDGE_VERSION selects or rolls
back the accepted snapshot without mutating a published index.

The default install excludes Chroma after the audit identified unpatched
server-side code-execution and authorization advisories. Its historical adapter
is retained as an optional legacy extra, excluded from the verified runtime.

## External acceptance still required

No real model credentials, carrier data, OSS/BSS accounts or cloud subscription
were supplied. Model-backed answer quality, enterprise identity, document ACLs,
multi-user authorization, network actions and Azure operation are therefore not
claimed as validated. The live release is one trusted operator domain with a
10,000-chunk capacity bound, suitable for an externally reviewed pilot after
those organization-specific requirements are resolved.

The delivery destination is the personal `mariusmiclau/telecom-genai-platform`
repository. No Mercury organization repositories are part of this release.
