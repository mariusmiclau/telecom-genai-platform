# Local delivery — 1.1.1

The current delivery target is local software. Infrastructure provisioning,
registry configuration and production rollout are deferred by the owner.

## Completed and measured

- 193 tests pass with PostgreSQL integration enabled; lint and type checks pass.
- The 50-question synthetic retrieval gate passes with Recall@5 A 78%, B 86%,
  C 100%. These are offline retrieval measurements, not real-model quality.
- The full local production topology passes certificate-verified HTTPS,
  authentication, source attribution, follow-up, unsupported-answer abstention,
  provider failure handling, retention/deletion and 150-chunk backup restoration.
- The API response trace ID correlates to actual exported spans for all nine
  required request stages. Raw query, context and entity attributes are rejected
  with content capture disabled. Arbitrary request-context values are no longer
  attached to spans; only their field count is recorded.
- Actual metrics expose answered, abstained, dependency-error and rejected-auth
  events from the same integration test.
- Versioned wheel, source distribution and a local Docker image are built.

The integration provider is a local OpenAI wire fixture. No external model key
or cloud account is required for these checks. Test cleanup removes only the
harness's isolated synthetic containers/volumes, not development databases.

## Run and reproduce

```sh
uv sync --locked --extra dev --python 3.11
docker compose --profile integration up -d --wait postgres redis
# PowerShell: $env:RUN_INTEGRATION='1'; uv run pytest -q
# POSIX shell: RUN_INTEGRATION=1 uv run pytest -q
uv run ruff check src tests scripts
uv run mypy src --ignore-missing-imports
uv run python -m src.evaluation.cli --compare-to reports/evaluation/baseline.json
docker compose build demo
uv run python scripts/accept_deployment.py
docker compose up -d --wait demo
uv run python scripts/verify_release.py
```

Open http://127.0.0.1:8000/demo. This is an offline historical-ticket demo, not
a live carrier operations console. Live action simulators remain disabled.

Acceptance evidence is written to `.release-check/deployment-acceptance.json`.
It records `real_model_quality_measured: false`, the exported stage names and
monitoring verification. Python distributions and the portable Docker archive
are delivered under `dist/`; the archive can be loaded using `docker load -i`.
Checksums accompany the artifacts. Source snapshots never include `.secrets`,
deployment environment files, test credentials or generated test volumes.

## Deferred, not silently marked complete

- Registry publication/provenance: the 10 November manual workflow passed image
  acceptance and the fixable-advisory gate, but the registry rejected the final
  push. No accepted registry digest or provenance attestation is claimed.
- Hosting, public DNS/TLS, cloud-specific infrastructure acceptance and receiving
  monitoring/on-call/backup destinations remain infrastructure-phase work.
- Real model access and real-ticket quality require an approved dataset and
  credentials. Fixture results cannot replace those acceptance checks.
- The image still has unfixed high/critical OS advisories; inspect the full scan
  report before production. A green fixable-advisory gate is not risk acceptance.
- Enterprise SSO, multiple trust domains and carrier OSS/BSS writes are outside
  the supported single-operator release, not hidden unfinished integrations.

See [the deployment runbook](deployment.md) when the infrastructure phase begins.
