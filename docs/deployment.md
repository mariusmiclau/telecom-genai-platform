# Deployment runbook

Bearer mode is a single-operator ticket assistant. Optional OIDC API identity,
ACLs, durable jobs and guarded local action simulation are described in
[the 1.2 completion record](roadmap-1.2.md); actual enterprise/provider acceptance
has not been performed. Deployment is
available on a Linux Docker host, or through Azure Container Apps and private
PostgreSQL. It does not execute carrier network changes. Enterprise SSO,
multi-tenant document permissions and operational adapters require a separate
integration scope before admitting multiple trust domains.

## Release gates

CI checks the locked install, lint, types, tests, known dependency advisories,
golden retrieval, distributions, the Docker demo, the production stack over TLS,
and Azure provider-schema validation plus two mocked infrastructure plans. The production acceptance harness uses a
local OpenAI wire fixture. It verifies HTTP integration and failure handling;
it is not evidence of real model answer quality. Its report explicitly records
`real_model_quality_measured: false`.

The image gate rejects fixable high/critical advisories and preserves the full
high/critical report, including unfixed findings. On 10 November 2026 the measured
Debian 13 runtime had 47 high/critical OS-package findings without published fixes,
zero fixable high/critical findings, and zero high/critical Python-package findings.
This is not a clean security bill: review the full CI artifact and obtain the
receiving organization's acceptance or rebuild with remediated packages before
production. Non-root, read-only filesystems, dropped capabilities and no-new-
privileges reduce exposure but do not establish that findings are unexploitable.

To reproduce:

```sh
uv sync --locked --extra dev --python 3.11
docker compose build demo
uv run python scripts/accept_deployment.py
```

The harness creates an isolated `telco-acceptance-*` project. It publishes 50
synthetic tickets through the actual operator command and embedding SDK,
verifies the TLS certificate against its generated CA, rejects anonymous
requests, checks a cited answer, follow-up memory, unsupported-answer abstention,
provider outage handling, deletion and a PostgreSQL backup/restore. It removes
only its own synthetic containers and volumes afterward. It never installs a
certificate in your host trust store and never calls a real model account.

## Publish the accepted container

After merging this release, manually run the **Publish verified deployment
image** workflow in your personal repository. It accepts the built image before
pushing that same image to GHCR and attaching a GitHub provenance attestation.
The job summary prints the immutable `ghcr.io/...@sha256:...` reference.
It runs only when dispatched; creating a PR does not publish an image.

Authenticate the deployment host to the registry if the package is private.
Use a token limited to package reads. Verify the attestation with:

```sh
gh attestation verify oci://YOUR_IMAGE_REFERENCE --repo mariusmiclau/telecom-genai-platform
```

Substitute the actual digest reference from the workflow. Keep the source
commit, image digest, model names, dimensions and knowledge version together
with every deployment record.

## Linux host deployment

Prerequisites: Docker Engine with Compose, a domain pointing to the host, ports
80/443 available, an approved ticket snapshot, and an OpenAI account with access
to the configured chat and embedding models. Retain the default model contract
unless an evaluation justifies changing it. Do not use a test provider in a
real deployment.

1. Set OPENAI_API_KEY in the shell without saving it in source control. Run
   `uv run python -m src.deployment init`. It writes unique operator/admin/database
   credentials into `.secrets/` and never prints their values. On Linux, the
   directory is owner-only; file mounts are readable by the container users.
   Secret files are not encrypted by Compose: protect the host disk and backups.
2. Copy `deploy.env.example` to `deploy.env`. Set APP_IMAGE to the accepted digest,
   DOMAIN to your hostname, KNOWLEDGE_VERSION to a new immutable name, and
   DATASET_PATH to the approved ticket directory containing `tickets.jsonl`.
   This file contains settings, not secrets.
3. Run `uv run python scripts/deploy.py publish`. Publication validates the source
   and embeds it, then commits the complete snapshot transactionally. Duplicate
   versions fail; the command never replaces an accepted index. Model calls
   consume the configured account's quota and budget.
4. Run `uv run python scripts/deploy.py apply`. It checks configuration, pulls the
   pinned image and waits for database/API readiness before starting the TLS
   gateway. Only the gateway publishes ports. PostgreSQL remains internal.
5. Run the credential-backed smoke check with an approved question and its
   expected ticket:

```sh
uv run python scripts/verify_live.py --url https://YOUR_DOMAIN \
  --question "Which incident fixed DIAMETER_TOO_BUSY on PCRF-LON-01?" \
  --expect-ticket INC20260001
```

Use that example only with the included synthetic snapshot. An abstention fails
the smoke acceptance; investigate evidence, citations, model behavior and the
selected snapshot. Public certificate issuance requires working DNS and inbound
80/443. Local acceptance uses Caddy's private CA instead of public issuance.

Conversation retention defaults to 30 days. Expired turns are excluded from
reads and physically purged on startup and successful conversation writes.
For inactive systems needing scheduled physical erasure, run a maintenance
restart or a scheduled database retention operation. An authenticated operator
can delete a session with `DELETE /api/v1/sessions/{session_id}`.

Resource bound: at most 10,000 chunks AND 2,000,000 stored embedding scalars per
snapshot. At 1,536 dimensions this is about 1,302 chunks. Publishing or starting
an oversized snapshot fails before serving traffic. Increase capacity only
after load/memory testing or replace the bounded scan with an indexed backend.

## Rollback

Preserve the previous non-secret deployment file, image digest and knowledge
snapshot before applying a new release. Run:

```sh
uv run python scripts/deploy.py rollback --env-file previous-deploy.env
```

Rollback changes the image/configuration/snapshot selector; it never deletes
database rows. Repeat readiness and the known-ticket smoke check. Never run
`down --volumes` against production. The application uses one process and one
replica so its admission quota remains global within the documented deployment.

## Backup and recovery

```sh
uv run python scripts/database_backup.py backup --file telecom-backup.dump
uv run python scripts/database_backup.py restore --file telecom-backup.dump --restore-database restore_drill
```

Backups use PostgreSQL custom format and owner-only file permissions. A backup
file is created exclusively and never overwritten. Restore only creates a new
`restore_*` database and fails if the name exists. It never drops a database.
Verify row counts and smoke queries against the restored database before
switching the mounted database_url secret. Store backups encrypted outside the
host, set retention, and perform scheduled recovery drills.

## Azure deployment

`infrastructure/terraform` now matches this runtime: private PostgreSQL 16,
Container Apps, ACR, a managed identity for image pulls/Key Vault reads, and a
manual publishing job. It does not provision an unused vector search product.
PostgreSQL uses certificate-verified TLS, private DNS and no public endpoint.
The API defaults to internal ingress and one replica. Database destruction is
blocked by Terraform lifecycle rules; backups retain 35 days.

Terraform 1.9+ is required. Configure Azure authentication, the subscription ID
and an encrypted remote-state backend with locking before planning. State
contains the generated database credential, so restrict state access. Existing
Key Vault secret URLs are used for the reader/admin/model keys; the module does
not read their values into Terraform. Required Azure providers must already be
registered because automatic registration is disabled.

Staged rollout avoids starting an API against an empty snapshot:

1. Initialize the backend and plan/apply infrastructure with both `deploy_api`
   and `deploy_publisher` false. Review cost, region and network policy first.
2. Push the accepted image to the output ACR registry. Set `container_image` to
   its immutable ACR digest; configure `key_vault_id` and `secret_ids` with keys
   `api-token`, `admin-token`, and `openai-key` pointing to existing secrets.
3. Plan/apply with `deploy_publisher=true`. Run the output publisher job from
   Azure and verify its success. The dataset must be baked into the accepted
   image at `dataset_path`; the default path is the synthetic demonstration set.
4. Plan/apply with `deploy_api=true`. Check the internal HTTPS output from an
   allowed network and run the live smoke check. Set external_ingress=true only
   when the intended API access policy is accepted.

Rollback by planning/applying the previous digest and knowledge_version.
Do not run `terraform destroy` to roll back an application. Terraform fmt/init/
validate are executed in CI without credentials. No Azure resources have been
created or a subscription-specific plan accepted by the agent.

## Monitoring

Set `OTEL_TRACES_EXPORTER=otlp` and `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` in
deploy.env to your approved private OTLP/HTTP collector. Terraform exposes the
equivalent `traces_exporter` and `traces_endpoint` variables. Export is disabled
by default because no receiving telemetry service is provisioned by this stack.
Keep raw query capture disabled and restrict collector access and retention.
The existing tracing tests verify every required request stage.

The authenticated `/metrics` surface includes answer outcomes, latency, token
usage and authentication failures. The Docker gateway hides it from public
traffic. Scrape the internal API using the reader credential file from your
monitoring network. `infrastructure/prometheus/alerts.yaml` contains initial
availability, dependency-failure and authentication rules. Route alerts to the
receiving team's on-call destination and adjust thresholds from measured load.
Do not put raw credentials in a scrape configuration committed to Git.

## Operator inputs still required at deployment time

An accepted image digest, actual model credentials, approved data, target domain
and host/cloud account are deployment inputs. They cannot be fabricated by a
test fixture. Real-provider quality, tenant policy and carrier integration
acceptance remain distinct from the repeatable software deployment checks.

References: [Docker secrets](https://docs.docker.com/compose/how-tos/use-secrets/),
[Caddy HTTPS](https://caddyserver.com/docs/automatic-https),
[OpenAI embeddings](https://developers.openai.com/api/reference/resources/embeddings/methods/create).
