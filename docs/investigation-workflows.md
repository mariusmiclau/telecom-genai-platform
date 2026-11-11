# Investigation workflows — 1.4

These controls run locally. They do not connect to network devices or authorize
changes. Historical evidence is not a current diagnosis.

## Clarification and structured investigation

`POST /api/v1/chat` and `/api/v1/demo/chat` accept optional scalar context fields:
`element_id`, `interface`, `region`, `software_version`, `equipment_family`.
Context is bounded to 16 fields of at most 256 UTF-8 bytes each. It cannot set
identity, ACLs, server policies, clocks, token budgets or model parameters.

Ambiguous operational wording returns `outcome: abstained`, no published answer,
and `clarification.required_fields` / `question`. Missing node/region, a requested
but unnamed interface, and an unspecified software version are checked before
retrieval or generation. Multiple candidate nodes in a region require a node.
Conceptual and explicitly identified historical-ticket questions need not specify
an operational asset. Rules recognize bounded telecom ID/interface patterns;
they are not a universal language-understanding or ambiguity detector. Supply
explicit context for identifiers outside those patterns.

Resubmit the original question with the confirmed scope. The demo has fields for
this. Scope influences retrieval and excludes mismatched asset/interface results;
it is never inferred from an arbitrary top hit. No silent scope inheritance is
performed from old conversation turns.

Published responses include `troubleshooting`:

- symptoms: the user's report, explicitly unverified;
- relevant_incidents: eligible historical tickets with citation links;
- likely_causes: extracted historical root causes, labelled hypotheses;
- suggested_read_only_checks: fixed, non-executed inspection suggestions;
- missing_evidence: live alarms/logs, time window, asset and compatibility gaps.

The investigation view is built from eligible evidence, not a second uncontrolled
LLM call. Generic check suggestions are labelled suggestions, not source-derived
facts. No commands, resets, configuration changes or network tools are executed.

## Engineer feedback and reviewed correction

Every completed chat response includes `answer_id`. Endpoint flow:

1. `POST /answers/{answer_id}/feedback` with `rating` (`helpful` or `wrong`),
   `incorrect_citation_ids`, `comment`, `proposed_correction`.
2. `GET /feedback/{feedback_id}` inspects the submitted item.
3. `POST /feedback/{feedback_id}/review` with `approve`, `reviewed_answer`,
   `rationale`. An independent verified OIDC identity with `feedback_review` (or
   admin) must review it. Shared bearer mode and the public demo cannot approve.
4. `GET /feedback/{feedback_id}/regression` exports an approved regression
   candidate with reference answer, expected document IDs, source snapshots and
   dataset identity. Pending or rejected feedback cannot be exported.

All paths have `/api/v1` prefix. Owners submit feedback on their own answers.
Reviewers must belong to the same tenant and retain access to every saved evidence
document. Wrong citations must identify citations actually supplied in the answer.
Reviewed answers must pass the existing lexical source-binding gate; this is not
proof of semantic correctness and does not replace the reviewer's judgment.
Corrections requiring new evidence need the separate approved ingestion process.

Nothing automatically changes prompts, knowledge, golden tests or model weights.
Exports also include the existing evaluator's `question_id`, `question`,
`expected_ticket_ids`, `expected_sources`, `reference_answer` and `tags` fields.
After repository review, append the case to a private golden JSONL for that same
dataset and run the existing runtime evaluator with `--questions <reviewed.jsonl>`.
Source/document IDs must exist in the selected snapshot. Keep the extra provenance
fields for auditing; do not confuse an approved expected answer with a measured
generated answer.
The export is data for a separate repository-reviewed regression change. Review
approval/rejection is one-shot and transactional; submitted/reviewed hashes are
recorded in an append-only audit table. Store these exports under the same data
access restrictions as their source material, not in a public repo by default.

Live records/feedback survive service restarts in PostgreSQL (migration 7).
Replay access rechecks document ACLs against the pinned immutable dataset.
Records expire after `ANSWER_RETENTION_DAYS` (30 by default); old records are
purged on new captures. Session deletion removes its answer records and feedback.
Audit events retain IDs, actors, timestamps and payload hashes, not correction
bodies. The local demo is deliberately ephemeral: at most 500 answer records and
1,000 feedback items, cleared on restart; all demo visitors share a local identity.
It is not a multi-user feedback service. Review and regression export require live
OIDC mode with distinct identities.

## Token, retry and resource budgets

Server settings (also available in production Compose):

| Setting | Default | Meaning |
|---|---:|---|
| MAX_REQUEST_TOKENS | 24000 | Sum of conservative reservations across paid attempts |
| MAX_PROMPT_TOKENS | 16000 | Maximum assembled prompt reservation per completion |
| MAX_OUTPUT_TOKENS | 1024 | Provider output cap for every completion |
| MAX_HISTORY_TOKENS | 2000 | Bounded history, additionally at most six recent messages |
| MODEL_RETRY_BUDGET | 0 | Additional transient completion attempts, at most two |
| REQUEST_TIMEOUT_SECONDS | 45 | Existing overall generation deadline |

UTF-8 bytes plus framing overhead provide a conservative accounting bound for the
configured byte-tokenized model family; this is not an exact tokenizer count or
a currency billing measurement. Input, source passages, history and all prompt
framing count. Oversized prompts fail closed with HTTP 429 *before* completion.
Each attempted completion reserves input plus the full output allowance, including
failed calls. A retry cannot exceed the total reservation. Live query embeddings
reserve input and have one application attempt; hidden SDK retries are disabled.
Ingestion has separate bounded jobs/retries and is not charged to chat budgets.

`consumption` reports these reservations, completion attempts and provider-reported
completion usage. Provider-reported usage is unavailable for failed calls; failed
attempts may still be billed. Embedding counts are reservations, not provider
billing. `currency_cost` is null, not a fabricated zero-dollar estimate.
Prometheus exposes `genai_budget_reserved_tokens_total` and
`genai_budget_rejections_total`. Existing concurrency, body/rate and time limits
remain. These are per-process/per-request controls, not an organization-wide
monetary spending cap. For cloud deployment enforce provider/account budgets too.

Admission and the request deadline cover generation **and** replay persistence;
replay/feedback routes share the same admission slots. A burst regression initially
showed eight operations overlapping despite two slots because persistence was
outside the generation slot. The lifecycle boundary was corrected and is locked
down by a test; increasing database pool size was not used to hide the defect.

This implements layered bounds consistent with [OWASP unbounded-consumption
guidance](https://genai.owasp.org/llmrisk/llm102025-unbounded-consumption/). The
existing gpt-4o Chat Completions path keeps its bounded `max_tokens` request;
new model families need separately verified parameter/tokenizer compatibility.
See the [official OpenAI API reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).

## Reproducible answer investigation

`GET /answers/{answer_id}` returns the saved replay manifest: dataset/source
snapshot hash, embedding model/dimensions, chunking/retrieval settings, application
and prompt template versions, assembled messages, completion model/temperature/
output limit, generated text, public evidence and final response. Provider secrets
and endpoint credentials are never included. Pattern-based PII/credential
redaction is applied and `prompt_redacted` discloses whether the prompt envelope
changed. The original-envelope and stored-record hashes aid comparison; they are
not signatures or protection against a privileged database administrator.

`POST /answers/{answer_id}/replay` validates the stored hash and re-runs the claim
gate on the recorded redacted evidence. It does not retrieve today's documents,
execute tools or call a model. Original outcomes and new gate results are shown
separately. This reproduces a bounded decision check, not the entire original
pipeline. Neural regeneration is deliberately not automatic: models are stochastic
and provider aliases may change. Retained messages and configuration allow a
separately authorized experiment, subject to redaction and provider availability.

Redaction is heuristic, not a guarantee that arbitrary secrets are detected.
Treat replay manifests as sensitive and keep access controls, disk encryption,
backups and retention appropriate to the operational dataset.

## Verification

Measured locally on 11 November 2026: 276 Python tests passed with PostgreSQL
integration enabled; 35/35 adversarial cases passed; lint, types and fake-DOM
renderer checks passed. The HTTPS acceptance fixture passed all eight concurrent
requests and the new investigation/feedback/replay contracts. These are local
software/fixture checks, not real-provider quality or production-scale results.

Run `uv run pytest tests/test_investigation_workflows.py -q`; set
`RUN_INTEGRATION=1` for the PostgreSQL restart/ACL/review/audit test. The main CI
also runs this file against real PostgreSQL. `node scripts/verify_demo_dom.cjs`
exercises the actual renderer in a fake DOM; it is not browser or visual QA.
The HTTPS acceptance harness verifies recorded prompts, consumption, clarification,
investigation sections, pending feedback, replay and deletion alongside existing
TLS/provider-failure/backup/rollback checks. No real-model quality measurement is
claimed by its local provider fixture.
