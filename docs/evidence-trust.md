# Evidence-first answers and source safety — 1.3.0

The three additions share one policy layer after authorization and before answer
generation. Cloud infrastructure and real model credentials are not required to
run their regression tests. None of these controls authorizes network actions.

Measured local verification: 252 Python tests with PostgreSQL integration pass,
35/35 safety regression cases pass, lint/types pass, and the renderer unit check
passes. Full-stack HTTPS acceptance verifies the new evidence fields alongside
the existing SDK retrieval, durable ingestion, rollback and backup restoration.

## Inspectable answers

The existing response/sources fields remain. New fields are:

- `answer_kind`: historical_resolution, advisory, insufficient_evidence or conflict.
- `evidence`: citation ID, source/document ID, supplied passage, document kind,
  review/effective dates, software/equipment metadata, freshness and redaction flag.
- `claims`: individual answer sentences and their source IDs. Support is labelled
  `lexical_match`, never independently verified.
- `warnings`: stable reason codes and explanations; no rejected source body.
- `conflicts`: scoped facts, competing values and their authorized source IDs.

Every published sentence must cite an in-range source and match that source's
supplied passage. Values found only in another document do not validate a wrong
citation. Extra unsupported citations, missing citations, invented values and
the tested negation/status inversions prevent publication. A rejected generation
is replaced by a safe abstention; its unsupported claims are not returned.

The demo renders claim links that open the corresponding evidence disclosure.
Passages and source names are inserted as text, never interpreted as HTML or
automatically fetched URLs. Historical incident resolutions are explicitly
labelled as history, not current instructions or operational approval.

Passages are limited to the same 1,500 characters supplied to the model. PII and
credential redaction also applies to nested evidence, claims, warnings and conflict
values. `redacted` flags transformed passages. `retrieved_content_sha256` hashes
the original retrieved text, not the original file bytes or a signed source.

## Source eligibility

Optional metadata on the existing ticket-shaped publication records:

| Field | Meaning |
|---|---|
| document_type | ticket (default), runbook, sla or configuration |
| reviewed_at | Last review timestamp; not proof an authorized review actually occurred |
| valid_from / valid_until | Effective interval [from, until) |
| superseded_by | Nonempty replacement identifier excludes this source; no automatic fetch |
| software_version / equipment_family | Compatibility labels |
| fact_scope | Asset/policy scope; otherwise element_id is used |
| facts | Up to 20 bounded string key/value facts for comparison |

Dates use ISO 8601. Date-only and naive values are interpreted as UTC. Invalid
metadata is rejected/quarantined; malformed pre-existing sources are excluded at
query time. Expired, not-yet-effective, superseded or incompatible sources are
not used. Non-ticket documents older than SOURCE_REVIEW_MAX_AGE_DAYS (default 365)
are excluded. A recent review timestamp means only “within policy by metadata,”
not independently current or authoritative. Missing review information produces
a freshness_unknown warning rather than an invented review date.

Ticket resolutions remain historical regardless of age. Explicit validity and
supersession metadata still applies. Client context may select software_version
or equipment_family, but cannot override the server clock, review-age policy,
identity, tenant or document groups. Metadata cannot grant operational permissions.

## Conflicts and compatibility

For eligible non-ticket sources, the policy compares declared facts and narrowly
recognized `Availability target: ...%` / `Maximum latency: ...ms` statements.
The grouping key includes asset/fact scope, software version and equipment family.
Equivalent numeric formatting/spacing is normalized without rounding. Different
historical incidents are not treated as simultaneous current policies.

Contradictory active facts cause escalation before an LLM call; competing values
and passages remain available for human review. Incompatible versions are not
mislabelled as factual contradictions: an unspecified version/family causes a
compatibility warning/escalation. Selecting a version excludes mismatched sources.
Stale or expired sources do not override eligible current evidence.

This is not general semantic contradiction detection. Missing scope/facts,
unrecognized units, prose ambiguity and source falsification still need review.

## Adversarial regression gate

```sh
uv run python -m src.evaluation.adversarial --output .release-check/adversarial.json
uv run pytest tests/test_evidence_trust.py -q
node scripts/verify_demo_dom.cjs
```

The committed 35 cases exercise the actual orchestration, prompt construction,
source policy and claim gate with a canned local completion client. They include
override/role/citation poisoning, Unicode obfuscation, source-path injection,
secret canaries, wrong/extra citations, uncited recommendations, unsupported
numbers, negation/status inversion, temporal/compatibility errors and conflicts.
Positive controls test legitimate historical security reports, defensive notices,
redacted placeholders, equivalent units and explicit version selection.

Known suspicious instruction/credential patterns are rejected before embedding
publication. Durable jobs quarantine those records without transmitting them to
the embedding provider. Previously stored poisoned sources are also excluded at
query time. A narrowly recognized historical “the note contained ignore previous
instructions” description is retained as a report, with the instruction phrase
removed from model evidence and the transformation disclosed. It is not a
blanket exemption for quoted instructions.

CI fails on a case failure and preserves the report. The report fixes its policy
clock to 10 November 2026 for reproducible date cases and records
`real_model_attack_resistance_measured: false`. The suite does not benchmark a
neural model's jailbreak resistance or establish a general attack success rate.
The [measured 1.3 regression report](../reports/evaluation/adversarial-1.3.json)
is committed with the cases and reproducible evaluator.

Additional PostgreSQL tests prove that unauthorized conflicting documents never
reach the policy or public evidence. Other tests cover quarantine, nested
redaction, canary-free failure logs, conflict responses and client-clock bypass.

## Verification and limits

API, orchestration, database and renderer-unit checks are automated. The renderer
unit test uses a fake DOM to verify text insertion, disclosure links and reset
behavior; it is not a live-browser or visual QA result. In this session both
browser automation paths were unavailable/stopped by their safety checks, so
visual verification is explicitly unclaimed. Open the local demo to review it.

The original grounding and the new source-binding checks remain lexical
heuristics. They cannot prove entailment, causal relevance, document authenticity
or that all malicious instructions have been detected. JSON quoting, eligibility
checks and claim gates are layered controls, not a complete security boundary.
Real models, real operational data, enterprise policy and high-impact advice
still require the external acceptance described in the 1.2 completion record.
