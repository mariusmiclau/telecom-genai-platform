# Retrieval evaluation methodology

## Goal

Measure retrieval changes before they are merged. The evaluator answers one question: does the expected source ticket appear in the ranked results, and at what position?

The corpus and all 50 questions are synthetic so they can be published safely. They model realistic NOC identifiers, interfaces, acronyms, symptoms, causes, and remediations, but the results must not be presented as production performance.

## Golden record schema

Each JSONL row in `evaluation/golden_questions.jsonl` contains:

- `question_id`
- `question`
- `expected_ticket_ids`
- `expected_sources`
- `reference_answer`
- `tags` (`entity_id`, `acronym`, or `semantic`)

Validation requires exactly 50 unique questions, existing ticket/source references, and all required fields.

## Configurations

### A — Recursive chunks + vector search

- Generic 240-character chunks with 40-character overlap.
- 384-dimensional deterministic TF-IDF feature-hashing embeddings.
- Cosine ranking, rolled up from chunk to ticket.

### B — Ticket-aware chunks + hybrid retrieval

- Three field groups per ticket: identity, incident, and resolution.
- The ticket ID and network identity are repeated across related groups.
- Vector results and Okapi BM25 results are combined using weighted reciprocal-rank fusion.

### C — Ticket-aware chunks + hybrid retrieval + reranker

- The same candidate generator as B.
- A deterministic reranker scores full-ticket token coverage, exact entity matches, exact ticket IDs, and field matches.
- This component is intentionally replaceable by a cross-encoder without changing the evaluator contract.

## Metrics

- `Recall@k`: fraction of questions whose expected ticket appears in the first k results.
- `Source Recall@5`: fraction whose expected source appears in the first five results.
- `MRR`: mean reciprocal rank of the first expected ticket.
- `nDCG@5`: position-sensitive quality for the single relevant ticket in each question.
- `Evidence coverage@5`: fraction of normalized reference-answer terms present in the first five retrieved tickets.
- p50/p95 latency: local retrieval latency, reported for context and not used as a cross-machine quality claim.

## Measured baseline

| Configuration | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| A | 64% | 78% | 80% | 0.690 | 0.710 |
| B | 74% | 86% | 96% | 0.796 | 0.802 |
| C | 100% | 100% | 100% | 1.000 | 1.000 |

The complete per-question output is committed in `reports/evaluation/baseline.json`; the human-readable table is in `reports/evaluation/baseline.md`.

## Interpretation limits

The hashing tokenizer deliberately removes numeric identifiers and some short
acronyms in its semantic mode. This is a constructed baseline, not a measured
property of OpenAI or SentenceTransformers embeddings. A versus B changes both
chunking and retrieval, so improvements cannot be attributed to either factor
alone. The 50 questions are not a held-out estimate of performance on unseen
carrier data. Perfect C retrieval does not imply perfect answer generation.

Live and offline paths share BM25 and field reranking policy. Candidate pools,
embedding providers and collection storage still differ. Real PostgreSQL integration
tests exercise the live implementation; a credential-backed, SME-reviewed model
evaluation remains necessary for external production acceptance.

## Reproduction and CI

```bash
python scripts/build_dataset_versions.py
python -m src.evaluation.cli \
  --output reports/evaluation/latest.json \
  --compare-to reports/evaluation/baseline.json \
  --max-recall-drop 0.0
```

CI runs the second command and fails if any configuration's Recall@5 drops below the committed baseline. Update the baseline only after reviewing per-question gains and regressions.
