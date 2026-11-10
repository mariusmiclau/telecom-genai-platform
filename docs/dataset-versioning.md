# Ticket dataset versioning

Dataset releases are immutable directories under `datasets/`. Each release contains `tickets.jsonl` and `manifest.json`.

```text
tickets-v1.0
  -> tickets-v1.1-cleaned
      -> tickets-v1.2-pii-redacted
          -> tickets-v2.0-ticket-aware-chunking
```

## Release contract

Every manifest records:

- source snapshot and parent version;
- snapshot path and SHA-256;
- cleaning rules;
- chunking configuration;
- embedding model and dimensions;
- document and rejected-record counts;
- creation timestamp;
- evaluation status, configuration metrics, and report path.

`src.evaluation.dataset.validate_repository_data` verifies hashes, counts, schemas, unique IDs, golden references, and the exact 50-question contract.

## Transformations

- `tickets-v1.0`: immutable synthetic ServiceNow-style export with test-only contact data.
- `tickets-v1.1-cleaned`: collapses whitespace and normalizes status/severity values.
- `tickets-v1.2-pii-redacted`: redacts reporter email, contact phone, and management IP.
- `tickets-v2.0-ticket-aware-chunking`: adds the canonical field-aware ticket text and manifest configuration used by the evaluator and ingester.

## Build and validate

```bash
python scripts/build_dataset_versions.py
python -m src.evaluation.cli --output reports/evaluation/latest.json
```

The builder is deterministic. Source data is synthetic and identified as such in every manifest; substitute an approved, access-controlled export in production.
