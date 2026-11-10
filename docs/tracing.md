# OpenTelemetry tracing

The API emits standard OpenTelemetry spans and exports them over OTLP/HTTP. Docker Compose runs Jaeger for local visualization at `http://localhost:16686`.

## Span tree

```text
POST /api/v1/chat
  query
    entity_extraction
    vector_retrieval
    keyword_retrieval
    rank_fusion
    reranker
    prompt_construction
    llm_call
    grounding_check
    final_response
```

Attributes include query length/hash, optional query text, extracted-entity count, candidate counts, prompt/source sizes, model parameters, token usage, grounding score, final confidence, and response size. Full prompts, retrieved ticket bodies, model responses, secrets, and tool payloads are not attached to spans.

## Local setup

```bash
docker compose up -d jaeger api
```

Open Jaeger, select the `telecom-genai-api` service, and search for the `POST /api/v1/chat` operation.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OTEL_TRACES_ENABLED` | `true` | Enable tracing helpers |
| `OTEL_TRACES_EXPORTER` | `none` in code, `otlp` in Compose | `none`, `console`, or `otlp` |
| `OTEL_SERVICE_NAME` | `telecom-genai-api` | Backend service name |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | `http://localhost:4318/v1/traces` | OTLP/HTTP endpoint |
| `OTEL_CAPTURE_CONTENT` | `false` | Opt in to raw query text only after privacy review |

Set `OTEL_CAPTURE_CONTENT=false` in production unless prompt capture has passed privacy and retention review. Query SHA-256 and length remain available for correlation without raw text.

OpenTelemetry failures never fail a user request. When packages or the backend are unavailable, manual span helpers safely become no-ops and application logs record the setup failure.
