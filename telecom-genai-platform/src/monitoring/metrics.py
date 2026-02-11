"""Prometheus metrics for observability.

Tracks key GenAI platform health indicators:
- Request latency and throughput
- Token usage and costs
- Hallucination/confidence scores
- Tool call success rates
- RAG retrieval quality
"""

from prometheus_client import Counter, Histogram, Gauge, Summary


# Request metrics
REQUEST_COUNT = Counter(
    "genai_requests_total",
    "Total API requests",
    ["endpoint", "method"],
)

REQUEST_LATENCY = Histogram(
    "genai_request_latency_seconds",
    "Request processing latency",
    ["endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

# LLM metrics
TOKEN_USAGE = Counter(
    "genai_tokens_total",
    "Total tokens consumed",
    ["type"],  # prompt, completion
)

LLM_LATENCY = Histogram(
    "genai_llm_latency_seconds",
    "LLM inference latency",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

# Quality metrics
HALLUCINATION_SCORE = Histogram(
    "genai_confidence_score",
    "Response confidence/grounding score distribution",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

GROUNDING_FAILURES = Counter(
    "genai_grounding_failures_total",
    "Responses that failed grounding threshold",
)

# Tool metrics
TOOL_CALLS = Counter(
    "genai_tool_calls_total",
    "Tool invocations",
    ["tool_name", "status"],  # status: success, error, timeout
)

TOOL_LATENCY = Histogram(
    "genai_tool_latency_seconds",
    "Tool execution latency",
    ["tool_name"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# RAG metrics
RAG_RETRIEVAL_COUNT = Histogram(
    "genai_rag_documents_retrieved",
    "Number of documents retrieved per query",
    buckets=[1, 2, 3, 5, 8, 10, 15, 20],
)

RAG_RELEVANCE_SCORE = Histogram(
    "genai_rag_relevance_score",
    "Average relevance score of retrieved documents",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# System metrics
ACTIVE_SESSIONS = Gauge(
    "genai_active_sessions",
    "Currently active conversation sessions",
)

KNOWLEDGE_BASE_SIZE = Gauge(
    "genai_knowledge_base_chunks",
    "Total chunks in vector database",
)


def startup_metrics():
    """Initialize metrics on application startup."""
    ACTIVE_SESSIONS.set(0)


def track_token_usage(tokens: int, prompt_tokens: int = 0, completion_tokens: int = 0):
    """Track token consumption."""
    if prompt_tokens:
        TOKEN_USAGE.labels(type="prompt").inc(prompt_tokens)
    if completion_tokens:
        TOKEN_USAGE.labels(type="completion").inc(completion_tokens)
    if tokens and not (prompt_tokens or completion_tokens):
        TOKEN_USAGE.labels(type="total").inc(tokens)
