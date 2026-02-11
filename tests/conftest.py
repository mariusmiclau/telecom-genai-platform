"""Pytest fixtures for Telecom GenAI Platform tests."""

import asyncio
import os
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

# Sample telecom documents for testing
SAMPLE_NETWORK_DOC = """
Network Operations Manual - Region A

1. Overview
The Region A network consists of 5 core routers, 12 distribution switches,
and 48 access switches serving approximately 10,000 enterprise customers.

2. Performance Metrics
- Target Availability: 99.95%
- Maximum Latency: 30ms
- Error Rate Threshold: 1%

3. Escalation Procedures
Critical incidents (P1) must be escalated within 15 minutes.
High priority incidents (P2) require escalation within 1 hour.
"""

SAMPLE_CONFIG_DOC = """
Router Configuration Standards

Security Requirements:
- All management access must use SSH (no telnet)
- Passwords must be encrypted (type 5 or 7)
- NTP synchronization is mandatory

QoS Policy:
- Voice traffic: Priority queue
- Video traffic: Guaranteed bandwidth
- Data traffic: Best effort
"""

SAMPLE_SLA_DOC = """
Service Level Agreement - Enterprise Tier

Availability: 99.95% monthly uptime guarantee
Latency: Maximum 30ms round-trip time
Packet Loss: Less than 0.1%
Error Rate: Below 1% threshold

Breach Penalties:
- 10% credit for availability below 99.9%
- 25% credit for availability below 99.5%
"""


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_telecom_documents() -> list[dict]:
    """Provide sample telecom documents for testing."""
    return [
        {
            "content": SAMPLE_NETWORK_DOC,
            "metadata": {
                "source": "network_operations_manual.pdf",
                "doc_type": "manual",
                "region": "region_a",
            },
        },
        {
            "content": SAMPLE_CONFIG_DOC,
            "metadata": {
                "source": "config_standards.pdf",
                "doc_type": "standard",
                "category": "configuration",
            },
        },
        {
            "content": SAMPLE_SLA_DOC,
            "metadata": {
                "source": "enterprise_sla.pdf",
                "doc_type": "sla",
                "tier": "enterprise",
            },
        },
    ]


@pytest.fixture
def sample_chunks() -> list[dict]:
    """Provide pre-chunked document samples."""
    return [
        {
            "text": "The Region A network consists of 5 core routers, 12 distribution switches.",
            "metadata": {"source": "network_manual.pdf", "chunk_id": 1},
        },
        {
            "text": "Target Availability: 99.95%. Maximum Latency: 30ms.",
            "metadata": {"source": "network_manual.pdf", "chunk_id": 2},
        },
        {
            "text": "Critical incidents (P1) must be escalated within 15 minutes.",
            "metadata": {"source": "network_manual.pdf", "chunk_id": 3},
        },
        {
            "text": "All management access must use SSH. Passwords must be encrypted.",
            "metadata": {"source": "config_standards.pdf", "chunk_id": 4},
        },
        {
            "text": "Service Level Agreement guarantees 99.95% monthly uptime.",
            "metadata": {"source": "enterprise_sla.pdf", "chunk_id": 5},
        },
    ]


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        "choices": [
            {
                "message": {
                    "content": "Based on the network documentation, Region A has 5 core routers with 99.95% target availability.",
                    "role": "assistant",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 150,
            "completion_tokens": 30,
            "total_tokens": 180,
        },
    }


@pytest.fixture
def mock_embedding_response():
    """Mock embedding response for testing."""
    # Return a 1536-dimensional vector (OpenAI ada-002 size)
    return {"data": [{"embedding": [0.1] * 1536}]}


@pytest.fixture
def mock_openai_client(mock_llm_response, mock_embedding_response):
    """Create a mock OpenAI client."""
    mock_client = MagicMock()

    # Mock chat completions
    mock_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content=mock_llm_response["choices"][0]["message"]["content"],
                        role="assistant",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=MagicMock(
                prompt_tokens=150,
                completion_tokens=30,
                total_tokens=180,
            ),
        )
    )

    # Mock embeddings
    mock_client.embeddings.create = AsyncMock(
        return_value=MagicMock(
            data=[MagicMock(embedding=[0.1] * 1536)]
        )
    )

    return mock_client


@pytest.fixture
def mock_chromadb_collection():
    """Create a mock ChromaDB collection."""
    mock_collection = MagicMock()

    mock_collection.query = MagicMock(
        return_value={
            "ids": [["doc1", "doc2", "doc3"]],
            "documents": [[
                "Region A has 5 core routers.",
                "Target availability is 99.95%.",
                "Critical incidents require escalation within 15 minutes.",
            ]],
            "metadatas": [[
                {"source": "network_manual.pdf"},
                {"source": "network_manual.pdf"},
                {"source": "network_manual.pdf"},
            ]],
            "distances": [[0.1, 0.2, 0.3]],
        }
    )

    mock_collection.add = MagicMock()
    mock_collection.count = MagicMock(return_value=100)

    return mock_collection


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client for caching tests."""
    mock_redis = MagicMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock(return_value=True)
    mock_redis.exists = AsyncMock(return_value=False)
    return mock_redis


@pytest.fixture
def mock_postgres_connection():
    """Create a mock PostgreSQL connection for memory tests."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    mock_cursor.fetchall = MagicMock(return_value=[])
    mock_cursor.fetchone = MagicMock(return_value=None)
    mock_cursor.execute = MagicMock()

    mock_conn.cursor = MagicMock(return_value=mock_cursor)
    mock_conn.commit = MagicMock()

    return mock_conn


@pytest.fixture
def test_session_id() -> str:
    """Provide a test session ID."""
    return "test_session_12345"


@pytest.fixture
def sample_chat_request() -> dict:
    """Provide a sample chat request."""
    return {
        "message": "What is the network status for Region A?",
        "session_id": "test_session_12345",
        "context": {"region": "A", "priority": "normal"},
    }


@pytest.fixture
def sample_ingest_request() -> dict:
    """Provide a sample ingest request."""
    return {
        "source_path": "/data/documents/network_manual.pdf",
        "doc_type": "pdf",
        "metadata": {"category": "network", "region": "region_a"},
    }


@pytest.fixture
def mock_tool_results():
    """Provide mock tool execution results."""
    return {
        "network_health": {
            "data": {
                "region": "region_a",
                "status": "healthy",
                "elements": [
                    {"id": "rtr-001", "status": "up", "cpu": 45, "memory": 60},
                    {"id": "rtr-002", "status": "up", "cpu": 38, "memory": 55},
                ],
            },
            "summary": "Region A: 2/2 elements healthy",
        },
        "sla_monitor": {
            "data": {
                "region": "region_a",
                "compliant": 3,
                "non_compliant": 0,
                "services": [
                    {
                        "service_id": "svc-001",
                        "is_compliant": True,
                        "current_availability": 99.98,
                    }
                ],
            },
            "summary": "Region A: 3/3 services compliant",
        },
        "ticket_creator": {
            "data": {
                "ticket_id": "INC12345678",
                "status": "new",
                "priority": "high",
                "assignee": "noc-region_a@telecom.net",
            },
            "summary": "Created incident ticket INC12345678",
        },
    }


@pytest.fixture
def environment_variables():
    """Set up test environment variables."""
    test_env = {
        "OPENAI_API_KEY": "test-api-key",
        "DATABASE_URL": "postgresql://test:test@localhost/test",
        "REDIS_URL": "redis://localhost:6379/0",
        "CHROMADB_HOST": "localhost",
        "CHROMADB_PORT": "8000",
        "LOG_LEVEL": "DEBUG",
    }
    with patch.dict(os.environ, test_env):
        yield test_env


# Async fixtures
@pytest.fixture
async def async_mock_openai():
    """Async fixture for mocking OpenAI calls."""
    with patch("openai.AsyncOpenAI") as mock:
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(content="Test response", role="assistant"),
                        finish_reason="stop",
                    )
                ]
            )
        )
        mock.return_value = client
        yield client
