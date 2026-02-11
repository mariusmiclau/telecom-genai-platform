"""Agent behavior tests."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.orchestrator import AgentOrchestrator
from src.agents.memory import ConversationMemory
from src.agents.reasoning import ReasoningEngine


class TestToolSelectionNetworkQuery:
    """Tests for tool selection on network queries."""

    @pytest.fixture
    def orchestrator(self):
        """Create an AgentOrchestrator instance."""
        with patch("src.agents.orchestrator.AsyncOpenAI"):
            return AgentOrchestrator()

    @pytest.mark.asyncio
    async def test_tool_selection_network_query(self, orchestrator):
        """Test that network queries select network_health tool."""
        message = "What is the current network status for Region A?"

        # Mock the reasoning engine
        orchestrator.reasoning = MagicMock()
        orchestrator.reasoning.classify_intent = AsyncMock(
            return_value=MagicMock(
                intent_type="diagnostic",
                confidence=0.95,
                suggested_tools=["network_health"],
            )
        )

        intent = await orchestrator.reasoning.classify_intent(message)

        assert "network_health" in intent.suggested_tools
        assert intent.intent_type == "diagnostic"

    @pytest.mark.asyncio
    async def test_network_health_check_selected(self, orchestrator):
        """Test network health check is selected for status queries."""
        queries = [
            "Check the health of routers in Region B",
            "Are there any network issues?",
            "Show me the current network metrics",
            "What's the status of switch sw-001?",
        ]

        orchestrator.reasoning = MagicMock()
        orchestrator.reasoning.classify_intent = AsyncMock(
            return_value=MagicMock(
                intent_type="diagnostic",
                suggested_tools=["network_health"],
            )
        )

        for query in queries:
            intent = await orchestrator.reasoning.classify_intent(query)
            assert "network_health" in intent.suggested_tools

    @pytest.mark.asyncio
    async def test_sla_tool_for_compliance_query(self, orchestrator):
        """Test SLA tool is selected for compliance queries."""
        message = "Are we meeting our SLA targets for Region A?"

        orchestrator.reasoning = MagicMock()
        orchestrator.reasoning.classify_intent = AsyncMock(
            return_value=MagicMock(
                intent_type="diagnostic",
                suggested_tools=["sla_monitor"],
            )
        )

        intent = await orchestrator.reasoning.classify_intent(message)

        assert "sla_monitor" in intent.suggested_tools


class TestToolSelectionTicketCreation:
    """Tests for tool selection on ticket creation."""

    @pytest.fixture
    def orchestrator(self):
        """Create an AgentOrchestrator instance."""
        with patch("src.agents.orchestrator.AsyncOpenAI"):
            return AgentOrchestrator()

    @pytest.mark.asyncio
    async def test_tool_selection_ticket_creation(self, orchestrator):
        """Test that incident reports select ticket_creator tool."""
        message = "Create an incident ticket for the router outage in Region A"

        orchestrator.reasoning = MagicMock()
        orchestrator.reasoning.classify_intent = AsyncMock(
            return_value=MagicMock(
                intent_type="action",
                confidence=0.92,
                suggested_tools=["ticket_creator"],
            )
        )

        intent = await orchestrator.reasoning.classify_intent(message)

        assert "ticket_creator" in intent.suggested_tools
        assert intent.intent_type == "action"

    @pytest.mark.asyncio
    async def test_ticket_query_tool_for_status(self, orchestrator):
        """Test ticket_query is selected for ticket status checks."""
        message = "What's the status of ticket INC12345?"

        orchestrator.reasoning = MagicMock()
        orchestrator.reasoning.classify_intent = AsyncMock(
            return_value=MagicMock(
                intent_type="query",
                suggested_tools=["ticket_query"],
            )
        )

        intent = await orchestrator.reasoning.classify_intent(message)

        assert "ticket_query" in intent.suggested_tools


class TestMemoryPersistence:
    """Tests for conversation memory persistence."""

    @pytest.fixture
    def memory(self, mock_postgres_connection):
        """Create a ConversationMemory with mocked database."""
        with patch("psycopg2.connect", return_value=mock_postgres_connection):
            return ConversationMemory()

    @pytest.mark.asyncio
    async def test_memory_persistence(self, memory, test_session_id):
        """Test that conversation history is persisted."""
        # Mock save method
        memory.save = AsyncMock()

        await memory.save(
            session_id=test_session_id,
            user_message="What is the network status?",
            assistant_response="The network is healthy with 99.95% uptime.",
            sources=["network_doc.pdf"],
        )

        memory.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_memory_retrieval(self, memory, test_session_id):
        """Test that conversation history can be retrieved."""
        memory.get_history = AsyncMock(
            return_value=MagicMock(
                messages=[
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ],
                session_id=test_session_id,
            )
        )

        history = await memory.get_history(test_session_id)

        assert len(history.messages) == 2
        assert history.session_id == test_session_id

    @pytest.mark.asyncio
    async def test_memory_max_messages_limit(self, memory, test_session_id):
        """Test that memory respects max messages limit."""
        memory.get_history = AsyncMock(
            return_value=MagicMock(
                messages=[{"role": "user", "content": f"Message {i}"} for i in range(10)],
                session_id=test_session_id,
            )
        )

        history = await memory.get_history(test_session_id, max_messages=10)

        assert len(history.messages) <= 10

    @pytest.mark.asyncio
    async def test_new_session_empty_history(self, memory):
        """Test that new sessions have empty history."""
        memory.get_history = AsyncMock(
            return_value=MagicMock(messages=[], session_id="new_session")
        )

        history = await memory.get_history("new_session")

        assert len(history.messages) == 0


class TestReasoningChainCompletion:
    """Tests for reasoning chain completion."""

    @pytest.fixture
    def reasoning_engine(self, mock_openai_client):
        """Create a ReasoningEngine with mocked client."""
        with patch("src.agents.reasoning.AsyncOpenAI", return_value=mock_openai_client):
            engine = ReasoningEngine()
            engine.client = mock_openai_client
            return engine

    @pytest.mark.asyncio
    async def test_reasoning_chain_completion(self, reasoning_engine):
        """Test that reasoning chain completes successfully."""
        message = "Diagnose network latency issues in Region A"

        reasoning_engine.classify_intent = AsyncMock(
            return_value=MagicMock(
                intent_type="diagnostic",
                confidence=0.9,
                reasoning_steps=[
                    "User is asking about latency issues",
                    "Need to check network health metrics",
                    "May need to examine specific routers",
                ],
                suggested_tools=["network_health", "sla_monitor"],
            )
        )

        intent = await reasoning_engine.classify_intent(message)

        assert intent.reasoning_steps is not None
        assert len(intent.reasoning_steps) > 0
        assert intent.confidence > 0

    @pytest.mark.asyncio
    async def test_multi_step_reasoning(self, reasoning_engine):
        """Test multi-step reasoning for complex queries."""
        message = "The network is slow and customers are complaining. Create a ticket and check SLA status."

        reasoning_engine.classify_intent = AsyncMock(
            return_value=MagicMock(
                intent_type="action",
                suggested_tools=["network_health", "sla_monitor", "ticket_creator"],
                reasoning_steps=[
                    "Multiple issues mentioned",
                    "Need diagnostic tools first",
                    "Then create ticket for tracking",
                ],
            )
        )

        intent = await reasoning_engine.classify_intent(message)

        # Should suggest multiple tools for complex request
        assert len(intent.suggested_tools) >= 2


class TestMaxIterationsLimit:
    """Tests for max iterations limit."""

    @pytest.fixture
    def orchestrator(self):
        """Create an AgentOrchestrator instance."""
        with patch("src.agents.orchestrator.AsyncOpenAI"):
            orch = AgentOrchestrator()
            orch.max_iterations = 5
            return orch

    @pytest.mark.asyncio
    async def test_max_iterations_limit(self, orchestrator):
        """Test that processing stops at max iterations."""
        orchestrator.process = AsyncMock(
            return_value=MagicMock(
                response="Processing completed",
                iterations=5,
                stopped_reason="max_iterations",
            )
        )

        result = await orchestrator.process(
            message="Complex query requiring many steps",
            session_id="test_session",
        )

        assert result.iterations <= orchestrator.max_iterations

    @pytest.mark.asyncio
    async def test_early_completion_before_limit(self, orchestrator):
        """Test that processing can complete before max iterations."""
        orchestrator.process = AsyncMock(
            return_value=MagicMock(
                response="Quick answer",
                iterations=2,
                stopped_reason="completed",
            )
        )

        result = await orchestrator.process(
            message="Simple query",
            session_id="test_session",
        )

        assert result.iterations < orchestrator.max_iterations
        assert result.stopped_reason == "completed"


class TestHumanEscalationTrigger:
    """Tests for human escalation triggers."""

    @pytest.fixture
    def orchestrator(self):
        """Create an AgentOrchestrator instance."""
        with patch("src.agents.orchestrator.AsyncOpenAI"):
            return AgentOrchestrator()

    @pytest.mark.asyncio
    async def test_human_escalation_trigger(self, orchestrator):
        """Test that low confidence triggers human escalation."""
        orchestrator.process = AsyncMock(
            return_value=MagicMock(
                response="I'm not confident about this answer.",
                confidence=0.3,
                requires_escalation=True,
                escalation_reason="low_confidence",
            )
        )

        result = await orchestrator.process(
            message="Complex question about undocumented feature",
            session_id="test_session",
        )

        assert result.requires_escalation is True
        assert result.confidence < 0.5

    @pytest.mark.asyncio
    async def test_escalation_on_sensitive_action(self, orchestrator):
        """Test escalation for sensitive actions."""
        orchestrator.process = AsyncMock(
            return_value=MagicMock(
                response="This action requires human approval.",
                requires_escalation=True,
                escalation_reason="sensitive_action",
            )
        )

        result = await orchestrator.process(
            message="Delete all configurations for Region A",
            session_id="test_session",
        )

        assert result.requires_escalation is True
        assert result.escalation_reason == "sensitive_action"

    @pytest.mark.asyncio
    async def test_escalation_on_repeated_failures(self, orchestrator):
        """Test escalation after repeated tool failures."""
        orchestrator.process = AsyncMock(
            return_value=MagicMock(
                response="Unable to complete the request after multiple attempts.",
                requires_escalation=True,
                escalation_reason="repeated_failures",
                failed_attempts=3,
            )
        )

        result = await orchestrator.process(
            message="Check status of offline system",
            session_id="test_session",
        )

        assert result.requires_escalation is True
        assert result.escalation_reason == "repeated_failures"

    @pytest.mark.asyncio
    async def test_no_escalation_for_confident_response(self, orchestrator):
        """Test no escalation for high-confidence responses."""
        orchestrator.process = AsyncMock(
            return_value=MagicMock(
                response="The network status is healthy.",
                confidence=0.95,
                requires_escalation=False,
            )
        )

        result = await orchestrator.process(
            message="What is the network status?",
            session_id="test_session",
        )

        assert result.requires_escalation is False
        assert result.confidence > 0.8


class TestIntentClassification:
    """Tests for intent classification."""

    @pytest.fixture
    def reasoning_engine(self, mock_openai_client):
        """Create a ReasoningEngine instance."""
        with patch("src.agents.reasoning.AsyncOpenAI", return_value=mock_openai_client):
            return ReasoningEngine()

    @pytest.mark.asyncio
    async def test_diagnostic_intent(self, reasoning_engine):
        """Test classification of diagnostic intent."""
        reasoning_engine.classify_intent = AsyncMock(
            return_value=MagicMock(intent_type="diagnostic")
        )

        intent = await reasoning_engine.classify_intent("Why is the network slow?")

        assert intent.intent_type == "diagnostic"

    @pytest.mark.asyncio
    async def test_action_intent(self, reasoning_engine):
        """Test classification of action intent."""
        reasoning_engine.classify_intent = AsyncMock(
            return_value=MagicMock(intent_type="action")
        )

        intent = await reasoning_engine.classify_intent("Create a ticket for this issue")

        assert intent.intent_type == "action"

    @pytest.mark.asyncio
    async def test_query_intent(self, reasoning_engine):
        """Test classification of query intent."""
        reasoning_engine.classify_intent = AsyncMock(
            return_value=MagicMock(intent_type="query")
        )

        intent = await reasoning_engine.classify_intent("What does the SLA say about uptime?")

        assert intent.intent_type == "query"

    @pytest.mark.asyncio
    async def test_escalation_intent(self, reasoning_engine):
        """Test classification of escalation intent."""
        reasoning_engine.classify_intent = AsyncMock(
            return_value=MagicMock(intent_type="escalation")
        )

        intent = await reasoning_engine.classify_intent(
            "I need to speak with a human operator"
        )

        assert intent.intent_type == "escalation"
