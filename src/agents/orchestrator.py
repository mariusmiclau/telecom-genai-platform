"""Agent Orchestrator - Multi-step reasoning with tool-calling.

This orchestrator manages the agentic workflow:
1. Receives user query
2. Retrieves relevant context from RAG pipeline
3. Determines if tools need to be called
4. Executes tool calls (network checks, ticket creation, etc.)
5. Synthesizes response with source attribution
6. Scores confidence based on source grounding
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

from src.rag_pipeline.retriever import HybridRetriever
from src.rag_pipeline.chain import RAGChain
from src.agents.memory import ConversationMemory
from src.agents.reasoning import ReasoningEngine
from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Record of a tool invocation."""

    name: str
    input_params: dict
    output: dict
    latency_ms: float


@dataclass
class AgentResult:
    """Result from agent processing."""

    response: str
    sources: list
    confidence: float
    tools_called: list[ToolCall] = field(default_factory=list)
    tokens_used: int = 0
    reasoning_steps: list[str] = field(default_factory=list)


class AgentOrchestrator:
    """Main agent orchestrator for telecom GenAI workflows.

    Architecture:
    - Uses ReAct-style reasoning (Reason + Act)
    - Tool-calling for real-time network data
    - RAG for knowledge base queries
    - Conversation memory for multi-turn context
    """

    def __init__(self):
        self.retriever = HybridRetriever()
        self.rag_chain = RAGChain()
        self.memory = ConversationMemory()
        self.reasoning = ReasoningEngine()
        self.tools = ToolRegistry()

        # Register telecom-specific tools
        self._register_tools()

    def _register_tools(self):
        """Register available tools for agent use."""
        from src.tools.network import NetworkHealthTool, NetworkConfigTool
        from src.tools.ticketing import TicketCreatorTool
        from src.tools.sla import SLAMonitorTool
        from src.tools.config import ConfigValidatorTool

        self.tools.register(NetworkHealthTool())
        self.tools.register(NetworkConfigTool())
        self.tools.register(TicketCreatorTool())
        self.tools.register(SLAMonitorTool())
        self.tools.register(ConfigValidatorTool())

    async def process(
        self,
        message: str,
        session_id: Optional[str] = None,
        context: Optional[dict] = None,
    ) -> AgentResult:
        """Process a user message through the agent pipeline.

        Args:
            message: User's input message
            session_id: Session identifier for conversation memory
            context: Additional context (region, priority, etc.)

        Returns:
            AgentResult with response, sources, confidence, and metadata
        """
        session_id = session_id or str(uuid.uuid4())
        reasoning_steps = []
        tools_called = []

        logger.info(f"Processing message for session {session_id}")

        # Step 1: Load conversation history
        history = await self.memory.get_history(session_id)
        reasoning_steps.append("Loaded conversation history")

        # Step 2: Determine intent and required actions
        intent = await self.reasoning.classify_intent(
            message=message,
            history=history,
            available_tools=self.tools.list_tools(),
        )
        reasoning_steps.append(f"Intent classified: {intent.category}")

        # Step 3: Retrieve relevant context from RAG
        rag_context = await self.retriever.search(
            query=message,
            top_k=5,
            filters=context,
        )
        reasoning_steps.append(f"Retrieved {len(rag_context.documents)} relevant documents")

        # Step 4: Execute tool calls if needed
        if intent.requires_tools:
            for tool_request in intent.tool_calls:
                logger.info(f"Calling tool: {tool_request.tool_name}")
                tool_result = await self.tools.execute(
                    tool_name=tool_request.tool_name,
                    params=tool_request.params,
                )
                tools_called.append(
                    ToolCall(
                        name=tool_request.tool_name,
                        input_params=tool_request.params,
                        output=tool_result.data,
                        latency_ms=tool_result.latency_ms,
                    )
                )
                reasoning_steps.append(
                    f"Tool '{tool_request.tool_name}' returned: {tool_result.summary}"
                )

        # Step 5: Generate response using RAG chain
        response = await self.rag_chain.generate(
            query=message,
            context_docs=rag_context.documents,
            tool_results=[t.output for t in tools_called],
            conversation_history=history,
            system_context=self._build_system_context(context),
        )

        # Step 6: Calculate confidence based on source grounding
        confidence = self._calculate_confidence(
            response=response,
            sources=rag_context.documents,
            tool_results=tools_called,
        )
        reasoning_steps.append(f"Confidence score: {confidence:.2f}")

        # Step 7: Save to conversation memory
        await self.memory.save(
            session_id=session_id,
            user_message=message,
            assistant_response=response.text,
            sources=[s.document_id for s in rag_context.documents],
        )

        return AgentResult(
            response=response.text,
            sources=response.sources,
            confidence=confidence,
            tools_called=tools_called,
            tokens_used=response.tokens_used,
            reasoning_steps=reasoning_steps,
        )

    def _build_system_context(self, context: Optional[dict]) -> str:
        """Build system context prompt for telecom domain."""
        base_context = (
            "You are a telecom operations AI assistant. You help network engineers "
            "diagnose issues, check system status, and manage incidents. Always ground "
            "your responses in retrieved documents and live tool data. If you're unsure, "
            "say so and recommend human escalation. Never guess about network "
            "configurations or status. Use tool calls to get real-time data."
        )

        if context:
            if "region" in context:
                base_context += f"\nFocus region: {context['region']}"
            if "priority" in context:
                base_context += f"\nPriority level: {context['priority']}"

        return base_context

    def _calculate_confidence(
        self,
        response,
        sources: list,
        tool_results: list[ToolCall],
    ) -> float:
        """Calculate response confidence based on grounding.

        Confidence is higher when:
        - Response is well-grounded in source documents
        - Tool calls returned valid data
        - No contradictions between sources and tool data
        """
        if not sources and not tool_results:
            return 0.3  # Low confidence without grounding

        source_score = min(len(sources) / 3, 1.0) * 0.5
        tool_score = min(len(tool_results) / 2, 1.0) * 0.3

        # Base confidence from having relevant sources
        relevance_scores = [getattr(s, "relevance_score", 0.5) for s in sources]
        avg_relevance = sum(relevance_scores) / max(len(relevance_scores), 1)

        confidence = source_score + tool_score + (avg_relevance * 0.2)
        return min(confidence, 1.0)
