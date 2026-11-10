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
from typing import Literal, Optional

from src.agents.memory import ConversationMemory
from src.agents.reasoning import ReasoningEngine
from src.evidence import (
    ClaimLink,
    EvidenceConflict,
    EvidenceItem,
    SafetyNotice,
    link_claims,
    select_evidence,
)
from src.guardrails.hallucination import HallucinationDetector
from src.monitoring.tracing import content_attributes, trace_span
from src.rag_pipeline.chain import RAGChain
from src.rag_pipeline.retriever import HybridRetriever
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
    grounding_score: float = 0.0
    tools_called: list[ToolCall] = field(default_factory=list)
    tokens_used: int = 0
    reasoning_steps: list[str] = field(default_factory=list)
    outcome: Literal["answered", "abstained", "escalated"] = "answered"
    evidence: list[EvidenceItem] = field(default_factory=list)
    claims: list[ClaimLink] = field(default_factory=list)
    warnings: list[SafetyNotice] = field(default_factory=list)
    conflicts: list[EvidenceConflict] = field(default_factory=list)
    answer_kind: Literal["historical_resolution", "advisory", "insufficient_evidence", "conflict"] = "advisory"


class AgentOrchestrator:
    """Main agent orchestrator for telecom GenAI workflows.

    Architecture:
    - Uses ReAct-style reasoning (Reason + Act)
    - Tool-calling for real-time network data
    - RAG for knowledge base queries
    - Conversation memory for multi-turn context
    """

    def __init__(self, retriever=None, rag_chain=None, memory=None, enable_tools=True, evidence_clock=None, source_review_max_age_days=365):
        self.retriever = retriever or HybridRetriever()
        self.rag_chain = rag_chain or RAGChain()
        self.memory = memory or ConversationMemory()
        self.enable_tools = enable_tools
        self.evidence_clock = evidence_clock
        self.source_review_max_age_days = source_review_max_age_days
        self.reasoning = ReasoningEngine()
        self.tools = ToolRegistry()
        self.grounding_detector = HallucinationDetector()

        # Register telecom-specific tools
        if enable_tools:
            self._register_tools()

    def _register_tools(self):
        """Register available tools for agent use."""
        from src.tools.config import ConfigValidatorTool
        from src.tools.network import NetworkConfigTool, NetworkHealthTool
        from src.tools.sla import SLAMonitorTool
        from src.tools.ticketing import TicketCreatorTool

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
        """Process and trace a user message through the complete agent pipeline."""
        session_id = session_id or str(uuid.uuid4())
        attributes = {
            **content_attributes(message),
            "session.id": session_id,
            "telecom.request.context_field_count": len(context or {}),
        }
        with trace_span("query", attributes) as span:
            result = await self._process(message, session_id, context)
            span.set_attribute("genai.response.confidence", result.confidence)
            span.set_attribute("genai.response.grounding_score", result.grounding_score)
            span.set_attribute("genai.response.source_count", len(result.sources))
            return result

    async def _process(
        self,
        message: str,
        session_id: str,
        context: Optional[dict] = None,
    ) -> AgentResult:
        """Execute the agent pipeline inside the request's root query span.

        Args:
            message: User's input message
            session_id: Session identifier for conversation memory
            context: Additional context (region, priority, etc.)

        Returns:
            AgentResult with response, sources, confidence, and metadata
        """
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

        if intent.requires_tools and not self.enable_tools:
            return AgentResult(
                response="This request requires an operational integration. Contact the network operations team; live actions are disabled in this release.",
                sources=[], confidence=0.0, reasoning_steps=["escalate: operational action"], outcome="escalated",
            )

        # Step 3: Retrieve relevant context from RAG
        rag_context = await self.retriever.search(
            query=message,
            top_k=5,
            filters=context,
        )
        reasoning_steps.append(f"Retrieved {len(rag_context.documents)} relevant documents")

        if not rag_context.documents:
            return AgentResult(
                response="No supporting evidence was found. Please refine the question or escalate to a network engineer.",
                sources=[], confidence=0.0, reasoning_steps=["abstain: no evidence"], outcome="abstained", answer_kind="insufficient_evidence",
            )

        with trace_span("evidence_policy") as policy_span:
            selection = select_evidence(rag_context.documents, context,
                now=self.evidence_clock() if self.evidence_clock else None, max_review_age_days=self.source_review_max_age_days)
            policy_span.set_attribute("evidence.eligible_count", len(selection.documents))
            policy_span.set_attribute("evidence.conflict_count", len(selection.conflicts))
        ambiguous = any(warning.code == "compatibility_ambiguous" for warning in selection.warnings)
        if selection.conflicts or ambiguous or not selection.documents:
            return AgentResult(
                response="Sources disagree on current facts. Review the conflicting evidence with a network engineer." if selection.conflicts else "Sources apply to different versions or equipment families. Specify the intended compatibility context." if ambiguous else "No eligible source remains after source safety, freshness and compatibility checks.",
                sources=[], confidence=0.0, outcome="escalated" if selection.conflicts or ambiguous else "abstained",
                answer_kind="conflict" if selection.conflicts else "insufficient_evidence",
                evidence=selection.evidence, warnings=selection.warnings, conflicts=selection.conflicts,
            )
        rag_context.documents = selection.documents

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

        if response.finish_reason == "error":
            raise RuntimeError("Model provider unavailable")

        # Step 6: Check grounding and calculate confidence
        source_evidence = [
            {
                "content": document.content,
                "source": document.source_path,
                "metadata": document.metadata,
            }
            for document in rag_context.documents
        ]
        with trace_span("grounding_check") as span:
            grounding = self.grounding_detector.check_grounding(
                response=response.text,
                source_documents=source_evidence,
                tool_results=[tool.output for tool in tools_called],
            )
            span.set_attribute("genai.grounding.is_grounded", grounding.is_grounded)
            span.set_attribute("genai.grounding.score", grounding.grounding_score)
            span.set_attribute("genai.grounding.ungrounded_claims", len(grounding.ungrounded_claims))

        retrieval_confidence = self._calculate_confidence(
            response=response,
            sources=rag_context.documents,
            tool_results=tools_called,
        )
        confidence = min(retrieval_confidence, grounding.grounding_score)
        claims, fully_linked = link_claims(response.text, selection.evidence)
        if not grounding.is_grounded or not response.sources or response.finish_reason != "stop" or not fully_linked:
            response.text = "The generated answer could not be verified against the retrieved evidence. Please escalate to a network engineer."
            response.sources = []
            confidence = 0.0
            claims = []
            selection.warnings.append(SafetyNotice(code="unsupported_claim", message="The generated answer could not be matched to its cited evidence; it was not published."))
            reasoning_steps.append("abstain: grounding, citation or completion check failed")
        reasoning_steps.append(f"Confidence score: {confidence:.2f}")

        # Step 7: Save to conversation memory
        await self.memory.save(
            session_id=session_id,
            user_message=message,
            assistant_response=response.text,
            sources=[s.document_id for s in rag_context.documents],
        )

        with trace_span(
            "final_response",
            {
                "genai.response.characters": len(response.text),
                "genai.response.confidence": confidence,
                "genai.response.finish_reason": response.finish_reason,
            },
        ):
            return AgentResult(
                response=response.text,
                sources=response.sources,
                confidence=confidence,
                grounding_score=grounding.grounding_score,
                tools_called=tools_called,
                tokens_used=response.tokens_used,
                reasoning_steps=reasoning_steps,
                outcome="answered" if response.sources else "abstained",
                answer_kind=("historical_resolution" if all(item.freshness == "historical" for item in selection.evidence) else "advisory") if response.sources else "insufficient_evidence",
                evidence=selection.evidence, claims=claims, warnings=selection.warnings,
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
