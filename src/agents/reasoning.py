"""ReAct-style Reasoning Engine for intent classification and action planning.

Implements Reason + Act pattern for telecom operations:
1. Analyze user intent from message and conversation history
2. Determine if tools are needed for real-time data
3. Plan tool calls with appropriate parameters
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolCallRequest:
    """Request to call a specific tool."""

    tool_name: str
    params: dict
    reason: str  # Why this tool is being called


@dataclass
class Intent:
    """Classified intent with action plan."""

    category: str  # query, action, diagnostic, escalation
    confidence: float
    requires_tools: bool
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    reasoning: str = ""


class ReasoningEngine:
    """Classifies user intent and plans tool usage.

    Intent Categories:
    - query: Information retrieval from knowledge base
    - action: Request to perform an operation (create ticket, change config)
    - diagnostic: Troubleshooting that requires live data
    - escalation: Request that needs human intervention

    Tool Planning:
    - Analyzes message for entities (regions, node IDs, metrics)
    - Maps intent to required tools
    - Constructs tool call parameters from context
    """

    # Intent patterns for telecom domain
    INTENT_PATTERNS = {
        "diagnostic": [
            r"what is the (status|health|state) of",
            r"check (the )?(network|node|router|element)",
            r"(is|are) .* (down|degraded|operational|working)",
            r"troubleshoot",
            r"diagnose",
            r"why is .* (slow|failing|down)",
            r"current (status|metrics|performance)",
        ],
        "action": [
            r"create (a )?(ticket|incident|case)",
            r"change (the )?config",
            r"update (the )?(setting|parameter|config)",
            r"restart",
            r"reboot",
            r"apply (the )?change",
            r"schedule (a )?(maintenance|outage)",
        ],
        "escalation": [
            r"escalate",
            r"need (a )?human",
            r"speak to (an? )?(operator|engineer|person)",
            r"emergency",
            r"critical (issue|problem|outage)",
            r"urgent",
        ],
        "query": [
            r"what is",
            r"how (do|does|to|can)",
            r"explain",
            r"tell me about",
            r"describe",
            r"documentation",
            r"procedure for",
            r"best practice",
            r"policy for",
        ],
    }

    # Entity extraction patterns
    ENTITY_PATTERNS = {
        "region": r"region[_\s]?([a-z]|\d+)",
        "node_id": r"node[_\-]?(\d+)",
        "element_id": r"(cr|er)[_\-]([a-z])[_\-](\d+)",
        "interface": r"(eth|ge|xe|et)\d+(/\d+)*",
        "ip_address": r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
        "ticket_id": r"(INC|CHG|PRB)\d+",
    }

    # Tool mapping by intent
    TOOL_MAPPING = {
        "diagnostic": ["network_health_check", "network_config"],
        "action": ["ticket_creator", "config_validator"],
        "escalation": [],  # No tools, escalate to human
        "query": [],  # RAG only, no live tools
    }

    def __init__(self):
        self.confidence_threshold = float(
            os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.6")
        )

    async def classify_intent(
        self,
        message: str,
        history: Optional[object] = None,
        available_tools: Optional[list[dict]] = None,
    ) -> Intent:
        """Classify user intent and plan required actions.

        Args:
            message: User's input message
            history: Conversation history for context
            available_tools: List of available tool schemas

        Returns:
            Intent with category, confidence, and tool call plan
        """
        message_lower = message.lower()

        # Step 1: Pattern-based intent classification
        intent_scores = self._score_intents(message_lower)
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        category, confidence = best_intent

        # Step 2: Extract entities from message
        entities = self._extract_entities(message)

        # Step 3: Check conversation context
        if history and hasattr(history, 'messages') and history.messages:
            # Boost confidence if continuing a topic
            context_boost = self._analyze_context(history, category)
            confidence = min(confidence + context_boost, 1.0)

        # Step 4: Plan tool calls if needed
        requires_tools = category in ["diagnostic", "action"] and confidence > 0.5
        tool_calls = []

        if requires_tools:
            tool_calls = self._plan_tool_calls(
                category=category,
                message=message,
                entities=entities,
                available_tools=available_tools or [],
            )

        reasoning = self._generate_reasoning(
            message=message,
            category=category,
            entities=entities,
            tool_calls=tool_calls,
        )

        logger.info(
            f"Intent classified: {category} (confidence: {confidence:.2f})",
            extra={
                "entities": entities,
                "requires_tools": requires_tools,
                "tool_calls": [t.tool_name for t in tool_calls],
            },
        )

        return Intent(
            category=category,
            confidence=confidence,
            requires_tools=requires_tools,
            tool_calls=tool_calls,
            reasoning=reasoning,
        )

    def _score_intents(self, message: str) -> dict[str, float]:
        """Score message against intent patterns."""
        scores = {intent: 0.0 for intent in self.INTENT_PATTERNS}

        for intent, patterns in self.INTENT_PATTERNS.items():
            matches = 0
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    matches += 1

            if matches > 0:
                # Score based on pattern matches and pattern specificity
                scores[intent] = min(0.4 + (matches * 0.2), 0.95)

        # Default to query if no strong signal
        if max(scores.values()) < 0.3:
            scores["query"] = 0.5

        return scores

    def _extract_entities(self, message: str) -> dict[str, list[str]]:
        """Extract telecom entities from message."""
        entities = {}

        for entity_type, pattern in self.ENTITY_PATTERNS.items():
            matches = re.findall(pattern, message, re.IGNORECASE)
            if matches:
                # Flatten tuples from group matches
                flat_matches = []
                for m in matches:
                    if isinstance(m, tuple):
                        flat_matches.append("-".join(m))
                    else:
                        flat_matches.append(m)
                entities[entity_type] = flat_matches

        return entities

    def _analyze_context(self, history: object, current_intent: str) -> float:
        """Analyze conversation history for context boost."""
        if not hasattr(history, 'messages') or not history.messages:
            return 0.0

        # Check recent messages for same topic
        recent_messages = history.messages[-4:] if len(history.messages) > 4 else history.messages

        topic_keywords = {
            "diagnostic": ["status", "health", "check", "performance"],
            "action": ["ticket", "change", "update", "create"],
            "escalation": ["escalate", "human", "urgent"],
            "query": ["what", "how", "explain", "documentation"],
        }

        keywords = topic_keywords.get(current_intent, [])
        context_matches = 0

        for msg in recent_messages:
            content = msg.content.lower() if hasattr(msg, 'content') else str(msg).lower()
            for keyword in keywords:
                if keyword in content:
                    context_matches += 1
                    break

        return min(context_matches * 0.05, 0.15)

    def _plan_tool_calls(
        self,
        category: str,
        message: str,
        entities: dict,
        available_tools: list[dict],
    ) -> list[ToolCallRequest]:
        """Plan tool calls based on intent and entities."""
        tool_calls = []
        suggested_tools = self.TOOL_MAPPING.get(category, [])

        # Get available tool names
        available_names = {
            t.get("function", {}).get("name", "")
            for t in available_tools
        }

        for tool_name in suggested_tools:
            if tool_name not in available_names:
                continue

            params = self._build_tool_params(tool_name, message, entities)
            reason = self._get_tool_reason(tool_name, category)

            tool_calls.append(ToolCallRequest(
                tool_name=tool_name,
                params=params,
                reason=reason,
            ))

        return tool_calls

    def _build_tool_params(
        self,
        tool_name: str,
        message: str,
        entities: dict,
    ) -> dict:
        """Build parameters for a tool call."""
        params = {}

        if tool_name == "network_health_check":
            # Extract region
            if "region" in entities:
                params["region"] = f"region_{entities['region'][0]}"
            else:
                # Try to infer from message
                region_match = re.search(r"region\s+([a-z])", message, re.I)
                if region_match:
                    params["region"] = f"region_{region_match.group(1).lower()}"
                else:
                    params["region"] = "region_a"  # Default

            # Check for specific element
            if "node_id" in entities:
                params["element_id"] = f"node-{entities['node_id'][0]}"
            elif "element_id" in entities:
                params["element_id"] = entities["element_id"][0]

        elif tool_name == "network_config":
            params["action"] = "get"
            if "element_id" in entities:
                params["element_id"] = entities["element_id"][0]
            elif "node_id" in entities:
                params["element_id"] = f"node-{entities['node_id'][0]}"
            else:
                params["element_id"] = "cr-a-01"  # Default core router

        elif tool_name == "ticket_creator":
            params["title"] = message[:100]
            params["priority"] = "high" if "urgent" in message.lower() else "medium"
            if "region" in entities:
                params["region"] = entities["region"][0]

        elif tool_name == "config_validator":
            params["action"] = "validate"

        return params

    def _get_tool_reason(self, tool_name: str, category: str) -> str:
        """Get reason for calling a tool."""
        reasons = {
            "network_health_check": "Checking live network status for current metrics",
            "network_config": "Retrieving current configuration for analysis",
            "ticket_creator": "Creating incident ticket as requested",
            "config_validator": "Validating proposed configuration changes",
            "sla_monitor": "Checking SLA compliance metrics",
        }
        return reasons.get(tool_name, f"Required for {category} operation")

    def _generate_reasoning(
        self,
        message: str,
        category: str,
        entities: dict,
        tool_calls: list[ToolCallRequest],
    ) -> str:
        """Generate explanation of reasoning process."""
        parts = [f"Classified as '{category}' intent."]

        if entities:
            entity_str = ", ".join(
                f"{k}: {v}" for k, v in entities.items()
            )
            parts.append(f"Extracted entities: {entity_str}.")

        if tool_calls:
            tool_str = ", ".join(t.tool_name for t in tool_calls)
            parts.append(f"Will call tools: {tool_str}.")
        else:
            parts.append("No tool calls needed, using knowledge base.")

        return " ".join(parts)
