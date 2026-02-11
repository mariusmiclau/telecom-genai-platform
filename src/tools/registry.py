"""Tool registry for agent tool-calling.

Manages registration, discovery, and execution of tools
available to the AI agent. Follows OpenAI function-calling
compatible schema for tool definitions.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from src.monitoring.metrics import TOOL_CALLS, TOOL_LATENCY

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a tool execution."""

    data: dict
    summary: str
    latency_ms: float
    error: Optional[str] = None


class BaseTool(ABC):
    """Base class for all agent tools."""

    name: str = ""
    description: str = ""
    parameters: dict = {}

    @abstractmethod
    async def execute(self, params: dict) -> ToolResult:
        """Execute the tool with given parameters."""
        pass

    def to_schema(self) -> dict:
        """Convert to OpenAI function-calling compatible schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                },
            },
        }


class ToolRegistry:
    """Registry for managing and executing agent tools."""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        """Register a tool for agent use."""
        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' already registered, overwriting")
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def list_tools(self) -> list[dict]:
        """List all available tools in function-calling schema format."""
        return [tool.to_schema() for tool in self._tools.values()]

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    async def execute(self, tool_name: str, params: dict) -> ToolResult:
        """Execute a tool by name with given parameters.

        Includes metrics tracking and error handling.
        """
        tool = self._tools.get(tool_name)
        if not tool:
            TOOL_CALLS.labels(tool_name=tool_name, status="error").inc()
            return ToolResult(
                data={},
                summary=f"Tool '{tool_name}' not found",
                latency_ms=0,
                error=f"Unknown tool: {tool_name}",
            )

        start = time.time()
        try:
            result = await tool.execute(params)
            latency = (time.time() - start) * 1000

            TOOL_CALLS.labels(tool_name=tool_name, status="success").inc()
            TOOL_LATENCY.labels(tool_name=tool_name).observe(latency / 1000)

            logger.info(
                f"Tool executed: {tool_name}",
                extra={"latency_ms": latency, "params": params},
            )
            return result

        except Exception as e:
            latency = (time.time() - start) * 1000
            TOOL_CALLS.labels(tool_name=tool_name, status="error").inc()

            logger.error(
                f"Tool execution failed: {tool_name}: {e}",
                extra={"params": params},
                exc_info=True,
            )
            return ToolResult(
                data={},
                summary=f"Tool '{tool_name}' failed: {str(e)}",
                latency_ms=latency,
                error=str(e),
            )
