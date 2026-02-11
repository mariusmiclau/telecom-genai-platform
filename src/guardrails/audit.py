"""Audit logging for agent interactions.

Logs all agent activities with timestamps, session IDs, and tool calls
to structured JSON for compliance and debugging.
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Record of a tool call."""

    tool_name: str
    parameters: dict[str, Any]
    result_summary: Optional[str] = None
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"


@dataclass
class AuditEntry:
    """A single audit log entry."""

    session_id: str
    timestamp: str
    event_type: str  # query, response, tool_call, error, escalation
    user_message: Optional[str] = None
    assistant_response: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    sources_used: list[str] = field(default_factory=list)
    confidence_score: Optional[float] = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert tool_calls to dicts
        data["tool_calls"] = [
            tc if isinstance(tc, dict) else asdict(tc)
            for tc in self.tool_calls
        ]
        return data


class AuditLogger:
    """Logger for agent interactions with structured JSON output.

    Logs all agent activities including:
    - User queries and session info
    - Agent responses with confidence scores
    - Tool calls with parameters and results
    - Errors and escalations
    - Performance metrics

    Supports file-based logging with rotation and optional streaming.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_file: str = "audit.jsonl",
        max_file_size_mb: int = 100,
        also_log_to_logger: bool = True,
    ):
        """Initialize the audit logger.

        Args:
            log_dir: Directory for audit logs (defaults to ./logs/audit)
            log_file: Name of the audit log file
            max_file_size_mb: Maximum file size before rotation
            also_log_to_logger: Also send audit entries to Python logger
        """
        self.log_dir = Path(log_dir or os.getenv("AUDIT_LOG_DIR", "./logs/audit"))
        self.log_file = log_file
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.also_log_to_logger = also_log_to_logger

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / self.log_file

    def log_query(
        self,
        session_id: str,
        message: str,
        correlation_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Log an incoming user query.

        Args:
            session_id: Session identifier
            message: User's message
            correlation_id: Request correlation ID
            metadata: Additional metadata
        """
        entry = AuditEntry(
            session_id=session_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            event_type="query",
            user_message=message,
            correlation_id=correlation_id,
            metadata=metadata or {},
        )
        self._write_entry(entry)

    def log_response(
        self,
        session_id: str,
        response: str,
        sources: list[str],
        confidence: float,
        tool_calls: list[ToolCall],
        latency_ms: float,
        correlation_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Log an agent response.

        Args:
            session_id: Session identifier
            response: Agent's response text
            sources: Sources used in response
            confidence: Confidence score
            tool_calls: Tools called during processing
            latency_ms: Total processing time
            correlation_id: Request correlation ID
            metadata: Additional metadata
        """
        entry = AuditEntry(
            session_id=session_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            event_type="response",
            assistant_response=response[:1000],  # Truncate long responses
            sources_used=sources,
            confidence_score=confidence,
            tool_calls=tool_calls,
            latency_ms=latency_ms,
            correlation_id=correlation_id,
            metadata=metadata or {},
        )
        self._write_entry(entry)

    def log_tool_call(
        self,
        session_id: str,
        tool_name: str,
        parameters: dict,
        result_summary: Optional[str] = None,
        latency_ms: float = 0.0,
        success: bool = True,
        error: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Log a single tool call.

        Args:
            session_id: Session identifier
            tool_name: Name of the tool called
            parameters: Tool parameters
            result_summary: Summary of tool result
            latency_ms: Tool execution time
            success: Whether tool call succeeded
            error: Error message if failed
            correlation_id: Request correlation ID
        """
        tool_call = ToolCall(
            tool_name=tool_name,
            parameters=parameters,
            result_summary=result_summary,
            latency_ms=latency_ms,
            success=success,
            error=error,
        )

        entry = AuditEntry(
            session_id=session_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            event_type="tool_call",
            tool_calls=[tool_call],
            latency_ms=latency_ms,
            correlation_id=correlation_id,
        )
        self._write_entry(entry)

    def log_error(
        self,
        session_id: str,
        error: str,
        error_type: str = "processing_error",
        correlation_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Log an error event.

        Args:
            session_id: Session identifier
            error: Error message
            error_type: Type/category of error
            correlation_id: Request correlation ID
            metadata: Additional metadata
        """
        entry = AuditEntry(
            session_id=session_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            event_type="error",
            correlation_id=correlation_id,
            metadata={
                **(metadata or {}),
                "error": error,
                "error_type": error_type,
            },
        )
        self._write_entry(entry)

    def log_escalation(
        self,
        session_id: str,
        reason: str,
        escalation_type: str = "human_handoff",
        correlation_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Log an escalation event.

        Args:
            session_id: Session identifier
            reason: Reason for escalation
            escalation_type: Type of escalation
            correlation_id: Request correlation ID
            metadata: Additional metadata
        """
        entry = AuditEntry(
            session_id=session_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            event_type="escalation",
            correlation_id=correlation_id,
            metadata={
                **(metadata or {}),
                "escalation_reason": reason,
                "escalation_type": escalation_type,
            },
        )
        self._write_entry(entry)

    def _write_entry(self, entry: AuditEntry) -> None:
        """Write an audit entry to the log file.

        Args:
            entry: The audit entry to write
        """
        try:
            # Check for rotation
            self._rotate_if_needed()

            # Write to file
            with open(self.log_path, "a", encoding="utf-8") as f:
                json.dump(entry.to_dict(), f, default=str)
                f.write("\n")

            # Also log to Python logger if enabled
            if self.also_log_to_logger:
                logger.info(
                    f"Audit: {entry.event_type}",
                    extra={
                        "session_id": entry.session_id,
                        "event_type": entry.event_type,
                        "correlation_id": entry.correlation_id,
                    },
                )

        except Exception as e:
            logger.error(f"Failed to write audit entry: {e}", exc_info=True)

    def _rotate_if_needed(self) -> None:
        """Rotate the log file if it exceeds max size."""
        if not self.log_path.exists():
            return

        if self.log_path.stat().st_size >= self.max_file_size_bytes:
            # Rotate with timestamp
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            rotated_name = f"{self.log_file}.{timestamp}"
            rotated_path = self.log_dir / rotated_name
            self.log_path.rename(rotated_path)
            logger.info(f"Rotated audit log to {rotated_name}")

    def get_session_history(self, session_id: str) -> list[AuditEntry]:
        """Retrieve all audit entries for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of audit entries for the session
        """
        entries = []
        if not self.log_path.exists():
            return entries

        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("session_id") == session_id:
                        entries.append(data)
                except json.JSONDecodeError:
                    continue

        return entries
