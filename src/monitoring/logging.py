"""Structured logging configuration for the platform.

Provides JSON logging with correlation IDs for distributed tracing.
"""

import json
import logging
import os
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Optional

# Context variable for correlation ID (thread-safe)
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID from context."""
    return correlation_id_var.get()


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """Set a correlation ID in context.

    Args:
        correlation_id: ID to set, or None to generate a new one

    Returns:
        The correlation ID that was set
    """
    cid = correlation_id or str(uuid.uuid4())
    correlation_id_var.set(cid)
    return cid


class JSONFormatter(logging.Formatter):
    """JSON log formatter with structured output.

    Outputs logs in JSON Lines format with consistent fields:
    - timestamp: ISO 8601 timestamp
    - level: Log level
    - logger: Logger name
    - message: Log message
    - correlation_id: Request correlation ID
    - extra: Additional context fields
    """

    def __init__(
        self,
        include_timestamp: bool = True,
        include_logger_name: bool = True,
        include_level: bool = True,
    ):
        """Initialize the JSON formatter.

        Args:
            include_timestamp: Include timestamp in output
            include_logger_name: Include logger name in output
            include_level: Include log level in output
        """
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_logger_name = include_logger_name
        self.include_level = include_level

        # Fields to exclude from extra (they're handled specially)
        self._reserved_fields = {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "exc_info",
            "exc_text",
            "thread",
            "threadName",
            "taskName",
            "message",
        }

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON.

        Args:
            record: The log record to format

        Returns:
            JSON-formatted log string
        """
        log_data: dict[str, Any] = {}

        # Core fields
        if self.include_timestamp:
            log_data["timestamp"] = datetime.utcnow().isoformat() + "Z"

        if self.include_level:
            log_data["level"] = record.levelname

        if self.include_logger_name:
            log_data["logger"] = record.name

        # Message
        log_data["message"] = record.getMessage()

        # Correlation ID from context
        correlation_id = get_correlation_id()
        if correlation_id:
            log_data["correlation_id"] = correlation_id

        # Extra fields
        extra = {}
        for key, value in record.__dict__.items():
            if key not in self._reserved_fields:
                try:
                    # Ensure value is JSON serializable
                    json.dumps(value)
                    extra[key] = value
                except (TypeError, ValueError):
                    extra[key] = str(value)

        if extra:
            log_data["extra"] = extra

        # Exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable console formatter with colors.

    For development use - provides colored, readable output.
    """

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record for console output.

        Args:
            record: The log record to format

        Returns:
            Formatted log string with colors
        """
        color = self.COLORS.get(record.levelname, "")
        timestamp = datetime.utcnow().strftime("%H:%M:%S")

        # Get correlation ID
        correlation_id = get_correlation_id()
        cid_str = f" [{correlation_id[:8]}]" if correlation_id else ""

        # Base message
        message = f"{color}{timestamp} {record.levelname:8}{self.RESET}{cid_str} {record.name}: {record.getMessage()}"

        # Add extra fields
        extra_fields = []
        for key, value in record.__dict__.items():
            if key not in self._get_reserved_fields():
                extra_fields.append(f"{key}={value}")

        if extra_fields:
            message += f" | {', '.join(extra_fields)}"

        # Add exception
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"

        return message

    def _get_reserved_fields(self) -> set:
        """Get reserved field names."""
        return {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs",
            "pathname", "process", "processName", "relativeCreated",
            "stack_info", "exc_info", "exc_text", "thread", "threadName",
            "taskName", "message",
        }


def setup_logging(
    level: str = "INFO",
    json_output: bool = True,
    log_file: Optional[str] = None,
    app_name: str = "telecom-genai",
) -> logging.Logger:
    """Configure structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: Use JSON format (True) or console format (False)
        log_file: Optional file path for logging
        app_name: Application name for the root logger

    Returns:
        Configured root logger
    """
    # Get log level from environment or parameter
    log_level = os.getenv("LOG_LEVEL", level).upper()
    numeric_level = getattr(logging, log_level, logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create formatter based on output type
    if json_output:
        formatter = JSONFormatter()
    else:
        formatter = ConsoleFormatter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (always JSON for parsing)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)

    # Configure third-party loggers to be less verbose
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)

    # Log startup
    app_logger = logging.getLogger(app_name)
    app_logger.info(
        "Logging configured",
        extra={
            "log_level": log_level,
            "json_output": json_output,
            "log_file": log_file,
        },
    )

    return app_logger


class LoggingMiddleware:
    """ASGI middleware for request logging with correlation IDs.

    Automatically sets correlation ID from X-Correlation-ID header
    or generates a new one for each request.
    """

    def __init__(self, app):
        """Initialize the middleware.

        Args:
            app: The ASGI application
        """
        self.app = app
        self.logger = logging.getLogger("telecom-genai.http")

    async def __call__(self, scope, receive, send):
        """Process a request.

        Args:
            scope: ASGI scope
            receive: ASGI receive callable
            send: ASGI send callable
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Extract or generate correlation ID
        headers = dict(scope.get("headers", []))
        correlation_id = headers.get(b"x-correlation-id", b"").decode() or None
        set_correlation_id(correlation_id)

        # Log request start
        path = scope.get("path", "/")
        method = scope.get("method", "UNKNOWN")

        self.logger.info(
            f"{method} {path}",
            extra={
                "event": "request_start",
                "method": method,
                "path": path,
            },
        )

        # Process request
        await self.app(scope, receive, send)
