"""Conversation Memory with PostgreSQL persistence.

Stores and retrieves conversation history for multi-turn context.
Uses PostgreSQL for durable storage with session-based partitioning.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message in conversation history."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    sources: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ConversationHistory:
    """Complete conversation history for a session."""

    session_id: str
    messages: list[Message]
    created_at: datetime
    updated_at: datetime

    def to_messages_format(self) -> list[dict]:
        """Convert to OpenAI-compatible messages format."""
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]

    def get_context_window(self, max_messages: int = 10) -> list[Message]:
        """Get recent messages within context window."""
        return self.messages[-max_messages:]


class ConversationMemory:
    """Manages conversation history with PostgreSQL storage.

    Features:
    - Session-based conversation tracking
    - Automatic history pruning for context window management
    - Source attribution storage for RAG traceability
    - Async-compatible interface

    Schema:
        conversations: session_id, created_at, updated_at, metadata
        messages: id, session_id, role, content, sources, timestamp
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        max_history_length: int = 50,
    ):
        self.connection_string = connection_string or os.getenv(
            "DATABASE_URL",
            "postgresql://genai:changeme@localhost:5432/telecom_genai"
        )
        self.max_history_length = max_history_length
        self._connection = None

    def _get_connection(self):
        """Get or create database connection."""
        if self._connection is None or self._connection.closed:
            try:
                self._connection = psycopg2.connect(
                    self.connection_string,
                    cursor_factory=RealDictCursor,
                )
                self._connection.autocommit = True
                self._ensure_schema()
            except Exception as e:
                logger.error(f"Database connection failed: {e}")
                raise
        return self._connection

    def _ensure_schema(self):
        """Create tables if they don't exist."""
        conn = self._connection
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    session_id VARCHAR(255) PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB DEFAULT '{}'
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) REFERENCES conversations(session_id),
                    role VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    sources JSONB DEFAULT '[]',
                    metadata JSONB DEFAULT '{}',
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, timestamp DESC)
            """)

        logger.info("Database schema initialized")

    async def get_history(
        self,
        session_id: str,
        max_messages: Optional[int] = None,
    ) -> ConversationHistory:
        """Retrieve conversation history for a session.

        Args:
            session_id: Unique session identifier
            max_messages: Maximum messages to retrieve (default: max_history_length)

        Returns:
            ConversationHistory with messages ordered by timestamp
        """
        limit = max_messages or self.max_history_length
        conn = self._get_connection()

        try:
            with conn.cursor() as cur:
                # Get or create conversation
                cur.execute("""
                    INSERT INTO conversations (session_id)
                    VALUES (%s)
                    ON CONFLICT (session_id) DO UPDATE
                    SET updated_at = CURRENT_TIMESTAMP
                    RETURNING created_at, updated_at
                """, (session_id,))
                conv_row = cur.fetchone()

                # Get messages
                cur.execute("""
                    SELECT role, content, sources, metadata, timestamp
                    FROM messages
                    WHERE session_id = %s
                    ORDER BY timestamp ASC
                    LIMIT %s
                """, (session_id, limit))

                messages = []
                for row in cur.fetchall():
                    messages.append(Message(
                        role=row["role"],
                        content=row["content"],
                        timestamp=row["timestamp"],
                        sources=row["sources"] or [],
                        metadata=row["metadata"] or {},
                    ))

                logger.debug(
                    f"Retrieved {len(messages)} messages for session {session_id}"
                )

                return ConversationHistory(
                    session_id=session_id,
                    messages=messages,
                    created_at=conv_row["created_at"],
                    updated_at=conv_row["updated_at"],
                )

        except Exception as e:
            logger.error(f"Failed to get history for session {session_id}: {e}")
            # Return empty history on error
            return ConversationHistory(
                session_id=session_id,
                messages=[],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

    async def save(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        sources: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Save a conversation turn (user message + assistant response).

        Args:
            session_id: Unique session identifier
            user_message: The user's input message
            assistant_response: The assistant's response
            sources: Document IDs used to generate response
            metadata: Additional metadata to store
        """
        conn = self._get_connection()
        sources = sources or []
        metadata = metadata or {}

        try:
            with conn.cursor() as cur:
                # Ensure conversation exists
                cur.execute("""
                    INSERT INTO conversations (session_id)
                    VALUES (%s)
                    ON CONFLICT (session_id) DO UPDATE
                    SET updated_at = CURRENT_TIMESTAMP
                """, (session_id,))

                # Insert user message
                cur.execute("""
                    INSERT INTO messages (session_id, role, content, sources, metadata)
                    VALUES (%s, 'user', %s, %s, %s)
                """, (session_id, user_message, json.dumps([]), json.dumps({})))

                # Insert assistant response
                cur.execute("""
                    INSERT INTO messages (session_id, role, content, sources, metadata)
                    VALUES (%s, 'assistant', %s, %s, %s)
                """, (
                    session_id,
                    assistant_response,
                    json.dumps(sources),
                    json.dumps(metadata),
                ))

                # Prune old messages if exceeding limit
                cur.execute("""
                    DELETE FROM messages
                    WHERE session_id = %s
                    AND id NOT IN (
                        SELECT id FROM messages
                        WHERE session_id = %s
                        ORDER BY timestamp DESC
                        LIMIT %s
                    )
                """, (session_id, session_id, self.max_history_length))

            logger.info(
                f"Saved conversation turn for session {session_id}",
                extra={"sources_count": len(sources)},
            )

        except Exception as e:
            logger.error(f"Failed to save message for session {session_id}: {e}")
            raise

    async def clear_session(self, session_id: str) -> None:
        """Clear all messages for a session."""
        conn = self._get_connection()

        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM messages WHERE session_id = %s",
                    (session_id,)
                )
                cur.execute(
                    "DELETE FROM conversations WHERE session_id = %s",
                    (session_id,)
                )
            logger.info(f"Cleared session {session_id}")

        except Exception as e:
            logger.error(f"Failed to clear session {session_id}: {e}")
            raise

    async def get_session_count(self) -> int:
        """Get total number of active sessions."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as count FROM conversations")
            return cur.fetchone()["count"]

    def close(self):
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("Database connection closed")
