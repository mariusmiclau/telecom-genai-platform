"""Transactional conversation persistence with a bounded PostgreSQL pool.

All blocking driver calls run off the event loop. Schema is installed explicitly
at startup, and complete turns are committed atomically.
"""

import asyncio
from datetime import datetime, timezone

from psycopg2.pool import ThreadedConnectionPool

from src.agents.memory import ConversationHistory, Message

SCHEMA = """
CREATE TABLE IF NOT EXISTS platform_messages (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS platform_messages_session
    ON platform_messages(session_id, id DESC);
"""


class PostgresConversationStore:
    def __init__(self, dsn: str, pool_size: int = 4):
        self.dsn = dsn
        self.pool_size = pool_size
        self.pool = None

    async def start(self):
        await asyncio.to_thread(self._start)

    def _start(self):
        self.pool = ThreadedConnectionPool(
            1, self.pool_size, self.dsn, connect_timeout=5,
            options="-c statement_timeout=5000",
        )
        self._transaction(SCHEMA)

    def _transaction(self, sql: str, parameters=None, fetch=False):
        if self.pool is None:
            raise RuntimeError("Conversation store has not started")
        connection = self.pool.getconn()
        try:
            with connection:
                with connection.cursor() as cursor:
                    cursor.execute(sql, parameters)
                    return cursor.fetchall() if fetch else None
        finally:
            self.pool.putconn(connection)

    async def get_history(self, session_id: str, max_messages: int = 10):
        rows = await asyncio.to_thread(
            self._transaction,
            "SELECT role, content, created_at FROM ("
            "SELECT id, role, content, created_at FROM platform_messages "
            "WHERE session_id=%s ORDER BY id DESC LIMIT %s"
            ") recent ORDER BY id ASC", (session_id, max_messages), True,
        )
        now = datetime.now(timezone.utc)
        messages = [Message(role=row[0], content=row[1], timestamp=row[2]) for row in rows]
        return ConversationHistory(session_id, messages, now, now)

    async def save(self, session_id, user_message, assistant_response, **kwargs):
        await asyncio.to_thread(
            self._transaction,
            "INSERT INTO platform_messages(session_id,role,content) "
            "VALUES (%s,'user',%s),(%s,'assistant',%s)",
            (session_id, user_message, session_id, assistant_response),
        )

    async def ready(self):
        await asyncio.to_thread(self._transaction, "SELECT 1", None, True)

    async def close(self):
        if self.pool:
            await asyncio.to_thread(self.pool.closeall)
            self.pool = None
