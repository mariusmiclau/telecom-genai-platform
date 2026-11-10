"""Transactional conversation persistence with a bounded PostgreSQL pool.

All blocking driver calls run off the event loop. Schema is installed explicitly
at startup, and complete turns are committed atomically.
"""

import asyncio
from datetime import datetime, timezone

from psycopg2.pool import ThreadedConnectionPool

from src.agents.memory import ConversationHistory, Message
from src.migrations import migrate
from src.security import current_principal


class PostgresConversationStore:
    def __init__(self, dsn: str, pool_size: int = 4, retention_days: int = 30):
        self.dsn = dsn
        self.pool_size = pool_size
        self.pool = None
        self.retention_days = retention_days

    async def start(self):
        await asyncio.to_thread(self._start)

    def _start(self):
        self.pool = ThreadedConnectionPool(
            1, self.pool_size, self.dsn, connect_timeout=5,
            options="-c statement_timeout=5000",
        )
        connection = self.pool.getconn()
        try:
            migrate(connection)
        finally:
            self.pool.putconn(connection)
        self._transaction("DELETE FROM platform_messages WHERE created_at < now() - (%s * interval '1 day')", (self.retention_days,))

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

    async def run(self, operation):
        """Run a trusted multi-statement operation atomically off the event loop."""
        return await asyncio.to_thread(self._run, operation)

    def _run(self, operation):
        if self.pool is None:
            raise RuntimeError("Store has not started")
        connection = self.pool.getconn()
        try:
            with connection:
                with connection.cursor() as cursor:
                    return operation(cursor)
        finally:
            self.pool.putconn(connection)

    async def get_history(self, session_id: str, max_messages: int = 10):
        principal = current_principal()
        rows = await asyncio.to_thread(
            self._transaction,
            "SELECT role, content, created_at FROM ("
            "SELECT id, role, content, created_at FROM platform_messages "
            "WHERE session_id=%s AND tenant_id=%s AND user_id=%s AND created_at >= now() - (%s * interval '1 day') ORDER BY id DESC LIMIT %s"
            ") recent ORDER BY id ASC", (session_id, principal.tenant, principal.subject, self.retention_days, max_messages), True,
        )
        now = datetime.now(timezone.utc)
        messages = [Message(role=row[0], content=row[1], timestamp=row[2]) for row in rows]
        return ConversationHistory(session_id, messages, now, now)

    async def save(self, session_id, user_message, assistant_response, **kwargs):
        principal = current_principal()
        await asyncio.to_thread(
            self._transaction,
            "INSERT INTO platform_messages(session_id,tenant_id,user_id,role,content) "
            "VALUES (%s,%s,%s,'user',%s),(%s,%s,%s,'assistant',%s); "
            "DELETE FROM platform_messages WHERE created_at < now() - (%s * interval '1 day')",
            (session_id, principal.tenant, principal.subject, user_message, session_id, principal.tenant, principal.subject, assistant_response, self.retention_days),
        )

    async def ready(self):
        await asyncio.to_thread(self._transaction, "SELECT 1", None, True)

    async def forget(self, session_id: str):
        principal = current_principal()
        await asyncio.to_thread(self._transaction, "DELETE FROM platform_messages WHERE session_id=%s AND tenant_id=%s AND user_id=%s", (session_id, principal.tenant, principal.subject))

    async def close(self):
        if self.pool:
            await asyncio.to_thread(self.pool.closeall)
            self.pool = None
