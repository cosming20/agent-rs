"""Short-term (session) and long-term (persistent) memory over Postgres."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from agentic_ra.core.errors import MemoryStoreError
from agentic_ra.stores.postgres.client import PostgresClient
from agentic_ra.stores.postgres.models import ChatMessage, MemoryEntry, MessageRole


class MemoryStore:
    """Agent memory abstraction over PostgreSQL.

    Provides short-term (session-scoped chat messages) and long-term
    (persistent key-value facts) memory operations.

    Args:
        pg_client: Async PostgreSQL client for database access.
    """

    def __init__(self, pg_client: PostgresClient) -> None:
        self._pg = pg_client

    # --- Short-term memory (session messages) ---

    async def add_message(
        self,
        session_id: uuid.UUID,
        role: MessageRole,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """Add a message to the session conversation history.

        Args:
            session_id: Session identifier.
            role: Message role (user/assistant/system).
            content: Message text content.
            metadata: Optional metadata dict.

        Returns:
            The persisted ChatMessage.
        """
        try:
            msg = ChatMessage(
                session_id=session_id,
                role=role,
                content=content,
                metadata_=metadata or {},
            )
            async with self._pg.session() as session:
                session.add(msg)
                await session.commit()
                await session.refresh(msg)
                session.expunge(msg)
            return msg
        except SQLAlchemyError as e:
            raise MemoryStoreError(
                "Failed to add message",
                session_id=str(session_id),
                operation="add_message",
                original_error=str(e),
            ) from e

    async def get_messages(
        self,
        session_id: uuid.UUID,
        limit: int = 50,
    ) -> list[ChatMessage]:
        """Retrieve recent messages for a session, ordered by creation time.

        Args:
            session_id: Session identifier.
            limit: Maximum messages to return.

        Returns:
            Messages ordered oldest-first.
        """
        try:
            stmt = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.asc())
                .limit(limit)
            )
            async with self._pg.session() as session:
                result = await session.execute(stmt)
                messages = list(result.scalars().all())
                for msg in messages:
                    session.expunge(msg)
                return messages
        except SQLAlchemyError as e:
            raise MemoryStoreError(
                "Failed to get messages",
                session_id=str(session_id),
                operation="get_messages",
                original_error=str(e),
            ) from e

    # --- Long-term memory (persistent facts) ---

    async def save_memory(
        self,
        user_id: str,
        key: str,
        value: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Save or update a long-term memory entry.

        If a memory with the same user_id + key exists, its value and
        metadata are updated. Otherwise a new entry is created.

        Args:
            user_id: User identifier.
            key: Memory lookup key.
            value: Memory content.
            metadata: Optional metadata dict.

        Returns:
            The persisted or updated MemoryEntry.
        """
        try:
            async with self._pg.session() as session:
                stmt = select(MemoryEntry).where(
                    MemoryEntry.user_id == user_id,
                    MemoryEntry.key == key,
                )
                result = await session.execute(stmt)
                entry = result.scalar_one_or_none()

                if entry is not None:
                    entry.value = value
                    entry.metadata_ = metadata or {}
                else:
                    entry = MemoryEntry(
                        user_id=user_id,
                        key=key,
                        value=value,
                        metadata_=metadata or {},
                    )
                    session.add(entry)

                await session.commit()
                await session.refresh(entry)
                session.expunge(entry)
                return entry
        except SQLAlchemyError as e:
            raise MemoryStoreError(
                "Failed to save memory",
                user_id=user_id,
                key=key,
                operation="save_memory",
                original_error=str(e),
            ) from e

    async def get_memory(
        self,
        user_id: str,
        key: str,
    ) -> MemoryEntry | None:
        """Retrieve a specific long-term memory entry.

        Args:
            user_id: User identifier.
            key: Memory lookup key.

        Returns:
            The memory entry, or None if not found.
        """
        try:
            stmt = select(MemoryEntry).where(
                MemoryEntry.user_id == user_id,
                MemoryEntry.key == key,
            )
            async with self._pg.session() as session:
                result = await session.execute(stmt)
                entry = result.scalar_one_or_none()
                if entry is not None:
                    session.expunge(entry)
                return entry
        except SQLAlchemyError as e:
            raise MemoryStoreError(
                "Failed to get memory",
                user_id=user_id,
                key=key,
                operation="get_memory",
                original_error=str(e),
            ) from e

    async def get_memories(
        self,
        user_id: str,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """Retrieve all long-term memories for a user.

        Args:
            user_id: User identifier.
            limit: Maximum entries to return.

        Returns:
            Memory entries ordered by most recently updated first.
        """
        try:
            stmt = (
                select(MemoryEntry)
                .where(MemoryEntry.user_id == user_id)
                .order_by(MemoryEntry.updated_at.desc())
                .limit(limit)
            )
            async with self._pg.session() as session:
                result = await session.execute(stmt)
                entries = list(result.scalars().all())
                for entry in entries:
                    session.expunge(entry)
                return entries
        except SQLAlchemyError as e:
            raise MemoryStoreError(
                "Failed to get memories",
                user_id=user_id,
                operation="get_memories",
                original_error=str(e),
            ) from e

    async def delete_memory(
        self,
        user_id: str,
        key: str,
    ) -> bool:
        """Delete a specific long-term memory entry.

        Args:
            user_id: User identifier.
            key: Memory lookup key.

        Returns:
            True if the entry was deleted, False if not found.
        """
        try:
            async with self._pg.session() as session:
                stmt = select(MemoryEntry).where(
                    MemoryEntry.user_id == user_id,
                    MemoryEntry.key == key,
                )
                result = await session.execute(stmt)
                entry = result.scalar_one_or_none()
                if entry is None:
                    return False
                await session.delete(entry)
                await session.commit()
                return True
        except SQLAlchemyError as e:
            raise MemoryStoreError(
                "Failed to delete memory",
                user_id=user_id,
                key=key,
                operation="delete_memory",
                original_error=str(e),
            ) from e
