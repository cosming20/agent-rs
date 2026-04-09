"""SQLAlchemy ORM models for the Agentic RAG system.

All models inherit from ``Base``. Alembic uses ``Base.metadata``
for migration autogeneration.
"""


import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


ANONYMOUS_USER_ID = "anonymous"
"""Default user ID for unauthenticated or system-initiated operations."""


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all Agentic RAG models."""


class MessageRole(enum.StrEnum):
    """Valid roles for chat messages."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(Base):
    """A single message in a chat conversation."""

    __tablename__ = "chat_messages"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
    )
    session_id: Mapped[uuid.UUID] = mapped_column(index=True)
    role: Mapped[MessageRole]
    content: Mapped[str] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class MemoryEntry(Base):
    """A persistent memory entry for long-term agent memory."""

    __tablename__ = "memory_entries"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    key: Mapped[str] = mapped_column(String(255), index=True)
    value: Mapped[str] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class ParsedDocument(Base):
    """A parsed document stored as DoclingDocument JSON for reuse.

    Parse once, reuse many times: re-chunk with different params,
    expand context after retrieval, trace provenance.
    """

    __tablename__ = "parsed_documents"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
    )
    source_path: Mapped[str] = mapped_column(Text, index=True)
    user_id: Mapped[str] = mapped_column(
        String(255), index=True, default=ANONYMOUS_USER_ID
    )
    docling_json: Mapped[dict[str, Any]] = mapped_column(JSONB)
    chunk_count: Mapped[int] = mapped_column(Integer)
    chunks_added: Mapped[int] = mapped_column(Integer, default=0)
    cognify_done: Mapped[bool] = mapped_column(default=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class PipelineRun(Base):
    """Tracks a single RAG pipeline execution for observability and auditing."""

    __tablename__ = "pipeline_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
    )
    session_id: Mapped[uuid.UUID] = mapped_column(index=True)
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    query: Mapped[str] = mapped_column(Text)
    rewritten_query: Mapped[str] = mapped_column(Text)
    strategy: Mapped[str] = mapped_column(String(50))
    answer: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    is_grounded: Mapped[bool]
    citation_count: Mapped[int] = mapped_column(Integer)
    retrieval_ms: Mapped[float] = mapped_column(Float)
    evidence_count: Mapped[int] = mapped_column(Integer)
    retries: Mapped[int] = mapped_column(Integer, default=0)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
