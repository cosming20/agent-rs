"""Parsed document persistence — store and retrieve DoclingDocument JSON."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError

from agentic_ra.core.errors import PostgresError
from agentic_ra.stores.postgres.client import PostgresClient
from agentic_ra.stores.postgres.models import ANONYMOUS_USER_ID, ParsedDocument


class DocumentStore:
    """Persists parsed DoclingDocument JSON in Postgres.

    Parse once, reuse many times: re-chunk with different parameters,
    expand context after retrieval, trace provenance back to source.

    Args:
        pg_client: Async PostgreSQL client.
    """

    def __init__(self, pg_client: PostgresClient) -> None:
        self._pg = pg_client

    async def save(
        self,
        source_path: str,
        docling_json: dict[str, Any],
        chunk_count: int,
        metadata: dict[str, Any] | None = None,
        user_id: str = ANONYMOUS_USER_ID,
    ) -> ParsedDocument:
        """Store a parsed DoclingDocument as JSON.

        Args:
            source_path: Original file path or URL.
            docling_json: Result of ``docling_doc.export_to_dict()``.
            chunk_count: Number of chunks produced from this document.
            metadata: Optional extra metadata.
            user_id: Owner of the document for per-user data isolation.

        Returns:
            The persisted ParsedDocument record.
        """
        try:
            entry = ParsedDocument(
                source_path=source_path,
                docling_json=docling_json,
                chunk_count=chunk_count,
                metadata_=metadata or {},
                user_id=user_id,
            )
            async with self._pg.session() as session:
                session.add(entry)
                await session.commit()
                await session.refresh(entry)
                session.expunge(entry)
            return entry
        except SQLAlchemyError as e:
            raise PostgresError(
                "Failed to save parsed document",
                source_path=source_path,
                operation="save_parsed_document",
                original_error=str(e),
            ) from e

    async def get_by_source(
        self, source_path: str, user_id: str
    ) -> ParsedDocument | None:
        """Retrieve a parsed document by its source path, scoped to a user.

        Enables cache-hit on re-ingest: if already parsed, skip parsing.

        Args:
            source_path: Original file path or URL.
            user_id: Owner of the document (prevents cross-user leakage).

        Returns:
            The ParsedDocument if found, None otherwise.
        """
        try:
            stmt = (
                select(ParsedDocument)
                .where(
                    ParsedDocument.source_path == source_path,
                    ParsedDocument.user_id == user_id,
                )
                .order_by(ParsedDocument.created_at.desc())
                .limit(1)
            )
            async with self._pg.session() as session:
                result = await session.execute(stmt)
                entry = result.scalar_one_or_none()
                if entry is not None:
                    session.expunge(entry)
                return entry
        except SQLAlchemyError as e:
            raise PostgresError(
                "Failed to get parsed document",
                source_path=source_path,
                operation="get_parsed_document",
                original_error=str(e),
            ) from e

    async def get_by_user(self, user_id: str, limit: int = 100) -> list[ParsedDocument]:
        """Retrieve all parsed documents for a user.

        Args:
            user_id: The user whose documents to retrieve.
            limit: Maximum number of results to return (most recent first).

        Returns:
            List of ParsedDocument records belonging to the user.
        """
        try:
            stmt = (
                select(ParsedDocument)
                .where(ParsedDocument.user_id == user_id)
                .order_by(ParsedDocument.created_at.desc())
                .limit(limit)
            )
            async with self._pg.session() as session:
                result = await session.execute(stmt)
                entries = list(result.scalars().all())
                for entry in entries:
                    session.expunge(entry)
                return entries
        except SQLAlchemyError as e:
            raise PostgresError(
                "Failed to get documents by user",
                user_id=user_id,
                operation="get_by_user",
                original_error=str(e),
            ) from e

    async def list_user_datasets(self) -> list[str]:
        """List all distinct user_ids that have documents (for admin fan-out).

        Returns:
            Sorted list of user_id strings present in parsed_documents.
        """
        try:
            stmt = select(ParsedDocument.user_id).distinct()
            async with self._pg.session() as session:
                result = await session.execute(stmt)
                return [row[0] for row in result.all()]
        except SQLAlchemyError as e:
            raise PostgresError(
                "Failed to list user datasets",
                operation="list_user_datasets",
                original_error=str(e),
            ) from e

    async def update_progress(
        self,
        doc_id: uuid.UUID,
        chunks_added: int,
        cognify_done: bool = False,
    ) -> None:
        """Update ingestion progress for a parsed document.

        Called after each batch of chunks is added to Cognee, and again
        when cognify completes.  Enables resume on crash.
        """
        try:
            stmt = (
                update(ParsedDocument)
                .where(ParsedDocument.id == doc_id)
                .values(chunks_added=chunks_added, cognify_done=cognify_done)
            )
            async with self._pg.session() as session:
                await session.execute(stmt)
                await session.commit()
        except SQLAlchemyError as e:
            raise PostgresError(
                "Failed to update ingestion progress",
                doc_id=str(doc_id),
                operation="update_progress",
                original_error=str(e),
            ) from e

    async def get_incomplete(self, user_id: str = ANONYMOUS_USER_ID) -> list[ParsedDocument]:
        """Get documents where ingestion didn't finish (for resume)."""
        try:
            stmt = (
                select(ParsedDocument)
                .where(
                    ParsedDocument.user_id == user_id,
                    ParsedDocument.cognify_done == False,  # noqa: E712
                )
                .order_by(ParsedDocument.created_at.asc())
            )
            async with self._pg.session() as session:
                result = await session.execute(stmt)
                entries = list(result.scalars().all())
                for entry in entries:
                    session.expunge(entry)
                return entries
        except SQLAlchemyError as e:
            raise PostgresError(
                "Failed to get incomplete documents",
                operation="get_incomplete",
                original_error=str(e),
            ) from e

    async def get_by_id(self, doc_id: uuid.UUID, user_id: str) -> ParsedDocument | None:
        """Retrieve a parsed document by ID, scoped to a user.

        Used for context expansion: chunk metadata references parent doc ID.

        Args:
            doc_id: ParsedDocument UUID.
            user_id: Owner of the document (prevents cross-user access).

        Returns:
            The ParsedDocument if found, None otherwise.
        """
        try:
            stmt = select(ParsedDocument).where(
                ParsedDocument.id == doc_id,
                ParsedDocument.user_id == user_id,
            )
            async with self._pg.session() as session:
                result = await session.execute(stmt)
                entry = result.scalar_one_or_none()
                if entry is not None:
                    session.expunge(entry)
                return entry
        except SQLAlchemyError as e:
            raise PostgresError(
                "Failed to get parsed document by ID",
                doc_id=str(doc_id),
                operation="get_parsed_document_by_id",
                original_error=str(e),
            ) from e
