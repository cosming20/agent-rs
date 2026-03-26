"""Parsed document persistence — store and retrieve DoclingDocument JSON."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from agentic_ra.core.errors import PostgresError
from agentic_ra.stores.postgres.client import PostgresClient
from agentic_ra.stores.postgres.models import ParsedDocument


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
    ) -> ParsedDocument:
        """Store a parsed DoclingDocument as JSON.

        Args:
            source_path: Original file path or URL.
            docling_json: Result of ``docling_doc.export_to_dict()``.
            chunk_count: Number of chunks produced from this document.
            metadata: Optional extra metadata.

        Returns:
            The persisted ParsedDocument record.
        """
        try:
            entry = ParsedDocument(
                source_path=source_path,
                docling_json=docling_json,
                chunk_count=chunk_count,
                metadata_=metadata or {},
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

    async def get_by_source(self, source_path: str) -> ParsedDocument | None:
        """Retrieve a parsed document by its source path.

        Enables cache-hit on re-ingest: if already parsed, skip parsing.

        Args:
            source_path: Original file path or URL.

        Returns:
            The ParsedDocument if found, None otherwise.
        """
        try:
            stmt = (
                select(ParsedDocument)
                .where(
                    ParsedDocument.source_path == source_path,
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

    async def get_by_id(self, doc_id: uuid.UUID) -> ParsedDocument | None:
        """Retrieve a parsed document by ID.

        Used for context expansion: chunk metadata references parent doc ID.

        Args:
            doc_id: ParsedDocument UUID.

        Returns:
            The ParsedDocument if found, None otherwise.
        """
        try:
            stmt = select(ParsedDocument).where(ParsedDocument.id == doc_id)
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
