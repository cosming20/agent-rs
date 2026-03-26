"""Pipeline run persistence — track each RAG query execution."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from agentic_ra.core.errors import PostgresError
from agentic_ra.stores.postgres.client import PostgresClient
from agentic_ra.stores.postgres.models import PipelineRun


class RunStore:
    """Persists pipeline run records for observability and auditing.

    Args:
        pg_client: Async PostgreSQL client.
    """

    def __init__(self, pg_client: PostgresClient) -> None:
        self._pg = pg_client

    async def save(
        self,
        session_id: uuid.UUID,
        user_id: str,
        query: str,
        rewritten_query: str,
        strategy: str,
        answer: str,
        confidence: float,
        is_grounded: bool,
        citation_count: int,
        retrieval_ms: float,
        evidence_count: int,
        retries: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> PipelineRun:
        """Persist a pipeline run record.

        Args:
            session_id: Conversation session ID.
            user_id: User identifier.
            query: Original user query.
            rewritten_query: Query after rewriting.
            strategy: Retrieval strategy used.
            answer: Generated answer.
            confidence: Verification confidence score.
            is_grounded: Whether answer passed grounding check.
            citation_count: Number of citations in response.
            retrieval_ms: Retrieval latency in milliseconds.
            evidence_count: Number of evidence documents retrieved.
            retries: Number of Stage-2 retries performed.
            metadata: Optional extra metadata.

        Returns:
            The persisted PipelineRun record.
        """
        try:
            run = PipelineRun(
                session_id=session_id,
                user_id=user_id,
                query=query,
                rewritten_query=rewritten_query,
                strategy=strategy,
                answer=answer,
                confidence=confidence,
                is_grounded=is_grounded,
                citation_count=citation_count,
                retrieval_ms=retrieval_ms,
                evidence_count=evidence_count,
                retries=retries,
                metadata_=metadata or {},
            )
            async with self._pg.session() as session:
                session.add(run)
                await session.commit()
                await session.refresh(run)
                session.expunge(run)
            return run
        except SQLAlchemyError as e:
            raise PostgresError(
                "Failed to save pipeline run",
                operation="save_pipeline_run",
                original_error=str(e),
            ) from e

    async def get_by_session(
        self,
        session_id: uuid.UUID,
        limit: int = 50,
    ) -> list[PipelineRun]:
        """Retrieve pipeline runs for a session."""
        try:
            stmt = (
                select(PipelineRun)
                .where(PipelineRun.session_id == session_id)
                .order_by(PipelineRun.created_at.desc())
                .limit(limit)
            )
            async with self._pg.session() as session:
                result = await session.execute(stmt)
                runs = list(result.scalars().all())
                for run in runs:
                    session.expunge(run)
                return runs
        except SQLAlchemyError as e:
            raise PostgresError(
                "Failed to get pipeline runs",
                session_id=str(session_id),
                operation="get_pipeline_runs",
                original_error=str(e),
            ) from e
