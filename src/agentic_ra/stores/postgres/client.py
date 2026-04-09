"""Async PostgreSQL client wrapping SQLAlchemy engine and session factory."""


from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from agentic_ra.core.errors import PostgresError


class PostgresClient:
    """Async PostgreSQL connection manager.

    Args:
        dsn: Async PostgreSQL connection string (``postgresql+asyncpg://...``).
        pool_size: Number of persistent connections in the pool.
        max_overflow: Max additional connections beyond pool_size.
    """

    def __init__(
        self,
        dsn: str,
        pool_size: int = 5,
        max_overflow: int = 10,
    ) -> None:
        try:
            self._engine = create_async_engine(
                dsn,
                pool_size=pool_size,
                max_overflow=max_overflow,
            )
        except SQLAlchemyError as e:
            raise PostgresError(
                "Failed to create PostgreSQL engine",
                operation="create_engine",
            ) from e
        self._session_factory = async_sessionmaker(
            self._engine,
            expire_on_commit=False,
        )

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Yield an async session, auto-closing on exit."""
        async with self._session_factory() as session:
            yield session

    async def close(self) -> None:
        """Dispose the engine and release all pooled connections."""
        await self._engine.dispose()
