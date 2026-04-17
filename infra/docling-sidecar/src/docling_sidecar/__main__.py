"""Entry point for the Docling gRPC sidecar.

Usage:
    python -m docling_sidecar
    docling-sidecar

Configuration via environment variables:
    DOCLING_GRPC_PORT         gRPC listen port (default: 1068)
    DOCLING_GRPC_HOST         gRPC listen host (default: 0.0.0.0)
    DOCLING_LOG_LEVEL         Logging level (default: INFO)
    DOCLING_MAX_PDF_BYTES     Max document bytes (default: 104857600 = 100 MiB)
    DOCLING_MAX_PAGES         Max pages per document (default: 500)
    DOCLING_PARSE_TIMEOUT_S   Parse wall-clock timeout (default: 300)
    DOCLING_STREAM_IDLE_TIMEOUT_S  Idle stream timeout (default: 600)
    DOCLING_MAX_CHUNKS_PER_DOC    Max chunks per doc (default: 10000)
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys


def _configure_logging() -> None:
    level_str = os.environ.get("DOCLING_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )


async def _run() -> None:
    from .chunker import ChunkerPool
    from .limits import ServerLimits
    from .server import create_server

    port = int(os.environ.get("DOCLING_GRPC_PORT", 1068))
    _configure_logging()

    logger = logging.getLogger(__name__)
    logger.info("Starting Docling gRPC sidecar on port %d", port)

    server_limits = ServerLimits.from_env()
    logger.info("Server limits: %s", server_limits)

    # Pre-load models at startup — this is intentionally synchronous so the
    # process does not accept connections until Docling is ready.
    loop = asyncio.get_running_loop()
    chunker_pool = await loop.run_in_executor(None, ChunkerPool)

    server = await create_server(port, chunker_pool, server_limits)
    await server.start()
    logger.info("Docling gRPC sidecar listening on 0.0.0.0:%d", port)

    # Handle graceful shutdown on SIGTERM / SIGINT.
    shutdown_event = asyncio.Event()

    def _handle_signal(sig: signal.Signals) -> None:
        logger.info("Received %s, initiating graceful shutdown...", sig.name)
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _handle_signal, sig)

    await shutdown_event.wait()
    logger.info("Stopping gRPC server (grace period: 10s)...")
    await server.stop(grace=10)
    logger.info("Server stopped. Goodbye.")


def main() -> None:
    """Synchronous entry point for the CLI script."""
    asyncio.run(_run())


if __name__ == "__main__":
    main()
