"""Exception hierarchy and IngestErrorCode mapping.

Maps Python exceptions to proto IngestErrorCode values and retryability flags.
The sidecar catches all exceptions in ChunkDocument and emits an IngestError
payload rather than closing the stream uncleanly.
"""

from __future__ import annotations

import sys
import traceback


# ---------------------------------------------------------------------------
# Domain exceptions
# ---------------------------------------------------------------------------


class DoclingIngestError(Exception):
    """Base class for all sidecar-domain errors."""


class RequestTooLargeError(DoclingIngestError):
    """Document exceeds configured size or page limits."""


class DoclingParseError(DoclingIngestError):
    """Docling DocumentConverter raised during conversion."""


class ChunkingError(DoclingIngestError):
    """HybridChunker raised during chunking."""


class InvalidMetadataError(DoclingIngestError):
    """sha256 mismatch or missing required metadata field."""


class UnsupportedMimetypeError(DoclingIngestError):
    """Mimetype not supported by Docling."""


class ModelDownloadError(DoclingIngestError):
    """Docling model download failed or unavailable."""


class OCRTimeoutError(DoclingIngestError):
    """OCR stage exceeded timeout."""


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

# Proto enum integers — mirrors IngestErrorCode in chunker.proto.
# We use raw ints here to avoid a circular import with generated stubs.
# The server.py maps these to the actual proto enum.
INGEST_ERROR_CODE_UNSPECIFIED = 0
INGEST_ERROR_CODE_PARSE_ERROR = 1
INGEST_ERROR_CODE_UNSUPPORTED_MIMETYPE = 2
INGEST_ERROR_CODE_OCR_TIMEOUT = 3
INGEST_ERROR_CODE_MODEL_DOWNLOAD_UNAVAILABLE = 4
INGEST_ERROR_CODE_CHUNKING_ERROR = 5
INGEST_ERROR_CODE_INVALID_METADATA = 6
INGEST_ERROR_CODE_REQUEST_TOO_LARGE = 7
INGEST_ERROR_CODE_REQUEST_TIMEOUT = 8
INGEST_ERROR_CODE_INTERNAL_ERROR = 99


def classify_exception(exc: BaseException) -> tuple[int, bool]:
    """Map a Python exception to (IngestErrorCode_int, retryable).

    Returns:
        A tuple of (error_code_int, retryable). The code is one of the
        INGEST_ERROR_CODE_* constants above, matching the proto enum.
    """
    if isinstance(exc, RequestTooLargeError):
        return (INGEST_ERROR_CODE_REQUEST_TOO_LARGE, False)
    if isinstance(exc, TimeoutError):
        return (INGEST_ERROR_CODE_REQUEST_TIMEOUT, True)
    if isinstance(exc, OCRTimeoutError):
        return (INGEST_ERROR_CODE_OCR_TIMEOUT, True)
    if isinstance(exc, ModelDownloadError):
        return (INGEST_ERROR_CODE_MODEL_DOWNLOAD_UNAVAILABLE, True)
    if isinstance(exc, UnsupportedMimetypeError):
        return (INGEST_ERROR_CODE_UNSUPPORTED_MIMETYPE, False)
    if isinstance(exc, InvalidMetadataError):
        return (INGEST_ERROR_CODE_INVALID_METADATA, False)
    if isinstance(exc, ChunkingError):
        return (INGEST_ERROR_CODE_CHUNKING_ERROR, False)
    if isinstance(exc, DoclingParseError):
        return (INGEST_ERROR_CODE_PARSE_ERROR, False)
    # Catch-all: any unexpected exception is an internal error with one retry.
    return (INGEST_ERROR_CODE_INTERNAL_ERROR, True)


def format_stack(exc: BaseException) -> str:
    """Format exception traceback as a string for the docling_stack field."""
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
