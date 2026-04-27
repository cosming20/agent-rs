"""Exception hierarchy for the Marker sidecar.

Mirrors the Docling sidecar's error model so both speak the same
proto ``IngestErrorCode`` values. That way, downstream Rust error
handling branches don't need to discriminate by backend.
"""

from __future__ import annotations

import traceback


class MarkerIngestError(Exception):
    """Base class for all Marker sidecar-domain errors."""


class RequestTooLargeError(MarkerIngestError):
    """Document exceeds configured size or page limits."""


class MarkerParseError(MarkerIngestError):
    """Marker converter raised during conversion."""


class ChunkingError(MarkerIngestError):
    """Post-render chunk assembly raised."""


class InvalidMetadataError(MarkerIngestError):
    """sha256 mismatch or missing required metadata field."""


class UnsupportedMimetypeError(MarkerIngestError):
    """Mimetype not supported by Marker."""


class ModelDownloadError(MarkerIngestError):
    """Marker model download failed or unavailable."""


# Proto enum integers — mirrors IngestErrorCode in chunker.proto.
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
    """Map a Python exception to ``(IngestErrorCode_int, retryable)``."""
    if isinstance(exc, RequestTooLargeError):
        return (INGEST_ERROR_CODE_REQUEST_TOO_LARGE, False)
    if isinstance(exc, TimeoutError):
        return (INGEST_ERROR_CODE_REQUEST_TIMEOUT, True)
    if isinstance(exc, ModelDownloadError):
        return (INGEST_ERROR_CODE_MODEL_DOWNLOAD_UNAVAILABLE, True)
    if isinstance(exc, UnsupportedMimetypeError):
        return (INGEST_ERROR_CODE_UNSUPPORTED_MIMETYPE, False)
    if isinstance(exc, InvalidMetadataError):
        return (INGEST_ERROR_CODE_INVALID_METADATA, False)
    if isinstance(exc, ChunkingError):
        return (INGEST_ERROR_CODE_CHUNKING_ERROR, False)
    if isinstance(exc, MarkerParseError):
        return (INGEST_ERROR_CODE_PARSE_ERROR, False)
    return (INGEST_ERROR_CODE_INTERNAL_ERROR, True)


def format_stack(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
