"""Deterministic ID generation for documents, chunks, pages, and sections.

All IDs use UUIDv5 with NAMESPACE_OID as the namespace, ensuring stable
IDs across re-ingestion of identical bytes.
"""

from __future__ import annotations

from uuid import UUID, uuid5

import blake3

# UUIDv5 namespace: OID (6ba7b812-9dad-11d1-80b4-00c04fd430c8)
NAMESPACE_OID = UUID("6ba7b812-9dad-11d1-80b4-00c04fd430c8")


def document_id(sha256_hex: str) -> UUID:
    """Derive a stable document ID from the content sha256."""
    return uuid5(NAMESPACE_OID, sha256_hex)


def chunk_id(doc_id: UUID, index: int, embed_text: str) -> UUID:
    """Derive a stable chunk ID.

    Incorporates a blake3 hash of the embed_text so that two chunks at the
    same index with different content produce different IDs (e.g. after edit).
    """
    h = blake3.blake3(embed_text.encode("utf-8")).hexdigest()[:32]
    return uuid5(NAMESPACE_OID, f"{doc_id}:{index}:{h}")


def page_id(doc_id: UUID, page_no: int) -> UUID:
    """Derive a stable page ID."""
    return uuid5(NAMESPACE_OID, f"{doc_id}:p{page_no}")


def section_id(doc_id: UUID, heading_path: list[str]) -> UUID:
    """Derive a stable section ID from the heading path."""
    return uuid5(NAMESPACE_OID, f"{doc_id}:s:" + ">".join(heading_path))


def table_id(doc_id: UUID, self_ref: str) -> UUID:
    """Derive a stable table ID from the docling self_ref."""
    return uuid5(NAMESPACE_OID, f"{doc_id}:{self_ref}")


def picture_id(doc_id: UUID, self_ref: str) -> UUID:
    """Derive a stable picture ID from the docling self_ref."""
    return uuid5(NAMESPACE_OID, f"{doc_id}:{self_ref}")
