"""Deterministic ID generation — identical to the Docling sidecar.

IDs are backend-agnostic so the same sha256 always resolves to the same
``document_id``, ``chunk_id`` and ``section_id`` whether the ingest ran
through Docling or Marker. That makes A/B retrieval comparisons on the
same PDF trivial: Qdrant point IDs collide intentionally.
"""

from __future__ import annotations

from uuid import UUID, uuid5

import blake3

NAMESPACE_OID = UUID("6ba7b812-9dad-11d1-80b4-00c04fd430c8")


def document_id(sha256_hex: str) -> UUID:
    return uuid5(NAMESPACE_OID, sha256_hex)


def chunk_id(doc_id: UUID, index: int, embed_text: str) -> UUID:
    h = blake3.blake3(embed_text.encode("utf-8")).hexdigest()[:32]
    return uuid5(NAMESPACE_OID, f"{doc_id}:{index}:{h}")


def page_id(doc_id: UUID, page_no: int) -> UUID:
    return uuid5(NAMESPACE_OID, f"{doc_id}:p{page_no}")


def section_id(doc_id: UUID, heading_path: list[str]) -> UUID:
    return uuid5(NAMESPACE_OID, f"{doc_id}:s:" + ">".join(heading_path))


def block_id(doc_id: UUID, self_ref: str) -> UUID:
    return uuid5(NAMESPACE_OID, f"{doc_id}:{self_ref}")
