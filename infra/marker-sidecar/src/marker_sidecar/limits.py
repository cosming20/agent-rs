"""Request limits enforcement for the Marker sidecar.

Same 0-as-default / clamp-to-server-cap semantics as the Docling sidecar.
"""

from __future__ import annotations

import dataclasses
import os


@dataclasses.dataclass(frozen=True)
class ServerLimits:
    max_pdf_bytes: int = 100 * 1024 * 1024  # 100 MiB
    max_pages: int = 500
    parse_timeout_s: int = 600
    stream_idle_timeout_s: int = 600
    max_chunks_per_doc: int = 10_000

    @classmethod
    def from_env(cls) -> "ServerLimits":
        return cls(
            max_pdf_bytes=int(os.environ.get("MARKER_MAX_PDF_BYTES", 100 * 1024 * 1024)),
            max_pages=int(os.environ.get("MARKER_MAX_PAGES", 500)),
            parse_timeout_s=int(os.environ.get("MARKER_PARSE_TIMEOUT_S", 600)),
            stream_idle_timeout_s=int(
                os.environ.get("MARKER_STREAM_IDLE_TIMEOUT_S", 600)
            ),
            max_chunks_per_doc=int(os.environ.get("MARKER_MAX_CHUNKS_PER_DOC", 10_000)),
        )

    def merge_with_request(
        self,
        req_max_pdf_bytes: int,
        req_max_pages: int,
        req_parse_timeout_s: int,
        req_stream_idle_timeout_s: int,
        req_max_chunks_per_doc: int,
    ) -> "ServerLimits":
        def bounded(server_cap: int, client_val: int) -> int:
            if client_val == 0:
                return server_cap
            return min(server_cap, client_val)

        return ServerLimits(
            max_pdf_bytes=bounded(self.max_pdf_bytes, req_max_pdf_bytes),
            max_pages=bounded(self.max_pages, req_max_pages),
            parse_timeout_s=bounded(self.parse_timeout_s, req_parse_timeout_s),
            stream_idle_timeout_s=bounded(
                self.stream_idle_timeout_s, req_stream_idle_timeout_s
            ),
            max_chunks_per_doc=bounded(self.max_chunks_per_doc, req_max_chunks_per_doc),
        )
