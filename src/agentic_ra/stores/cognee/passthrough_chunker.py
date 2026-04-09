"""PassthroughChunker — yields each document's text as a single chunk, no re-splitting.

Cognee's default TextChunker always runs chunk_by_paragraph → chunk_by_sentence →
chunk_by_word, which can split pre-chunked text at sentence boundaries even when
chunk_size is set very high.  This chunker bypasses that entirely: each document
becomes exactly one DocumentChunk, preserving our upstream chunking decisions.
"""


from uuid import NAMESPACE_OID, uuid5

from cognee.modules.chunking.Chunker import Chunker
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk


class PassthroughChunker(Chunker):
    """Chunker that passes each document through as a single chunk unchanged."""

    async def read(self):  # type: ignore[override]
        full_text = ""
        async for content_text in self.get_text():
            full_text += content_text

        if not full_text.strip():
            return

        # Count tokens via embedding engine if available, else approximate
        try:
            from cognee.infrastructure.databases.vector.embeddings import (
                get_embedding_engine,
            )

            engine = get_embedding_engine()
            if engine.tokenizer:
                chunk_size = engine.tokenizer.count_tokens(full_text)
            else:
                chunk_size = len(full_text.split())
        except Exception:
            chunk_size = len(full_text.split())

        yield DocumentChunk(
            id=uuid5(NAMESPACE_OID, f"{str(self.document.id)}-0"),
            text=full_text,
            chunk_size=chunk_size,
            is_part_of=self.document,
            chunk_index=0,
            cut_type="passthrough",
            contains=[],
            metadata={"index_fields": ["text"]},
        )
