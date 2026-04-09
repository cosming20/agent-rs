"""Evidence formatting utilities for tool responses."""


from haystack import Document

_MAX_EVIDENCE_CHARS = 16000
"""Soft cap on formatted evidence text sent back to the LLM."""

_MAX_CONTEXT_CHARS = 300
"""Max characters of conversation context prepended to search queries."""


def format_evidence(documents: list[Document] | tuple[Document, ...]) -> str:
    """Format a list of Documents as numbered, readable text for the LLM.

    Each entry includes the source metadata (if present) and the content,
    truncated to ``_MAX_EVIDENCE_CHARS`` total to stay within token budgets.
    """
    if not documents:
        return "No documents found."

    parts: list[str] = []
    total_chars = 0

    for idx, doc in enumerate(documents, start=1):
        source = doc.meta.get("source", "unknown")
        header = f"[{idx}] (source={source})"

        content = doc.content or "(empty)"
        entry = f"{header}\n{content}"

        if total_chars + len(entry) > _MAX_EVIDENCE_CHARS:
            remaining = _MAX_EVIDENCE_CHARS - total_chars
            if remaining > 80:
                parts.append(entry[:remaining] + "\n... (truncated)")
            break

        parts.append(entry)
        total_chars += len(entry)

    return "\n\n".join(parts)


def build_contextual_query(query: str, conversation_context: str) -> str:
    """Augment a search query with conversation context for better retrieval.

    Prepends a brief conversation summary so embedding models and graph
    traversal can anchor on entities/topics from the ongoing conversation.
    Keeps context short to avoid diluting the query signal.

    Args:
        query: The raw search query.
        conversation_context: Brief summary of recent conversation turns.

    Returns:
        Augmented query string, or the original if no context is available.
    """
    if not conversation_context:
        return query
    ctx = conversation_context[:_MAX_CONTEXT_CHARS]
    return f"Context: {ctx}\nQuery: {query}"


def parse_brave_results(raw_content: str, provider: str) -> list[Document]:
    """Parse Brave Search MCP text results into Documents.

    Brave MCP returns results as text content. Each result typically
    includes title, URL, and description. We parse these into Documents
    with structured metadata.

    Args:
        raw_content: Raw text content from Brave MCP tool result.
        provider: Provider tag for metadata (``"brave_web"`` or ``"brave_news"``).

    Returns:
        List of Documents with source and provider metadata.
    """
    if not raw_content.strip():
        return []

    documents: list[Document] = []
    blocks = [b.strip() for b in raw_content.split("\n\n") if b.strip()]

    for idx, block in enumerate(blocks):
        url = ""
        for line in block.split("\n"):
            line_stripped = line.strip()
            if line_stripped.startswith(("http://", "https://")):
                url = line_stripped
                break

        documents.append(
            Document(
                content=block,
                meta={
                    "source": url or f"{provider}_result_{idx}",
                    "provider": provider,
                    "score": 1.0 / (idx + 1),
                },
            )
        )

    return documents
