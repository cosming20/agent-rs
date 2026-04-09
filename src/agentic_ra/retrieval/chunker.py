"""Text chunking strategies for RAG pipelines.

Two-stage approach:
  1. Topic segmentation (TextTiling) — finds natural topic boundaries
  2. Sliding window — splits large topics into overlapping chunks

Operates on plain text — parser-agnostic.
"""


import logging
import re

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    max_words: int = 200,
    step_words: int = 150,
) -> list[str]:
    """Chunk text using topic segmentation + sliding window.

    1. TextTiling splits text at topic boundaries (no mid-topic splits)
    2. Large topics are further split with a sliding window (overlap for context)
    3. Small topics are kept as-is

    Args:
        text: Input text to chunk.
        max_words: Maximum words per chunk.
        step_words: Step size for sliding window (overlap = max_words - step_words).

    Returns:
        List of text chunks. Empty input returns empty list.
    """
    if not text or not text.strip():
        return []

    # Stage 1: topic segmentation
    topics = _topic_segment(text)

    # Stage 2: sliding window on oversized topics
    chunks: list[str] = []
    for topic in topics:
        words = topic.split()
        if len(words) <= max_words:
            chunks.append(topic)
        else:
            chunks.extend(_sliding_window(topic, max_words, step_words))

    return [c for c in chunks if c.strip()]


def _topic_segment(text: str) -> list[str]:
    """Split text into topic-coherent segments using TextTiling.

    Falls back to paragraph splitting if TextTiling fails (e.g. text
    is too short or lacks sufficient structure).
    """
    try:
        import nltk
        for res in ("stopwords", "punkt_tab"):
            try:
                nltk.data.find(f"corpora/{res}" if res == "stopwords" else f"tokenizers/{res}")
            except LookupError:
                nltk.download(res, quiet=True)
        from nltk.tokenize import TextTilingTokenizer
        tokenizer = TextTilingTokenizer()
        segments = tokenizer.tokenize(text)
        if segments:
            return [s.strip() for s in segments if s.strip()]
    except (ImportError, ValueError, LookupError) as e:
        logger.debug("TextTiling unavailable or failed (%s), using paragraph split", e)

    # Fallback: split on paragraph boundaries
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _sliding_window(
    text: str,
    window_size: int,
    step: int,
) -> list[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    if len(words) <= window_size:
        return [text]

    chunks: list[str] = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + window_size])
        chunks.append(chunk)
        # Stop if we've consumed all words
        if i + window_size >= len(words):
            break

    return chunks
