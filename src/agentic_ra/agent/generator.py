"""LLM-based answer generation and query rewriting using OpenAI."""

from __future__ import annotations

import logging

from haystack import Document
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from agentic_ra.core.errors import OrchestratorError

logger = logging.getLogger(__name__)

_RAG_SYSTEM_PROMPT = """\
You are a precise, helpful research assistant. Answer the user's question \
based ONLY on the provided evidence. Follow these rules strictly:

1. Use ONLY information from the evidence to answer. Do not use prior knowledge.
2. If the evidence does not contain enough information, say so explicitly.
3. Be concise and direct. Avoid filler.
4. When referencing evidence, cite by [index] number.
5. If evidence is in a foreign language, answer in that language unless the user asks in English.
"""

_QUERY_REWRITE_PROMPT = """\
You are a search query optimizer. Given a user question and optional conversation history, \
rewrite it into a better search query for a retrieval system. Rules:

1. Expand abbreviations and resolve pronouns from context.
2. Add relevant keywords that improve recall.
3. Keep it concise — one query, no explanations.
4. Return ONLY the rewritten query, nothing else.
"""


class LLMGenerator:
    """LLM-based answer generation and query rewriting.

    Uses OpenAI chat models via Haystack's OpenAIChatGenerator.

    Args:
        model: OpenAI chat model name.
        api_key: OpenAI API key.
        temperature: Sampling temperature for generation.
        max_tokens: Maximum tokens in the generated answer.
    """

    def __init__(
        self,
        model: str = "gpt-5.4-mini-2026-03-17",
        api_key: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        self._model = model
        api_secret = (
            Secret.from_token(api_key)
            if api_key
            else Secret.from_env_var("OPENAI_API_KEY")
        )

        self._generator = OpenAIChatGenerator(
            model=model,
            api_key=api_secret,
            generation_kwargs={
                "temperature": temperature,
                "max_completion_tokens": max_tokens,
            },
        )
        self._rewriter = OpenAIChatGenerator(
            model=model,
            api_key=api_secret,
            generation_kwargs={"temperature": 0.0, "max_completion_tokens": 256},
        )

    def generate_answer(
        self,
        query: str,
        evidence: tuple[Document, ...] | list[Document],
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Generate an answer from query and evidence using the LLM.

        Args:
            query: User query text.
            evidence: Ranked evidence documents.
            conversation_history: Optional prior messages for context.

        Returns:
            Generated answer text.
        """
        if not evidence:
            return f"No relevant evidence found for: {query}"

        # Build evidence context
        evidence_text = self._format_evidence(evidence)

        messages = [ChatMessage.from_system(_RAG_SYSTEM_PROMPT)]

        # Add conversation history if available
        if conversation_history:
            for msg in conversation_history[-6:]:  # Last 6 messages for context
                if msg["role"] == "user":
                    messages.append(ChatMessage.from_user(msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(ChatMessage.from_assistant(msg["content"]))

        user_message = (
            f"Evidence:\n{evidence_text}\n\n"
            f"Question: {query}\n\n"
            f"Answer based on the evidence above:"
        )
        messages.append(ChatMessage.from_user(user_message))

        try:
            result = self._generator.run(messages=messages)
            replies: list[ChatMessage] = result["replies"]
            if replies:
                return replies[0].text or ""
            return "Failed to generate answer."
        except Exception as e:
            raise OrchestratorError(
                "LLM answer generation failed",
                model=self._model,
                operation="generate_answer",
                original_error=str(e),
            ) from e

    def rewrite_query(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Rewrite a query for better retrieval.

        Args:
            query: Original user query.
            conversation_history: Optional prior messages for context.

        Returns:
            Optimized search query.
        """
        messages = [ChatMessage.from_system(_QUERY_REWRITE_PROMPT)]

        context = ""
        if conversation_history:
            recent = conversation_history[-4:]
            context = "\n".join(f"{m['role']}: {m['content'][:200]}" for m in recent)
            context = f"Conversation context:\n{context}\n\n"

        messages.append(
            ChatMessage.from_user(
                f"{context}Original query: {query}\n\nRewritten query:"
            )
        )

        try:
            result = self._rewriter.run(messages=messages)
            replies: list[ChatMessage] = result["replies"]
            if replies and replies[0].text:
                rewritten = replies[0].text.strip().strip('"').strip("'")
                logger.info("Query rewritten: %r -> %r", query, rewritten)
                return rewritten
            return query
        except Exception as e:
            logger.warning("Query rewrite failed, using original: %s", e)
            return query

    def _format_evidence(
        self,
        evidence: tuple[Document, ...] | list[Document],
    ) -> str:
        """Format evidence documents for the LLM prompt."""
        parts: list[str] = []
        for idx, doc in enumerate(evidence, start=1):
            content = doc.meta.get("chunk_text", doc.content or "")
            source = doc.meta.get("source_path", doc.meta.get("source", "unknown"))
            headings = doc.meta.get("headings", [])

            header = f"[{idx}]"
            if headings:
                header += f" ({' > '.join(headings)})"
            header += f" [source: {source}]"

            parts.append(f"{header}\n{content}")

        return "\n\n".join(parts)
