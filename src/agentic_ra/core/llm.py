"""LiteLLM-backed ChatGenerator for Haystack + factory function.

Provides a Haystack-compatible ChatGenerator that uses LiteLLM under the
hood, enabling provider/model switching via config strings (e.g.
``openai/gpt-4o``, ``anthropic/claude-sonnet-4-20250514``, ``ollama/llama3``).

Every call flows through LiteLLM's callback system for cost tracking.
"""


import logging
from typing import Any

import litellm
from haystack import component
from haystack.dataclasses import ChatMessage, ChatRole, ToolCall
from haystack.tools import Tool

logger = logging.getLogger(__name__)


@component
class LiteLLMChatGenerator:
    """Haystack-compatible ChatGenerator backed by LiteLLM.

    Drop-in replacement for ``OpenAIChatGenerator`` that supports 100+
    LLM providers through LiteLLM's unified API.

    Args:
        model: LiteLLM model string (e.g. ``openai/gpt-4o``).
        api_key: Provider API key.
        generation_kwargs: Default generation parameters (temperature, etc.).
        purpose: Cost tracking label (``agent``, ``verifier``, ``chunk_enrichment``).
    """

    def __init__(
        self,
        model: str,
        api_key: str = "",
        generation_kwargs: dict[str, Any] | None = None,
        purpose: str = "unknown",
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.generation_kwargs = generation_kwargs or {}
        self.purpose = purpose

    def warm_up(self) -> None:
        """No-op for API-based generators."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize for Haystack pipeline snapshots."""
        return {
            "type": "agentic_ra.core.llm.LiteLLMChatGenerator",
            "init_parameters": {
                "model": self.model,
                "generation_kwargs": self.generation_kwargs,
                "purpose": self.purpose,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LiteLLMChatGenerator":
        """Deserialize from Haystack pipeline snapshot."""
        params = data.get("init_parameters", {})
        return cls(
            model=params.get("model", ""),
            api_key=params.get("api_key", ""),
            generation_kwargs=params.get("generation_kwargs", {}),
            purpose=params.get("purpose", "unknown"),
        )

    @component.output_types(replies=list[ChatMessage])
    def run(
        self,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None = None,
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> dict[str, list[ChatMessage]]:
        """Run LLM completion via LiteLLM.

        Args:
            messages: Haystack ChatMessages (conversation history).
            generation_kwargs: Override generation parameters.
            tools: Haystack Tool definitions (converted to OpenAI format).

        Returns:
            Dict with ``replies`` key containing response ChatMessages.
        """
        merged_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        # Convert Haystack messages to LiteLLM/OpenAI format
        litellm_messages = self._convert_messages(messages)

        # Convert Haystack tools to OpenAI function format
        litellm_tools = None
        if tools:
            litellm_tools = [self._convert_tool(t) for t in tools]

        # Call LiteLLM
        response = litellm.completion(
            model=self.model,
            messages=litellm_messages,
            api_key=self.api_key or None,
            tools=litellm_tools,
            metadata={"purpose": self.purpose},
            **merged_kwargs,
        )

        # Convert response to Haystack ChatMessages
        replies = self._convert_response(response)
        return {"replies": replies}

    @staticmethod
    def _convert_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
        """Convert Haystack ChatMessages to LiteLLM/OpenAI format."""
        result: list[dict[str, Any]] = []
        for msg in messages:
            entry: dict[str, Any] = {
                "role": msg.role.value,
                "content": msg.text or "",
            }
            # Include tool calls if present (for assistant messages)
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": tc.arguments
                            if isinstance(tc.arguments, str)
                            else __import__("json").dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            # Include tool_call_id for tool result messages
            if msg.tool_call_result and msg.tool_call_result.origin:
                entry["role"] = "tool"
                entry["tool_call_id"] = msg.tool_call_result.origin.id
                entry["content"] = msg.tool_call_result.result
            result.append(entry)
        return result

    @staticmethod
    def _convert_tool(tool: Tool) -> dict[str, Any]:
        """Convert a Haystack Tool to OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    @staticmethod
    def _convert_response(response: Any) -> list[ChatMessage]:
        """Convert LiteLLM response to Haystack ChatMessages."""
        replies: list[ChatMessage] = []
        for choice in response.choices:
            msg = choice.message

            # Extract tool calls if present
            tool_calls: list[ToolCall] | None = None
            if msg.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.id,
                        tool_name=tc.function.name,
                        arguments=tc.function.arguments,
                    )
                    for tc in msg.tool_calls
                ]

            chat_msg = ChatMessage.from_assistant(
                text=msg.content or "",
                tool_calls=tool_calls,
            )
            # Attach usage metadata
            chat_msg._meta.update(
                {
                    "model": response.model,
                    "finish_reason": choice.finish_reason,
                    "usage": dict(response.usage) if response.usage else {},
                    "index": choice.index,
                }
            )
            replies.append(chat_msg)
        return replies


def create_generator(
    model: str,
    api_key: str = "",
    purpose: str = "unknown",
    **generation_kwargs: Any,
) -> LiteLLMChatGenerator:
    """Create a cost-tracked LLM generator.

    Single place to configure provider + model. Change the ``model`` string
    to switch providers (e.g. ``openai/gpt-4o`` → ``anthropic/claude-sonnet-4-20250514``).

    Args:
        model: LiteLLM model string.
        api_key: Provider API key.
        purpose: Cost tracking label.
        **generation_kwargs: Default generation parameters.

    Returns:
        Configured LiteLLMChatGenerator.
    """
    gen = LiteLLMChatGenerator(
        model=model,
        api_key=api_key,
        generation_kwargs=generation_kwargs,
        purpose=purpose,
    )
    logger.info("Created LLM generator: model=%s, purpose=%s", model, purpose)
    return gen
