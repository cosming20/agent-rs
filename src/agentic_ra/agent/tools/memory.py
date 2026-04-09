"""Memory tool -- save, retrieve, and list long-term user memories."""


import logging
from typing import Any

from agentic_ra.core.errors import ToolError

from ._async_bridge import run_async
from ._types import ToolConfig

logger = logging.getLogger(__name__)


def make_remember_tool(
    memory_store: Any,  # MemoryStore -- typed as Any to avoid circular import
    user_id: str,
) -> ToolConfig:
    """Create the ``remember`` tool for the Haystack Agent.

    Provides save, get, and list operations on long-term user memory.
    The ``user_id`` is captured via closure so the LLM cannot impersonate
    other users.

    Args:
        memory_store: MemoryStore instance for persistent memory.
        user_id: User identifier bound into the closure.

    Returns:
        Tool descriptor dict with name, description, function, parameters.
    """

    def remember(action: str, key: str = "", value: str = "") -> dict[str, str]:
        """Save, retrieve, or list long-term memories."""
        action_lower = action.lower()

        try:
            if action_lower == "save":
                if not key:
                    raise ToolError(
                        "Memory 'save' requires a key",
                        operation="remember",
                    )
                run_async(
                    memory_store.save_memory(user_id=user_id, key=key, value=value)
                )
                logger.info("remember: saved key=%s for user=%s", key, user_id)
                return {"text": f"Saved memory '{key}'."}

            if action_lower == "get":
                if not key:
                    raise ToolError(
                        "Memory 'get' requires a key",
                        operation="remember",
                    )
                entry = run_async(memory_store.get_memory(user_id=user_id, key=key))
                if entry is None:
                    return {"text": f"No memory found for key '{key}'."}
                return {"text": f"{entry.key}: {entry.value}"}

            if action_lower == "list":
                entries = run_async(memory_store.get_memories(user_id=user_id))
                if not entries:
                    return {"text": "No memories stored."}
                lines = [f"- {e.key}: {e.value}" for e in entries]
                return {"text": "\n".join(lines)}

            valid_actions = "save, get, list"
            raise ToolError(
                f"Invalid memory action: {action!r}",
                valid_actions=valid_actions,
                operation="remember",
            )

        except ToolError:
            raise
        except Exception as e:
            raise ToolError(
                "Memory operation failed",
                action=action,
                key=key,
                user_id=user_id,
                operation="remember",
                original_error=str(e),
            ) from e

    return {
        "name": "remember",
        "description": (
            "Save, retrieve, or list long-term memories for the current user. "
            "Use action='save' to store a fact, 'get' to retrieve by key, "
            "or 'list' to show all stored memories."
        ),
        "function": remember,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Memory operation: save, get, or list.",
                    "enum": ["save", "get", "list"],
                },
                "key": {
                    "type": "string",
                    "description": "Memory key (required for save and get).",
                    "default": "",
                },
                "value": {
                    "type": "string",
                    "description": "Value to store (required for save).",
                    "default": "",
                },
            },
            "required": ["action"],
        },
    }
