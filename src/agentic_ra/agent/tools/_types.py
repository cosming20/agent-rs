"""Tool configuration type shared by all tool factories."""


from typing import Any, TypedDict


class ToolConfig(TypedDict):
    """Descriptor returned by each ``make_*`` tool factory."""

    name: str
    description: str
    function: Any  # Callable -- can't be more specific due to varying signatures
    parameters: dict[str, Any]
