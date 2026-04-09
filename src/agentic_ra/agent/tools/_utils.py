"""Shared utility functions for agent tools."""


import re

_MAX_USER_ID_LEN = 128
"""Maximum length for sanitized user IDs in dataset names."""


def sanitize_user_id(user_id: str) -> str:
    """Sanitize user_id for use in Cognee dataset names.

    Strips any characters that are not alphanumeric, underscore, or hyphen
    to prevent injection into dataset names.

    Args:
        user_id: Raw user identifier.

    Returns:
        Sanitized string safe for dataset name construction.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "", user_id)
    sanitized = sanitized[:_MAX_USER_ID_LEN]
    return sanitized or "anonymous"
