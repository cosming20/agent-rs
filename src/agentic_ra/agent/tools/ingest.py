"""Document ingestion tool -- adds text to the Cognee knowledge graph."""


import logging
from typing import Any

from agentic_ra.core.errors import ToolError

from ._async_bridge import run_async
from ._types import ToolConfig

logger = logging.getLogger(__name__)


def make_ingest_tool(
    cognee_store: Any,  # CogneeStore -- typed as Any to avoid circular import
    dataset: str = "user_anonymous",
) -> ToolConfig:
    """Create the ``ingest_document`` tool for the Haystack Agent.

    Ingests text into the Cognee knowledge graph via ``add_and_cognify``,
    scoped to the given *dataset* (typically ``"user_{user_id}"``).

    Args:
        cognee_store: CogneeStore instance for knowledge graph operations.
        dataset: Cognee dataset name for the current user.

    Returns:
        Tool descriptor dict with name, description, function, parameters.
    """

    def ingest_document(text: str) -> dict[str, str]:
        """Ingest text into the knowledge graph."""
        if not text.strip():
            raise ToolError(
                "Cannot ingest empty text",
                operation="ingest_document",
            )

        try:
            run_async(cognee_store.add_and_cognify(text=text, dataset=dataset))

            logger.info(
                "ingest_document ingested %d characters (dataset=%s)",
                len(text),
                dataset,
            )
            return {"text": f"Successfully ingested {len(text)} characters."}
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(
                "Document ingestion failed",
                text_length=len(text),
                operation="ingest_document",
                original_error=str(e),
            ) from e

    return {
        "name": "ingest_document",
        "description": (
            "Ingest text into the knowledge graph. The text is chunked, "
            "entities and relations are extracted, and the graph is updated."
        ),
        "function": ingest_document,
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The raw text content to ingest.",
                },
            },
            "required": ["text"],
        },
    }
