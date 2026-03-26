"""Ingest sample documents for testing."""

import json
import sys

import httpx

BASE_URL = "http://127.0.0.1:1059"

SAMPLE_DOCS = [
    {
        "content": "Python is a popular programming language used for web development, data science, and artificial intelligence. It was created by Guido van Rossum.",
        "metadata": {"source": "wiki"},
    },
    {
        "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data without being explicitly programmed. Deep learning uses neural networks.",
        "metadata": {"source": "textbook"},
    },
    {
        "content": "PostgreSQL is an advanced open-source relational database known for reliability, data integrity, and extensibility. It supports JSON and full-text search.",
        "metadata": {"source": "docs"},
    },
    {
        "content": "Vector databases like Qdrant store high-dimensional embeddings for efficient similarity search. They power modern AI applications including RAG systems.",
        "metadata": {"source": "blog"},
    },
    {
        "content": "OpenSearch provides full-text search, log analytics, and observability capabilities. It uses BM25 scoring for keyword relevance ranking.",
        "metadata": {"source": "docs"},
    },
]


def main() -> None:
    resp = httpx.post(
        f"{BASE_URL}/api/v1/ingest",
        json={"documents": SAMPLE_DOCS, "source_path": "sample_docs"},
        timeout=120,
    )
    if resp.status_code != 200:
        print(f"Error: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(resp.json(), indent=2))


if __name__ == "__main__":
    main()
