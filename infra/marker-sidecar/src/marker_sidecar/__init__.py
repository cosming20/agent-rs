"""Marker-based gRPC sidecar for the agent-rs ingestion pipeline.

Speaks the same ``agentic_rag.v1.DoclingChunker`` service as the Docling
sidecar so the Rust ``indexer-worker`` can swap backends via the
``AGENT_RS__INGEST_BACKEND`` env var without any code change.

Marker runs MPS-natively on Apple Silicon (surya layout + texify for
formulas), which is ~10-20× faster on a Mac than Docling's CodeFormulaV2
while matching quality on well-typeset printed math.
"""
