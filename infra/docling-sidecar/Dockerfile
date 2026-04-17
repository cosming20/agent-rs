# syntax=docker/dockerfile:1
# =============================================================================
# Docling gRPC sidecar
# Build context: REPO ROOT (infra/docker-compose.yml sets context: ..)
# Paths below are relative to repo root.
# =============================================================================

FROM python:3.12-slim AS base

# OS deps required by Docling's PDF pipeline and image processing.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        poppler-utils \
        curl \
    && rm -rf /var/lib/apt/lists/*

# grpc_health_probe: standard binary for gRPC healthchecks in Docker/k8s.
# Pinned version for reproducible builds.
RUN curl -fsSL -o /usr/local/bin/grpc_health_probe \
    https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/v0.4.25/grpc_health_probe-linux-amd64 \
    && chmod +x /usr/local/bin/grpc_health_probe

WORKDIR /app

# Copy Python package files from repo root context.
COPY infra/docling-sidecar/pyproject.toml ./pyproject.toml
COPY infra/docling-sidecar/src ./src

# Install Python dependencies (no-cache keeps the layer lean).
RUN pip install --no-cache-dir .

# Pre-download Docling models into a cache directory that will be mounted
# as a named volume in production so downloads only happen on first boot.
# Explicitly force a real PDF conversion so both the layout model AND the
# TableFormer ACCURATE model are downloaded at build time — just
# constructing DocumentConverter() does NOT load TableFormer weights.
# Per W7 review: failure here MUST fail the build (no `|| echo` fallback),
# otherwise first-request cold-start quietly pays the download cost + OOM
# risk under load.
ENV DOCLING_CACHE_DIR=/root/.cache/docling
RUN python -c "\
import logging, io; \
logging.basicConfig(level=logging.INFO); \
from docling.datamodel.base_models import InputFormat; \
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode; \
from docling.document_converter import DocumentConverter, PdfFormatOption; \
opts = PdfPipelineOptions(do_ocr=False, do_table_structure=True, force_backend_text=True); \
opts.table_structure_options.mode = TableFormerMode.ACCURATE; \
converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}); \
print('[models] Docling layout + tableformer-accurate pre-downloaded OK')"

# =============================================================================
# Compile protobuf stubs into the installed package.
# The proto file is copied from the Rust workspace (repo root context).
# =============================================================================
COPY agent-rs/proto/agentic_rag/v1/chunker.proto /workdir/proto/agentic_rag/v1/chunker.proto

# Ensure the generated package directory exists with an __init__.py.
RUN mkdir -p /app/src/agentic_rag/v1 \
    && touch /app/src/agentic_rag/__init__.py \
    && touch /app/src/agentic_rag/v1/__init__.py \
    && python -m grpc_tools.protoc \
        --proto_path=/workdir/proto \
        --python_out=/app/src \
        --grpc_python_out=/app/src \
        /workdir/proto/agentic_rag/v1/chunker.proto

# The generated grpc stub already uses absolute package imports
# (from agentic_rag.v1 import ...) which work correctly with PYTHONPATH=/app/src.
# No import fixup needed.

ENV PYTHONPATH=/app/src

EXPOSE 1068

# Healthcheck uses the standard grpc_health_probe binary baked above.
# Checks overall server health ("" service name) which the health servicer sets to SERVING.
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=3 \
    CMD /usr/local/bin/grpc_health_probe -addr=:1068 -rpc-timeout=5s || exit 1

ENTRYPOINT ["python", "-m", "docling_sidecar"]
