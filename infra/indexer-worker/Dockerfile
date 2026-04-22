# syntax=docker/dockerfile:1.7
# Multi-stage build for the `indexer-worker` binary.
#
# - Builder: rust:1.85-slim compiles the workspace against the pinned
#   toolchain; only the indexer-worker binary is emitted so the final
#   layer does not ship the agent-service binary.
# - Runtime: debian:bookworm-slim with ca-certificates (TLS for OpenAI /
#   Qdrant / MinIO) + libssl3 (aws-sdk-s3 rustls suffices but libssl is
#   cheap and enables pg-native if we ever switch backends).

FROM rust:1.85-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        pkg-config libssl-dev protobuf-compiler cmake git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . .

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/workspace/target \
    cargo build --release --bin indexer-worker \
    && cp target/release/indexer-worker /usr/local/bin/indexer-worker

FROM debian:bookworm-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/indexer-worker /usr/local/bin/indexer-worker

ENV RUST_LOG=info

ENTRYPOINT ["/usr/local/bin/indexer-worker"]
