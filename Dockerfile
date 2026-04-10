# Build stage: compile the agent-rs binary in release mode.
FROM rust:latest AS builder

WORKDIR /app

# Copy workspace manifest and all crate manifests first for better layer caching.
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
COPY src/ src/
COPY proto/ proto/

RUN cargo build --release

# Runtime stage: minimal image with only the compiled binary.
FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/agent-rs /usr/local/bin/agent-rs

# gRPC server port
EXPOSE 1059

ENTRYPOINT ["agent-rs"]
