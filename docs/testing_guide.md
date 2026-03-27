# Agentic RAG — Testing Guide

This guide explains how to test and verify the Agentic RAG system as it is being built. Following the project's **"Verification Over Testing"** philosophy, we prioritize real-world evidence of correctness over isolated unit tests.

---

## 🏗️ 1. Infrastructure Setup

Before any testing or verification, the underlying services must be running.

```bash
# Start all infrastructure services (Qdrant, OpenSearch, Postgres, OPA, OTel)
docker-compose -f infra/docker-compose.yml up -d

# Verify all services are healthy
docker-compose -f infra/docker-compose.yml ps
```

---

## ✅ 2. Module Verification (Current State)

Since the project is currently in the **Foundation Phase**, we focus on verifying the core and store modules.

### Database Layer (Postgres)
We verify that the Postgres client can connect, run migrations, and interact with models.

```bash
# Verify Postgres connection and models
python scripts/verify_db.py
```

### Infrastructure Health
We verify that all services defined in `docker-compose.yml` are reachable and responding on their assigned ports.

```bash
# Run infrastructure health check
python scripts/verify_infra.py
```

---

## 🚀 3. Future E2E Testing (Playwright)

Once the **API Gateway** and **Pipelines** are implemented, all system-level testing will be performed using **Playwright**.

### Strategy:
- **Location**: `tests/e2e/`
- **Scope**: Full request-response cycles (e.g., `POST /v1/chat`).
- **Data**: Verified against live infrastructure (Qdrant collections, Postgres sessions).
- **Automation**: Integrated into the `ralph-loop` for continuous validation.

---

## 🛠️ 4. Verification Protocol

Before declaring any module "complete", you must provide evidence of its correctness:

1.  **Run the verification script** for the module (e.g., `verify_db.py`).
2.  **Observe the output**: Success means all assertions passed; Failure means a detailed traceback.
3.  **Check Traces**: Verify that telemetry is being captured in **Jaeger** (`http://localhost:16686`).

> [!IMPORTANT]
> Do NOT write unit tests inside source files (`test_*.py`). Use external verification scripts or E2E tests only.
