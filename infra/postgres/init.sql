-- ============================================================
-- Agentic RAG — Initial Database Setup
-- ============================================================
-- Executed once on first container start via
-- /docker-entrypoint-initdb.d/init.sql.
-- Table schemas are managed by Alembic (Phase 2).
-- ============================================================

-- Enable useful extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Verify the database is ready (used by health check)
SELECT 1;
