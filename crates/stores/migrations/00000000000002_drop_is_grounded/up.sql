-- Drop `is_grounded` columns. The LLM-based grounding verifier was
-- removed in commit 8cacafe; the column has been written with a
-- heuristic-derived value since then but no consumer reads it.
--
-- Two columns share the name and the same dead concept:
--   * `query_runs.is_grounded`           (BOOLEAN NOT NULL) -- per-run flag
--   * `strategy_performance.is_grounded` (BOOLEAN NULL)     -- adaptive
--                                                              learning mirror
--
-- Both come from the verifier output; the `strategy_performance` table
-- has no insertable / queryable struct in the codebase, so its column
-- is also dead.
--
-- Idempotent via `IF EXISTS` so the migration can run on a fresh DB
-- (where the column never existed beyond the initial schema) and on an
-- existing DB (where it must be dropped).

ALTER TABLE query_runs DROP COLUMN IF EXISTS is_grounded;

ALTER TABLE strategy_performance DROP COLUMN IF EXISTS is_grounded;
