-- Re-add the `is_grounded` columns dropped in the corresponding `up.sql`.
--
-- The original nullability is preserved:
--   * `query_runs.is_grounded`           BOOLEAN NOT NULL DEFAULT FALSE
--   * `strategy_performance.is_grounded` BOOLEAN NULL
--
-- `ADD COLUMN IF NOT EXISTS` is supported on Postgres 9.6+, which keeps
-- the rollback idempotent across fresh and partially-rolled-back DBs.
-- The `DEFAULT FALSE` on the NOT NULL column lets existing rows
-- backfill cleanly without a separate `UPDATE` step.

ALTER TABLE query_runs
    ADD COLUMN IF NOT EXISTS is_grounded BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE strategy_performance
    ADD COLUMN IF NOT EXISTS is_grounded BOOLEAN NULL;
