-- Query runs: records of individual query executions with results.
-- confidence is stored as DOUBLE PRECISION and constrained to [0.0, 1.0].
CREATE TABLE query_runs (
    id           UUID              PRIMARY KEY,
    session_id   UUID              NOT NULL,
    user_id      TEXT              NOT NULL,
    query        TEXT              NOT NULL,
    answer       TEXT              NOT NULL,
    strategy     TEXT              NOT NULL,
    confidence   DOUBLE PRECISION  NOT NULL,
    is_grounded  BOOLEAN           NOT NULL,
    created_at   TIMESTAMP         NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_confidence_range CHECK (confidence >= 0.0 AND confidence <= 1.0)
);

CREATE INDEX idx_query_runs_session_id ON query_runs (session_id);
CREATE INDEX idx_query_runs_user_id    ON query_runs (user_id);
