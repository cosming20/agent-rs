-- Evidence items: individual evidence pieces supporting a query run answer.
-- Each item is foreign-keyed to query_runs and cascade-deleted with the parent run.
CREATE TABLE evidence_items (
    id          UUID              PRIMARY KEY,
    run_id      UUID              NOT NULL REFERENCES query_runs(id) ON DELETE CASCADE,
    content     TEXT              NOT NULL,
    source      TEXT              NOT NULL,
    score       DOUBLE PRECISION  NOT NULL,
    created_at  TIMESTAMP         NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_evidence_items_run_id ON evidence_items (run_id);
