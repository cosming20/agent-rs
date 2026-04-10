-- Memories: persistent long-term memory entries associated with a user.
CREATE TABLE memories (
    id          UUID        PRIMARY KEY,
    user_id     TEXT        NOT NULL,
    content     TEXT        NOT NULL,
    created_at  TIMESTAMP   NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_memories_user_id ON memories (user_id);
