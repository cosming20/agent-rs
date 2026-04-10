-- Chat messages: stores each message in a conversation session.
CREATE TABLE chat_messages (
    id          UUID        PRIMARY KEY,
    session_id  UUID        NOT NULL,
    role        TEXT        NOT NULL,
    content     TEXT        NOT NULL,
    created_at  TIMESTAMP   NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_chat_messages_session_id ON chat_messages (session_id);
CREATE INDEX idx_chat_messages_created_at ON chat_messages (session_id, created_at DESC);
