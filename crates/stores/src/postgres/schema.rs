// @generated — matches migrations hand-written in crates/stores/migrations/.
// Regenerate with `diesel print-schema` once a live DB is available.

diesel::table! {
    chat_messages (id) {
        id         -> Uuid,
        session_id -> Uuid,
        role       -> Text,
        content    -> Text,
        created_at -> Timestamp,
    }
}

diesel::table! {
    memories (id) {
        id         -> Uuid,
        user_id    -> Text,
        content    -> Text,
        created_at -> Timestamp,
    }
}

diesel::table! {
    query_runs (id) {
        id          -> Uuid,
        session_id  -> Uuid,
        user_id     -> Text,
        query       -> Text,
        answer      -> Text,
        strategy    -> Text,
        confidence  -> Double,
        is_grounded -> Bool,
        created_at  -> Timestamp,
    }
}

diesel::table! {
    evidence_items (id) {
        id         -> Uuid,
        run_id     -> Uuid,
        content    -> Text,
        source     -> Text,
        score      -> Double,
        created_at -> Timestamp,
    }
}

diesel::joinable!(evidence_items -> query_runs (run_id));

diesel::allow_tables_to_appear_in_same_query!(chat_messages, evidence_items, memories, query_runs,);
