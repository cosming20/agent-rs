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

diesel::table! {
    ingested_documents (document_id) {
        document_id              -> Uuid,
        sha256                   -> Bpchar,
        source                   -> Text,
        ingest_status            -> Text,
        ingest_run_id            -> Uuid,
        failure_reason           -> Nullable<Text>,
        docling_version          -> Nullable<Text>,
        docling_core_version     -> Nullable<Text>,
        schema_version           -> Nullable<Integer>,
        n_pages                  -> Nullable<Integer>,
        n_chunks                 -> Nullable<Integer>,
        n_tables                 -> Nullable<Integer>,
        n_pictures               -> Nullable<Integer>,
        n_formulas               -> Nullable<Integer>,
        language                 -> Nullable<Text>,
        title                    -> Nullable<Text>,
        mimetype                 -> Nullable<Text>,
        parse_latency_ms         -> Nullable<BigInt>,
        total_ingest_latency_ms  -> Nullable<BigInt>,
        started_at               -> Timestamp,
        completed_at             -> Nullable<Timestamp>,
        updated_at               -> Timestamp,
    }
}

diesel::allow_tables_to_appear_in_same_query!(
    chat_messages,
    evidence_items,
    ingested_documents,
    memories,
    query_runs,
);
