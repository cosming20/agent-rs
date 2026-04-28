// @generated automatically by Diesel CLI.

diesel::table! {
    document_index_state (minio_object_key) {
        minio_object_key -> Text,
        user_id -> Text,
        state -> Text,
        #[max_length = 64]
        sha256 -> Nullable<Bpchar>,
        n_pages -> Nullable<Int4>,
        n_chunks -> Nullable<Int4>,
        error_message -> Nullable<Text>,
        queued_at -> Timestamptz,
        started_at -> Nullable<Timestamptz>,
        completed_at -> Nullable<Timestamptz>,
        updated_at -> Timestamptz,
    }
}

diesel::table! {
    evidence_items (id) {
        id -> Uuid,
        run_id -> Uuid,
        content -> Text,
        source -> Text,
        score -> Float8,
        created_at -> Timestamptz,
    }
}

diesel::table! {
    memories (id) {
        id -> Uuid,
        user_id -> Text,
        content -> Text,
        created_at -> Timestamptz,
    }
}

diesel::table! {
    query_runs (id) {
        id -> Uuid,
        trace_id -> Nullable<Uuid>,
        user_id -> Text,
        query -> Text,
        answer -> Text,
        strategy -> Text,
        confidence -> Float8,
        created_at -> Timestamptz,
    }
}

diesel::table! {
    strategy_performance (id) {
        id -> Int8,
        #[max_length = 64]
        query_hash -> Bpchar,
        query_intent -> Nullable<Text>,
        strategy -> Text,
        evidence_count -> Int4,
        avg_relevance_score -> Nullable<Float4>,
        confidence -> Nullable<Float4>,
        latency_ms -> Int8,
        required_retry -> Bool,
        sub_query_count -> Int4,
        used_hyde -> Bool,
        used_metadata_filter -> Bool,
        used_context_expansion -> Bool,
        linked_entities -> Int4,
        created_at -> Timestamptz,
    }
}

diesel::joinable!(evidence_items -> query_runs (run_id));

diesel::allow_tables_to_appear_in_same_query!(
    document_index_state,
    evidence_items,
    memories,
    query_runs,
    strategy_performance,
);
