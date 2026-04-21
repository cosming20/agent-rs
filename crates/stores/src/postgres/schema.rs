// @generated automatically by Diesel CLI.

diesel::table! {
    chat_messages (id) {
        id -> Uuid,
        session_id -> Uuid,
        role -> Text,
        content -> Text,
        created_at -> Timestamp,
        user_id -> Text,
    }
}

diesel::table! {
    evidence_items (id) {
        id -> Uuid,
        run_id -> Uuid,
        content -> Text,
        source -> Text,
        score -> Float8,
        created_at -> Timestamp,
    }
}

diesel::table! {
    ingested_documents (document_id) {
        document_id -> Uuid,
        #[max_length = 64]
        sha256 -> Bpchar,
        source -> Text,
        ingest_status -> Text,
        ingest_run_id -> Uuid,
        failure_reason -> Nullable<Text>,
        docling_version -> Nullable<Text>,
        docling_core_version -> Nullable<Text>,
        schema_version -> Nullable<Int4>,
        n_pages -> Nullable<Int4>,
        n_chunks -> Nullable<Int4>,
        n_tables -> Nullable<Int4>,
        n_pictures -> Nullable<Int4>,
        n_formulas -> Nullable<Int4>,
        language -> Nullable<Text>,
        title -> Nullable<Text>,
        mimetype -> Nullable<Text>,
        parse_latency_ms -> Nullable<Int8>,
        total_ingest_latency_ms -> Nullable<Int8>,
        started_at -> Timestamp,
        completed_at -> Nullable<Timestamp>,
        updated_at -> Timestamp,
        user_id -> Text,
        minio_object_key -> Nullable<Text>,
    }
}

diesel::table! {
    memories (id) {
        id -> Uuid,
        user_id -> Text,
        content -> Text,
        created_at -> Timestamp,
    }
}

diesel::table! {
    query_runs (id) {
        id -> Uuid,
        session_id -> Uuid,
        user_id -> Text,
        query -> Text,
        answer -> Text,
        strategy -> Text,
        confidence -> Float8,
        is_grounded -> Bool,
        created_at -> Timestamp,
    }
}

diesel::table! {
    session_active_documents (session_id, document_id) {
        session_id -> Uuid,
        document_id -> Uuid,
        activated_at -> Timestamptz,
    }
}

diesel::table! {
    sessions (session_id) {
        session_id -> Uuid,
        user_id -> Text,
        created_at -> Timestamptz,
        updated_at -> Timestamptz,
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
        is_grounded -> Nullable<Bool>,
        confidence -> Nullable<Float4>,
        latency_ms -> Int8,
        required_retry -> Bool,
        sub_query_count -> Int4,
        used_hyde -> Bool,
        used_metadata_filter -> Bool,
        used_context_expansion -> Bool,
        linked_entities -> Int4,
        created_at -> Timestamp,
    }
}

diesel::joinable!(evidence_items -> query_runs (run_id));
diesel::joinable!(session_active_documents -> ingested_documents (document_id));
diesel::joinable!(session_active_documents -> sessions (session_id));

diesel::allow_tables_to_appear_in_same_query!(
    chat_messages,
    evidence_items,
    ingested_documents,
    memories,
    query_runs,
    session_active_documents,
    sessions,
    strategy_performance,
);
