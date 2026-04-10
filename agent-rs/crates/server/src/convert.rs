//! Type conversion between proto-generated types and domain types.
//!
//! All mapping is explicit — no implicit `From` impls that hide contract
//! mismatches. Call sites import the free functions directly.

use agent_rs_agent::synthesis::{Citation as DomainCitation, SynthesizedResponse};
use agent_rs_core::types::RetrievalStrategy;

use crate::proto::agentic_rag::v1::{
    Citation as ProtoCitation, QueryResponse, Strategy as ProtoStrategy,
};

// ---- Strategy conversions ----

/// Convert a proto [`Strategy`] value to a domain [`RetrievalStrategy`].
///
/// `STRATEGY_UNSPECIFIED` maps to `Hybrid` as the sensible default.
pub(crate) fn proto_strategy_to_domain(proto: i32) -> RetrievalStrategy {
    match ProtoStrategy::try_from(proto).unwrap_or(ProtoStrategy::Unspecified) {
        ProtoStrategy::Unspecified => RetrievalStrategy::Hybrid,
        ProtoStrategy::Semantic => RetrievalStrategy::Semantic,
        ProtoStrategy::Keyword => RetrievalStrategy::Keyword,
        ProtoStrategy::Hybrid => RetrievalStrategy::Hybrid,
        ProtoStrategy::Graph => RetrievalStrategy::Graph,
        ProtoStrategy::GraphHybrid => RetrievalStrategy::GraphHybrid,
        ProtoStrategy::RagCompletion => RetrievalStrategy::RagCompletion,
        ProtoStrategy::Summaries => RetrievalStrategy::Summaries,
        ProtoStrategy::Temporal => RetrievalStrategy::Temporal,
        ProtoStrategy::GraphCot => RetrievalStrategy::GraphCoT,
        ProtoStrategy::GraphConstrained => RetrievalStrategy::GraphConstrained,
    }
}

/// Convert a domain [`RetrievalStrategy`] to the proto [`Strategy`] i32 value.
pub(crate) fn domain_strategy_to_proto(strategy: RetrievalStrategy) -> i32 {
    let proto = match strategy {
        RetrievalStrategy::Semantic => ProtoStrategy::Semantic,
        RetrievalStrategy::Keyword => ProtoStrategy::Keyword,
        RetrievalStrategy::Hybrid => ProtoStrategy::Hybrid,
        RetrievalStrategy::Graph => ProtoStrategy::Graph,
        RetrievalStrategy::GraphHybrid => ProtoStrategy::GraphHybrid,
        RetrievalStrategy::RagCompletion => ProtoStrategy::RagCompletion,
        RetrievalStrategy::Summaries => ProtoStrategy::Summaries,
        RetrievalStrategy::Temporal => ProtoStrategy::Temporal,
        RetrievalStrategy::GraphCoT => ProtoStrategy::GraphCot,
        RetrievalStrategy::GraphConstrained => ProtoStrategy::GraphConstrained,
    };
    proto as i32
}

// ---- Citation conversion ----

/// Convert a domain [`Citation`] to a proto [`Citation`].
pub(crate) fn domain_citation_to_proto(c: &DomainCitation) -> ProtoCitation {
    ProtoCitation {
        index: c.index as i32,
        content_snippet: c.content_snippet.clone(),
        source: c.source.clone(),
        score: c.score,
    }
}

// ---- Full response conversion ----

/// Convert a [`SynthesizedResponse`] to a proto [`QueryResponse`].
pub(crate) fn synthesized_to_query_response(resp: &SynthesizedResponse) -> QueryResponse {
    QueryResponse {
        answer: resp.answer.clone(),
        citations: resp
            .citations
            .iter()
            .map(domain_citation_to_proto)
            .collect(),
        confidence: resp.confidence.value(),
        is_grounded: resp.is_grounded,
        strategy: domain_strategy_to_proto(resp.strategy),
    }
}
