//! Retrieval strategy registry.
//!
//! Maps each [`RetrievalStrategy`] variant to its configuration and determines
//! which search back-ends need to be invoked for a given strategy.

use agent_rs_core::RetrievalStrategy;

/// Registry of strategy configurations.
///
/// Currently a zero-sized type exposing pure functions. Will be extended to
/// carry per-strategy tunables (e.g. reranking flags, timeout budgets) when
/// the router is implemented.
#[derive(Debug, Clone, Default)]
pub struct StrategyRegistry;

impl StrategyRegistry {
    /// Return the set of search back-ends required for a given strategy.
    ///
    /// The returned slice enumerates the logical back-ends the router must
    /// dispatch to. The router combines their results via RRF fusion.
    pub fn required_backends(strategy: RetrievalStrategy) -> &'static [SearchBackend] {
        match strategy {
            RetrievalStrategy::Semantic => &[SearchBackend::Vector],
            RetrievalStrategy::Keyword => &[SearchBackend::Keyword],
            RetrievalStrategy::Hybrid => &[SearchBackend::Vector, SearchBackend::Keyword],
            RetrievalStrategy::Graph
            | RetrievalStrategy::RagCompletion
            | RetrievalStrategy::GraphCoT
            | RetrievalStrategy::GraphConstrained => &[SearchBackend::Graph],
            RetrievalStrategy::GraphHybrid => &[
                SearchBackend::Vector,
                SearchBackend::Keyword,
                SearchBackend::Graph,
            ],
            RetrievalStrategy::Summaries => &[SearchBackend::Summaries],
            RetrievalStrategy::Temporal => &[SearchBackend::Temporal],
        }
    }
}

/// Logical search back-end identifiers used by the strategy registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SearchBackend {
    Vector,
    Keyword,
    Graph,
    Summaries,
    Temporal,
}
