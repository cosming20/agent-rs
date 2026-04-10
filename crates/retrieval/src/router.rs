//! Retrieval router.
//!
//! Rewrites incoming queries and dispatches to the appropriate search
//! back-ends based on the selected [`RetrievalStrategy`].
//!
//! # Status
//!
//! Stub implementation — full dispatch logic will be added in Phase 3 Step 9.

use agent_rs_core::RetrievalStrategy;
use tracing::instrument;

/// Routes queries to the appropriate search back-ends.
///
/// The router is responsible for:
/// 1. Optional query rewriting (HyDE, expansion, decomposition).
/// 2. Strategy resolution (user-supplied or auto-detected).
/// 3. Parallel dispatch to individual search back-ends.
/// 4. Collecting results for the ranker.
#[derive(Debug, Clone, Default)]
pub struct RetrievalRouter;

impl RetrievalRouter {
    /// Rewrite a raw user query for improved retrieval quality.
    ///
    /// Returns the (possibly modified) query string. When no rewriting is
    /// configured this is a no-op returning the input unchanged.
    #[instrument(skip(self))]
    pub fn rewrite_query(&self, query: &str, _strategy: RetrievalStrategy) -> String {
        // TODO(Phase 3, Step 9): LLM-based HyDE / expansion rewriting.
        query.to_owned()
    }
}
