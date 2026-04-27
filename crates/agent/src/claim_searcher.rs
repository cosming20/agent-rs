//! Claim-directed retrieval interface used by the grounding gate.
//!
//! Decouples [`crate::steps::grounding_gate::GroundingGate`] from the
//! concrete retrieval router so the gate can be unit-tested with a mock
//! and so the retrieval implementation can evolve (graph + vector
//! hybrid, reranker swap, weighted-keyword integration, …) without
//! touching the gate.
//!
//! The trait is the seam Phase 3 introduces to break the **circular
//! grounding bug** described in the rewrite spec: when the verifier
//! flags claims as `supported = false`, the gate now runs a focused
//! retrieval per claim before re-running the agent. Only when the
//! claim-directed search produces zero new evidence does the gate fall
//! through to a full agent rerun (the legacy behaviour, kept under
//! [`agent_rs_core::config::RetryStrategy::AgentRerun`]).

use std::sync::Arc;

use agent_rs_core::errors::AgentError;
use agent_rs_core::types::{RetrievalStrategy, UserId};
use agent_rs_retrieval::{RetrievalRouter, router::SectionFilter};
use agent_rs_stores::types::SearchResult;
use async_trait::async_trait;

/// Searches the knowledge base for evidence supporting a single claim.
///
/// Implementations should rewrite the claim into a retrieval-friendly
/// query if needed, run dense + keyword (and optionally graph) fusion,
/// and return up to `top_k` documents. An empty result indicates "no
/// evidence found" — the caller decides what to do (typically fall back
/// to a full agent rerun).
///
/// # Errors
///
/// Returns [`AgentError`] when the underlying retrieval back-end fails
/// in a non-recoverable way. Soft failures (empty result set, partial
/// branch outage) should be reported as `Ok(Vec::new())` and logged.
#[async_trait]
pub trait ClaimSearcher: Send + Sync {
    /// Run a focused retrieval for `claim`.
    ///
    /// # Parameters
    ///
    /// - `claim` — the verbatim claim text the verifier flagged as
    ///   unsupported. Implementations may rewrite or trim it for
    ///   retrieval but should not lose its semantic content.
    /// - `user_id` — tenant scope; every search branch must enforce it
    ///   at the store layer.
    /// - `active_document_keys` — MinIO keys of documents pinned to the
    ///   current turn (`active ∪ history`). Implementations may use
    ///   this to scope retrieval to the user's pinned set or pass it
    ///   through verbatim.
    /// - `strategy` — retrieval strategy hint (typically the same one
    ///   the agent used for the original turn).
    /// - `top_k` — maximum number of documents to return. The
    ///   `claim_search_top_k` knob in
    ///   [`agent_rs_core::config::GroundingConfig`] controls this
    ///   value.
    async fn search_for_claim(
        &self,
        claim: &str,
        user_id: &UserId,
        active_document_keys: &[String],
        strategy: RetrievalStrategy,
        top_k: usize,
    ) -> Result<Vec<SearchResult>, AgentError>;
}

/// Production [`ClaimSearcher`] backed by a shared
/// [`agent_rs_retrieval::RetrievalRouter`].
///
/// Delegates to [`RetrievalRouter::retrieve_filtered`] — the lowest-arity
/// entry point that already integrates dense + keyword + graph fusion,
/// optional cross-encoder reranking, and the post-fusion section / user
/// filter chain. The claim text is passed through unchanged; the router
/// owns query rewriting / HyDE / metadata extraction internally.
pub struct RouterClaimSearcher {
    router: Arc<RetrievalRouter>,
}

impl RouterClaimSearcher {
    /// Create a new [`RouterClaimSearcher`] sharing `router` with the
    /// rest of the pipeline.
    pub fn new(router: Arc<RetrievalRouter>) -> Self {
        Self { router }
    }
}

#[async_trait]
impl ClaimSearcher for RouterClaimSearcher {
    async fn search_for_claim(
        &self,
        claim: &str,
        user_id: &UserId,
        active_document_keys: &[String],
        strategy: RetrievalStrategy,
        top_k: usize,
    ) -> Result<Vec<SearchResult>, AgentError> {
        // When exactly one document is pinned, scope the claim search
        // to it via a `SectionFilter { doc_id, .. }` so the retrieved
        // evidence stays within the user's intended corpus. With zero
        // or multiple pinned documents we omit the filter and let the
        // router's user-scoped branches do their normal work.
        let section_filter: Option<SectionFilter> = match active_document_keys.len() {
            1 => Some(SectionFilter {
                doc_id: Some(active_document_keys[0].clone()),
                section_id: None,
            }),
            _ => None,
        };

        // The retrieval crate exposes its own `RetrievalError`. The
        // canonical wire-format error in the agent layer is
        // `agent_rs_core::RetrievalError`, and `AgentError` only has
        // `From` for the core variant. Translate through the existing
        // `From<retrieval::RetrievalError> for core::RetrievalError`
        // impl so the claim search reports the same way the rest of
        // the agent does.
        let pack = self
            .router
            .retrieve_filtered(user_id, claim, strategy, top_k, section_filter)
            .await
            .map_err(|e| AgentError::from(agent_rs_core::errors::RetrievalError::from(e)))?;

        Ok(pack.documents)
    }
}
