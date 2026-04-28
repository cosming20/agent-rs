//! Agent orchestration crate for agent-rs.
//!
//! Provides the pipeline that processes user queries:
//! context building, LLM agent execution via rig-core, evidence fusion,
//! grounding verification, response synthesis with citations, and
//! result persistence.
//!
//! # Module layout
//!
//! - [`context`] -- [`QueryContext`] mutable pipeline state
//! - [`orchestrator`] -- [`AgentOrchestrator`] step pipeline coordinator
//! - [`verifier`] -- [`GroundingVerifier`] faithfulness checker
//! - [`synthesis`] -- [`ResponseSynthesizer`] citation builder
//! - [`memory`] -- [`MemoryManager`] session + persistent memory
//! - [`prompts`] -- System prompt constants
//! - [`steps`] -- Pipeline step implementations

pub mod claim_searcher;
pub mod context;
pub mod coverage;
pub mod lang;
pub mod orchestrator;
pub mod prompts;
pub mod steps;
pub mod synthesis;
pub mod verifier;

pub use claim_searcher::{ClaimSearcher, RouterClaimSearcher};
pub use context::QueryContext;
pub use orchestrator::{AgentOrchestrator, OrchestratorDeps, QueryRequest};
pub use synthesis::{Citation, ResponseSynthesizer, SynthesizedResponse};
pub use verifier::{GroundingVerifier, VerificationResult};
