//! Agent orchestration crate for agent-rs.
//!
//! Provides the pipeline that processes user queries:
//! context building, LLM agent execution via rig-core, evidence fusion,
//! deterministic coverage verification, response synthesis with
//! citations, and result persistence.
//!
//! The previous LLM-based grounding verifier (with its retry loop and
//! claim-directed retrieval back-end) was deleted in the synthesizer-
//! streams refactor. Grounding is now enforced implicitly by the
//! synthesis prompt itself ("only claim what's in evidence"), with a
//! deterministic [`subject_check`] pre-filter as the only safety net
//! that runs before synthesis.
//!
//! # Module layout
//!
//! - [`context`] -- [`QueryContext`] mutable pipeline state
//! - [`orchestrator`] -- [`AgentOrchestrator`] step pipeline coordinator
//! - [`synthesis`] -- [`ResponseSynthesizer`] citation builder + heuristic confidence
//! - [`subject_check`] -- deterministic pre-synth subject-presence gate
//! - [`prompts`] -- System prompt constants
//! - [`steps`] -- Pipeline step implementations

pub mod agent_hook;
pub mod context;
pub mod coverage;
pub mod lang;
pub mod orchestrator;
pub mod prompts;
pub mod steps;
pub mod subject_check;
pub mod synthesis;

pub use agent_hook::{AgentRsHook, COMPONENT_AGENT_RUNNER};
pub use context::QueryContext;
pub use orchestrator::{AgentOrchestrator, OrchestratorDeps, QueryRequest};
pub use synthesis::{Citation, ResponseSynthesizer, SynthesizedResponse};
