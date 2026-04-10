//! Agent orchestrator -- slim coordinator over pipeline step structs.
//!
//! Wires together context building, agent execution, evidence fusion,
//! grounding verification, and result persistence into the full query
//! processing pipeline.

/// Coordinates the full query processing pipeline.
///
/// Constructed once at startup. Each call to a future `process_query`
/// method creates a [`QueryContext`] and runs it through the step
/// pipeline. Implemented in a later task.
pub struct AgentOrchestrator;
