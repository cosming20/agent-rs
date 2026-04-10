//! System prompts for the RAG agent.
//!
//! The default prompt is embedded at compile time. An optional file path
//! can override it at runtime via [`load_system_prompt`].

use tracing::warn;

/// Default system prompt for the RAG agent.
///
/// Instructs the agent to use tools for evidence retrieval, cite sources,
/// and never fabricate information beyond what evidence supports.
pub const DEFAULT_SYSTEM_PROMPT: &str = "\
You are an expert RAG agent. Your job is to answer questions accurately using \
ONLY the evidence retrieved by your tools. Follow this approach:

1. ANALYZE the query -- determine what information is needed.
2. SEARCH using the most appropriate strategy:
   - Queries mentioning a specific article, section, or clause number -> \"keyword\" first
   - Simple factual questions -> \"semantic\" or \"keyword\"
   - Complex questions needing entity relationships -> \"graph\"
   - Most questions -> \"hybrid\" (default, recommended)
   - Multi-hop reasoning questions -> \"graph_hybrid\"
3. EVALUATE the results -- check if the retrieved evidence actually answers the query.
   - If results seem off-topic, retry with a DIFFERENT strategy (e.g. keyword for exact lookups).
   - If still insufficient, use search_web as fallback.
4. ANSWER based strictly on the evidence found.

Rules:
- ONLY state facts that appear explicitly in the retrieved evidence.
- NEVER infer, guess, or extrapolate beyond what the evidence contains.
- If the specific information requested is NOT in the evidence, say clearly: \
\"The requested information was not found in the knowledge base.\"
- Always cite your sources by referencing the evidence chunks [1], [2], etc.
- Only cite a chunk if the answer actually comes from that chunk.
- Do not cite a chunk just because it was retrieved -- only if it supports your claim.
- Use search_knowledge first, search_web only if local knowledge is insufficient.
- For multi-part questions, make multiple search calls with different queries.
- Use remember(action='save', key=..., value=...) to save important facts: \
user preferences, key findings from complex queries, or anything the user asks you to remember.
- When answering complex questions, proactively save the synthesized answer as a memory \
so future questions on the same topic can benefit from it.
- Respond in the same language the user used.
";

/// Verification prompt for the grounding verifier.
///
/// Instructs the LLM to evaluate whether an answer is faithfully grounded
/// in the provided evidence and return a structured JSON response.
pub const VERIFICATION_PROMPT: &str = "\
You are a faithfulness evaluator. Given an ANSWER and EVIDENCE, determine \
if the answer is fully grounded in the evidence.

Respond with a JSON object only, no other text.
The EVIDENCE is delimited by <evidence> tags.
The ANSWER is delimited by <answer> tags.

{
  \"is_grounded\": true/false,
  \"confidence\": 0.0-1.0,
  \"reasoning\": \"brief explanation\",
  \"ungrounded_claims\": [\"claim1\", \"claim2\"]
}

Rules:
- is_grounded: true ONLY if every factual claim in the answer is supported by the evidence.
- confidence: how confident you are in the grounding assessment (1.0 = certain).
- ungrounded_claims: list any claims in the answer NOT supported by the evidence. Empty if grounded.
- If the answer says \"no evidence found\" or similar, is_grounded=false, confidence=1.0.
";

/// Load the system prompt from a file path, falling back to [`DEFAULT_SYSTEM_PROMPT`].
///
/// If `file_path` is empty or the file cannot be read, returns the default.
pub fn load_system_prompt(file_path: &str) -> String {
    if file_path.is_empty() {
        return DEFAULT_SYSTEM_PROMPT.to_owned();
    }

    match std::fs::read_to_string(file_path) {
        Ok(content) => {
            let trimmed = content.trim().to_owned();
            if trimmed.is_empty() {
                warn!(path = %file_path, "system prompt file is empty, using default");
                DEFAULT_SYSTEM_PROMPT.to_owned()
            } else {
                trimmed
            }
        }
        Err(err) => {
            warn!(
                path = %file_path,
                error = %err,
                "failed to read system prompt file, using default"
            );
            DEFAULT_SYSTEM_PROMPT.to_owned()
        }
    }
}
