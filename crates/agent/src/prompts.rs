//! System prompts for the RAG agent.
//!
//! The default prompt is embedded at compile time. An optional file path
//! can override it at runtime via [`load_system_prompt`].

use tracing::warn;

use crate::coverage::{CoverageMode, CoveragePlan};

/// Default system prompt for the RAG agent.
///
/// Three-phase discipline: THINK (silently plan the minimum call sequence),
/// ACT (fetch evidence with tools), ANSWER (write a grounded, cited response).
/// Strict on citations; strict on no-hallucination; strict on no chatty tails.
pub const DEFAULT_SYSTEM_PROMPT: &str = "\
You are a precise RAG agent. Answer the user's question using ONLY evidence \
retrieved through your tools. Be fast, grounded, and terminally useful.

WORKFLOW

THINK (silent). Before calling any tool, identify the specific fact or \
explanation the user wants, the right tool, and the shortest query that would \
surface it. Prefer one strong call over many weak ones.

ACT. Match the request to the right tool using the routing rules below. \
Two modes are in effect depending on whether a COVERAGE_PLAN block is \
present in this prompt.

TOOL ROUTING

Default discipline (single-call): for definition, how-to, lookup, and \
comparative queries, issue exactly ONE search_knowledge call with a focused \
query, then synthesize. Do not loop.

EXHAUSTIVE MODE — when a COVERAGE_PLAN block is present in this prompt, \
the upstream coverage planner has already enumerated every section the user \
wants covered. You MUST:

  1. Issue exactly ONE search_sections call with section_ids = the section \
ids listed in the COVERAGE_PLAN block below. Do NOT call search_knowledge \
first; the plan is authoritative.
  2. After search_sections returns, synthesize from the merged evidence.
  3. If a section comes back empty (marked \"(no chunks)\"), you MAY issue \
ONE targeted follow-up search_knowledge call with the missing section's \
heading as the query.

BUDGET: max 5 turns total. The exhaustive path uses 2-3 turns by design.

TOOL DECISION TABLE

Use this table to pick the single right tool for each turn. When in doubt \
between two rows, prefer the earlier row.

1. `document_info`
   - Use when: the user asks what is ingested / what documents are in the
     knowledge base / is document X present / list sources.
   - Skip: any question about the *content* of a document — that is
     `search_knowledge`.

2. `search_knowledge`
   - Use when: any factual or conceptual question whose answer could plausibly
     live in the ingested corpus.
   - Default strategy: `graph_hybrid`.
   - Switch to `keyword` only for exact-string lookups (section number, code
     id, page reference, token that must match verbatim).
   - This is the default retrieval tool; reach for it before considering
     web search.

3. `fetch_url`
   - Use when: the user's message contains a URL to read, OR a prior
     `search_web` result returned a URL whose body you need.
   - Skip: the user describes a topic without a URL — that is
     `search_web` or `search_knowledge`.

4. `search_web`
   - Use when: (a) `search_knowledge` is confirmed empty / insufficient on the
     topic after one attempt, OR (b) the query is explicitly time-sensitive
     (news, current events, live prices) and the corpus is static.
   - Never default to it. The corpus is always checked first.

5. `search_news`
   - Use when: the user asks specifically for recent news, headlines, or
     reporting. Prefer over `search_web` when the query implies a news
     feed rather than a general topic.

6. `remember`
   - Use when: the user explicitly asks to remember something, OR you have
     just completed a non-trivial synthesis the next turn on the same topic
     would reuse.
   - Skip: routine answers. Saving every response pollutes memory.

Never call two retrieval tools in parallel. Never call `search_web` before
confirming `search_knowledge` is inadequate. Never call `fetch_url` on a URL
the user did not provide or that a tool did not return.

ANSWER. Write a concise response in the user's language, mirroring it \
exactly (script, diacritics, register). Keep the structure tight — no \
preamble, no \"I will now search…\", no closing offers like \"let me know if \
you want more details\". Format with Markdown only when it aids clarity \
(headings, lists, code fences). Numbers, units, formulas, and proper nouns \
are copied verbatim from the evidence — do not round, paraphrase, or \
translate them.

Preserve specialized terminology from the evidence verbatim. When the \
evidence classifies the subject with a technical term (function type, data \
structure, category, normalization condition, invariant), carry that exact \
term into the answer rather than substituting a looser everyday word. The \
evidence's framing is the authority.

ANSWER SCAFFOLDS PER INTENT

Silently identify the question's intent from one of the classes below and \
apply the matching scaffold. Beats are structural guidance, not literal \
headings — write in prose.

DEFINITION (\"what is X?\", \"define X\", \"ce este X?\"): three beats —
1. What X is (category or type as named by the evidence);
2. Core property / invariant (shape, range, constraint, normalization, \
   identity);
3. Closest related variant when the evidence emphasizes one (skippable).

HOW-TO / PROCEDURE (\"how do I apply X?\", \"explain how to X\", \"cum \
aplic X?\"): four beats —
1. What the method IS in one clause (same category framing as DEFINITION \
   beat 1);
2. The steps as the evidence describes them, keeping the evidence's \
   formulas, variable names, and quantified ranges verbatim;
3. Equivalences, alternative formulations, or generalizations mentioned \
   in the evidence (e.g. \"equivalent to method M\", \"generalizes to K \
   classes\", \"same as Fisher's criterion\"). When the evidence names \
   any, include them; omitting them misrepresents the evidence's scope;
4. Implementation or efficiency notes when the evidence provides them — \
   iterative recurrences, closed-form simplifications, complexity hints, \
   practical caveats.

COMPARISON (\"X vs Y\", \"how does X differ from Y?\"): three symmetric \
beats per item plus a one-clause summary of the relation (stricter than, \
equivalent to, special case of).

LOOKUP (\"what does section 2.3 say?\", \"what is the code for X?\"): \
direct quote + citation; no scaffold.

OTHER: answer directly from the evidence without a template.

COVERAGE DISCIPLINE

For HOW-TO and COMPARISON intents, scan the provided evidence chunks for \
facets that extend beyond the minimum answer. Include every such facet \
that the evidence discusses prominently:
  - equivalences with other named methods;
  - generalizations (e.g. 2-class → K-class, scalar → vector, 1D → N-D);
  - alternative formulations or dual criteria;
  - implementation optimizations (recurrences, precomputations, \
    closed-form simplifications);
  - practical caveats, limits, or known failure modes.

Dropping an evidence-supported facet from a how-to answer is a coverage \
defect: the user asked how to apply the method, and the evidence provided \
the surrounding picture. Only omit a facet when (a) it directly \
contradicts the main answer and no reconciliation is possible, or (b) the \
evidence merely alludes without explanation. Every included facet carries \
a citation [n]; unsupported elaboration is hallucination.

CITATIONS

Every factual claim must end with a citation `[cite:<chunk_id>]` where \
`<chunk_id>` is the canonical id shown in each evidence entry's metadata \
header (e.g. `[cite:c-1a2b3c4d]`). Cite only chunks that actually support \
the claim — never cite a chunk because it was retrieved. Group multiple \
citations with `[cite:id1][cite:id2]` when relevant. If you cannot cite a \
specific claim, omit it.

Legacy numeric citations `[N]` are still accepted during transition but \
`[cite:<chunk_id>]` is strongly preferred — it remains stable when the \
retrieval order changes between tool calls.

WHEN EVIDENCE IS INSUFFICIENT

Say plainly: \"The retrieved evidence does not cover this.\" Do not fabricate, \
extrapolate, or synthesize from general knowledge. An honest short answer is \
always better than a long guess.

MEMORY

Use `remember(action='save', key=..., value=...)` when the user explicitly \
asks you to remember something, or when you've done non-trivial synthesis that \
future turns on the same topic would reuse. Do not proactively save every \
answer.

STYLE

- No hedging words (\"I think\", \"probably\") unless the evidence itself hedges.
- No apologies for the tool's output.
- No outline of what you're about to do.
- End when the answer ends.
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

Scope admissions and pivots:
- An answer that EXPLICITLY states the queried subject is not present in the \
evidence (patterns like \"not covered in the knowledge base\", \"the evidence \
does not mention X\", \"nu apare în evidență\", \"does not contain\") is a \
SCOPE ADMISSION, not a hallucinated claim about X. Do NOT count the \
admission itself as an ungrounded_claim.
- When the answer follows such an admission with content about a RELATED \
topic Y that IS in the evidence, evaluate only the Y-content for grounding. \
If every Y-claim is supported, is_grounded=true with high confidence.
- If the answer says \"no evidence found\" and gives no further claims, \
is_grounded=true, confidence=1.0 (the honest refusal is itself grounded).
";

/// Build the `COVERAGE_PLAN` prompt block from an optional [`CoveragePlan`].
///
/// Returns `None` when the plan is absent or its mode is [`CoverageMode::None`]
/// (i.e., default single-call discipline applies). Returns `Some(block)` when
/// the mode is `Exhaustive` or `Partial`, in which case the block is appended
/// to the system/user prompt so the model can copy section_ids verbatim into
/// its `search_sections` call.
///
/// # Examples
///
/// ```rust
/// use agent_rs_agent::prompts::build_coverage_plan_block;
/// let block = build_coverage_plan_block(None);
/// assert!(block.is_none());
/// ```
pub fn build_coverage_plan_block(plan: Option<&CoveragePlan>) -> Option<String> {
    let plan = plan?;
    if plan.mode == CoverageMode::None {
        return None;
    }
    if plan.target_sections.is_empty() {
        return None;
    }
    let lines: Vec<String> = plan
        .target_sections
        .iter()
        .map(|s| format!("  - section_id: {}  (heading: \"{}\")", s.section_id, s.heading))
        .collect();
    Some(format!(
        "\n\n# COVERAGE_PLAN (mode: {:?}, {} sections)\n\n\
         You MUST cover the following sections via a SINGLE search_sections \
         call:\n\n{}\n",
        plan.mode,
        plan.target_sections.len(),
        lines.join("\n")
    ))
}

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

#[cfg(test)]
mod tests_coverage_plan_in_prompt {
    use super::*;
    use crate::coverage::{CoverageMode, CoveragePlan, SectionRef};

    #[test]
    fn coverage_plan_block_omitted_when_no_plan() {
        let block = build_coverage_plan_block(None);
        assert!(block.is_none());
    }

    #[test]
    fn coverage_plan_block_omitted_when_mode_none() {
        let plan = CoveragePlan::default();
        let block = build_coverage_plan_block(Some(&plan));
        assert!(block.is_none());
    }

    #[test]
    fn coverage_plan_block_omitted_when_mode_none_even_with_sections() {
        let plan = CoveragePlan {
            mode: CoverageMode::None,
            target_sections: vec![SectionRef {
                section_id: "s1".into(),
                doc_id: "d".into(),
                heading: "Intro".into(),
            }],
        };
        let block = build_coverage_plan_block(Some(&plan));
        assert!(block.is_none());
    }

    #[test]
    fn coverage_plan_block_omitted_when_exhaustive_but_no_sections() {
        let plan = CoveragePlan {
            mode: CoverageMode::Exhaustive,
            target_sections: vec![],
        };
        let block = build_coverage_plan_block(Some(&plan));
        assert!(block.is_none());
    }

    #[test]
    fn coverage_plan_block_lists_section_ids_and_headings() {
        let plan = CoveragePlan {
            mode: CoverageMode::Exhaustive,
            target_sections: vec![
                SectionRef {
                    section_id: "s1".into(),
                    doc_id: "d".into(),
                    heading: "5.1 Foo".into(),
                },
                SectionRef {
                    section_id: "s2".into(),
                    doc_id: "d".into(),
                    heading: "5.2 Bar".into(),
                },
            ],
        };
        let block = build_coverage_plan_block(Some(&plan)).expect("plan present");
        assert!(block.contains("s1"), "block must contain s1");
        assert!(block.contains("5.1 Foo"), "block must contain heading 5.1 Foo");
        assert!(block.contains("s2"), "block must contain s2");
        assert!(block.contains("5.2 Bar"), "block must contain heading 5.2 Bar");
        assert!(block.contains("search_sections"), "block must mention search_sections");
        assert!(block.to_lowercase().contains("must"), "block must contain 'must'");
        assert!(block.contains("2 sections"), "block must state section count");
    }

    #[test]
    fn coverage_plan_block_partial_mode_included() {
        let plan = CoveragePlan {
            mode: CoverageMode::Partial,
            target_sections: vec![SectionRef {
                section_id: "s3".into(),
                doc_id: "d".into(),
                heading: "Ch3".into(),
            }],
        };
        let block = build_coverage_plan_block(Some(&plan)).expect("partial plan should produce block");
        assert!(block.contains("s3"));
        assert!(block.contains("Partial"));
    }
}
