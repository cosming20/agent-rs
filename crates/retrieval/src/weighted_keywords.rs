//! LLM-backed weighted-keyword extractor and pure-function consumer helpers.
//!
//! From an input query, the extractor produces a ranked, weight-normalized
//! view of information-bearing terms. The same value feeds three consumers:
//! BM25 query rewrite, post-rerank coverage-ratio adjustment, and
//! overlap-first entity-linker candidate ordering.
//!
//! See `docs/superpowers/specs/2026-04-17-weighted-keywords-design.md`.
//!
//! All failure paths collapse to [`WeightedKeywords::empty`], and every
//! consumer's empty-input branch is byte-identical to pre-feature behaviour.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use backon::{ExponentialBuilder, Retryable};
use regex::Regex;
use reqwest::StatusCode;
use tracing::{instrument, warn};
use unicode_normalization::UnicodeNormalization;

use crate::error::RetrievalError;

// ---------------------------------------------------------------------------
// Public output types
// ---------------------------------------------------------------------------

/// Ranked, weight-normalized view of a query's information-bearing terms.
///
/// Stable public type — future enhancements (per-sub-query refinement)
/// add new methods on [`WeightedKeywordExtractor`] rather than widening
/// this type.
///
/// # Invariants
///
/// - Every [`WeightedTerm::weight`] is in `(0.0, 1.0]`.
/// - If non-empty, the maximum weight equals exactly `1.0` (linear
///   renormalisation against the emitted max).
/// - Sorted by weight descending; stable on ties.
/// - Deduplicated by NFKC + case-folded `term`; on collision the
///   higher-weight entry wins.
/// - Every `term` is non-empty after normalisation.
///
/// # Examples
///
/// ```
/// use agent_rs_retrieval::weighted_keywords::{WeightedKeywords, WeightedTerm};
/// let kw = WeightedKeywords::from_raw(vec![
///     WeightedTerm { term: "foo".into(), weight: 0.4, is_phrase: false },
///     WeightedTerm { term: "bar".into(), weight: 0.8, is_phrase: false },
/// ]);
/// // Max weight after renormalisation is exactly 1.0.
/// assert!((kw.terms[0].weight - 1.0).abs() < 1e-6);
/// assert_eq!(kw.terms[0].term, "bar");
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WeightedKeywords {
    /// Ranked terms; see struct-level invariants.
    pub terms: Vec<WeightedTerm>,
}

/// One weighted keyword entry.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedTerm {
    /// Surface form after NFKC normalisation and trimming. Guaranteed
    /// non-empty.
    pub term: String,
    /// Ordinal importance score in `(0.0, 1.0]`. **Not a calibrated
    /// probability, confidence, or corpus-aware relevance estimate** —
    /// only ordering and relative magnitude are meaningful.
    pub weight: f32,
    /// True when this entry is a multi-token phrase to be treated as a
    /// contiguous token group by downstream consumers.
    pub is_phrase: bool,
}

impl WeightedKeywords {
    /// The safe-fallback empty value.
    ///
    /// All consumer empty-input paths are byte-identical to pre-feature
    /// behaviour when they receive this.
    pub fn empty() -> Self {
        Self { terms: Vec::new() }
    }

    /// Sum of all weights.
    pub fn total_weight(&self) -> f32 {
        self.terms.iter().map(|t| t.weight).sum()
    }

    /// First `k` entries (already sorted by weight descending).
    pub fn top_k(&self, k: usize) -> &[WeightedTerm] {
        let n = self.terms.len().min(k);
        &self.terms[..n]
    }

    /// Convenience iterator over terms.
    pub fn iter(&self) -> impl Iterator<Item = &WeightedTerm> {
        self.terms.iter()
    }

    /// Normalise a raw vector of [`WeightedTerm`] into a canonical
    /// `WeightedKeywords`, enforcing all struct-level invariants.
    ///
    /// Steps:
    /// 1. NFC-normalise and trim each `term`; drop entries whose normalised
    ///    term is empty.
    /// 2. Drop entries whose weight is NaN, infinite, or ≤ 0.
    /// 3. Deduplicate by lowercased `term`; on collision keep the higher
    ///    weight and carry its `is_phrase`.
    /// 4. Renormalise so the max weight equals exactly `1.0`.
    /// 5. Stable-sort by weight descending.
    pub fn from_raw(raw: Vec<WeightedTerm>) -> Self {
        let mut cleaned: Vec<WeightedTerm> = raw
            .into_iter()
            .filter_map(|t| {
                let n: String = t.term.nfc().collect::<String>().trim().to_owned();
                if n.is_empty() {
                    return None;
                }
                if !t.weight.is_finite() || t.weight <= 0.0 {
                    return None;
                }
                Some(WeightedTerm {
                    term: n,
                    weight: t.weight,
                    is_phrase: t.is_phrase,
                })
            })
            .collect();

        // Dedupe by lowercased term: sort, then fold duplicates.
        cleaned.sort_by_key(|a| a.term.to_lowercase());
        let mut deduped: Vec<WeightedTerm> = Vec::with_capacity(cleaned.len());
        for t in cleaned {
            match deduped.last_mut() {
                Some(prev) if prev.term.to_lowercase() == t.term.to_lowercase() => {
                    if t.weight > prev.weight {
                        prev.weight = t.weight;
                        prev.is_phrase = t.is_phrase;
                        prev.term = t.term;
                    }
                }
                _ => deduped.push(t),
            }
        }

        if deduped.is_empty() {
            return Self::empty();
        }

        // Renormalise so max == 1.0.
        let max = deduped.iter().map(|t| t.weight).fold(f32::MIN, f32::max);
        if max > 0.0 && (max - 1.0).abs() > 1e-6 {
            for t in &mut deduped {
                t.weight /= max;
            }
        }

        // Stable sort by weight descending.
        deduped.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Self { terms: deduped }
    }
}

// ---------------------------------------------------------------------------
// Extractor configuration
// ---------------------------------------------------------------------------

/// Typed configuration for [`WeightedKeywordExtractor`].
#[derive(Debug, Clone)]
pub struct WeightedKeywordExtractorConfig {
    /// OpenAI-compatible API key (e.g. `sk-...`).
    pub api_key: String,
    /// Model identifier. Default matches `query_rewriter` for prompt-cache
    /// alignment.
    pub model: String,
    /// Max retries on transient failure (429 / 5xx).
    pub max_retries: u32,
    /// Per-call LLM timeout.
    pub timeout: Duration,
    /// Override endpoint (tests only). Production uses OpenAI's chat
    /// completions URL when this is `None`.
    pub endpoint: Option<String>,
}

impl WeightedKeywordExtractorConfig {
    /// Construct with default model, retries, and timeout.
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: DEFAULT_MODEL.to_owned(),
            max_retries: 2,
            timeout: Duration::from_secs(10),
            endpoint: None,
        }
    }
}

const DEFAULT_MODEL: &str = "gpt-5.4-mini-2026-03-17";
const OPENAI_CHAT_URL: &str = "https://api.openai.com/v1/chat/completions";

// ---------------------------------------------------------------------------
// Cross-consumer integration config
// ---------------------------------------------------------------------------

/// Per-consumer integration toggles. `None` disables that consumer without
/// touching the extractor.
#[derive(Debug, Clone)]
pub struct WeightedKeywordIntegrationConfig {
    pub bm25: Option<BM25WeightingConfig>,
    pub rerank: Option<RerankAdjustmentConfig>,
    pub entity_linker: Option<EntityLinkerPrioritizationConfig>,
}

impl Default for WeightedKeywordIntegrationConfig {
    fn default() -> Self {
        Self {
            bm25: Some(BM25WeightingConfig::default()),
            rerank: Some(RerankAdjustmentConfig::default()),
            entity_linker: Some(EntityLinkerPrioritizationConfig::default()),
        }
    }
}

/// Config for BM25 query rewrite.
#[derive(Debug, Clone)]
pub struct BM25WeightingConfig {
    /// Drop terms whose weight is strictly less than this.
    pub min_weight: f32,
    /// Per-token repetition cap applied after cross-term aggregation.
    pub max_repeats: u32,
    /// Minimum Unicode-codepoint count for a single-token term to be kept.
    /// Language-agnostic — drops short tokens by length alone. Phrase terms
    /// are exempt at the phrase level; this applies to tokens produced by
    /// phrase expansion.
    pub min_term_chars: usize,
    /// When `true`, phrase terms are emitted as both the full phrase (once)
    /// and component tokens at reduced repetition.
    pub phrase_token_expansion: bool,
}

impl Default for BM25WeightingConfig {
    fn default() -> Self {
        Self {
            min_weight: 0.3,
            max_repeats: 3,
            min_term_chars: 3,
            phrase_token_expansion: true,
        }
    }
}

/// Config for the rerank coverage-ratio adjustment.
#[derive(Debug, Clone)]
pub struct RerankAdjustmentConfig {
    /// Blend factor — maximum fractional boost when coverage is 1.0.
    pub alpha: f32,
    /// Multiplier applied to phrase-term weights in the coverage ratio;
    /// rewards contiguous phrase matches over equivalent-mass token
    /// matches. Default ≥ 1.0.
    pub phrase_bonus: f32,
}

impl Default for RerankAdjustmentConfig {
    fn default() -> Self {
        Self {
            alpha: 0.2,
            phrase_bonus: 1.5,
        }
    }
}

/// Config for entity-linker overlap-first prioritisation.
#[derive(Debug, Clone)]
pub struct EntityLinkerPrioritizationConfig {
    /// Max fallback-probe count in tier 3.
    pub top_k: usize,
}

impl Default for EntityLinkerPrioritizationConfig {
    fn default() -> Self {
        Self { top_k: 5 }
    }
}

// ---------------------------------------------------------------------------
// Extractor
// ---------------------------------------------------------------------------

/// LLM-backed extractor producing [`WeightedKeywords`] from a user query.
pub struct WeightedKeywordExtractor {
    config: WeightedKeywordExtractorConfig,
    http: Arc<reqwest::Client>,
}

impl WeightedKeywordExtractor {
    /// Construct an extractor with the given config and HTTP client.
    pub fn new(config: WeightedKeywordExtractorConfig, http: Arc<reqwest::Client>) -> Self {
        Self { config, http }
    }

    /// Wrap in an `Arc` — convenience for the router, which shares one
    /// extractor across all requests.
    pub fn into_arc(self) -> Arc<Self> {
        Arc::new(self)
    }

    /// Extract weighted keywords for the given query.
    ///
    /// **Failure semantics:** every failure mode (HTTP error, timeout,
    /// malformed JSON, exhausted retries) collapses to
    /// `Ok(WeightedKeywords::empty())` with a `tracing::warn!` — the
    /// retrieval pipeline is never blocked by this component.
    ///
    /// # Errors
    ///
    /// Returns `Ok(...)` in all practical paths; the `Result` is preserved
    /// for forward-compat with future hard-error modes (e.g. invalid
    /// config detected at call time).
    #[instrument(skip(self, query), fields(query_hash = %query_hash(query)))]
    pub async fn extract(&self, query: &str) -> Result<WeightedKeywords, RetrievalError> {
        let endpoint = self.config.endpoint.as_deref().unwrap_or(OPENAI_CHAT_URL);

        let body = serde_json::json!({
            "model": self.config.model,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "weighted_keywords",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "terms": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": false,
                                    "properties": {
                                        "term": {"type": "string"},
                                        "weight": {"type": "number"},
                                        "is_phrase": {"type": "boolean"}
                                    },
                                    "required": ["term", "weight", "is_phrase"]
                                }
                            }
                        },
                        "required": ["terms"]
                    }
                }
            },
            "messages": [
                {"role": "system", "content": WEIGHTED_KEYWORDS_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        });

        let call = || async {
            let resp = self
                .http
                .post(endpoint)
                .bearer_auth(&self.config.api_key)
                .json(&body)
                .timeout(self.config.timeout)
                .send()
                .await
                .map_err(ExtractError::Http)?;
            let status = resp.status();
            if !status.is_success() {
                let body = resp.text().await.unwrap_or_default();
                return Err(ExtractError::Status { status, body });
            }
            let full: serde_json::Value = resp.json().await.map_err(ExtractError::Http)?;
            let content = full
                .pointer("/choices/0/message/content")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    ExtractError::Schema("missing choices[0].message.content".into())
                })?;
            #[derive(serde::Deserialize)]
            struct Raw {
                terms: Vec<RawTerm>,
            }
            #[derive(serde::Deserialize)]
            struct RawTerm {
                term: String,
                weight: f32,
                is_phrase: bool,
            }
            let raw: Raw = serde_json::from_str(content).map_err(ExtractError::Json)?;
            let terms = raw
                .terms
                .into_iter()
                .map(|r| WeightedTerm {
                    term: r.term,
                    weight: r.weight,
                    is_phrase: r.is_phrase,
                })
                .collect();
            Ok::<_, ExtractError>(WeightedKeywords::from_raw(terms))
        };

        let backoff = ExponentialBuilder::default()
            .with_max_times(self.config.max_retries as usize);

        match call.retry(backoff).when(is_retryable).await {
            Ok(kw) => Ok(kw),
            Err(e) => {
                warn!(error = %e, "weighted keyword extraction failed; returning empty");
                Ok(WeightedKeywords::empty())
            }
        }
    }
}

fn is_retryable(e: &ExtractError) -> bool {
    match e {
        ExtractError::Http(_) => true,
        ExtractError::Status { status, .. } => {
            *status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
        }
        ExtractError::Json(_) | ExtractError::Schema(_) => false,
    }
}

#[derive(Debug, thiserror::Error)]
enum ExtractError {
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),
    #[error("http status {status}: {body}")]
    Status { status: StatusCode, body: String },
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("schema: {0}")]
    Schema(String),
}

fn query_hash(q: &str) -> String {
    let h = blake3::hash(q.as_bytes());
    h.to_hex()[..12].to_owned()
}

const WEIGHTED_KEYWORDS_SYSTEM_PROMPT: &str = r#"You are an expert information-retrieval query analyst.

Your task: from a user query, produce a ranked list of the most
information-bearing terms and short phrases, with an importance weight on
each.

Rules:
1. Return at most 8 terms total.
2. For each term emit: { "term": string, "weight": number, "is_phrase": bool }.
3. Weight is an ORDINAL importance score in (0, 1] — relative importance,
   not a probability.
4. Prefer rare / discriminative / content-bearing terms (names, identifiers,
   domain-specific vocabulary, numbered references) over common function
   words.
5. Use phrases (is_phrase: true) when a multi-word group carries meaning
   that the individual tokens do not (e.g. "connection timeout",
   "fourth edition"). Single words use is_phrase: false.
6. Preserve the user's language; do not translate.
7. Do not assume any specific domain; rank purely by information content
   inside the given query.

Return ONLY JSON matching the schema. No other text."#;

// ---------------------------------------------------------------------------
// BM25 query rewrite (consumer #1)
// ---------------------------------------------------------------------------

/// Construct a BM25 query string from an original user query and its
/// weighted-keyword view.
///
/// Pure function — no I/O, no state.
///
/// - Empty keywords → returns the original query unchanged.
/// - Otherwise: filters by `min_weight` and `min_term_chars`, preserves
///   phrases intact, optionally expands phrases to component tokens at
///   half repetition, aggregates per-token counts, and clamps to
///   `max_repeats`.
///
/// See spec §7.1.
///
/// # Examples
///
/// ```
/// use agent_rs_retrieval::weighted_keywords::{
///     BM25WeightingConfig, WeightedKeywords, WeightedTerm, weighted_bm25_query,
/// };
/// let kw = WeightedKeywords::from_raw(vec![
///     WeightedTerm { term: "compiler".into(), weight: 0.9, is_phrase: false },
/// ]);
/// let rewritten = weighted_bm25_query("anything", &kw, &BM25WeightingConfig::default());
/// assert!(rewritten.contains("compiler"));
/// ```
pub fn weighted_bm25_query(
    original_query: &str,
    kw: &WeightedKeywords,
    cfg: &BM25WeightingConfig,
) -> String {
    if kw.terms.is_empty() {
        return original_query.to_owned();
    }

    let mut per_token: BTreeMap<String, u32> = BTreeMap::new();
    let mut intact_phrases: Vec<String> = Vec::new();

    for t in &kw.terms {
        if t.weight < cfg.min_weight {
            continue;
        }

        if t.is_phrase {
            intact_phrases.push(t.term.clone());
            if cfg.phrase_token_expansion {
                let rep = ((t.weight * cfg.max_repeats as f32) / 2.0).ceil() as u32;
                let rep = rep.max(1);
                for token in t.term.split_whitespace() {
                    if token.chars().count() < cfg.min_term_chars {
                        continue;
                    }
                    *per_token.entry(token.to_owned()).or_insert(0) += rep;
                }
            }
        } else {
            if t.term.chars().count() < cfg.min_term_chars {
                continue;
            }
            let rep = (t.weight * cfg.max_repeats as f32).ceil() as u32;
            let rep = rep.max(1);
            *per_token.entry(t.term.clone()).or_insert(0) += rep;
        }
    }

    // Clamp per-token aggregated counts at max_repeats.
    for count in per_token.values_mut() {
        if *count > cfg.max_repeats {
            *count = cfg.max_repeats;
        }
    }

    // Intact phrases first (so a downstream BM25 tokenizer sees them as
    // consecutive terms), then aggregated single tokens.
    let mut parts: Vec<String> = intact_phrases;
    for (token, count) in per_token {
        for _ in 0..count {
            parts.push(token.clone());
        }
    }
    parts.join(" ")
}

// ---------------------------------------------------------------------------
// Rerank coverage-ratio adjustment (consumer #2)
// ---------------------------------------------------------------------------

/// Adjust a cross-encoder reranker score by how well `candidate_text`
/// covers the user's weighted keywords.
///
/// Pure function — no I/O. See spec §7.2.
///
/// Output is bounded in `[reranker_score, reranker_score * (1 + alpha)]`.
///
/// # Examples
///
/// ```
/// use agent_rs_retrieval::weighted_keywords::{
///     RerankAdjustmentConfig, WeightedKeywords, WeightedTerm, adjust_for_weighted_keywords,
/// };
/// let kw = WeightedKeywords::from_raw(vec![
///     WeightedTerm { term: "foo".into(), weight: 1.0, is_phrase: false },
/// ]);
/// let adjusted = adjust_for_weighted_keywords(
///     1.0, "some foo here", &kw, &RerankAdjustmentConfig::default(),
/// );
/// assert!(adjusted > 1.0);
/// ```
pub fn adjust_for_weighted_keywords(
    reranker_score: f32,
    candidate_text: &str,
    kw: &WeightedKeywords,
    cfg: &RerankAdjustmentConfig,
) -> f32 {
    if kw.terms.is_empty() || cfg.alpha == 0.0 {
        return reranker_score;
    }
    let text_lc = candidate_text.to_lowercase();
    let mut matched = 0.0_f32;
    let mut total = 0.0_f32;
    for t in &kw.terms {
        let bonus = if t.is_phrase { cfg.phrase_bonus } else { 1.0 };
        let mass = t.weight * bonus;
        total += mass;
        let term_lc = t.term.to_lowercase();
        let hit = if t.is_phrase {
            phrase_matches(&text_lc, &term_lc)
        } else {
            single_token_matches(&text_lc, &term_lc)
        };
        if hit {
            matched += mass;
        }
    }
    if total <= 0.0 {
        return reranker_score;
    }
    let coverage = (matched / total).clamp(0.0, 1.0);
    reranker_score * (1.0 + cfg.alpha * coverage)
}

/// Unicode word-boundary match for a single-token term. `regex`'s default
/// `\b` is Unicode-aware, handling Latin, Cyrillic, Greek, Arabic, and
/// CJK uniformly.
fn single_token_matches(text_lc: &str, token_lc: &str) -> bool {
    let escaped = regex::escape(token_lc);
    let pattern = format!(r"(?u)\b{}\b", escaped);
    Regex::new(&pattern)
        .map(|r| r.is_match(text_lc))
        .unwrap_or(false)
}

/// Contiguous token-sequence match for a phrase, boundary-aware at the
/// phrase edges, tolerant of variable internal whitespace.
fn phrase_matches(text_lc: &str, phrase_lc: &str) -> bool {
    let escaped = regex::escape(phrase_lc);
    // `regex::escape` emits `\ ` for space; relax to `\s+` so any internal
    // whitespace in the candidate matches the phrase.
    let normalised = escaped.replace("\\ ", r"\s+");
    let pattern = format!(r"(?u)\b{}\b", normalised);
    Regex::new(&pattern)
        .map(|r| r.is_match(text_lc))
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Entity-linker overlap-first tiering (consumer #3)
// ---------------------------------------------------------------------------

/// Order entity-lookup candidates using overlap-first precedence.
///
/// Three tiers:
/// 1. Weighted keywords that also appear among the LLM-extracted entity
///    candidates — highest priority, ordered by weight descending.
/// 2. Extracted entity candidates not present in tier 1, original order.
/// 3. Remaining weighted keywords, top `cfg.top_k` by weight.
///
/// See spec §7.3.
///
/// # Examples
///
/// ```
/// use agent_rs_retrieval::weighted_keywords::{
///     EntityLinkerPrioritizationConfig, WeightedKeywords, WeightedTerm,
///     order_candidates_overlap_first,
/// };
/// let kw = WeightedKeywords::from_raw(vec![
///     WeightedTerm { term: "webhook".into(), weight: 0.9, is_phrase: false },
///     WeightedTerm { term: "retry".into(), weight: 0.7, is_phrase: false },
/// ]);
/// let extracted = vec!["webhook".to_owned(), "delivery".to_owned()];
/// let cfg = EntityLinkerPrioritizationConfig { top_k: 5 };
/// let ordered = order_candidates_overlap_first(&kw, &extracted, &cfg);
/// assert_eq!(ordered[0], "webhook");
/// ```
pub fn order_candidates_overlap_first(
    kw: &WeightedKeywords,
    extracted: &[String],
    cfg: &EntityLinkerPrioritizationConfig,
) -> Vec<String> {
    use std::collections::HashSet;

    let extracted_lc: HashSet<String> = extracted.iter().map(|e| e.to_lowercase()).collect();

    // Tier 1: intersection, weight-sorted (kw.terms is already sorted).
    let tier1: Vec<String> = kw
        .iter()
        .filter(|t| extracted_lc.contains(&t.term.to_lowercase()))
        .map(|t| t.term.clone())
        .collect();

    let tier1_lc: HashSet<String> = tier1.iter().map(|s| s.to_lowercase()).collect();

    // Tier 2: extracted entries not in tier 1, original order.
    let tier2: Vec<String> = extracted
        .iter()
        .filter(|e| !tier1_lc.contains(&e.to_lowercase()))
        .cloned()
        .collect();

    // Tier 3: remaining weighted keywords, capped at top_k.
    let tier3: Vec<String> = kw
        .iter()
        .filter(|t| !extracted_lc.contains(&t.term.to_lowercase()))
        .map(|t| t.term.clone())
        .take(cfg.top_k)
        .collect();

    let mut out = tier1;
    out.extend(tier2);
    out.extend(tier3);
    out
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(term: &str, w: f32, is_phrase: bool) -> WeightedTerm {
        WeightedTerm {
            term: term.to_owned(),
            weight: w,
            is_phrase,
        }
    }

    // ----- WeightedKeywords::from_raw -----

    #[test]
    fn from_raw_normalises_weights_so_max_is_one() {
        let kw = WeightedKeywords::from_raw(vec![raw("a", 0.2, false), raw("b", 0.4, false)]);
        let max = kw.terms.iter().map(|t| t.weight).fold(0f32, f32::max);
        assert!((max - 1.0).abs() < 1e-6);
    }

    #[test]
    fn from_raw_sorts_by_weight_descending() {
        let kw = WeightedKeywords::from_raw(vec![
            raw("low", 0.1, false),
            raw("high", 0.9, false),
            raw("mid", 0.5, false),
        ]);
        let ws: Vec<f32> = kw.terms.iter().map(|t| t.weight).collect();
        for pair in ws.windows(2) {
            assert!(pair[0] >= pair[1]);
        }
    }

    #[test]
    fn from_raw_deduplicates_case_insensitively() {
        let kw = WeightedKeywords::from_raw(vec![
            raw("Foo", 0.5, false),
            raw("foo", 0.8, false),
            raw("FOO", 0.3, false),
        ]);
        assert_eq!(kw.terms.len(), 1);
        assert_eq!(kw.terms[0].term.to_lowercase(), "foo");
    }

    #[test]
    fn from_raw_drops_empty_or_whitespace_terms() {
        let kw = WeightedKeywords::from_raw(vec![
            raw("", 0.9, false),
            raw("   ", 0.8, false),
            raw("actual", 0.5, false),
        ]);
        assert_eq!(kw.terms.len(), 1);
        assert_eq!(kw.terms[0].term, "actual");
    }

    #[test]
    fn from_raw_drops_invalid_weights() {
        let kw = WeightedKeywords::from_raw(vec![
            raw("neg", -0.5, false),
            raw("zero", 0.0, false),
            raw("nan", f32::NAN, false),
            raw("inf", f32::INFINITY, false),
            raw("ok", 0.7, false),
        ]);
        let names: Vec<String> = kw.terms.iter().map(|t| t.term.clone()).collect();
        assert_eq!(names, vec!["ok".to_owned()]);
        assert!((kw.terms[0].weight - 1.0).abs() < 1e-6);
    }

    #[test]
    fn empty_round_trip() {
        let kw = WeightedKeywords::empty();
        assert!(kw.terms.is_empty());
        assert_eq!(kw.total_weight(), 0.0);
        assert!(kw.top_k(5).is_empty());
    }

    #[test]
    fn top_k_respects_cap() {
        let kw = WeightedKeywords::from_raw(vec![
            raw("a", 0.9, false),
            raw("b", 0.8, false),
            raw("c", 0.7, false),
        ]);
        assert_eq!(kw.top_k(2).len(), 2);
        assert_eq!(kw.top_k(10).len(), 3);
    }

    // ----- weighted_bm25_query -----

    #[test]
    fn bm25_empty_keywords_returns_original() {
        let q = "what is a monad";
        let out = weighted_bm25_query(q, &WeightedKeywords::empty(), &BM25WeightingConfig::default());
        assert_eq!(out, q);
    }

    #[test]
    fn bm25_filters_below_min_weight() {
        let kw = WeightedKeywords::from_raw(vec![
            raw("important", 0.9, false),
            raw("noise", 0.1, false),
        ]);
        let cfg = BM25WeightingConfig {
            min_weight: 0.3,
            ..Default::default()
        };
        let out = weighted_bm25_query("_", &kw, &cfg);
        assert!(out.contains("important"));
        assert!(!out.contains("noise"));
    }

    #[test]
    fn bm25_respects_min_term_chars_for_single_tokens() {
        // After renormalisation both become weight 1.0 (same max). We set
        // min_weight to 0.0 so both pass that filter and only length matters.
        let kw = WeightedKeywords::from_raw(vec![raw("ok", 0.8, false), raw("fine", 0.8, false)]);
        let cfg = BM25WeightingConfig {
            min_term_chars: 3,
            min_weight: 0.0,
            ..Default::default()
        };
        let out = weighted_bm25_query("_", &kw, &cfg);
        assert!(!out.split_whitespace().any(|t| t == "ok"));
        assert!(out.split_whitespace().any(|t| t == "fine"));
    }

    #[test]
    fn bm25_does_not_drop_common_english_function_words_by_hardcoded_list() {
        // Regression guard: no hardcoded stopword list. "the" and "and" with
        // high weight and len >= min_term_chars must still be emitted.
        let kw = WeightedKeywords::from_raw(vec![raw("the", 0.9, false), raw("and", 0.8, false)]);
        let cfg = BM25WeightingConfig {
            min_term_chars: 3,
            min_weight: 0.1,
            ..Default::default()
        };
        let out = weighted_bm25_query("_", &kw, &cfg);
        assert!(out.split_whitespace().any(|t| t == "the"));
        assert!(out.split_whitespace().any(|t| t == "and"));
    }

    #[test]
    fn bm25_phrase_preserved_when_expansion_false() {
        let kw = WeightedKeywords::from_raw(vec![raw("error code", 0.9, true)]);
        let cfg = BM25WeightingConfig {
            phrase_token_expansion: false,
            ..Default::default()
        };
        let out = weighted_bm25_query("_", &kw, &cfg);
        // The intact phrase appears exactly once; no standalone "error" or
        // "code" tokens beyond what's inside the phrase.
        assert!(out.contains("error code"));
        let after = out.replacen("error code", "", 1);
        assert!(!after.contains("error"));
        assert!(!after.contains("code"));
    }

    #[test]
    fn bm25_phrase_expanded_when_flag_true() {
        let kw = WeightedKeywords::from_raw(vec![raw("error code", 1.0, true)]);
        let cfg = BM25WeightingConfig {
            phrase_token_expansion: true,
            max_repeats: 3,
            min_term_chars: 3,
            min_weight: 0.0,
        };
        let out = weighted_bm25_query("_", &kw, &cfg);
        assert!(out.contains("error code"));
        let error_count = out.split_whitespace().filter(|t| *t == "error").count();
        let code_count = out.split_whitespace().filter(|t| *t == "code").count();
        // Phrase expansion at w=1.0, max_repeats=3 → rep = ceil(1*3/2) = 2.
        // Plus 1 occurrence inside the intact phrase.
        assert!((2..=3).contains(&error_count), "error_count = {error_count}");
        assert!((2..=3).contains(&code_count), "code_count = {code_count}");
    }

    #[test]
    fn bm25_token_level_aggregation_caps_at_max_repeats() {
        let kw = WeightedKeywords::from_raw(vec![
            raw("foo", 1.0, false),
            raw("foo bar", 1.0, true),
        ]);
        let cfg = BM25WeightingConfig {
            phrase_token_expansion: true,
            max_repeats: 3,
            min_term_chars: 3,
            min_weight: 0.0,
        };
        let out = weighted_bm25_query("_", &kw, &cfg);
        // Count standalone "foo" tokens only (not those inside the phrase).
        // The phrase "foo bar" contributes one "foo" inside it; single-token
        // aggregation caps that count separately.
        let all_foo = out.split_whitespace().filter(|t| *t == "foo").count();
        // Intact phrase adds 1; aggregated single-token is capped at 3.
        assert!(all_foo <= cfg.max_repeats as usize + 1);
    }

    // ----- adjust_for_weighted_keywords -----

    #[test]
    fn adjust_empty_is_noop() {
        let out = adjust_for_weighted_keywords(
            0.5,
            "anything",
            &WeightedKeywords::empty(),
            &RerankAdjustmentConfig::default(),
        );
        assert!((out - 0.5).abs() < 1e-6);
    }

    #[test]
    fn adjust_alpha_zero_is_noop() {
        let kw = WeightedKeywords::from_raw(vec![raw("foo", 0.9, false)]);
        let cfg = RerankAdjustmentConfig {
            alpha: 0.0,
            phrase_bonus: 1.5,
        };
        let out = adjust_for_weighted_keywords(0.5, "foo bar", &kw, &cfg);
        assert!((out - 0.5).abs() < 1e-6);
    }

    #[test]
    fn adjust_no_matches_returns_base_score() {
        let kw = WeightedKeywords::from_raw(vec![raw("foo", 0.9, false)]);
        let out = adjust_for_weighted_keywords(
            0.5,
            "nothing relevant",
            &kw,
            &RerankAdjustmentConfig::default(),
        );
        assert!((out - 0.5).abs() < 1e-6);
    }

    #[test]
    fn adjust_bounded_at_full_coverage() {
        let kw = WeightedKeywords::from_raw(vec![raw("foo", 0.9, false), raw("bar", 0.5, false)]);
        let cfg = RerankAdjustmentConfig {
            alpha: 0.2,
            phrase_bonus: 1.0,
        };
        let out = adjust_for_weighted_keywords(1.0, "foo bar and more", &kw, &cfg);
        // Full coverage → adjustment = 1.0 * (1 + 0.2 * 1.0) = 1.2
        assert!((out - 1.2).abs() < 1e-6);
    }

    #[test]
    fn adjust_word_boundary_not_substring() {
        let kw = WeightedKeywords::from_raw(vec![raw("art", 1.0, false)]);
        let cfg = RerankAdjustmentConfig {
            alpha: 0.2,
            phrase_bonus: 1.5,
        };
        // "article" must NOT match the query term "art".
        let out = adjust_for_weighted_keywords(1.0, "article twelve", &kw, &cfg);
        assert!((out - 1.0).abs() < 1e-6);
    }

    #[test]
    fn adjust_phrase_requires_contiguous_match() {
        let kw = WeightedKeywords::from_raw(vec![raw("error code", 1.0, true)]);
        let cfg = RerankAdjustmentConfig {
            alpha: 0.2,
            phrase_bonus: 1.5,
        };
        // Non-contiguous: no match.
        let out = adjust_for_weighted_keywords(1.0, "error then the code", &kw, &cfg);
        assert!((out - 1.0).abs() < 1e-6);
        // Contiguous: match.
        let out2 = adjust_for_weighted_keywords(1.0, "see error code 504", &kw, &cfg);
        assert!(out2 > 1.0);
    }

    // ----- order_candidates_overlap_first -----

    #[test]
    fn overlap_first_orders_intersection_before_others() {
        let kw = WeightedKeywords::from_raw(vec![
            raw("webhook", 0.9, false),
            raw("retry", 0.7, false),
            raw("timeout", 0.5, false),
        ]);
        let extracted = vec!["webhook".to_owned(), "delivery".to_owned()];
        let cfg = EntityLinkerPrioritizationConfig { top_k: 5 };
        let ordered = order_candidates_overlap_first(&kw, &extracted, &cfg);
        assert_eq!(ordered[0], "webhook");
        assert_eq!(ordered[1], "delivery");
        assert_eq!(ordered[2], "retry");
        assert_eq!(ordered[3], "timeout");
    }

    #[test]
    fn overlap_first_empty_weighted_preserves_extracted() {
        let extracted = vec!["a".to_owned(), "b".to_owned()];
        let cfg = EntityLinkerPrioritizationConfig { top_k: 5 };
        let ordered =
            order_candidates_overlap_first(&WeightedKeywords::empty(), &extracted, &cfg);
        assert_eq!(ordered, extracted);
    }

    #[test]
    fn overlap_first_caps_fallback_tier_at_top_k() {
        let kw = WeightedKeywords::from_raw(vec![
            raw("a", 1.0, false),
            raw("b", 0.9, false),
            raw("c", 0.8, false),
            raw("d", 0.7, false),
            raw("e", 0.6, false),
            raw("f", 0.5, false),
        ]);
        let extracted: Vec<String> = vec![];
        let cfg = EntityLinkerPrioritizationConfig { top_k: 3 };
        let ordered = order_candidates_overlap_first(&kw, &extracted, &cfg);
        assert_eq!(ordered.len(), 3);
        assert_eq!(ordered[0], "a");
        assert_eq!(ordered[2], "c");
    }

    // ----- Extractor (mock HTTP) -----

    #[tokio::test]
    async fn extract_parses_structured_response() {
        let mut server = mockito::Server::new_async().await;
        let body = r#"{
          "choices":[{"message":{"content":
            "{\"terms\":[{\"term\":\"compiler\",\"weight\":0.9,\"is_phrase\":false},{\"term\":\"static types\",\"weight\":0.6,\"is_phrase\":true}]}"
          }}]
        }"#;
        let _m = server
            .mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_body(body)
            .create_async()
            .await;

        let http = Arc::new(reqwest::Client::new());
        let mut cfg = WeightedKeywordExtractorConfig::new("sk-test".into());
        cfg.endpoint = Some(format!("{}/v1/chat/completions", server.url()));
        let ex = WeightedKeywordExtractor::new(cfg, http);

        let kw = ex
            .extract("is a compiler faster than an interpreter?")
            .await
            .unwrap();
        assert_eq!(kw.terms.len(), 2);
        assert_eq!(kw.terms[0].term, "compiler");
    }

    #[tokio::test]
    async fn extract_falls_back_to_empty_on_http_error() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("POST", "/v1/chat/completions")
            .with_status(500)
            .with_body("internal")
            .expect_at_least(1)
            .create_async()
            .await;

        let http = Arc::new(reqwest::Client::new());
        let mut cfg = WeightedKeywordExtractorConfig::new("sk-test".into());
        cfg.endpoint = Some(format!("{}/v1/chat/completions", server.url()));
        cfg.max_retries = 1;
        let ex = WeightedKeywordExtractor::new(cfg, http);

        let result = ex.extract("anything").await;
        assert!(result.is_ok());
        assert!(result.unwrap().terms.is_empty());
    }

    // ----- Property tests -----

    proptest::proptest! {
        #[test]
        fn bm25_empty_input_identity(q in "[a-z ]{0,50}") {
            let out = weighted_bm25_query(
                &q, &WeightedKeywords::empty(), &BM25WeightingConfig::default()
            );
            proptest::prop_assert_eq!(out, q);
        }

        #[test]
        fn bm25_no_token_exceeds_max_plus_one(
            terms in proptest::collection::vec(
                ("[a-z]{3,10}( [a-z]{3,10}){0,2}", 0.1f32..=1.0, proptest::bool::ANY),
                0..6
            )
        ) {
            let terms: Vec<WeightedTerm> = terms.into_iter()
                .map(|(term, weight, is_phrase)| WeightedTerm { term, weight, is_phrase })
                .collect();
            let kw = WeightedKeywords::from_raw(terms);
            let cfg = BM25WeightingConfig {
                max_repeats: 3, min_weight: 0.0, min_term_chars: 3,
                phrase_token_expansion: true,
            };
            let out = weighted_bm25_query("_", &kw, &cfg);
            let mut counts = std::collections::HashMap::<String, u32>::new();
            for w in out.split_whitespace() {
                *counts.entry(w.to_owned()).or_insert(0) += 1;
            }
            for (_tok, c) in counts {
                // Cap is per aggregated token; intact phrase can add at most
                // one additional occurrence, giving max_repeats + 1 worst case.
                proptest::prop_assert!(c <= cfg.max_repeats + 1);
            }
        }

        #[test]
        fn adjust_is_bounded(
            score in 0.0f32..5.0,
            alpha in 0.0f32..0.5,
            terms in proptest::collection::vec(
                ("[a-z]{3,10}", 0.1f32..=1.0, proptest::bool::ANY),
                0..4
            ),
            text in "[a-z ]{0,60}",
        ) {
            let terms: Vec<WeightedTerm> = terms.into_iter()
                .map(|(term, weight, is_phrase)| WeightedTerm { term, weight, is_phrase })
                .collect();
            let kw = WeightedKeywords::from_raw(terms);
            let cfg = RerankAdjustmentConfig { alpha, phrase_bonus: 1.5 };
            let out = adjust_for_weighted_keywords(score, &text, &kw, &cfg);
            proptest::prop_assert!(out >= score - 1e-4);
            proptest::prop_assert!(out <= score * (1.0 + alpha + 1e-4));
        }
    }
}
