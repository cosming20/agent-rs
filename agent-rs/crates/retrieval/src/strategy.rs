//! Strategy registry — maps [`RetrievalStrategy`] variants to their execution
//! descriptors.
//!
//! # Overview
//!
//! Each retrieval strategy specifies which search back-ends to engage, whether
//! those back-ends should run in parallel, and whether the query benefits from
//! context augmentation (i.e. graph or summary context injected into the
//! synthesis prompt).
//!
//! The registry is a pure function with no I/O; it is the single authoritative
//! source of truth for strategy → behaviour mapping.
//!
//! # Usage
//!
//! ```rust,no_run
//! use agent_rs_core::RetrievalStrategy;
//! use agent_rs_retrieval::strategy::get_descriptor;
//!
//! let desc = get_descriptor(RetrievalStrategy::GraphHybrid);
//! assert!(desc.parallel);
//! ```

use agent_rs_core::RetrievalStrategy;

// ---- SearchType ----

/// Identifies which storage back-end a strategy engages.
///
/// `Summaries` and `Temporal` are not separate back-ends — both route to
/// the vector store via specialised search functions. They are represented
/// here as `Vector` in `search_types`; the router inspects the original
/// [`RetrievalStrategy`] to select the correct vector search variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SearchType {
    /// Dense vector ANN search (Qdrant dense named vector).
    Vector,
    /// Sparse / BM25 lexical search (Qdrant sparse named vector).
    Keyword,
    /// Cypher graph traversal (Neo4j).
    Graph,
}

// ---- StrategyDescriptor ----

/// Describes how a [`RetrievalStrategy`] executes at runtime.
///
/// The router uses this descriptor to decide which search functions to call,
/// whether to fan them out concurrently, and whether to pass graph/summary
/// context to the LLM synthesis step.
#[derive(Debug, Clone)]
pub struct StrategyDescriptor {
    /// Ordered list of back-end types this strategy touches.
    ///
    /// When `parallel` is `true`, all listed back-ends are dispatched
    /// concurrently via `tokio::join!`. Single-element lists always have
    /// `parallel = false`.
    pub search_types: Vec<SearchType>,

    /// Whether the back-ends in `search_types` run concurrently.
    pub parallel: bool,

    /// Whether retrieved context should be injected into the synthesis prompt.
    ///
    /// `true` for strategies that build graph subgraphs or summaries that
    /// benefit from being surfaced explicitly in the final answer.
    pub context_augmented: bool,
}

// ---- StrategyRegistry ----

/// Zero-size token type that groups strategy descriptor lookups.
///
/// All methods are stateless and zero-cost. Use [`get_descriptor`] as a free
/// function if you do not need to name this type.
#[derive(Debug, Clone, Default)]
pub struct StrategyRegistry;

impl StrategyRegistry {
    /// Return the [`StrategyDescriptor`] for `strategy`.
    ///
    /// Delegates to [`get_descriptor`].
    #[inline]
    pub fn descriptor(strategy: RetrievalStrategy) -> StrategyDescriptor {
        get_descriptor(strategy)
    }
}

// ---- get_descriptor ----

/// Return the execution descriptor for `strategy`.
///
/// This is the single authoritative mapping of all ten [`RetrievalStrategy`]
/// variants to their runtime behaviour. Every variant is handled exhaustively —
/// adding a new variant without updating this function is a compile error.
///
/// # Mapping table
///
/// | Strategy           | Back-ends                | Parallel | Context-augmented |
/// |--------------------|--------------------------|----------|-------------------|
/// | `Semantic`         | Vector                   | false    | true              |
/// | `Keyword`          | Keyword                  | false    | false             |
/// | `Hybrid`           | Vector + Keyword         | true     | true              |
/// | `Graph`            | Graph                    | false    | true              |
/// | `GraphHybrid`      | Vector + Keyword + Graph | true     | true              |
/// | `RagCompletion`    | Graph                    | false    | false             |
/// | `Summaries`        | Vector                   | false    | false             |
/// | `Temporal`         | Vector                   | false    | false             |
/// | `GraphCoT`         | Graph                    | false    | true              |
/// | `GraphConstrained` | Graph + Vector           | true     | true              |
pub fn get_descriptor(strategy: RetrievalStrategy) -> StrategyDescriptor {
    match strategy {
        RetrievalStrategy::Semantic => StrategyDescriptor {
            search_types: vec![SearchType::Vector],
            parallel: false,
            context_augmented: true,
        },
        RetrievalStrategy::Keyword => StrategyDescriptor {
            search_types: vec![SearchType::Keyword],
            parallel: false,
            context_augmented: false,
        },
        RetrievalStrategy::Hybrid => StrategyDescriptor {
            search_types: vec![SearchType::Vector, SearchType::Keyword],
            parallel: true,
            context_augmented: true,
        },
        RetrievalStrategy::Graph => StrategyDescriptor {
            search_types: vec![SearchType::Graph],
            parallel: false,
            context_augmented: true,
        },
        RetrievalStrategy::GraphHybrid => StrategyDescriptor {
            search_types: vec![SearchType::Vector, SearchType::Keyword, SearchType::Graph],
            parallel: true,
            context_augmented: true,
        },
        RetrievalStrategy::RagCompletion => StrategyDescriptor {
            search_types: vec![SearchType::Graph],
            parallel: false,
            context_augmented: false,
        },
        RetrievalStrategy::Summaries => StrategyDescriptor {
            search_types: vec![SearchType::Vector],
            parallel: false,
            context_augmented: false,
        },
        RetrievalStrategy::Temporal => StrategyDescriptor {
            search_types: vec![SearchType::Vector],
            parallel: false,
            context_augmented: false,
        },
        RetrievalStrategy::GraphCoT => StrategyDescriptor {
            search_types: vec![SearchType::Graph],
            parallel: false,
            context_augmented: true,
        },
        RetrievalStrategy::GraphConstrained => StrategyDescriptor {
            search_types: vec![SearchType::Graph, SearchType::Vector],
            parallel: true,
            context_augmented: true,
        },
    }
}
