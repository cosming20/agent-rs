//! Web tool budget tracker.
//!
//! Thread-safe atomic counters that enforce per-query budget limits
//! for web search, news search, and URL fetch operations.
//!
//! Initialized from [`WebBudgetConfig`] and shared across tool
//! invocations within a single query via `Arc<WebBudgetTracker>`.

use std::sync::atomic::{AtomicUsize, Ordering};

use agent_rs_core::ToolError;
use agent_rs_core::config::WebBudgetConfig;

/// Thread-safe budget tracker for web tool usage within a single query.
///
/// Each query gets its own tracker instance. The tracker enforces
/// maximum counts for web searches, news searches, and URL fetches.
/// All methods use relaxed atomic ordering since exact precision
/// across threads is not critical -- slight over-counting is
/// acceptable and cheaper than sequential consistency.
#[derive(Debug)]
pub struct WebBudgetTracker {
    max_searches: usize,
    max_news: usize,
    max_fetches: usize,
    searches: AtomicUsize,
    news: AtomicUsize,
    fetches: AtomicUsize,
}

impl WebBudgetTracker {
    /// Create a new tracker from the given budget configuration.
    pub fn new(config: &WebBudgetConfig) -> Self {
        Self {
            max_searches: config.max_searches,
            max_news: config.max_news,
            max_fetches: config.max_fetches,
            searches: AtomicUsize::new(0),
            news: AtomicUsize::new(0),
            fetches: AtomicUsize::new(0),
        }
    }

    /// Check whether another web search is allowed within the budget.
    pub fn can_search(&self) -> bool {
        self.searches.load(Ordering::Relaxed) < self.max_searches
    }

    /// Check whether another news search is allowed within the budget.
    pub fn can_news(&self) -> bool {
        self.news.load(Ordering::Relaxed) < self.max_news
    }

    /// Check whether another URL fetch is allowed within the budget.
    pub fn can_fetch(&self) -> bool {
        self.fetches.load(Ordering::Relaxed) < self.max_fetches
    }

    /// Record a web search invocation. Returns `Ok(())` if within budget,
    /// or `Err(ToolError::WebBudgetExhausted)` if the limit has been reached.
    ///
    /// Uses `compare_exchange_weak` to atomically increment only when below
    /// the limit, eliminating the TOCTOU window of the previous
    /// `fetch_add` + rollback pattern.
    pub fn record_search(&self) -> Result<(), ToolError> {
        loop {
            let current = self.searches.load(Ordering::Relaxed);
            if current >= self.max_searches {
                return Err(ToolError::WebBudgetExhausted {
                    resource: "web_searches".into(),
                    used: current,
                    max: self.max_searches,
                });
            }
            if self
                .searches
                .compare_exchange_weak(current, current + 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return Ok(());
            }
        }
    }

    /// Record a news search invocation. Returns `Ok(())` if within budget,
    /// or `Err(ToolError::WebBudgetExhausted)` if the limit has been reached.
    pub fn record_news(&self) -> Result<(), ToolError> {
        loop {
            let current = self.news.load(Ordering::Relaxed);
            if current >= self.max_news {
                return Err(ToolError::WebBudgetExhausted {
                    resource: "news_searches".into(),
                    used: current,
                    max: self.max_news,
                });
            }
            if self
                .news
                .compare_exchange_weak(current, current + 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return Ok(());
            }
        }
    }

    /// Record a URL fetch invocation. Returns `Ok(())` if within budget,
    /// or `Err(ToolError::WebBudgetExhausted)` if the limit has been reached.
    pub fn record_fetch(&self) -> Result<(), ToolError> {
        loop {
            let current = self.fetches.load(Ordering::Relaxed);
            if current >= self.max_fetches {
                return Err(ToolError::WebBudgetExhausted {
                    resource: "url_fetches".into(),
                    used: current,
                    max: self.max_fetches,
                });
            }
            if self
                .fetches
                .compare_exchange_weak(current, current + 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return Ok(());
            }
        }
    }

    /// Return the current number of web searches used.
    pub fn searches_used(&self) -> usize {
        self.searches.load(Ordering::Relaxed)
    }

    /// Return the current number of news searches used.
    pub fn news_used(&self) -> usize {
        self.news.load(Ordering::Relaxed)
    }

    /// Return the current number of URL fetches used.
    pub fn fetches_used(&self) -> usize {
        self.fetches.load(Ordering::Relaxed)
    }
}
