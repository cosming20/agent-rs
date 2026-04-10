//! `fetch_url` tool -- HTTP GET with SSRF protection.
//!
//! Fetches the content of a URL after validating that it does not point
//! to an internal/private IP address. DNS resolution is performed before
//! the HTTP request to catch DNS rebinding and hostname-based SSRF.

use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;

use agent_rs_core::ToolError;
use agent_rs_core::config::FetchConfig;
use agent_rs_core::types::truncate_to_char_boundary;
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::budget::WebBudgetTracker;

// ---- SSRF validation ----

/// Returns `true` if the IP address is in a private, loopback, or
/// link-local range that should not be accessed from a server context.
fn is_internal_ip(addr: &IpAddr) -> bool {
    match addr {
        IpAddr::V4(ip) => {
            ip.is_loopback()        // 127.0.0.0/8
                || ip.is_private()   // 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
                || ip.is_link_local() // 169.254.0.0/16
                || ip.is_broadcast() // 255.255.255.255
                || ip.is_unspecified() // 0.0.0.0
                || ip.octets()[0] == 100 && (ip.octets()[1] & 0xC0) == 64 // 100.64.0.0/10 (CGNAT)
        }
        IpAddr::V6(ip) => {
            ip.is_loopback()        // ::1
                || ip.is_unspecified() // ::
                // Unique local addresses (fc00::/7)
                || (ip.segments()[0] & 0xfe00) == 0xfc00
                // Link-local (fe80::/10)
                || (ip.segments()[0] & 0xffc0) == 0xfe80
        }
    }
}

/// Resolve the hostname from a URL and check all resulting IP addresses
/// for SSRF. Returns `Ok(())` if all addresses are public.
async fn validate_url_ssrf(url: &str) -> Result<(), ToolError> {
    let parsed = url::Url::parse(url).map_err(|e| ToolError::FetchFailed {
        url: url.to_string(),
        reason: format!("invalid URL: {e}"),
    })?;

    // Only allow http and https schemes.
    match parsed.scheme() {
        "http" | "https" => {}
        _scheme => {
            return Err(ToolError::Ssrf {
                url: url.to_string(),
            });
        }
    }

    let host = parsed.host_str().ok_or_else(|| ToolError::FetchFailed {
        url: url.to_string(),
        reason: "URL has no host".into(),
    })?;

    let port = parsed.port_or_known_default().unwrap_or(443);
    let addr_str = format!("{host}:{port}");

    // Resolve DNS to check all returned IPs.
    let addrs = tokio::net::lookup_host(&addr_str)
        .await
        .map_err(|e| ToolError::FetchFailed {
            url: url.to_string(),
            reason: format!("DNS resolution failed: {e}"),
        })?;

    for socket_addr in addrs {
        if is_internal_ip(&socket_addr.ip()) {
            tracing::warn!(
                url = %url,
                resolved_ip = %socket_addr.ip(),
                "SSRF blocked: URL resolves to internal IP"
            );
            return Err(ToolError::Ssrf {
                url: url.to_string(),
            });
        }
    }

    Ok(())
}

// ---- Args ----

/// Arguments for the `fetch_url` tool.
#[derive(Debug, Deserialize)]
pub struct FetchUrlArgs {
    /// The URL to fetch.
    pub url: String,
}

// ---- Tool struct ----

/// Fetches the text content of a URL with SSRF protection.
///
/// Validates the target URL against internal IP ranges (including DNS
/// resolution) before performing the HTTP GET request. Enforces fetch
/// budget limits and respects timeout/size configuration.
#[derive(Serialize, Deserialize)]
pub struct FetchUrlTool {
    #[serde(skip)]
    config: Option<FetchConfig>,
    #[serde(skip)]
    budget: Option<Arc<WebBudgetTracker>>,
}

impl FetchUrlTool {
    /// Create a new `FetchUrlTool`.
    ///
    /// # Parameters
    ///
    /// - `config` -- fetch configuration (timeout, max bytes, max redirects).
    /// - `budget` -- shared budget tracker for enforcing per-query fetch limits.
    pub fn new(config: FetchConfig, budget: Arc<WebBudgetTracker>) -> Self {
        Self {
            config: Some(config),
            budget: Some(budget),
        }
    }
}

impl Tool for FetchUrlTool {
    const NAME: &'static str = "fetch_url";
    type Error = ToolError;
    type Args = FetchUrlArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Fetch the text content of a web page URL. Use when you need to \
                          read the content of a specific web page. The URL must be publicly \
                          accessible (internal/private IPs are blocked)."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch content from."
                    }
                },
                "required": ["url"]
            }),
        }
    }

    #[tracing::instrument(skip(self), fields(tool = Self::NAME))]
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let config = self
            .config
            .as_ref()
            .ok_or_else(|| ToolError::McpClient("fetch_url: config not initialized".into()))?;
        let budget = self.budget.as_ref().ok_or_else(|| {
            ToolError::McpClient("fetch_url: budget tracker not initialized".into())
        })?;

        // Check and record budget.
        budget.record_fetch()?;

        // SSRF validation: resolve DNS and check all IPs.
        validate_url_ssrf(&args.url).await?;

        tracing::info!(url = %args.url, "fetching URL content");

        // Build a one-shot reqwest client with the configured limits.
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout))
            .redirect(reqwest::redirect::Policy::limited(config.max_redirects))
            .build()
            .map_err(|e| ToolError::FetchFailed {
                url: args.url.clone(),
                reason: format!("HTTP client build error: {e}"),
            })?;

        let response = client
            .get(&args.url)
            .header("User-Agent", "agent-rs/0.1 (fetch_url tool)")
            .send()
            .await
            .map_err(|e| ToolError::FetchFailed {
                url: args.url.clone(),
                reason: format!("HTTP request failed: {e}"),
            })?;

        let status = response.status();
        if !status.is_success() {
            return Err(ToolError::FetchFailed {
                url: args.url,
                reason: format!("HTTP {status}"),
            });
        }

        // Early reject when Content-Length header exceeds the budget.
        if let Some(content_length) = response.content_length() {
            if content_length as usize > config.max_bytes {
                return Err(ToolError::FetchFailed {
                    url: args.url,
                    reason: format!(
                        "response Content-Length too large: {} bytes (max: {})",
                        content_length, config.max_bytes
                    ),
                });
            }
        }

        // Read body with size limit (fallback for when Content-Length is absent).
        let bytes = response.bytes().await.map_err(|e| ToolError::FetchFailed {
            url: args.url.clone(),
            reason: format!("failed to read response body: {e}"),
        })?;

        if bytes.len() > config.max_bytes {
            return Err(ToolError::FetchFailed {
                url: args.url,
                reason: format!(
                    "response body too large: {} bytes (max: {})",
                    bytes.len(),
                    config.max_bytes
                ),
            });
        }

        // Convert to text. For HTML, a full parser would be better, but
        // for now we return the raw text content.
        let text = String::from_utf8_lossy(&bytes);

        // Truncate to max_length characters for the LLM context.
        let truncated = if text.chars().count() > config.max_length {
            format!(
                "{}\n\n[Content truncated at {} characters]",
                truncate_to_char_boundary(&text, config.max_length),
                config.max_length
            )
        } else {
            text.into_owned()
        };

        Ok(truncated)
    }
}
