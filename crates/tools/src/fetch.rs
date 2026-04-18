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
            // Fix 1a: IPv4-mapped IPv6 (::ffff:0:0/96). An attacker can bypass v4
            // SSRF checks by encoding a private IPv4 address as an IPv6 mapped
            // address (e.g. ::ffff:169.254.169.254 → AWS metadata endpoint).
            // Re-check the embedded v4 address using the full v4 ruleset.
            if let Some(v4) = ip.to_ipv4_mapped() {
                return is_internal_ip(&IpAddr::V4(v4));
            }

            ip.is_loopback()        // ::1
                || ip.is_unspecified() // ::
                // Unique local addresses (fc00::/7) — RFC 4193 private ranges
                || (ip.segments()[0] & 0xfe00) == 0xfc00
                // Fix 1b: Link-local (fe80::/10) — already present but documented explicitly.
                // Covers link-local unicast; hosts on the same link, not routable.
                || (ip.segments()[0] & 0xffc0) == 0xfe80
                // Fix 1c: Multicast (ff00::/8) — RFC 4291 §2.7.
                // Multicast addresses should never be the target of a fetch.
                || (ip.segments()[0] & 0xff00) == 0xff00
                // Fix 1d: Discard-only (100::/64) — RFC 6666.
                // Traffic to this range is intentionally discarded; never a valid destination.
                || (ip.segments()[0] == 0x0100
                    && ip.segments()[1] == 0x0000
                    && ip.segments()[2] == 0x0000
                    && ip.segments()[3] == 0x0000)
                // Fix 1e: Documentation ranges:
                //   2001:db8::/32 — RFC 3849, reserved for docs/examples.
                //   2001:0002::/48 — RFC 5180, benchmarking (BMWG).
                // Neither should be reachable in production; block to prevent
                // accidental exfiltration to reserved space.
                || (ip.segments()[0] == 0x2001 && ip.segments()[1] == 0x0db8)
                || (ip.segments()[0] == 0x2001
                    && ip.segments()[1] == 0x0002
                    && ip.segments()[2] == 0x0000)
        }
    }
}

/// SSRF-validated host: carries the hostname and a validated public socket address.
///
/// The caller MUST use `reqwest::ClientBuilder::resolve()` to pin the
/// returned address, eliminating the DNS-rebinding TOCTOU window.
struct SsrfValidated {
    host: String,
    addr: std::net::SocketAddr,
}

/// Resolve the hostname from a URL and check all resulting IP addresses
/// for SSRF. Returns a validated host+address pair pinned for the HTTP client.
async fn validate_url_ssrf(url: &str) -> Result<SsrfValidated, ToolError> {
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
    let addrs: Vec<std::net::SocketAddr> = tokio::net::lookup_host(&addr_str)
        .await
        .map_err(|e| ToolError::FetchFailed {
            url: url.to_string(),
            reason: format!("DNS resolution failed: {e}"),
        })?
        .collect();

    let mut first_public: Option<std::net::SocketAddr> = None;
    for socket_addr in &addrs {
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
        if first_public.is_none() {
            first_public = Some(*socket_addr);
        }
    }

    let addr = first_public.ok_or_else(|| ToolError::FetchFailed {
        url: url.to_string(),
        reason: "DNS resolution returned no addresses".into(),
    })?;

    Ok(SsrfValidated {
        host: host.to_string(),
        addr,
    })
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
            description: "Fetch and return the plain-text body of a PUBLIC web URL. Use ONLY \
                          when: (a) the user's message contained a URL, or (b) a prior \
                          search_web / search_news call returned a URL whose full body is \
                          needed. Never synthesize a URL. Never use this tool to 'browse' or \
                          'google' — use search_web for discovery. Private / internal IP ranges \
                          and non-HTTP(S) schemes are blocked; size is capped; a web-budget \
                          tracker prevents runaway fetches."
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

        // SSRF validation: resolve DNS, check all IPs, return pinned address.
        let validated = validate_url_ssrf(&args.url).await?;

        tracing::info!(url = %args.url, pinned_ip = %validated.addr.ip(), "fetching URL content");

        // Fix 2: Custom redirect policy that re-validates each redirect target.
        //
        // reqwest::Client::builder().resolve() pins DNS only for the *initial*
        // hostname. On a 30x redirect to a *different* hostname reqwest
        // re-resolves via the system resolver, which opens a DNS-rebinding
        // window. We close that window with a custom policy that:
        //   1. Rejects non-http(s) schemes (blocks file://, gopher://, etc.)
        //   2. Runs full SSRF validation (scheme + resolved IPs) on the
        //      redirect target before following it.
        //   3. Enforces the configured max_redirects cap.
        let max_redirects = config.max_redirects;
        let redirect_policy = reqwest::redirect::Policy::custom(move |attempt| {
            // Enforce the redirect cap first.
            if attempt.previous().len() >= max_redirects {
                return attempt.stop();
            }

            let location = attempt.url();

            // Only permit http/https on redirects; block file://, gopher://, etc.
            match location.scheme() {
                "http" | "https" => {}
                scheme => {
                    tracing::warn!(
                        redirect_url = %location,
                        scheme = %scheme,
                        "SSRF blocked: redirect to non-http(s) scheme"
                    );
                    return attempt.stop();
                }
            }

            // Validate the redirect target's host against internal IP ranges.
            // We cannot do async DNS resolution inside this sync closure, so we
            // use the synchronous std resolver as a best-effort check. The async
            // validate_url_ssrf path (pre-request) remains the primary guard.
            let host = match location.host_str() {
                Some(h) => h.to_string(),
                None => {
                    tracing::warn!(redirect_url = %location, "SSRF blocked: redirect URL has no host");
                    return attempt.stop();
                }
            };
            let port = location.port_or_known_default().unwrap_or(443);
            let addr_str = format!("{host}:{port}");

            match std::net::ToSocketAddrs::to_socket_addrs(&addr_str.as_str()) {
                Ok(addrs) => {
                    for socket_addr in addrs {
                        if is_internal_ip(&socket_addr.ip()) {
                            tracing::warn!(
                                redirect_url = %location,
                                resolved_ip = %socket_addr.ip(),
                                "SSRF blocked: redirect resolves to internal IP"
                            );
                            return attempt.stop();
                        }
                    }
                    attempt.follow()
                }
                Err(e) => {
                    tracing::warn!(
                        redirect_url = %location,
                        error = %e,
                        "SSRF blocked: redirect DNS resolution failed"
                    );
                    attempt.stop()
                }
            }
        });

        // Build per-request client with DNS pinning to eliminate rebinding TOCTOU.
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout))
            .redirect(redirect_policy)
            .resolve(&validated.host, validated.addr)
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

        // Fix 3: Early reject when Content-Length header exceeds the budget.
        //
        // Previously compared as `content_length as usize > config.max_bytes`.
        // On 32-bit targets, casting a u64 ≥ 2³² to usize wraps to a small
        // value (e.g. 2³²+1 → 1), silently bypassing this gate. Compare
        // entirely in u64 to avoid any truncation on any target width.
        if let Some(content_length) = response.content_length() {
            if content_length > config.max_bytes as u64 {
                return Err(ToolError::FetchFailed {
                    url: args.url,
                    reason: format!(
                        "response Content-Length too large: {} bytes (max: {})",
                        content_length, config.max_bytes
                    ),
                });
            }
        }

        // Stream body in chunks to enforce size limit without buffering the
        // entire response (protects against OOM when Content-Length is absent).
        let max_bytes = config.max_bytes;
        let mut body = Vec::new();
        let mut response = response;
        while let Some(chunk) = response.chunk().await.map_err(|e| ToolError::FetchFailed {
            url: args.url.clone(),
            reason: format!("failed to read response chunk: {e}"),
        })? {
            body.extend_from_slice(&chunk);
            if body.len() > max_bytes {
                return Err(ToolError::FetchFailed {
                    url: args.url,
                    reason: format!(
                        "response body exceeded {} bytes during streaming",
                        max_bytes
                    ),
                });
            }
        }
        let bytes = body;

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
