//! Minimal circuit breaker for retrieval backend graceful degradation.
//!
//! Three-state machine:
//!
//! - **Closed** (default): requests flow normally. A streak of
//!   `failure_threshold` consecutive failures transitions the breaker to
//!   **Open**.
//! - **Open**: every [`CircuitBreaker::allow`] returns `false` and callers
//!   must short-circuit to a fallback path. After `recovery_after` has
//!   elapsed since the breaker entered Open, the state moves to
//!   **HalfOpen**.
//! - **HalfOpen**: at most `half_open_max_probes` requests are admitted
//!   to test recovery. The next [`CircuitBreaker::record_success`]
//!   transitions to Closed; any [`CircuitBreaker::record_failure`]
//!   trips the breaker back to Open and re-arms `recovery_after`.
//!
//! # Thread safety
//!
//! All operations are lock-free, backed by atomics. The success path
//! (the common case) takes a single Relaxed read for `opened_at_ms == 0`
//! and then a Relaxed reset of the failure counter. Hot retrieval paths
//! pay no Mutex.
//!
//! # Memory ordering
//!
//! - Failure increments use [`Ordering::Relaxed`] until the threshold is
//!   met; the open transition publishes `opened_at_ms` with
//!   [`Ordering::Release`] paired with [`Ordering::Acquire`] reads in
//!   [`CircuitBreaker::allow`] / [`CircuitBreaker::state`] so HalfOpen
//!   admission decisions observe a consistent view of the open
//!   timestamp.
//! - Probe admission uses a CAS-style `fetch_add` so two concurrent
//!   probes cannot both squeak past `half_open_max_probes`.
//!
//! # Time source
//!
//! Wall-clock is read via [`std::time::SystemTime`]. A monotonic clock
//! would be theoretically preferable, but [`std::time::Instant`] is not
//! `Copy`-into-atomic and the breaker tolerates wall-clock skew up to
//! seconds — operationally indistinguishable from a slightly longer or
//! shorter recovery window. Tests use a fake `now_ms` argument to drive
//! deterministic transitions.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Snapshot of the breaker state for diagnostics and logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakerState {
    /// Requests flow normally.
    Closed,
    /// Requests short-circuit; the recovery window is still active.
    Open,
    /// At most `half_open_max_probes` requests are admitted to test
    /// recovery.
    HalfOpen,
}

impl std::fmt::Display for BreakerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BreakerState::Closed => f.write_str("closed"),
            BreakerState::Open => f.write_str("open"),
            BreakerState::HalfOpen => f.write_str("half_open"),
        }
    }
}

/// Sentinel value for `opened_at_ms` meaning "never opened or already
/// recovered to Closed". Real timestamps are unix-epoch milliseconds and
/// will never be zero in practice (epoch was decades ago).
const OPENED_AT_MS_NONE: u64 = 0;

/// Lock-free circuit breaker. See module-level docs for the state-machine
/// contract.
#[derive(Debug)]
pub struct CircuitBreaker {
    /// Number of consecutive failures that trip Closed → Open.
    failure_threshold: u32,
    /// Wall-clock duration the breaker stays Open before HalfOpen probes
    /// are admitted.
    recovery_after: Duration,
    /// Maximum number of probes admitted in HalfOpen before transitioning
    /// back to Closed (on success) or Open (on failure).
    half_open_max_probes: u32,

    /// Consecutive failure streak. Reset to zero on every success.
    consecutive_failures: AtomicU32,
    /// Unix epoch milliseconds at which the breaker last entered Open.
    /// [`OPENED_AT_MS_NONE`] = never opened or already Closed.
    opened_at_ms: AtomicU64,
    /// Number of probes admitted while in HalfOpen. Reset by every Open
    /// transition.
    half_open_probes_admitted: AtomicU32,
}

impl CircuitBreaker {
    /// Construct a new breaker in the [`BreakerState::Closed`] state.
    ///
    /// `failure_threshold` of `0` is silently clamped to `1` — a breaker
    /// that opens after zero failures would never admit any traffic.
    /// `half_open_max_probes` of `0` is also clamped to `1` so the
    /// breaker is guaranteed to make forward progress out of HalfOpen.
    pub fn new(
        failure_threshold: u32,
        recovery_after: Duration,
        half_open_max_probes: u32,
    ) -> Self {
        Self {
            failure_threshold: failure_threshold.max(1),
            recovery_after,
            half_open_max_probes: half_open_max_probes.max(1),
            consecutive_failures: AtomicU32::new(0),
            opened_at_ms: AtomicU64::new(OPENED_AT_MS_NONE),
            half_open_probes_admitted: AtomicU32::new(0),
        }
    }

    /// Decide whether a request may proceed.
    ///
    /// Returns `true` when the breaker is Closed, or HalfOpen with a
    /// probe slot still available. Returns `false` when the breaker is
    /// Open (caller must fall back).
    pub fn allow(&self) -> bool {
        self.allow_at(now_ms())
    }

    /// Test-friendly variant of [`Self::allow`] that takes a unix-epoch
    /// millisecond timestamp instead of reading the system clock.
    fn allow_at(&self, now: u64) -> bool {
        let opened_at = self.opened_at_ms.load(Ordering::Acquire);
        if opened_at == OPENED_AT_MS_NONE {
            // Closed: fast path.
            return true;
        }

        // Open or HalfOpen.
        let recovery_ms = self.recovery_after.as_millis() as u64;
        let elapsed_ok = now.saturating_sub(opened_at) >= recovery_ms;
        if !elapsed_ok {
            // Still inside the Open window.
            return false;
        }

        // Past the recovery window — HalfOpen. Try to claim a probe slot.
        let admitted = self
            .half_open_probes_admitted
            .fetch_add(1, Ordering::AcqRel);
        if admitted < self.half_open_max_probes {
            true
        } else {
            // Probe budget exhausted — keep short-circuiting until a
            // probe records an outcome and resolves the state.
            //
            // We don't decrement here: another probe may resolve the
            // breaker imminently, after which `record_success` /
            // `record_failure` resets the counter.
            false
        }
    }

    /// Record a successful call.
    ///
    /// In Closed state, resets the consecutive-failure counter.
    /// In HalfOpen, transitions to Closed and clears all state.
    pub fn record_success(&self) {
        // The success path is intentionally cheap: drop the failure
        // streak unconditionally, then close the breaker if it was open.
        self.consecutive_failures.store(0, Ordering::Relaxed);
        // Publish closure with Release so observers reading
        // `opened_at_ms` with Acquire see the cleared probe counter.
        self.half_open_probes_admitted.store(0, Ordering::Relaxed);
        self.opened_at_ms
            .store(OPENED_AT_MS_NONE, Ordering::Release);
    }

    /// Record a failed call.
    ///
    /// In Closed state, increments the consecutive-failure counter and
    /// transitions to Open if the threshold is reached. In HalfOpen,
    /// re-trips to Open and re-arms `recovery_after`.
    pub fn record_failure(&self) {
        self.record_failure_at(now_ms());
    }

    /// Test-friendly variant of [`Self::record_failure`].
    fn record_failure_at(&self, now: u64) {
        let opened_at = self.opened_at_ms.load(Ordering::Acquire);
        if opened_at != OPENED_AT_MS_NONE {
            // Either Open or HalfOpen probe failed → re-arm Open with a
            // fresh timestamp and clear the probe counter.
            self.half_open_probes_admitted.store(0, Ordering::Relaxed);
            self.opened_at_ms.store(now, Ordering::Release);
            return;
        }

        // Closed: bump the streak and trip if threshold reached.
        let failures = self
            .consecutive_failures
            .fetch_add(1, Ordering::Relaxed)
            .saturating_add(1);
        if failures >= self.failure_threshold {
            self.half_open_probes_admitted.store(0, Ordering::Relaxed);
            self.opened_at_ms.store(now, Ordering::Release);
        }
    }

    /// Snapshot the current breaker state for diagnostics.
    ///
    /// The returned value is a logical view at the moment of the call;
    /// it may already be stale by the time the caller inspects it.
    pub fn state(&self) -> BreakerState {
        self.state_at(now_ms())
    }

    /// Test-friendly variant of [`Self::state`].
    fn state_at(&self, now: u64) -> BreakerState {
        let opened_at = self.opened_at_ms.load(Ordering::Acquire);
        if opened_at == OPENED_AT_MS_NONE {
            return BreakerState::Closed;
        }
        let recovery_ms = self.recovery_after.as_millis() as u64;
        if now.saturating_sub(opened_at) >= recovery_ms {
            BreakerState::HalfOpen
        } else {
            BreakerState::Open
        }
    }
}

/// Read the current wall-clock time in unix-epoch milliseconds.
///
/// On the rare failure path where the system clock is set before the
/// unix epoch, we return `1` (not zero) so the value is distinguishable
/// from [`OPENED_AT_MS_NONE`]. The breaker is robust to such skew —
/// pretending we're at "1 ms past epoch" simply means the recovery
/// window expires on the next sane clock read.
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: pin a "logical now" so tests don't depend on the system
    /// clock — keeps CI deterministic.
    const T0: u64 = 1_700_000_000_000; // arbitrary epoch ms anchor

    #[test]
    fn closed_to_open_after_threshold_failures() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(30), 1);
        assert_eq!(cb.state_at(T0), BreakerState::Closed);
        assert!(cb.allow_at(T0));

        cb.record_failure_at(T0);
        cb.record_failure_at(T0);
        assert_eq!(cb.state_at(T0), BreakerState::Closed);
        // Third failure trips the breaker.
        cb.record_failure_at(T0);
        assert_eq!(cb.state_at(T0), BreakerState::Open);
        assert!(!cb.allow_at(T0));
    }

    #[test]
    fn open_short_circuits_within_recovery_window() {
        let cb = CircuitBreaker::new(1, Duration::from_secs(60), 1);
        cb.record_failure_at(T0);
        assert_eq!(cb.state_at(T0), BreakerState::Open);
        // 30 s later — still open.
        assert!(!cb.allow_at(T0 + 30_000));
        assert_eq!(cb.state_at(T0 + 30_000), BreakerState::Open);
    }

    #[test]
    fn open_to_half_open_after_recovery_duration() {
        let cb = CircuitBreaker::new(1, Duration::from_secs(60), 2);
        cb.record_failure_at(T0);
        // Just after recovery_after — first probe is admitted.
        let after = T0 + 60_001;
        assert_eq!(cb.state_at(after), BreakerState::HalfOpen);
        assert!(cb.allow_at(after));
    }

    #[test]
    fn half_open_to_closed_on_success() {
        let cb = CircuitBreaker::new(1, Duration::from_secs(60), 1);
        cb.record_failure_at(T0);
        let after = T0 + 60_001;
        assert!(cb.allow_at(after));
        cb.record_success();
        assert_eq!(cb.state(), BreakerState::Closed);
        assert!(cb.allow_at(after));
    }

    #[test]
    fn half_open_to_open_on_failure() {
        let cb = CircuitBreaker::new(1, Duration::from_secs(60), 1);
        cb.record_failure_at(T0);
        let after = T0 + 60_001;
        assert!(cb.allow_at(after));
        // Probe fails → re-arm Open with a fresh timestamp.
        cb.record_failure_at(after);
        // 30 s past the new open instant — still Open.
        assert_eq!(cb.state_at(after + 30_000), BreakerState::Open);
        assert!(!cb.allow_at(after + 30_000));
    }

    #[test]
    fn half_open_probe_limit_enforced() {
        let cb = CircuitBreaker::new(1, Duration::from_secs(60), 2);
        cb.record_failure_at(T0);
        let after = T0 + 60_001;
        // Two probes admitted.
        assert!(cb.allow_at(after));
        assert!(cb.allow_at(after));
        // Third probe rejected — budget exhausted.
        assert!(!cb.allow_at(after));
    }

    #[test]
    fn success_in_closed_resets_failure_streak() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(30), 1);
        cb.record_failure_at(T0);
        cb.record_failure_at(T0);
        // A success cleared the streak; threshold not reached.
        cb.record_success();
        cb.record_failure_at(T0);
        cb.record_failure_at(T0);
        assert_eq!(cb.state_at(T0), BreakerState::Closed);
    }

    #[test]
    fn closed_with_no_traffic_stays_closed() {
        let cb = CircuitBreaker::new(5, Duration::from_secs(30), 1);
        assert_eq!(cb.state_at(T0), BreakerState::Closed);
        assert!(cb.allow_at(T0));
    }

    #[test]
    fn breaker_state_displays_human_readable_label() {
        assert_eq!(BreakerState::Closed.to_string(), "closed");
        assert_eq!(BreakerState::Open.to_string(), "open");
        assert_eq!(BreakerState::HalfOpen.to_string(), "half_open");
    }

    #[test]
    fn zero_threshold_is_clamped_to_one() {
        // A degenerate config of `failure_threshold=0` must not produce
        // a permanently-open breaker; the constructor clamps it.
        let cb = CircuitBreaker::new(0, Duration::from_secs(30), 0);
        assert_eq!(cb.state_at(T0), BreakerState::Closed);
        cb.record_failure_at(T0);
        assert_eq!(cb.state_at(T0), BreakerState::Open);
    }
}
