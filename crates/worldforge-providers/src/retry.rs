//! Retry with exponential backoff for WorldForge providers.
//!
//! Provides configurable retry policies with jitter to avoid thundering-herd
//! problems and automatic detection of retryable errors.

use std::future::Future;
use std::time::Duration;

use worldforge_core::error::{Result, WorldForgeError};

/// Retry policy configuration.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts (0 = no retries, just the initial attempt).
    pub max_retries: u32,
    /// Initial delay before the first retry.
    pub initial_delay: Duration,
    /// Maximum delay between retries.
    pub max_delay: Duration,
    /// Multiplier applied to the delay after each retry.
    pub backoff_multiplier: f64,
    /// Whether to add jitter to delays to avoid thundering herd.
    pub jitter: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }
}

impl RetryPolicy {
    /// Create a policy with no retries (execute once).
    pub fn no_retry() -> Self {
        Self {
            max_retries: 0,
            ..Self::default()
        }
    }

    /// Create an aggressive retry policy for critical operations.
    pub fn aggressive() -> Self {
        Self {
            max_retries: 5,
            initial_delay: Duration::from_millis(200),
            max_delay: Duration::from_secs(60),
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }

    /// Compute the delay for a given attempt number (0-based).
    fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let base = self.initial_delay.as_secs_f64()
            * self.backoff_multiplier.powi(attempt as i32);
        let capped = base.min(self.max_delay.as_secs_f64());

        if self.jitter {
            // Deterministic jitter based on attempt number to avoid needing rand.
            // Full jitter: uniform in [0, capped].
            // We use a simple hash-like approach for reproducible but varied delays.
            let jitter_factor = pseudo_jitter(attempt);
            Duration::from_secs_f64(capped * jitter_factor)
        } else {
            Duration::from_secs_f64(capped)
        }
    }
}

/// Simple pseudo-random jitter factor in [0.5, 1.0) based on attempt number.
/// Not cryptographically secure — just enough to decorrelate retries.
fn pseudo_jitter(attempt: u32) -> f64 {
    // Mix bits to get a varied but deterministic factor per attempt.
    let mut x = attempt as u64;
    x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    x ^= x >> 16;
    // Map to [0.5, 1.0)
    0.5 + (x % 1000) as f64 / 2000.0
}

/// Check whether a `WorldForgeError` is retryable.
pub fn is_retryable(err: &WorldForgeError) -> bool {
    matches!(
        err,
        WorldForgeError::ProviderTimeout { .. }
            | WorldForgeError::ProviderRateLimited { .. }
            | WorldForgeError::ProviderUnavailable { .. }
    )
}

/// Execute an async operation with retry according to the given policy.
///
/// The `operation` closure is called repeatedly until it succeeds, a
/// non-retryable error occurs, or retries are exhausted.
///
/// # Arguments
///
/// * `provider_name` — for logging context
/// * `policy` — the retry policy to apply
/// * `operation` — async closure to execute
///
/// # Errors
///
/// Returns the last error if all retries are exhausted.
pub async fn retry_with_policy<F, Fut, T>(
    provider_name: &str,
    policy: &RetryPolicy,
    operation: F,
) -> Result<T>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let mut last_error = None;

    for attempt in 0..=policy.max_retries {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(err) => {
                if attempt == policy.max_retries || !is_retryable(&err) {
                    return Err(err);
                }

                let delay = policy.delay_for_attempt(attempt);
                tracing::warn!(
                    provider = provider_name,
                    attempt = attempt + 1,
                    max_retries = policy.max_retries,
                    delay_ms = delay.as_millis() as u64,
                    error = %err,
                    "retryable error, backing off"
                );
                last_error = Some(err);
                tokio::time::sleep(delay).await;
            }
        }
    }

    Err(last_error.unwrap_or_else(|| WorldForgeError::ProviderUnavailable {
        provider: provider_name.to_string(),
        reason: "retry exhausted with no error captured".to_string(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn test_default_policy() {
        let policy = RetryPolicy::default();
        assert_eq!(policy.max_retries, 3);
        assert!(policy.jitter);
    }

    #[test]
    fn test_no_retry_policy() {
        let policy = RetryPolicy::no_retry();
        assert_eq!(policy.max_retries, 0);
    }

    #[test]
    fn test_delay_increases_with_backoff() {
        let policy = RetryPolicy {
            jitter: false,
            ..RetryPolicy::default()
        };
        let d0 = policy.delay_for_attempt(0);
        let d1 = policy.delay_for_attempt(1);
        let d2 = policy.delay_for_attempt(2);
        assert!(d1 > d0, "d1={d1:?} should be > d0={d0:?}");
        assert!(d2 > d1, "d2={d2:?} should be > d1={d1:?}");
    }

    #[test]
    fn test_delay_capped_at_max() {
        let policy = RetryPolicy {
            max_delay: Duration::from_secs(1),
            backoff_multiplier: 100.0,
            jitter: false,
            ..RetryPolicy::default()
        };
        let delay = policy.delay_for_attempt(10);
        assert!(delay <= Duration::from_secs(1));
    }

    #[test]
    fn test_jitter_varies_by_attempt() {
        let j0 = pseudo_jitter(0);
        let j1 = pseudo_jitter(1);
        assert!(j0 >= 0.5 && j0 < 1.0);
        assert!(j1 >= 0.5 && j1 < 1.0);
        // Different attempts should (usually) give different factors.
        // This is not guaranteed but extremely likely.
        assert_ne!(
            (j0 * 1000.0) as u32,
            (j1 * 1000.0) as u32,
            "jitter should vary between attempts"
        );
    }

    #[test]
    fn test_is_retryable() {
        assert!(is_retryable(&WorldForgeError::ProviderTimeout {
            provider: "test".into(),
            timeout_ms: 1000,
        }));
        assert!(is_retryable(&WorldForgeError::ProviderRateLimited {
            provider: "test".into(),
            retry_after_ms: 1000,
        }));
        assert!(is_retryable(&WorldForgeError::ProviderUnavailable {
            provider: "test".into(),
            reason: "server error".into(),
        }));
        assert!(!is_retryable(&WorldForgeError::ProviderAuthError(
            "bad key".into()
        )));
        assert!(!is_retryable(&WorldForgeError::ProviderNotFound(
            "missing".into()
        )));
    }

    #[tokio::test]
    async fn test_retry_succeeds_first_try() {
        let policy = RetryPolicy::no_retry();
        let result = retry_with_policy("test", &policy, || async { Ok(42u32) }).await;
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_retry_succeeds_after_failures() {
        let counter = AtomicU32::new(0);
        let policy = RetryPolicy {
            max_retries: 3,
            initial_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(10),
            backoff_multiplier: 1.0,
            jitter: false,
        };

        let result = retry_with_policy("test", &policy, || {
            let attempt = counter.fetch_add(1, Ordering::SeqCst);
            async move {
                if attempt < 2 {
                    Err(WorldForgeError::ProviderTimeout {
                        provider: "test".into(),
                        timeout_ms: 100,
                    })
                } else {
                    Ok("success")
                }
            }
        })
        .await;

        assert_eq!(result.unwrap(), "success");
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_retry_non_retryable_fails_immediately() {
        let counter = AtomicU32::new(0);
        let policy = RetryPolicy {
            max_retries: 5,
            initial_delay: Duration::from_millis(1),
            ..RetryPolicy::default()
        };

        let result: Result<()> = retry_with_policy("test", &policy, || {
            counter.fetch_add(1, Ordering::SeqCst);
            async {
                Err(WorldForgeError::ProviderAuthError(
                    "invalid key".to_string(),
                ))
            }
        })
        .await;

        assert!(result.is_err());
        // Should not have retried — auth errors are not retryable.
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_retry_exhaustion() {
        let counter = AtomicU32::new(0);
        let policy = RetryPolicy {
            max_retries: 2,
            initial_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(5),
            backoff_multiplier: 1.0,
            jitter: false,
        };

        let result: Result<()> = retry_with_policy("test", &policy, || {
            counter.fetch_add(1, Ordering::SeqCst);
            async {
                Err(WorldForgeError::ProviderTimeout {
                    provider: "test".into(),
                    timeout_ms: 100,
                })
            }
        })
        .await;

        assert!(result.is_err());
        // 1 initial + 2 retries = 3 total attempts.
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }
}
