//! Token bucket rate limiter for WorldForge providers.
//!
//! Provides per-provider rate limiting that supports both request-count and
//! token-count rate limits using a token bucket algorithm.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Mutex;

/// Configuration for a rate limiter.
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum number of tokens the bucket can hold.
    pub capacity: u32,
    /// Number of tokens refilled per second.
    pub refill_rate: f64,
    /// Initial token count (defaults to capacity if `None`).
    pub initial_tokens: Option<u32>,
}

impl RateLimitConfig {
    /// Create a rate limit allowing `requests_per_second` requests per second.
    pub fn requests_per_second(rps: f64) -> Self {
        let capacity = rps.ceil() as u32;
        Self {
            capacity,
            refill_rate: rps,
            initial_tokens: None,
        }
    }

    /// Create a rate limit allowing `count` requests per `period`.
    pub fn requests_per_period(count: u32, period: Duration) -> Self {
        let rps = count as f64 / period.as_secs_f64();
        Self {
            capacity: count,
            refill_rate: rps,
            initial_tokens: None,
        }
    }
}

/// Token bucket rate limiter.
///
/// Thread-safe and cheaply cloneable (wraps inner state in `Arc<Mutex<_>>`).
#[derive(Debug, Clone)]
pub struct TokenBucket {
    inner: Arc<Mutex<TokenBucketInner>>,
}

#[derive(Debug)]
struct TokenBucketInner {
    capacity: f64,
    tokens: f64,
    refill_rate: f64,
    last_refill: Instant,
}

impl TokenBucket {
    /// Create a new token bucket from the given configuration.
    pub fn new(config: &RateLimitConfig) -> Self {
        let initial = config
            .initial_tokens
            .map(|t| t as f64)
            .unwrap_or(config.capacity as f64);
        Self {
            inner: Arc::new(Mutex::new(TokenBucketInner {
                capacity: config.capacity as f64,
                tokens: initial,
                refill_rate: config.refill_rate,
                last_refill: Instant::now(),
            })),
        }
    }

    /// Acquire `count` tokens, waiting if necessary.
    ///
    /// Returns once the requested number of tokens are available. For simple
    /// request-count limiting, call `acquire(1)`.
    pub async fn acquire(&self, count: u32) {
        let count = count as f64;
        loop {
            let wait_duration = {
                let mut inner = self.inner.lock().await;
                inner.refill();

                if inner.tokens >= count {
                    inner.tokens -= count;
                    return;
                }

                let deficit = count - inner.tokens;
                Duration::from_secs_f64(deficit / inner.refill_rate)
            };

            tokio::time::sleep(wait_duration).await;
        }
    }

    /// Try to acquire `count` tokens without waiting.
    ///
    /// Returns `true` if tokens were acquired, `false` if insufficient tokens.
    pub async fn try_acquire(&self, count: u32) -> bool {
        let count = count as f64;
        let mut inner = self.inner.lock().await;
        inner.refill();

        if inner.tokens >= count {
            inner.tokens -= count;
            true
        } else {
            false
        }
    }

    /// Return the current number of available tokens (approximate).
    pub async fn available_tokens(&self) -> u32 {
        let mut inner = self.inner.lock().await;
        inner.refill();
        inner.tokens as u32
    }
}

impl TokenBucketInner {
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        if elapsed > 0.0 {
            self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity);
            self.last_refill = now;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_acquire_immediate() {
        let bucket = TokenBucket::new(&RateLimitConfig {
            capacity: 10,
            refill_rate: 10.0,
            initial_tokens: Some(10),
        });
        // Should return immediately — 10 tokens available.
        bucket.acquire(5).await;
        assert_eq!(bucket.available_tokens().await, 5);
    }

    #[tokio::test]
    async fn test_try_acquire_success() {
        let bucket = TokenBucket::new(&RateLimitConfig {
            capacity: 5,
            refill_rate: 1.0,
            initial_tokens: Some(3),
        });
        assert!(bucket.try_acquire(2).await);
        assert!(!bucket.try_acquire(2).await); // Only 1 left.
        assert!(bucket.try_acquire(1).await);
    }

    #[tokio::test]
    async fn test_try_acquire_insufficient() {
        let bucket = TokenBucket::new(&RateLimitConfig {
            capacity: 5,
            refill_rate: 1.0,
            initial_tokens: Some(1),
        });
        assert!(!bucket.try_acquire(3).await);
        // Tokens should not have been consumed.
        assert_eq!(bucket.available_tokens().await, 1);
    }

    #[tokio::test]
    async fn test_refill_over_time() {
        let bucket = TokenBucket::new(&RateLimitConfig {
            capacity: 10,
            refill_rate: 100.0, // 100 tokens/sec for fast test
            initial_tokens: Some(0),
        });
        assert_eq!(bucket.available_tokens().await, 0);
        tokio::time::sleep(Duration::from_millis(60)).await;
        let tokens = bucket.available_tokens().await;
        // Should have refilled some tokens (at least a few at 100/sec over 60ms).
        assert!(tokens >= 3, "expected >=3 tokens, got {tokens}");
    }

    #[tokio::test]
    async fn test_acquire_waits_for_refill() {
        let bucket = TokenBucket::new(&RateLimitConfig {
            capacity: 5,
            refill_rate: 100.0, // Fast refill for testing
            initial_tokens: Some(0),
        });
        let start = Instant::now();
        bucket.acquire(1).await;
        let elapsed = start.elapsed();
        // Should have waited briefly for refill.
        assert!(elapsed.as_millis() < 500, "waited too long: {elapsed:?}");
    }

    #[tokio::test]
    async fn test_requests_per_second_config() {
        let config = RateLimitConfig::requests_per_second(5.0);
        assert_eq!(config.capacity, 5);
        assert!((config.refill_rate - 5.0).abs() < f64::EPSILON);
        let bucket = TokenBucket::new(&config);
        // Full capacity initially.
        assert_eq!(bucket.available_tokens().await, 5);
    }

    #[tokio::test]
    async fn test_requests_per_period_config() {
        let config = RateLimitConfig::requests_per_period(60, Duration::from_secs(60));
        assert_eq!(config.capacity, 60);
        assert!((config.refill_rate - 1.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_capacity_not_exceeded() {
        let bucket = TokenBucket::new(&RateLimitConfig {
            capacity: 5,
            refill_rate: 1000.0,
            initial_tokens: Some(0),
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        let tokens = bucket.available_tokens().await;
        assert!(tokens <= 5, "tokens {tokens} exceeded capacity 5");
    }
}
