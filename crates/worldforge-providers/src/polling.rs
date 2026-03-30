//! Shared async polling infrastructure for submit/poll/download API patterns.
//!
//! Many video generation APIs (KLING, MiniMax, Sora 2, Veo 3) follow the same
//! pattern: submit a generation request, receive a task ID, poll for completion,
//! then download the result. This module provides reusable helpers for that flow.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use worldforge_core::error::{Result, WorldForgeError};

/// Configuration for async polling behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PollingConfig {
    /// Initial delay before first poll.
    pub initial_delay: Duration,
    /// Maximum delay between polls.
    pub max_delay: Duration,
    /// Backoff multiplier applied after each poll.
    pub backoff_factor: f32,
    /// Maximum number of poll attempts before timeout.
    pub max_attempts: u32,
}

impl Default for PollingConfig {
    fn default() -> Self {
        Self {
            initial_delay: Duration::from_secs(2),
            max_delay: Duration::from_secs(30),
            backoff_factor: 1.5,
            max_attempts: 60,
        }
    }
}

/// Status returned by a poll function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PollStatus<T> {
    /// Task is still in progress.
    Pending,
    /// Task completed successfully.
    Complete(T),
    /// Task failed with an error message.
    Failed(String),
}

/// Poll an async task until it completes, fails, or times out.
///
/// The `poll_fn` is called repeatedly with exponential backoff. It should
/// return `PollStatus::Pending` while the task is in progress,
/// `PollStatus::Complete(T)` when done, or `PollStatus::Failed(reason)`
/// on error.
///
/// # Errors
///
/// Returns `ProviderTimeout` if `max_attempts` is exhausted, or propagates
/// errors from the poll function.
pub async fn poll_until_complete<F, Fut, T>(
    provider_name: &str,
    config: &PollingConfig,
    poll_fn: F,
) -> Result<T>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<PollStatus<T>>>,
{
    let mut delay = config.initial_delay;

    for attempt in 0..config.max_attempts {
        if attempt > 0 {
            tokio::time::sleep(delay).await;
            delay = Duration::from_secs_f64(
                (delay.as_secs_f64() * config.backoff_factor as f64)
                    .min(config.max_delay.as_secs_f64()),
            );
        }

        match poll_fn().await? {
            PollStatus::Pending => {
                tracing::debug!(
                    provider = provider_name,
                    attempt = attempt + 1,
                    max_attempts = config.max_attempts,
                    next_delay_ms = delay.as_millis() as u64,
                    "task still pending, will retry"
                );
            }
            PollStatus::Complete(result) => return Ok(result),
            PollStatus::Failed(reason) => {
                return Err(WorldForgeError::ProviderUnavailable {
                    provider: provider_name.to_string(),
                    reason,
                });
            }
        }
    }

    Err(WorldForgeError::ProviderTimeout {
        provider: provider_name.to_string(),
        timeout_ms: (config.max_attempts as u64) * config.max_delay.as_millis() as u64,
    })
}

/// Common response handling for providers that return HTTP status codes.
///
/// Maps standard HTTP error codes to appropriate WorldForgeError variants.
pub fn check_http_response(
    provider_name: &str,
    status: reqwest::StatusCode,
    body: &str,
) -> Result<()> {
    match status {
        s if s.is_success() => Ok(()),
        reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN => Err(
            WorldForgeError::ProviderAuthError(format!("{provider_name}: {body}")),
        ),
        reqwest::StatusCode::TOO_MANY_REQUESTS => Err(WorldForgeError::ProviderRateLimited {
            provider: provider_name.to_string(),
            retry_after_ms: 5000,
        }),
        _ => Err(WorldForgeError::ProviderUnavailable {
            provider: provider_name.to_string(),
            reason: format!("HTTP {status}: {body}"),
        }),
    }
}

/// Build a synthetic video clip from provider metadata.
///
/// Used by providers that return video URLs or frame data. Creates a
/// deterministic stub `VideoClip` with proper metadata for downstream
/// consumption.
pub fn build_stub_video_clip(
    resolution: (u32, u32),
    fps: f32,
    duration_seconds: f64,
    marker: u64,
) -> worldforge_core::types::VideoClip {
    use worldforge_core::types::{DType, Device, Frame, SimTime, Tensor, TensorData, VideoClip};

    let num_frames = (fps as f64 * duration_seconds).ceil() as usize;
    let num_frames = num_frames.max(1);
    let (w, h) = resolution;
    let pixel_count = (w * h * 3) as usize;

    let frames: Vec<Frame> = (0..num_frames)
        .map(|i| {
            let t = i as f64 / fps as f64;
            // Deterministic pixel value from marker + frame index
            let value = ((marker.wrapping_add(i as u64)) % 200 + 30) as u8;
            Frame {
                data: Tensor {
                    data: TensorData::UInt8(vec![value; pixel_count]),
                    shape: vec![h as usize, w as usize, 3],
                    dtype: DType::UInt8,
                    device: Device::Cpu,
                },
                timestamp: SimTime {
                    step: i as u64,
                    seconds: t,
                    dt: 1.0 / fps as f64,
                },
                camera: None,
                depth: None,
                segmentation: None,
            }
        })
        .collect();

    VideoClip {
        frames,
        fps,
        resolution,
        duration: duration_seconds,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[tokio::test]
    async fn test_poll_completes_immediately() {
        let config = PollingConfig {
            initial_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(10),
            backoff_factor: 1.0,
            max_attempts: 5,
        };
        let result: Result<String> = poll_until_complete("test", &config, || async {
            Ok(PollStatus::Complete("done".to_string()))
        })
        .await;
        assert_eq!(result.unwrap(), "done");
    }

    #[tokio::test]
    async fn test_poll_retries_then_completes() {
        let counter = AtomicU32::new(0);
        let config = PollingConfig {
            initial_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(5),
            backoff_factor: 1.0,
            max_attempts: 10,
        };
        let result: Result<u32> = poll_until_complete("test", &config, || {
            let count = counter.fetch_add(1, Ordering::SeqCst);
            async move {
                if count < 3 {
                    Ok(PollStatus::Pending)
                } else {
                    Ok(PollStatus::Complete(count))
                }
            }
        })
        .await;
        assert_eq!(result.unwrap(), 3);
    }

    #[tokio::test]
    async fn test_poll_times_out() {
        let config = PollingConfig {
            initial_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(1),
            backoff_factor: 1.0,
            max_attempts: 3,
        };
        let result: Result<()> = poll_until_complete("test-provider", &config, || async {
            Ok(PollStatus::Pending)
        })
        .await;
        match result {
            Err(WorldForgeError::ProviderTimeout { provider, .. }) => {
                assert_eq!(provider, "test-provider");
            }
            other => panic!("expected ProviderTimeout, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_poll_propagates_failure() {
        let config = PollingConfig::default();
        let result: Result<()> = poll_until_complete("test-provider", &config, || async {
            Ok(PollStatus::Failed("generation failed".to_string()))
        })
        .await;
        match result {
            Err(WorldForgeError::ProviderUnavailable { reason, .. }) => {
                assert_eq!(reason, "generation failed");
            }
            other => panic!("expected ProviderUnavailable, got {other:?}"),
        }
    }

    #[test]
    fn test_check_http_response_success() {
        assert!(check_http_response("test", reqwest::StatusCode::OK, "").is_ok());
    }

    #[test]
    fn test_check_http_response_auth_error() {
        let result = check_http_response("test", reqwest::StatusCode::UNAUTHORIZED, "bad key");
        assert!(matches!(result, Err(WorldForgeError::ProviderAuthError(_))));
    }

    #[test]
    fn test_check_http_response_rate_limited() {
        let result = check_http_response("test", reqwest::StatusCode::TOO_MANY_REQUESTS, "");
        assert!(matches!(
            result,
            Err(WorldForgeError::ProviderRateLimited { .. })
        ));
    }

    #[test]
    fn test_build_stub_video_clip() {
        let clip = build_stub_video_clip((640, 480), 24.0, 2.0, 42);
        assert_eq!(clip.resolution, (640, 480));
        assert_eq!(clip.fps, 24.0);
        assert_eq!(clip.duration, 2.0);
        assert_eq!(clip.frames.len(), 48);
    }
}
