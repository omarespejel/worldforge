//! Generic async job submission and polling for WorldForge providers.
//!
//! Extends the existing `polling` module with a higher-level abstraction for
//! the common submit → poll → download pattern used by video generation APIs.
//!
//! Re-exports [`polling::PollingConfig`], [`polling::PollStatus`], and
//! [`polling::poll_until_complete`] for convenience, and adds a [`AsyncJobRunner`]
//! that encapsulates the full lifecycle.

use std::future::Future;
use std::time::Duration;

use worldforge_core::error::{Result, WorldForgeError};

pub use crate::polling::{PollStatus, PollingConfig, poll_until_complete};

/// Status of an async job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobStatus {
    /// Job has been submitted and is queued.
    Queued,
    /// Job is actively being processed.
    Processing,
    /// Job completed successfully.
    Completed,
    /// Job failed with the given reason.
    Failed(String),
    /// Job was cancelled.
    Cancelled,
}

impl JobStatus {
    /// Whether the job is in a terminal state (completed, failed, or cancelled).
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed(_) | Self::Cancelled)
    }
}

/// A tracked async job with its ID and current status.
#[derive(Debug, Clone)]
pub struct AsyncJob<T> {
    /// Provider-assigned job/task ID.
    pub job_id: String,
    /// Current job status.
    pub status: JobStatus,
    /// The result, available once the job completes.
    pub result: Option<T>,
}

/// Runner for the async submit → poll → collect pattern.
///
/// Encapsulates the full lifecycle of an async provider job:
/// 1. Submit the job and get a job ID
/// 2. Poll until complete with exponential backoff
/// 3. Collect the final result
///
/// # Type Parameters
///
/// * `T` — the final result type (e.g., video URL, `VideoClip`)
#[derive(Debug, Clone)]
pub struct AsyncJobRunner {
    /// Provider name for logging and error context.
    provider_name: String,
    /// Polling configuration.
    poll_config: PollingConfig,
    /// Overall timeout for the entire job lifecycle.
    overall_timeout: Option<Duration>,
}

impl AsyncJobRunner {
    /// Create a new runner for the given provider.
    pub fn new(provider_name: impl Into<String>) -> Self {
        Self {
            provider_name: provider_name.into(),
            poll_config: PollingConfig::default(),
            overall_timeout: None,
        }
    }

    /// Set custom polling configuration.
    pub fn with_poll_config(mut self, config: PollingConfig) -> Self {
        self.poll_config = config;
        self
    }

    /// Set an overall timeout for the entire job lifecycle.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.overall_timeout = Some(timeout);
        self
    }

    /// Submit a job and poll until completion.
    ///
    /// # Arguments
    ///
    /// * `submit_fn` — async closure that submits the job and returns a job ID
    /// * `poll_fn` — async closure that checks job status given a job ID
    ///
    /// # Returns
    ///
    /// The completed `AsyncJob<T>` with result populated.
    pub async fn run<S, SF, P, PF, T>(
        &self,
        submit_fn: S,
        poll_fn: P,
    ) -> Result<AsyncJob<T>>
    where
        S: FnOnce() -> SF,
        SF: Future<Output = Result<String>>,
        P: Fn(String) -> PF,
        PF: Future<Output = Result<PollStatus<T>>>,
    {
        // Step 1: Submit
        tracing::info!(provider = %self.provider_name, "submitting async job");
        let job_id = submit_fn().await?;
        tracing::info!(
            provider = %self.provider_name,
            job_id = %job_id,
            "job submitted, starting poll"
        );

        // Step 2: Poll with optional overall timeout
        let poll_id = job_id.clone();
        let poll_future = poll_until_complete(&self.provider_name, &self.poll_config, || {
            let id = poll_id.clone();
            poll_fn(id)
        });

        let result = if let Some(timeout) = self.overall_timeout {
            tokio::time::timeout(timeout, poll_future)
                .await
                .map_err(|_| WorldForgeError::ProviderTimeout {
                    provider: self.provider_name.clone(),
                    timeout_ms: timeout.as_millis() as u64,
                })?
        } else {
            poll_future.await
        }?;

        tracing::info!(
            provider = %self.provider_name,
            job_id = %job_id,
            "async job completed"
        );

        Ok(AsyncJob {
            job_id,
            status: JobStatus::Completed,
            result: Some(result),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn test_job_status_terminal() {
        assert!(!JobStatus::Queued.is_terminal());
        assert!(!JobStatus::Processing.is_terminal());
        assert!(JobStatus::Completed.is_terminal());
        assert!(JobStatus::Failed("err".into()).is_terminal());
        assert!(JobStatus::Cancelled.is_terminal());
    }

    #[tokio::test]
    async fn test_runner_submit_and_poll() {
        let runner = AsyncJobRunner::new("test-provider").with_poll_config(PollingConfig {
            initial_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(5),
            backoff_factor: 1.0,
            max_attempts: 10,
        });

        let poll_count = AtomicU32::new(0);
        let job = runner
            .run(
                || async { Ok("job-123".to_string()) },
                |job_id| {
                    let count = poll_count.fetch_add(1, Ordering::SeqCst);
                    async move {
                        assert_eq!(job_id, "job-123");
                        if count < 2 {
                            Ok(PollStatus::Pending)
                        } else {
                            Ok(PollStatus::Complete("video-url".to_string()))
                        }
                    }
                },
            )
            .await
            .unwrap();

        assert_eq!(job.job_id, "job-123");
        assert_eq!(job.status, JobStatus::Completed);
        assert_eq!(job.result.unwrap(), "video-url");
    }

    #[tokio::test]
    async fn test_runner_submit_failure() {
        let runner = AsyncJobRunner::new("test-provider");

        let result: Result<AsyncJob<String>> = runner
            .run(
                || async {
                    Err(WorldForgeError::ProviderAuthError("bad key".into()))
                },
                |_| async { Ok(PollStatus::Pending) },
            )
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_runner_with_timeout() {
        let runner = AsyncJobRunner::new("test-provider")
            .with_poll_config(PollingConfig {
                initial_delay: Duration::from_millis(50),
                max_delay: Duration::from_millis(50),
                backoff_factor: 1.0,
                max_attempts: 1000,
            })
            .with_timeout(Duration::from_millis(100));

        let result: Result<AsyncJob<String>> = runner
            .run(
                || async { Ok("job-timeout".to_string()) },
                |_| async { Ok(PollStatus::Pending) },
            )
            .await;

        assert!(matches!(
            result,
            Err(WorldForgeError::ProviderTimeout { .. })
        ));
    }
}
