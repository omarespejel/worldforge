//! Shared HTTP client builder for WorldForge providers.
//!
//! Provides a common `reqwest::Client` configuration with timeout, user-agent,
//! rustls TLS, and helpers for authentication, header injection, and response
//! error mapping.

use std::time::Duration;

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use worldforge_core::error::{Result, WorldForgeError};

/// Default request timeout for provider API calls.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);

/// Default user-agent sent with all provider requests.
const DEFAULT_USER_AGENT: &str = concat!("worldforge/", env!("CARGO_PKG_VERSION"));

/// Authentication method for provider APIs.
#[derive(Debug, Clone)]
pub enum AuthMethod {
    /// Bearer token in the `Authorization` header.
    Bearer(String),
    /// Custom header name and value (e.g., `X-Api-Key: <key>`).
    CustomHeader { name: String, value: String },
    /// No authentication.
    None,
}

/// Builder for creating a configured `reqwest::Client` with common provider defaults.
#[derive(Debug, Clone)]
pub struct HttpClientBuilder {
    timeout: Duration,
    user_agent: String,
    auth: AuthMethod,
    default_headers: HeaderMap,
}

impl Default for HttpClientBuilder {
    fn default() -> Self {
        Self {
            timeout: DEFAULT_TIMEOUT,
            user_agent: DEFAULT_USER_AGENT.to_string(),
            auth: AuthMethod::None,
            default_headers: HeaderMap::new(),
        }
    }
}

impl HttpClientBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the request timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set a custom user-agent string.
    pub fn user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.user_agent = user_agent.into();
        self
    }

    /// Set the authentication method.
    pub fn auth(mut self, auth: AuthMethod) -> Self {
        self.auth = auth;
        self
    }

    /// Convenience: set Bearer token authentication.
    pub fn bearer_token(self, token: impl Into<String>) -> Self {
        self.auth(AuthMethod::Bearer(token.into()))
    }

    /// Convenience: set a custom authentication header.
    pub fn api_key_header(self, header_name: impl Into<String>, key: impl Into<String>) -> Self {
        self.auth(AuthMethod::CustomHeader {
            name: header_name.into(),
            value: key.into(),
        })
    }

    /// Add a default header to all requests.
    pub fn default_header(mut self, name: &str, value: &str) -> Self {
        if let (Ok(n), Ok(v)) = (
            HeaderName::from_bytes(name.as_bytes()),
            HeaderValue::from_str(value),
        ) {
            self.default_headers.insert(n, v);
        }
        self
    }

    /// Build the `reqwest::Client`.
    ///
    /// # Errors
    ///
    /// Returns `WorldForgeError::ProviderUnavailable` if the client cannot be
    /// constructed (e.g., invalid header values).
    pub fn build(self) -> Result<reqwest::Client> {
        let mut headers = self.default_headers;

        match &self.auth {
            AuthMethod::Bearer(token) => {
                let value = HeaderValue::from_str(&format!("Bearer {token}")).map_err(|e| {
                    WorldForgeError::ProviderUnavailable {
                        provider: "http_client".to_string(),
                        reason: format!("invalid bearer token: {e}"),
                    }
                })?;
                headers.insert(reqwest::header::AUTHORIZATION, value);
            }
            AuthMethod::CustomHeader { name, value } => {
                let header_name =
                    HeaderName::from_bytes(name.as_bytes()).map_err(|e| {
                        WorldForgeError::ProviderUnavailable {
                            provider: "http_client".to_string(),
                            reason: format!("invalid header name '{name}': {e}"),
                        }
                    })?;
                let header_value =
                    HeaderValue::from_str(value).map_err(|e| {
                        WorldForgeError::ProviderUnavailable {
                            provider: "http_client".to_string(),
                            reason: format!("invalid header value: {e}"),
                        }
                    })?;
                headers.insert(header_name, header_value);
            }
            AuthMethod::None => {}
        }

        reqwest::Client::builder()
            .timeout(self.timeout)
            .user_agent(&self.user_agent)
            .default_headers(headers)
            .build()
            .map_err(|e| WorldForgeError::ProviderUnavailable {
                provider: "http_client".to_string(),
                reason: format!("failed to build HTTP client: {e}"),
            })
    }
}

/// Map a `reqwest::Response` to a `WorldForgeError` if the status is not successful.
///
/// Consumes the response body for error context. Returns `Ok(response)` on 2xx.
pub async fn check_response(
    provider_name: &str,
    response: reqwest::Response,
) -> Result<reqwest::Response> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }

    let body = response.text().await.unwrap_or_default();

    match status {
        reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN => {
            Err(WorldForgeError::ProviderAuthError(format!(
                "{provider_name}: {body}"
            )))
        }
        reqwest::StatusCode::TOO_MANY_REQUESTS => Err(WorldForgeError::ProviderRateLimited {
            provider: provider_name.to_string(),
            retry_after_ms: 5000,
        }),
        s if s.is_server_error() => Err(WorldForgeError::ProviderUnavailable {
            provider: provider_name.to_string(),
            reason: format!("server error HTTP {status}: {body}"),
        }),
        _ => Err(WorldForgeError::ProviderUnavailable {
            provider: provider_name.to_string(),
            reason: format!("HTTP {status}: {body}"),
        }),
    }
}

/// Check whether an error is retryable (5xx, timeout, rate limit).
pub fn is_retryable_status(status: reqwest::StatusCode) -> bool {
    status.is_server_error() || status == reqwest::StatusCode::TOO_MANY_REQUESTS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_defaults() {
        let client = HttpClientBuilder::new().build();
        assert!(client.is_ok());
    }

    #[test]
    fn test_builder_with_bearer() {
        let client = HttpClientBuilder::new()
            .bearer_token("test-token-123")
            .timeout(Duration::from_secs(30))
            .build();
        assert!(client.is_ok());
    }

    #[test]
    fn test_builder_with_custom_header() {
        let client = HttpClientBuilder::new()
            .api_key_header("X-Api-Key", "my-secret")
            .default_header("Content-Type", "application/json")
            .build();
        assert!(client.is_ok());
    }

    #[test]
    fn test_builder_with_user_agent() {
        let client = HttpClientBuilder::new()
            .user_agent("custom-agent/1.0")
            .build();
        assert!(client.is_ok());
    }

    #[test]
    fn test_is_retryable_status() {
        assert!(is_retryable_status(reqwest::StatusCode::INTERNAL_SERVER_ERROR));
        assert!(is_retryable_status(reqwest::StatusCode::BAD_GATEWAY));
        assert!(is_retryable_status(reqwest::StatusCode::SERVICE_UNAVAILABLE));
        assert!(is_retryable_status(reqwest::StatusCode::TOO_MANY_REQUESTS));
        assert!(!is_retryable_status(reqwest::StatusCode::OK));
        assert!(!is_retryable_status(reqwest::StatusCode::BAD_REQUEST));
        assert!(!is_retryable_status(reqwest::StatusCode::NOT_FOUND));
    }

    #[test]
    fn test_auth_method_none() {
        let client = HttpClientBuilder::new()
            .auth(AuthMethod::None)
            .build();
        assert!(client.is_ok());
    }
}
