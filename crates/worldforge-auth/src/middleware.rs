use std::collections::HashMap;

use crate::{AuthError, Result};

/// Represents an authenticated identity extracted from request headers.
#[derive(Debug, Clone)]
pub enum AuthIdentity {
    ApiKey(String),
    Bearer(String),
}

/// Extract authentication from HTTP headers.
///
/// Supports:
/// - `X-API-Key: wf_...` header for API key auth
/// - `Authorization: Bearer <token>` header for JWT auth
pub fn extract_auth_from_headers(headers: &HashMap<String, String>) -> Result<AuthIdentity> {
    // Check X-API-Key first
    if let Some(api_key) = headers.get("x-api-key").or_else(|| headers.get("X-API-Key")) {
        if api_key.is_empty() {
            return Err(AuthError::InvalidApiKey);
        }
        return Ok(AuthIdentity::ApiKey(api_key.clone()));
    }

    // Check Authorization: Bearer
    if let Some(auth_header) = headers
        .get("authorization")
        .or_else(|| headers.get("Authorization"))
    {
        if let Some(token) = auth_header.strip_prefix("Bearer ") {
            if token.is_empty() {
                return Err(AuthError::InvalidToken("Empty bearer token".into()));
            }
            return Ok(AuthIdentity::Bearer(token.to_string()));
        }
        return Err(AuthError::InvalidToken(
            "Authorization header must use Bearer scheme".into(),
        ));
    }

    Err(AuthError::MissingAuth)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_api_key() {
        let mut headers = HashMap::new();
        headers.insert("X-API-Key".to_string(), "wf_abc123".to_string());

        match extract_auth_from_headers(&headers).unwrap() {
            AuthIdentity::ApiKey(key) => assert_eq!(key, "wf_abc123"),
            other => panic!("Expected ApiKey, got {:?}", other),
        }
    }

    #[test]
    fn test_extract_bearer_token() {
        let mut headers = HashMap::new();
        headers.insert(
            "Authorization".to_string(),
            "Bearer eyJhbGciOiJSUzI1NiJ9.test".to_string(),
        );

        match extract_auth_from_headers(&headers).unwrap() {
            AuthIdentity::Bearer(token) => assert_eq!(token, "eyJhbGciOiJSUzI1NiJ9.test"),
            other => panic!("Expected Bearer, got {:?}", other),
        }
    }

    #[test]
    fn test_api_key_takes_precedence() {
        let mut headers = HashMap::new();
        headers.insert("X-API-Key".to_string(), "wf_key".to_string());
        headers.insert("Authorization".to_string(), "Bearer token".to_string());

        match extract_auth_from_headers(&headers).unwrap() {
            AuthIdentity::ApiKey(key) => assert_eq!(key, "wf_key"),
            other => panic!("Expected ApiKey, got {:?}", other),
        }
    }

    #[test]
    fn test_missing_auth() {
        let headers = HashMap::new();
        match extract_auth_from_headers(&headers) {
            Err(AuthError::MissingAuth) => {}
            other => panic!("Expected MissingAuth, got {:?}", other),
        }
    }

    #[test]
    fn test_invalid_auth_scheme() {
        let mut headers = HashMap::new();
        headers.insert("Authorization".to_string(), "Basic dXNlcjpwYXNz".to_string());

        match extract_auth_from_headers(&headers) {
            Err(AuthError::InvalidToken(_)) => {}
            other => panic!("Expected InvalidToken, got {:?}", other),
        }
    }

    #[test]
    fn test_lowercase_headers() {
        let mut headers = HashMap::new();
        headers.insert("x-api-key".to_string(), "wf_lower".to_string());

        match extract_auth_from_headers(&headers).unwrap() {
            AuthIdentity::ApiKey(key) => assert_eq!(key, "wf_lower"),
            other => panic!("Expected ApiKey, got {:?}", other),
        }
    }
}
