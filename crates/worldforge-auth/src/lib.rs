pub mod api_key;
pub mod jwt;
pub mod middleware;
pub mod rbac;

pub use api_key::{ApiKeyAuth, ApiKeyIdentity, ApiKeyMetadata, ApiKeyStore, InMemoryApiKeyStore};
pub use jwt::{Claims, JwtAuth};
pub use middleware::{extract_auth_from_headers, AuthIdentity};
pub use rbac::{Permission, Role, role_has_permission};

/// Auth error types.
#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("invalid API key")]
    InvalidApiKey,
    #[error("API key has been revoked")]
    RevokedApiKey,
    #[error("invalid JWT token: {0}")]
    InvalidToken(String),
    #[error("token expired")]
    TokenExpired,
    #[error("missing authentication")]
    MissingAuth,
    #[error("insufficient permissions")]
    InsufficientPermissions,
    #[error("internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, AuthError>;
