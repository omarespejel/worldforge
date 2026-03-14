//! Error types for WorldForge.
//!
//! All fallible operations return `Result<T, WorldForgeError>`.

use crate::types::WorldId;
use serde::{Deserialize, Serialize};

/// Top-level error type for all WorldForge operations.
#[derive(Debug, Clone, thiserror::Error, Serialize, Deserialize)]
pub enum WorldForgeError {
    // -- Provider errors --
    /// The requested provider was not found in the registry.
    #[error("provider not found: {0}")]
    ProviderNotFound(String),

    /// The provider is currently unavailable.
    #[error("provider unavailable: {provider} — {reason}")]
    ProviderUnavailable { provider: String, reason: String },

    /// The provider call timed out.
    #[error("provider timeout: {provider} after {timeout_ms}ms")]
    ProviderTimeout { provider: String, timeout_ms: u64 },

    /// The provider rate-limited the request.
    #[error("provider rate limited: {provider}, retry after {retry_after_ms}ms")]
    ProviderRateLimited {
        provider: String,
        retry_after_ms: u64,
    },

    /// Authentication failed for the provider.
    #[error("provider auth error: {0}")]
    ProviderAuthError(String),

    // -- Capability errors --
    /// The action is not supported by the provider.
    #[error("unsupported action on {provider}: {action}")]
    UnsupportedAction { provider: String, action: String },

    /// The capability is not supported by the provider.
    #[error("unsupported capability on {provider}: {capability}")]
    UnsupportedCapability {
        provider: String,
        capability: String,
    },

    // -- State errors --
    /// The requested world was not found.
    #[error("world not found: {0}")]
    WorldNotFound(WorldId),

    /// The world state is invalid.
    #[error("invalid state: {0}")]
    InvalidState(String),

    /// The world state is corrupted.
    #[error("state corrupted for world {world_id}: {details}")]
    StateCorrupted { world_id: WorldId, details: String },

    // -- Guardrail errors --
    /// A guardrail was violated.
    #[error("guardrail violation: {guardrail} — {details}")]
    GuardrailViolation { guardrail: String, details: String },

    /// A blocking guardrail prevented the operation.
    #[error("guardrail blocked operation: {reason}")]
    GuardrailBlocked { reason: String },

    // -- Planning errors --
    /// Planning failed.
    #[error("planning failed: {reason}")]
    PlanningFailed { reason: String },

    /// Planning timed out.
    #[error("planning timeout after {elapsed_ms}ms")]
    PlanningTimeout { elapsed_ms: u64 },

    /// No feasible plan was found.
    #[error("no feasible plan for goal '{goal}': {reason}")]
    NoFeasiblePlan { goal: String, reason: String },

    // -- Verification errors --
    /// ZK verification failed.
    #[error("verification failed ({proof_type}): {details}")]
    VerificationFailed { proof_type: String, details: String },

    // -- General errors --
    /// Serialization or deserialization failed.
    #[error("serialization error: {0}")]
    SerializationError(String),

    /// A network error occurred.
    #[error("network error: {0}")]
    NetworkError(String),

    /// An internal error occurred.
    #[error("internal error: {0}")]
    InternalError(String),
}

/// Convenience alias for WorldForge results.
pub type Result<T> = std::result::Result<T, WorldForgeError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let e = WorldForgeError::ProviderNotFound("cosmos".to_string());
        assert_eq!(e.to_string(), "provider not found: cosmos");
    }

    #[test]
    fn test_error_serialization_roundtrip() {
        let e = WorldForgeError::ProviderTimeout {
            provider: "cosmos".to_string(),
            timeout_ms: 5000,
        };
        let json = serde_json::to_string(&e).unwrap();
        let e2: WorldForgeError = serde_json::from_str(&json).unwrap();
        assert_eq!(e.to_string(), e2.to_string());
    }
}
