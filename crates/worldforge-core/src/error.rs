//! Error types for WorldForge.
//!
//! All fallible operations return `Result<T, WorldForgeError>`.

use crate::guardrail::GuardrailResult;
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
    #[error(
        "guardrail blocked operation: {}",
        format_guardrail_violations(.violations)
    )]
    GuardrailBlocked {
        /// Blocking guardrail results that caused the operation to be rejected.
        violations: Vec<GuardrailResult>,
    },

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

fn format_guardrail_violations(violations: &[GuardrailResult]) -> String {
    if violations.is_empty() {
        return "blocking violation".to_string();
    }

    violations
        .iter()
        .map(|result| {
            let details = result
                .violation_details
                .as_deref()
                .unwrap_or("violation detected");
            format!("{}: {}", result.guardrail_name, details)
        })
        .collect::<Vec<_>>()
        .join("; ")
}

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

    #[test]
    fn test_guardrail_blocked_error_serialization_roundtrip() {
        let e = WorldForgeError::GuardrailBlocked {
            violations: vec![GuardrailResult {
                guardrail_name: "NoCollisions".to_string(),
                passed: false,
                violation_details: Some("collision between 'left' and 'right'".to_string()),
                severity: crate::guardrail::ViolationSeverity::Blocking,
            }],
        };
        let json = serde_json::to_string(&e).unwrap();
        let e2: WorldForgeError = serde_json::from_str(&json).unwrap();
        assert_eq!(e.to_string(), e2.to_string());
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn arb_error() -> impl Strategy<Value = WorldForgeError> {
            prop_oneof![
                ".*".prop_map(WorldForgeError::ProviderNotFound),
                (".*", ".*").prop_map(|(p, r)| WorldForgeError::ProviderUnavailable {
                    provider: p,
                    reason: r,
                }),
                (".*", any::<u64>()).prop_map(|(p, t)| WorldForgeError::ProviderTimeout {
                    provider: p,
                    timeout_ms: t,
                }),
                ".*".prop_map(WorldForgeError::ProviderAuthError),
                (".*", ".*").prop_map(|(name, details)| WorldForgeError::GuardrailBlocked {
                    violations: vec![GuardrailResult {
                        guardrail_name: name,
                        passed: false,
                        violation_details: Some(details),
                        severity: crate::guardrail::ViolationSeverity::Blocking,
                    }],
                }),
                ".*".prop_map(WorldForgeError::SerializationError),
                ".*".prop_map(WorldForgeError::NetworkError),
                ".*".prop_map(WorldForgeError::InternalError),
            ]
        }

        proptest! {
            #[test]
            fn error_serialization_roundtrip(e in arb_error()) {
                let json = serde_json::to_string(&e).unwrap();
                let e2: WorldForgeError = serde_json::from_str(&json).unwrap();
                prop_assert_eq!(e.to_string(), e2.to_string());
            }
        }
    }
}
