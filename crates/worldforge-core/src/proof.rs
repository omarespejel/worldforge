//! Shared proof metadata types used across planning and verification.
//!
//! The concrete proving and verification logic lives in `worldforge-verify`.
//! This module only defines the serializable proof payloads that other crates
//! need to reference without introducing a crate cycle.

use std::str::FromStr;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::WorldForgeError;

/// ZK proof type describing what is being proved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZkProofType {
    /// Verify that inference was computed correctly.
    InferenceVerification {
        /// Hash of the model weights.
        model_hash: [u8; 32],
        /// Hash of the input state.
        input_hash: [u8; 32],
        /// Hash of the output state.
        output_hash: [u8; 32],
    },
    /// Verify guardrail compliance for a plan.
    GuardrailCompliance {
        /// Hash of the plan.
        plan_hash: [u8; 32],
        /// Hashes of individual guardrails checked.
        guardrail_hashes: Vec<[u8; 32]>,
        /// Whether all guardrails passed.
        all_passed: bool,
    },
    /// Verify data provenance.
    DataProvenance {
        /// Hash of the data.
        data_hash: [u8; 32],
        /// Timestamp of the data.
        timestamp: u64,
        /// Commitment to the data source.
        source_commitment: [u8; 32],
    },
}

/// Backend used for proof generation and verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationBackend {
    /// EZKL for ML model inference proofs.
    Ezkl,
    /// Cairo/STARK for guardrail and general computation proofs.
    Stark,
    /// Mock backend for testing.
    Mock,
}

impl VerificationBackend {
    /// Canonical lowercase backend identifier used by user-facing surfaces.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ezkl => "ezkl",
            Self::Stark => "stark",
            Self::Mock => "mock",
        }
    }
}

impl Serialize for VerificationBackend {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for VerificationBackend {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct BackendVisitor;

        impl<'de> Visitor<'de> for BackendVisitor {
            type Value = VerificationBackend;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a verification backend string")
            }

            fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
            where
                E: de::Error,
            {
                value.parse::<VerificationBackend>().map_err(E::custom)
            }
        }

        deserializer.deserialize_str(BackendVisitor)
    }
}

impl FromStr for VerificationBackend {
    type Err = WorldForgeError;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "ezkl" => Ok(Self::Ezkl),
            "stark" => Ok(Self::Stark),
            "mock" => Ok(Self::Mock),
            other => Err(WorldForgeError::InvalidState(format!(
                "unknown verification backend: {other}. Available: mock, ezkl, stark"
            ))),
        }
    }
}

/// A ZK proof with its type and serialized proof data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZkProof {
    /// Type of proof.
    pub proof_type: ZkProofType,
    /// Serialized proof data (backend-specific format).
    pub proof_data: Vec<u8>,
    /// Backend that generated this proof.
    pub backend: VerificationBackend,
    /// Time taken to generate the proof in milliseconds.
    pub generation_time_ms: u64,
}

/// Verification result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Whether the proof is valid.
    pub valid: bool,
    /// Time taken to verify in milliseconds.
    pub verification_time_ms: u64,
    /// Human-readable details.
    pub details: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verification_backend_parse_accepts_known_values() {
        assert_eq!(
            "mock".parse::<VerificationBackend>().unwrap(),
            VerificationBackend::Mock
        );
        assert_eq!(
            "EZKL".parse::<VerificationBackend>().unwrap(),
            VerificationBackend::Ezkl
        );
        assert_eq!(
            "stark".parse::<VerificationBackend>().unwrap(),
            VerificationBackend::Stark
        );
    }

    #[test]
    fn test_verification_backend_parse_rejects_unknown_value() {
        let error = "invalid".parse::<VerificationBackend>().unwrap_err();
        assert!(error.to_string().contains("unknown verification backend"));
    }

    #[test]
    fn test_verification_backend_serializes_lowercase() {
        let json = serde_json::to_string(&VerificationBackend::Mock).unwrap();
        assert_eq!(json, r#""mock""#);

        let json = serde_json::to_string(&VerificationBackend::Ezkl).unwrap();
        assert_eq!(json, r#""ezkl""#);

        let json = serde_json::to_string(&VerificationBackend::Stark).unwrap();
        assert_eq!(json, r#""stark""#);
    }

    #[test]
    fn test_verification_backend_deserializes_legacy_capitalized_values() {
        let mock: VerificationBackend = serde_json::from_str(r#""Mock""#).unwrap();
        let ezkl: VerificationBackend = serde_json::from_str(r#""Ezkl""#).unwrap();
        let stark: VerificationBackend = serde_json::from_str(r#""Stark""#).unwrap();

        assert_eq!(mock, VerificationBackend::Mock);
        assert_eq!(ezkl, VerificationBackend::Ezkl);
        assert_eq!(stark, VerificationBackend::Stark);
    }
}
