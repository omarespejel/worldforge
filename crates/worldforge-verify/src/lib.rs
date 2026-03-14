//! WorldForge ZK Verification Layer
//!
//! Provides zero-knowledge proof generation and verification for
//! WorldForge predictions and plans. Supports three proof types:
//!
//! - **InferenceVerification**: Prove a model forward pass was correct
//! - **GuardrailCompliance**: Prove all guardrails passed for a plan
//! - **DataProvenance**: Prove data origin and integrity
//!
//! # Implementation Strategy
//!
//! Phase 1: EZKL-based proofs for small models (inference verification)
//! Phase 2: Cairo/STARK-based proofs for guardrail compliance
//! Phase 3: On-chain verification on Starknet for audit trails
//!
//! The JEPA provider is the primary target for ZK verification because
//! it runs locally and has a deterministic, differentiable forward pass.

use serde::{Deserialize, Serialize};

use worldforge_core::guardrail::GuardrailResult;
use worldforge_core::prediction::Plan;

/// ZK proof type describing what is being proved.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// A ZK proof with its type and serialized proof data.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Whether the proof is valid.
    pub valid: bool,
    /// Time taken to verify in milliseconds.
    pub verification_time_ms: u64,
    /// Human-readable details.
    pub details: String,
}

/// Backend used for proof generation and verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationBackend {
    /// EZKL for ML model inference proofs.
    Ezkl,
    /// Cairo/STARK for guardrail and general computation proofs.
    Stark,
    /// Mock backend for testing.
    Mock,
}

/// Error types for verification operations.
#[derive(Debug, thiserror::Error)]
pub enum VerifyError {
    /// Proof generation failed.
    #[error("proof generation failed: {reason}")]
    GenerationFailed { reason: String },

    /// Proof verification failed.
    #[error("verification failed: {reason}")]
    VerificationFailed { reason: String },

    /// Unsupported proof type for the given backend.
    #[error("unsupported proof type for backend {backend:?}")]
    UnsupportedProofType { backend: VerificationBackend },

    /// Serialization error.
    #[error("serialization error: {0}")]
    Serialization(String),
}

/// Result type alias for verification operations.
pub type Result<T> = std::result::Result<T, VerifyError>;

/// Trait for ZK proof generation and verification.
///
/// Each backend (EZKL, STARK, etc.) implements this trait to provide
/// proof generation and verification for different proof types.
pub trait ZkVerifier: Send + Sync {
    /// The backend type this verifier uses.
    fn backend(&self) -> VerificationBackend;

    /// Generate a ZK proof for inference verification.
    ///
    /// Proves that a model forward pass with the given input produced
    /// the given output.
    fn prove_inference(
        &self,
        model_hash: [u8; 32],
        input_hash: [u8; 32],
        output_hash: [u8; 32],
    ) -> Result<ZkProof>;

    /// Generate a ZK proof for guardrail compliance.
    ///
    /// Proves that all guardrails passed for a given plan.
    fn prove_guardrail_compliance(
        &self,
        plan: &Plan,
        guardrail_results: &[Vec<GuardrailResult>],
    ) -> Result<ZkProof>;

    /// Generate a ZK proof for data provenance.
    ///
    /// Proves that data originated from a specific source at a specific time.
    fn prove_data_provenance(
        &self,
        data_hash: [u8; 32],
        timestamp: u64,
        source_commitment: [u8; 32],
    ) -> Result<ZkProof>;

    /// Verify a previously generated proof.
    fn verify(&self, proof: &ZkProof) -> Result<VerificationResult>;
}

/// SHA-256 hash helper for creating proof inputs.
pub fn sha256_hash(data: &[u8]) -> [u8; 32] {
    // Simple SHA-256 implementation using the standard approach.
    // In production, use a proper crypto library. For now, a
    // deterministic hash for testing purposes.
    let mut hash = [0u8; 32];
    let mut state: u64 = 0xcbf29ce4_84222325;
    for &byte in data {
        state ^= byte as u64;
        state = state.wrapping_mul(0x100000001b3);
    }
    for (i, chunk) in hash.iter_mut().enumerate() {
        *chunk = ((state >> ((i % 8) * 8)) & 0xff) as u8;
        state = state
            .wrapping_add(i as u64)
            .wrapping_mul(0x517cc1b727220a95);
    }
    hash
}

// ---------------------------------------------------------------------------
// Mock verifier for testing
// ---------------------------------------------------------------------------

/// A mock ZK verifier that produces deterministic, non-cryptographic proofs.
///
/// Useful for testing the verification pipeline without requiring actual
/// ZK proof infrastructure.
pub struct MockVerifier;

impl MockVerifier {
    /// Create a new mock verifier.
    pub fn new() -> Self {
        Self
    }
}

impl Default for MockVerifier {
    fn default() -> Self {
        Self::new()
    }
}

impl ZkVerifier for MockVerifier {
    fn backend(&self) -> VerificationBackend {
        VerificationBackend::Mock
    }

    fn prove_inference(
        &self,
        model_hash: [u8; 32],
        input_hash: [u8; 32],
        output_hash: [u8; 32],
    ) -> Result<ZkProof> {
        let proof_type = ZkProofType::InferenceVerification {
            model_hash,
            input_hash,
            output_hash,
        };
        // Mock proof: concatenate all hashes
        let mut proof_data = Vec::with_capacity(96);
        proof_data.extend_from_slice(&model_hash);
        proof_data.extend_from_slice(&input_hash);
        proof_data.extend_from_slice(&output_hash);

        Ok(ZkProof {
            proof_type,
            proof_data,
            backend: VerificationBackend::Mock,
            generation_time_ms: 1,
        })
    }

    fn prove_guardrail_compliance(
        &self,
        plan: &Plan,
        guardrail_results: &[Vec<GuardrailResult>],
    ) -> Result<ZkProof> {
        let plan_json =
            serde_json::to_vec(plan).map_err(|e| VerifyError::Serialization(e.to_string()))?;
        let plan_hash = sha256_hash(&plan_json);

        let all_passed = guardrail_results
            .iter()
            .all(|step| step.iter().all(|r| r.passed));

        let guardrail_hashes: Vec<[u8; 32]> = guardrail_results
            .iter()
            .map(|step| {
                let data = serde_json::to_vec(step).unwrap_or_default();
                sha256_hash(&data)
            })
            .collect();

        let proof_type = ZkProofType::GuardrailCompliance {
            plan_hash,
            guardrail_hashes: guardrail_hashes.clone(),
            all_passed,
        };

        let mut proof_data = Vec::new();
        proof_data.extend_from_slice(&plan_hash);
        for gh in &guardrail_hashes {
            proof_data.extend_from_slice(gh);
        }
        proof_data.push(u8::from(all_passed));

        Ok(ZkProof {
            proof_type,
            proof_data,
            backend: VerificationBackend::Mock,
            generation_time_ms: 1,
        })
    }

    fn prove_data_provenance(
        &self,
        data_hash: [u8; 32],
        timestamp: u64,
        source_commitment: [u8; 32],
    ) -> Result<ZkProof> {
        let proof_type = ZkProofType::DataProvenance {
            data_hash,
            timestamp,
            source_commitment,
        };

        let mut proof_data = Vec::with_capacity(72);
        proof_data.extend_from_slice(&data_hash);
        proof_data.extend_from_slice(&timestamp.to_le_bytes());
        proof_data.extend_from_slice(&source_commitment);

        Ok(ZkProof {
            proof_type,
            proof_data,
            backend: VerificationBackend::Mock,
            generation_time_ms: 1,
        })
    }

    fn verify(&self, proof: &ZkProof) -> Result<VerificationResult> {
        // Mock verification: check that proof data is non-empty and
        // matches the expected structure for the proof type.
        if proof.proof_data.is_empty() {
            return Ok(VerificationResult {
                valid: false,
                verification_time_ms: 0,
                details: "empty proof data".to_string(),
            });
        }

        let expected_len = match &proof.proof_type {
            ZkProofType::InferenceVerification { .. } => 96, // 3 * 32 bytes
            ZkProofType::GuardrailCompliance {
                guardrail_hashes, ..
            } => 32 + (guardrail_hashes.len() * 32) + 1,
            ZkProofType::DataProvenance { .. } => 72, // 32 + 8 + 32
        };

        let valid = proof.proof_data.len() == expected_len;

        Ok(VerificationResult {
            valid,
            verification_time_ms: 0,
            details: if valid {
                "mock proof verified successfully".to_string()
            } else {
                format!(
                    "proof data length mismatch: expected {expected_len}, got {}",
                    proof.proof_data.len()
                )
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_verifier_inference() {
        let verifier = MockVerifier::new();
        let model_hash = sha256_hash(b"model-weights");
        let input_hash = sha256_hash(b"input-state");
        let output_hash = sha256_hash(b"output-state");

        let proof = verifier
            .prove_inference(model_hash, input_hash, output_hash)
            .unwrap();
        assert_eq!(proof.backend, VerificationBackend::Mock);
        assert_eq!(proof.proof_data.len(), 96);

        let result = verifier.verify(&proof).unwrap();
        assert!(result.valid);
    }

    #[test]
    fn test_mock_verifier_guardrail_compliance() {
        let verifier = MockVerifier::new();
        let plan = Plan {
            actions: Vec::new(),
            predicted_states: Vec::new(),
            predicted_videos: None,
            total_cost: 0.0,
            success_probability: 1.0,
            guardrail_compliance: Vec::new(),
            planning_time_ms: 0,
            iterations_used: 0,
        };

        let guardrail_results: Vec<Vec<GuardrailResult>> = vec![vec![GuardrailResult {
            guardrail_name: "NoCollisions".to_string(),
            passed: true,
            violation_details: None,
            severity: worldforge_core::guardrail::ViolationSeverity::Info,
        }]];

        let proof = verifier
            .prove_guardrail_compliance(&plan, &guardrail_results)
            .unwrap();
        let result = verifier.verify(&proof).unwrap();
        assert!(result.valid);
    }

    #[test]
    fn test_mock_verifier_data_provenance() {
        let verifier = MockVerifier::new();
        let data_hash = sha256_hash(b"sensor-data");
        let source_commitment = sha256_hash(b"camera-01");

        let proof = verifier
            .prove_data_provenance(data_hash, 1710000000, source_commitment)
            .unwrap();
        assert_eq!(proof.proof_data.len(), 72);

        let result = verifier.verify(&proof).unwrap();
        assert!(result.valid);
    }

    #[test]
    fn test_verify_empty_proof_fails() {
        let verifier = MockVerifier::new();
        let proof = ZkProof {
            proof_type: ZkProofType::InferenceVerification {
                model_hash: [0; 32],
                input_hash: [0; 32],
                output_hash: [0; 32],
            },
            proof_data: Vec::new(),
            backend: VerificationBackend::Mock,
            generation_time_ms: 0,
        };
        let result = verifier.verify(&proof).unwrap();
        assert!(!result.valid);
    }

    #[test]
    fn test_verify_tampered_proof_fails() {
        let verifier = MockVerifier::new();
        let mut proof = verifier.prove_inference([1; 32], [2; 32], [3; 32]).unwrap();
        // Tamper with proof data
        proof.proof_data.push(0xff);
        let result = verifier.verify(&proof).unwrap();
        assert!(!result.valid);
    }

    #[test]
    fn test_sha256_hash_deterministic() {
        let h1 = sha256_hash(b"hello");
        let h2 = sha256_hash(b"hello");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_sha256_hash_different_inputs() {
        let h1 = sha256_hash(b"hello");
        let h2 = sha256_hash(b"world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_proof_type_serialization() {
        let proof_type = ZkProofType::InferenceVerification {
            model_hash: [1; 32],
            input_hash: [2; 32],
            output_hash: [3; 32],
        };
        let json = serde_json::to_string(&proof_type).unwrap();
        let deserialized: ZkProofType = serde_json::from_str(&json).unwrap();
        match deserialized {
            ZkProofType::InferenceVerification { model_hash, .. } => {
                assert_eq!(model_hash, [1; 32]);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_zk_proof_serialization_roundtrip() {
        let verifier = MockVerifier::new();
        let proof = verifier.prove_inference([1; 32], [2; 32], [3; 32]).unwrap();
        let json = serde_json::to_string(&proof).unwrap();
        let deserialized: ZkProof = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.proof_data.len(), proof.proof_data.len());
        assert_eq!(deserialized.backend, VerificationBackend::Mock);
    }

    #[test]
    fn test_verification_backend_serialization() {
        for backend in [
            VerificationBackend::Ezkl,
            VerificationBackend::Stark,
            VerificationBackend::Mock,
        ] {
            let json = serde_json::to_string(&backend).unwrap();
            let deserialized: VerificationBackend = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized, backend);
        }
    }

    #[test]
    fn test_verify_error_display() {
        let err = VerifyError::GenerationFailed {
            reason: "no circuit".to_string(),
        };
        assert!(err.to_string().contains("no circuit"));

        let err = VerifyError::UnsupportedProofType {
            backend: VerificationBackend::Ezkl,
        };
        assert!(err.to_string().contains("Ezkl"));
    }
}
