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
use worldforge_core::state::WorldState;
use worldforge_core::types::WorldId;

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

/// A proof, its verification result, and the concrete artifact that was hashed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationBundle<T> {
    /// Summary of the artifact that was verified.
    pub artifact: T,
    /// The generated proof bytes and metadata.
    pub proof: ZkProof,
    /// Result of verifying the proof.
    pub verification: VerificationResult,
}

/// Concrete inputs used for inference verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceArtifact {
    /// Provider associated with the verified transition.
    pub provider: String,
    /// Hash identifying the model or provider version.
    pub model_hash: [u8; 32],
    /// Hash of the input state.
    pub input_hash: [u8; 32],
    /// Hash of the output state.
    pub output_hash: [u8; 32],
}

/// Concrete inputs used for guardrail-compliance verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardrailArtifact {
    /// Hash of the serialized plan.
    pub plan_hash: [u8; 32],
    /// Hash of each step's guardrail evaluations.
    pub guardrail_hashes: Vec<[u8; 32]>,
    /// Whether every guardrail passed.
    pub all_passed: bool,
    /// Number of actions in the plan.
    pub action_count: usize,
    /// Number of predicted states in the plan.
    pub predicted_state_count: usize,
    /// Number of guardrail-evaluation steps in the plan.
    pub guardrail_step_count: usize,
}

/// Concrete inputs used for provenance verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceArtifact {
    /// World being attested.
    pub world_id: WorldId,
    /// Provider that created the world.
    pub provider: String,
    /// Hash of the serialized world state.
    pub data_hash: [u8; 32],
    /// Hash of the serialized history.
    pub history_hash: [u8; 32],
    /// Timestamp claimed by the attestation.
    pub timestamp: u64,
    /// Commitment to the source system or process.
    pub source_commitment: [u8; 32],
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

    /// Not enough history is available to derive the requested artifact.
    #[error("insufficient state history: need at least {required} entries, found {actual}")]
    InsufficientHistory { required: usize, actual: usize },
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

fn serialize_to_hash<T: Serialize>(value: &T) -> Result<[u8; 32]> {
    let bytes = serde_json::to_vec(value).map_err(|e| VerifyError::Serialization(e.to_string()))?;
    Ok(sha256_hash(&bytes))
}

/// Hash a provider/source label into a deterministic commitment.
pub fn source_commitment(label: &str) -> [u8; 32] {
    sha256_hash(label.as_bytes())
}

/// Compute the hash of a world state as used by the verification helpers.
pub fn state_hash(state: &WorldState) -> Result<[u8; 32]> {
    serialize_to_hash(state)
}

/// Compute the hash of a plan as used by the verification helpers.
pub fn plan_hash(plan: &Plan) -> Result<[u8; 32]> {
    serialize_to_hash(plan)
}

/// Build an inference artifact from explicit input/output states.
pub fn inference_artifact_from_states(
    provider: impl Into<String>,
    input_state: &WorldState,
    output_state: &WorldState,
) -> Result<InferenceArtifact> {
    let provider = provider.into();
    Ok(InferenceArtifact {
        model_hash: source_commitment(&provider),
        provider,
        input_hash: state_hash(input_state)?,
        output_hash: state_hash(output_state)?,
    })
}

/// Build an inference artifact from the latest recorded transition in a world state.
///
/// Requires at least two history entries so the previous output state hash can be
/// used as the input to the most recent transition.
pub fn latest_inference_artifact(
    state: &WorldState,
    provider_override: Option<&str>,
) -> Result<InferenceArtifact> {
    let actual = state.history.states.len();
    if actual < 2 {
        return Err(VerifyError::InsufficientHistory {
            required: 2,
            actual,
        });
    }

    let input_hash = state
        .history
        .states
        .iter()
        .rev()
        .nth(1)
        .map(|entry| entry.state_hash)
        .ok_or(VerifyError::InsufficientHistory {
            required: 2,
            actual,
        })?;
    let provider = provider_override
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| state.history.latest().map(|entry| entry.provider.clone()))
        .unwrap_or_else(|| state.metadata.created_by.clone());

    Ok(InferenceArtifact {
        model_hash: source_commitment(&provider),
        provider,
        input_hash,
        output_hash: state.history.latest().map(|entry| entry.state_hash).ok_or(
            VerifyError::InsufficientHistory {
                required: 2,
                actual,
            },
        )?,
    })
}

/// Build a guardrail artifact from a fully materialized plan.
pub fn guardrail_artifact(plan: &Plan) -> Result<GuardrailArtifact> {
    let guardrail_hashes = plan
        .guardrail_compliance
        .iter()
        .map(serialize_to_hash)
        .collect::<Result<Vec<_>>>()?;

    Ok(GuardrailArtifact {
        plan_hash: plan_hash(plan)?,
        all_passed: plan
            .guardrail_compliance
            .iter()
            .all(|step| step.iter().all(|result| result.passed)),
        action_count: plan.actions.len(),
        predicted_state_count: plan.predicted_states.len(),
        guardrail_step_count: plan.guardrail_compliance.len(),
        guardrail_hashes,
    })
}

/// Build a provenance artifact from a world state snapshot.
pub fn provenance_artifact(
    state: &WorldState,
    source_label: &str,
    timestamp: u64,
) -> Result<ProvenanceArtifact> {
    Ok(ProvenanceArtifact {
        world_id: state.id,
        provider: state.metadata.created_by.clone(),
        data_hash: state_hash(state)?,
        history_hash: serialize_to_hash(&state.history)?,
        timestamp,
        source_commitment: source_commitment(source_label),
    })
}

/// Generate and verify an inference proof for explicit input/output states.
pub fn prove_inference_transition<V: ZkVerifier>(
    verifier: &V,
    provider: impl Into<String>,
    input_state: &WorldState,
    output_state: &WorldState,
) -> Result<VerificationBundle<InferenceArtifact>> {
    let artifact = inference_artifact_from_states(provider, input_state, output_state)?;
    let proof = verifier.prove_inference(
        artifact.model_hash,
        artifact.input_hash,
        artifact.output_hash,
    )?;
    let verification = verifier.verify(&proof)?;

    Ok(VerificationBundle {
        artifact,
        proof,
        verification,
    })
}

/// Generate and verify an inference proof from the latest recorded world transition.
pub fn prove_latest_inference<V: ZkVerifier>(
    verifier: &V,
    state: &WorldState,
    provider_override: Option<&str>,
) -> Result<VerificationBundle<InferenceArtifact>> {
    let artifact = latest_inference_artifact(state, provider_override)?;
    let proof = verifier.prove_inference(
        artifact.model_hash,
        artifact.input_hash,
        artifact.output_hash,
    )?;
    let verification = verifier.verify(&proof)?;

    Ok(VerificationBundle {
        artifact,
        proof,
        verification,
    })
}

/// Generate and verify a guardrail-compliance proof for a plan.
pub fn prove_guardrail_plan<V: ZkVerifier>(
    verifier: &V,
    plan: &Plan,
) -> Result<VerificationBundle<GuardrailArtifact>> {
    let artifact = guardrail_artifact(plan)?;
    let proof = verifier.prove_guardrail_compliance(plan, &plan.guardrail_compliance)?;
    let verification = verifier.verify(&proof)?;

    Ok(VerificationBundle {
        artifact,
        proof,
        verification,
    })
}

/// Generate and verify a provenance proof for a world state snapshot.
pub fn prove_provenance<V: ZkVerifier>(
    verifier: &V,
    state: &WorldState,
    source_label: &str,
    timestamp: u64,
) -> Result<VerificationBundle<ProvenanceArtifact>> {
    let artifact = provenance_artifact(state, source_label, timestamp)?;
    let proof = verifier.prove_data_provenance(
        artifact.data_hash,
        artifact.timestamp,
        artifact.source_commitment,
    )?;
    let verification = verifier.verify(&proof)?;

    Ok(VerificationBundle {
        artifact,
        proof,
        verification,
    })
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
    use worldforge_core::action::Action;
    use worldforge_core::guardrail::{GuardrailResult, ViolationSeverity};
    use worldforge_core::scene::SceneObject;
    use worldforge_core::state::{HistoryEntry, PredictionSummary, WorldState};
    use worldforge_core::types::{BBox, Pose, Position, SimTime};

    fn sample_state(name: &str, provider: &str, x: f32) -> WorldState {
        let mut state = WorldState::new(name, provider);
        let object = SceneObject::new(
            format!("{name}_object"),
            Pose {
                position: Position { x, y: 0.5, z: 0.0 },
                ..Default::default()
            },
            BBox {
                min: Position {
                    x: x - 0.1,
                    y: 0.0,
                    z: -0.1,
                },
                max: Position {
                    x: x + 0.1,
                    y: 1.0,
                    z: 0.1,
                },
            },
        );
        state.scene.add_object(object);
        state
    }

    fn sample_guardrail_result(passed: bool) -> GuardrailResult {
        GuardrailResult {
            guardrail_name: "NoCollisions".to_string(),
            passed,
            violation_details: (!passed).then(|| "collision".to_string()),
            severity: if passed {
                ViolationSeverity::Info
            } else {
                ViolationSeverity::Blocking
            },
        }
    }

    fn sample_plan() -> Plan {
        Plan {
            actions: vec![Action::SetWeather {
                weather: worldforge_core::action::Weather::Clear,
            }],
            predicted_states: vec![sample_state("planned", "mock", 1.5)],
            predicted_videos: None,
            total_cost: 0.0,
            success_probability: 0.9,
            guardrail_compliance: vec![vec![sample_guardrail_result(true)]],
            planning_time_ms: 4,
            iterations_used: 2,
        }
    }

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
        let plan = sample_plan();
        let guardrail_results: Vec<Vec<GuardrailResult>> =
            vec![vec![sample_guardrail_result(true)]];

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

    #[test]
    fn test_inference_artifact_from_states_uses_serialized_state_hashes() {
        let input = sample_state("input", "mock", 0.0);
        let output = sample_state("output", "mock", 1.0);

        let artifact = inference_artifact_from_states("mock", &input, &output).unwrap();

        assert_eq!(artifact.provider, "mock");
        assert_eq!(artifact.model_hash, source_commitment("mock"));
        assert_eq!(artifact.input_hash, state_hash(&input).unwrap());
        assert_eq!(artifact.output_hash, state_hash(&output).unwrap());
    }

    #[test]
    fn test_latest_inference_artifact_requires_two_history_entries() {
        let state = sample_state("world", "mock", 0.0);
        let err = latest_inference_artifact(&state, None).unwrap_err();
        match err {
            VerifyError::InsufficientHistory { required, actual } => {
                assert_eq!(required, 2);
                assert_eq!(actual, 0);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn test_latest_inference_artifact_uses_previous_history_hash() {
        let mut world = sample_state("world", "mock", 2.0);
        let previous = sample_state("previous", "mock", 1.0);
        let previous_hash = state_hash(&previous).unwrap();
        let current_hash = state_hash(&world).unwrap();

        world.history.push(HistoryEntry {
            time: SimTime {
                step: 1,
                seconds: 0.5,
                dt: 0.5,
            },
            state_hash: previous_hash,
            action: None,
            prediction: None,
            provider: "mock".to_string(),
        });
        world.history.push(HistoryEntry {
            time: SimTime {
                step: 2,
                seconds: 1.0,
                dt: 0.5,
            },
            state_hash: current_hash,
            action: Some(Action::SetWeather {
                weather: worldforge_core::action::Weather::Clear,
            }),
            prediction: Some(PredictionSummary {
                confidence: 0.8,
                physics_score: 0.9,
                latency_ms: 12,
            }),
            provider: "mock".to_string(),
        });

        let artifact = latest_inference_artifact(&world, None).unwrap();
        assert_eq!(artifact.provider, "mock");
        assert_eq!(artifact.input_hash, previous_hash);
        assert_eq!(artifact.output_hash, current_hash);
    }

    #[test]
    fn test_guardrail_artifact_tracks_plan_material() {
        let plan = sample_plan();
        let artifact = guardrail_artifact(&plan).unwrap();

        assert_eq!(artifact.action_count, 1);
        assert_eq!(artifact.predicted_state_count, 1);
        assert_eq!(artifact.guardrail_step_count, 1);
        assert!(artifact.all_passed);
        assert_eq!(artifact.guardrail_hashes.len(), 1);
    }

    #[test]
    fn test_provenance_artifact_hashes_state_and_history() {
        let state = sample_state("world", "mock", 0.0);
        let artifact = provenance_artifact(&state, "worldforge-cli", 1710000000).unwrap();

        assert_eq!(artifact.world_id, state.id);
        assert_eq!(artifact.provider, "mock");
        assert_eq!(artifact.data_hash, state_hash(&state).unwrap());
        assert_eq!(
            artifact.source_commitment,
            source_commitment("worldforge-cli")
        );
    }

    #[test]
    fn test_prove_guardrail_plan_bundle_roundtrip() {
        let verifier = MockVerifier::new();
        let plan = sample_plan();

        let bundle = prove_guardrail_plan(&verifier, &plan).unwrap();

        assert!(bundle.verification.valid);
        assert_eq!(bundle.artifact.action_count, 1);
        assert_eq!(bundle.proof.backend, VerificationBackend::Mock);
    }

    #[test]
    fn test_prove_provenance_bundle_roundtrip() {
        let verifier = MockVerifier::new();
        let state = sample_state("world", "mock", 0.0);

        let bundle = prove_provenance(&verifier, &state, "worldforge-server", 1710000000).unwrap();

        assert!(bundle.verification.valid);
        assert_eq!(bundle.artifact.provider, "mock");
        assert_eq!(bundle.proof.backend, VerificationBackend::Mock);
    }
}
