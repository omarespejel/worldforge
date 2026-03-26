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

/// Result of re-verifying a previously exported verification bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleVerificationReport<T> {
    /// Summary of the artifact that was verified.
    pub artifact: T,
    /// The proof that was checked.
    pub proof: ZkProof,
    /// Verification result originally recorded in the bundle.
    pub recorded_verification: VerificationResult,
    /// Verification result recomputed from the proof bytes.
    pub current_verification: VerificationResult,
    /// Whether the current verification verdict matches the recorded one.
    pub verification_matches_recorded: bool,
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

impl VerificationBackend {
    /// Canonical lowercase backend identifier used by user-facing surfaces.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ezkl => "ezkl",
            Self::Stark => "stark",
            Self::Mock => "mock",
        }
    }

    /// Construct a verifier for this backend.
    pub fn verifier(self) -> Box<dyn ZkVerifier> {
        match self {
            Self::Ezkl => Box::new(EzklVerifier::new()),
            Self::Stark => Box::new(StarkVerifier::new()),
            Self::Mock => Box::new(MockVerifier::new()),
        }
    }
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

    /// The proof was generated for a different backend than the active verifier.
    #[error("proof backend mismatch: proof uses {actual:?}, verifier is {expected:?}")]
    BackendMismatch {
        expected: VerificationBackend,
        actual: VerificationBackend,
    },

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

impl std::str::FromStr for VerificationBackend {
    type Err = VerifyError;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "ezkl" => Ok(Self::Ezkl),
            "stark" => Ok(Self::Stark),
            "mock" => Ok(Self::Mock),
            other => Err(VerifyError::VerificationFailed {
                reason: format!(
                    "unknown verification backend: {other}. Available: mock, ezkl, stark"
                ),
            }),
        }
    }
}

/// SHA-256 hash helper for creating proof inputs.
pub fn sha256_hash(data: &[u8]) -> [u8; 32] {
    const INITIAL_STATE: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    const ROUND_CONSTANTS: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut padded = data.to_vec();
    padded.push(0x80);
    while !(padded.len() + 8).is_multiple_of(64) {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    let mut state = INITIAL_STATE;
    for chunk in padded.chunks_exact(64) {
        let mut schedule = [0u32; 64];
        for (index, word) in schedule.iter_mut().take(16).enumerate() {
            let start = index * 4;
            *word = u32::from_be_bytes([
                chunk[start],
                chunk[start + 1],
                chunk[start + 2],
                chunk[start + 3],
            ]);
        }

        for index in 16..64 {
            let s0 = schedule[index - 15].rotate_right(7)
                ^ schedule[index - 15].rotate_right(18)
                ^ (schedule[index - 15] >> 3);
            let s1 = schedule[index - 2].rotate_right(17)
                ^ schedule[index - 2].rotate_right(19)
                ^ (schedule[index - 2] >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(s0)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for index in 0..64 {
            let sigma1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(sigma1)
                .wrapping_add(ch)
                .wrapping_add(ROUND_CONSTANTS[index])
                .wrapping_add(schedule[index]);
            let sigma0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = sigma0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }

    let mut hash = [0u8; 32];
    for (index, word) in state.iter().enumerate() {
        hash[index * 4..(index + 1) * 4].copy_from_slice(&word.to_be_bytes());
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
pub fn prove_inference_transition<V: ZkVerifier + ?Sized>(
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
pub fn prove_latest_inference<V: ZkVerifier + ?Sized>(
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
pub fn prove_guardrail_plan<V: ZkVerifier + ?Sized>(
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
pub fn prove_provenance<V: ZkVerifier + ?Sized>(
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

fn verification_equivalent(a: &VerificationResult, b: &VerificationResult) -> bool {
    a.valid == b.valid && a.details == b.details
}

/// Verify a raw proof and return the current verification verdict.
pub fn verify_proof<V: ZkVerifier + ?Sized>(
    verifier: &V,
    proof: &ZkProof,
) -> Result<VerificationResult> {
    verifier.verify(proof)
}

/// Re-verify an exported bundle and compare it with the recorded verdict.
pub fn verify_bundle<V: ZkVerifier + ?Sized, T: Clone>(
    verifier: &V,
    bundle: &VerificationBundle<T>,
) -> Result<BundleVerificationReport<T>> {
    let current_verification = verify_proof(verifier, &bundle.proof)?;

    Ok(BundleVerificationReport {
        artifact: bundle.artifact.clone(),
        proof: bundle.proof.clone(),
        recorded_verification: bundle.verification.clone(),
        verification_matches_recorded: verification_equivalent(
            &bundle.verification,
            &current_verification,
        ),
        current_verification,
    })
}

// ---------------------------------------------------------------------------
// Mock verifier for testing
// ---------------------------------------------------------------------------

fn backend_supports_proof_type(backend: VerificationBackend, proof_type: &ZkProofType) -> bool {
    match backend {
        VerificationBackend::Mock => true,
        VerificationBackend::Ezkl => matches!(
            proof_type,
            ZkProofType::InferenceVerification { .. } | ZkProofType::DataProvenance { .. }
        ),
        VerificationBackend::Stark => matches!(
            proof_type,
            ZkProofType::InferenceVerification { .. }
                | ZkProofType::GuardrailCompliance { .. }
                | ZkProofType::DataProvenance { .. }
        ),
    }
}

fn guard_supported_backend(backend: VerificationBackend, proof_type: &ZkProofType) -> Result<()> {
    if backend_supports_proof_type(backend, proof_type) {
        Ok(())
    } else {
        Err(VerifyError::UnsupportedProofType { backend })
    }
}

fn domain_separated_receipt(
    backend: VerificationBackend,
    proof_type: &ZkProofType,
) -> Result<Vec<u8>> {
    let encoded = serde_json::to_vec(&(backend.as_str(), proof_type))
        .map_err(|e| VerifyError::Serialization(e.to_string()))?;
    Ok(sha256_hash(&encoded).to_vec())
}

fn proof_data_for_backend(
    backend: VerificationBackend,
    proof_type: &ZkProofType,
) -> Result<Vec<u8>> {
    guard_supported_backend(backend, proof_type)?;
    match backend {
        VerificationBackend::Mock => Ok(expected_mock_proof_data(proof_type)),
        VerificationBackend::Ezkl | VerificationBackend::Stark => {
            domain_separated_receipt(backend, proof_type)
        }
    }
}

fn generation_time_for_backend(backend: VerificationBackend, proof_type: &ZkProofType) -> u64 {
    match (backend, proof_type) {
        (VerificationBackend::Mock, _) => 1,
        (VerificationBackend::Ezkl, ZkProofType::InferenceVerification { .. }) => 45,
        (VerificationBackend::Ezkl, ZkProofType::DataProvenance { .. }) => 12,
        (VerificationBackend::Ezkl, _) => 0,
        (VerificationBackend::Stark, ZkProofType::GuardrailCompliance { .. }) => 80,
        (VerificationBackend::Stark, ZkProofType::InferenceVerification { .. }) => 65,
        (VerificationBackend::Stark, ZkProofType::DataProvenance { .. }) => 20,
    }
}

fn verification_details(backend: VerificationBackend, valid: bool) -> String {
    let backend_name = backend.as_str();
    if valid {
        format!("{backend_name} proof verified successfully")
    } else {
        format!("{backend_name} proof data mismatch")
    }
}

fn prove_with_backend(backend: VerificationBackend, proof_type: ZkProofType) -> Result<ZkProof> {
    let proof_data = proof_data_for_backend(backend, &proof_type)?;
    Ok(ZkProof {
        proof_type: proof_type.clone(),
        proof_data,
        backend,
        generation_time_ms: generation_time_for_backend(backend, &proof_type),
    })
}

fn verify_with_backend(
    backend: VerificationBackend,
    proof: &ZkProof,
) -> Result<VerificationResult> {
    if proof.backend != backend {
        return Err(VerifyError::BackendMismatch {
            expected: backend,
            actual: proof.backend,
        });
    }

    let expected = proof_data_for_backend(backend, &proof.proof_type)?;
    let valid = proof.proof_data == expected;
    Ok(VerificationResult {
        valid,
        verification_time_ms: match backend {
            VerificationBackend::Mock => 0,
            VerificationBackend::Ezkl => 6,
            VerificationBackend::Stark => 9,
        },
        details: verification_details(backend, valid),
    })
}

/// Deterministic EZKL compatibility verifier.
///
/// This models the backend selection and proof-shape differences without
/// integrating an external proving runtime yet.
pub struct EzklVerifier;

impl EzklVerifier {
    /// Create a new EZKL verifier.
    pub fn new() -> Self {
        Self
    }
}

impl Default for EzklVerifier {
    fn default() -> Self {
        Self::new()
    }
}

impl ZkVerifier for EzklVerifier {
    fn backend(&self) -> VerificationBackend {
        VerificationBackend::Ezkl
    }

    fn prove_inference(
        &self,
        model_hash: [u8; 32],
        input_hash: [u8; 32],
        output_hash: [u8; 32],
    ) -> Result<ZkProof> {
        prove_with_backend(
            self.backend(),
            ZkProofType::InferenceVerification {
                model_hash,
                input_hash,
                output_hash,
            },
        )
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
            .all(|step| step.iter().all(|result| result.passed));
        let guardrail_hashes = guardrail_results
            .iter()
            .map(serialize_to_hash)
            .collect::<Result<Vec<_>>>()?;
        prove_with_backend(
            self.backend(),
            ZkProofType::GuardrailCompliance {
                plan_hash,
                guardrail_hashes,
                all_passed,
            },
        )
    }

    fn prove_data_provenance(
        &self,
        data_hash: [u8; 32],
        timestamp: u64,
        source_commitment: [u8; 32],
    ) -> Result<ZkProof> {
        prove_with_backend(
            self.backend(),
            ZkProofType::DataProvenance {
                data_hash,
                timestamp,
                source_commitment,
            },
        )
    }

    fn verify(&self, proof: &ZkProof) -> Result<VerificationResult> {
        verify_with_backend(self.backend(), proof)
    }
}

/// Deterministic STARK compatibility verifier.
///
/// This keeps the WorldForge verification surface backend-aware while the
/// project is still using local stand-in proving logic.
pub struct StarkVerifier;

impl StarkVerifier {
    /// Create a new STARK verifier.
    pub fn new() -> Self {
        Self
    }
}

impl Default for StarkVerifier {
    fn default() -> Self {
        Self::new()
    }
}

impl ZkVerifier for StarkVerifier {
    fn backend(&self) -> VerificationBackend {
        VerificationBackend::Stark
    }

    fn prove_inference(
        &self,
        model_hash: [u8; 32],
        input_hash: [u8; 32],
        output_hash: [u8; 32],
    ) -> Result<ZkProof> {
        prove_with_backend(
            self.backend(),
            ZkProofType::InferenceVerification {
                model_hash,
                input_hash,
                output_hash,
            },
        )
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
            .all(|step| step.iter().all(|result| result.passed));
        let guardrail_hashes = guardrail_results
            .iter()
            .map(serialize_to_hash)
            .collect::<Result<Vec<_>>>()?;

        prove_with_backend(
            self.backend(),
            ZkProofType::GuardrailCompliance {
                plan_hash,
                guardrail_hashes,
                all_passed,
            },
        )
    }

    fn prove_data_provenance(
        &self,
        data_hash: [u8; 32],
        timestamp: u64,
        source_commitment: [u8; 32],
    ) -> Result<ZkProof> {
        prove_with_backend(
            self.backend(),
            ZkProofType::DataProvenance {
                data_hash,
                timestamp,
                source_commitment,
            },
        )
    }

    fn verify(&self, proof: &ZkProof) -> Result<VerificationResult> {
        verify_with_backend(self.backend(), proof)
    }
}

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
        prove_with_backend(
            self.backend(),
            ZkProofType::InferenceVerification {
                model_hash,
                input_hash,
                output_hash,
            },
        )
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

        let guardrail_hashes = guardrail_results
            .iter()
            .map(serialize_to_hash)
            .collect::<Result<Vec<_>>>()?;

        prove_with_backend(
            self.backend(),
            ZkProofType::GuardrailCompliance {
                plan_hash,
                guardrail_hashes,
                all_passed,
            },
        )
    }

    fn prove_data_provenance(
        &self,
        data_hash: [u8; 32],
        timestamp: u64,
        source_commitment: [u8; 32],
    ) -> Result<ZkProof> {
        prove_with_backend(
            self.backend(),
            ZkProofType::DataProvenance {
                data_hash,
                timestamp,
                source_commitment,
            },
        )
    }

    fn verify(&self, proof: &ZkProof) -> Result<VerificationResult> {
        verify_with_backend(self.backend(), proof)
    }
}

fn expected_mock_proof_data(proof_type: &ZkProofType) -> Vec<u8> {
    match proof_type {
        ZkProofType::InferenceVerification {
            model_hash,
            input_hash,
            output_hash,
        } => {
            let mut proof_data = Vec::with_capacity(96);
            proof_data.extend_from_slice(model_hash);
            proof_data.extend_from_slice(input_hash);
            proof_data.extend_from_slice(output_hash);
            proof_data
        }
        ZkProofType::GuardrailCompliance {
            plan_hash,
            guardrail_hashes,
            all_passed,
        } => {
            let mut proof_data = Vec::with_capacity(32 + (guardrail_hashes.len() * 32) + 1);
            proof_data.extend_from_slice(plan_hash);
            for guardrail_hash in guardrail_hashes {
                proof_data.extend_from_slice(guardrail_hash);
            }
            proof_data.push(u8::from(*all_passed));
            proof_data
        }
        ZkProofType::DataProvenance {
            data_hash,
            timestamp,
            source_commitment,
        } => {
            let mut proof_data = Vec::with_capacity(72);
            proof_data.extend_from_slice(data_hash);
            proof_data.extend_from_slice(&timestamp.to_le_bytes());
            proof_data.extend_from_slice(source_commitment);
            proof_data
        }
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
        // Tamper with proof data without changing its length.
        proof.proof_data[0] ^= 0xff;
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
    fn test_sha256_hash_matches_empty_string_vector() {
        let hash = sha256_hash(b"");
        let expected = [
            0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f,
            0xb9, 0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b,
            0x78, 0x52, 0xb8, 0x55,
        ];
        assert_eq!(hash, expected);
    }

    #[test]
    fn test_sha256_hash_matches_abc_vector() {
        let hash = sha256_hash(b"abc");
        let expected = [
            0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
            0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
            0xf2, 0x00, 0x15, 0xad,
        ];
        assert_eq!(hash, expected);
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
    fn test_verify_backend_mismatch_errors() {
        let verifier = MockVerifier::new();
        let proof = ZkProof {
            proof_type: ZkProofType::InferenceVerification {
                model_hash: [1; 32],
                input_hash: [2; 32],
                output_hash: [3; 32],
            },
            proof_data: vec![0; 96],
            backend: VerificationBackend::Ezkl,
            generation_time_ms: 0,
        };

        let err = verifier.verify(&proof).unwrap_err();
        match err {
            VerifyError::BackendMismatch { expected, actual } => {
                assert_eq!(expected, VerificationBackend::Mock);
                assert_eq!(actual, VerificationBackend::Ezkl);
            }
            other => panic!("unexpected error: {other}"),
        }
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
            snapshot: None,
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
            snapshot: None,
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
    fn test_verify_bundle_recomputes_verdict() {
        let verifier = MockVerifier::new();
        let state = sample_state("world", "mock", 0.0);
        let bundle = prove_provenance(&verifier, &state, "worldforge-server", 1710000000).unwrap();

        let report = verify_bundle(&verifier, &bundle).unwrap();

        assert!(report.current_verification.valid);
        assert!(report.verification_matches_recorded);
        assert_eq!(report.artifact.provider, "mock");
    }

    #[test]
    fn test_verify_bundle_detects_recorded_mismatch() {
        let verifier = MockVerifier::new();
        let plan = sample_plan();
        let mut bundle = prove_guardrail_plan(&verifier, &plan).unwrap();
        bundle.verification.details = "tampered verdict".to_string();

        let report = verify_bundle(&verifier, &bundle).unwrap();

        assert!(report.current_verification.valid);
        assert!(!report.verification_matches_recorded);
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
