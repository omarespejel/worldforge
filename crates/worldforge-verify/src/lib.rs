//! WorldForge ZK Verification Layer (stub).
//!
//! This crate will provide zero-knowledge proof generation and verification
//! for WorldForge predictions and plans. Currently a placeholder — ZK
//! functionality is not part of the initial MVP.

use serde::{Deserialize, Serialize};

/// ZK proof type (placeholder).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ZkProofType {
    /// Verify that inference was computed correctly.
    InferenceVerification {
        model_hash: [u8; 32],
        input_hash: [u8; 32],
        output_hash: [u8; 32],
    },
    /// Verify guardrail compliance.
    GuardrailCompliance {
        plan_hash: [u8; 32],
        guardrail_hashes: Vec<[u8; 32]>,
        all_passed: bool,
    },
    /// Verify data provenance.
    DataProvenance {
        data_hash: [u8; 32],
        timestamp: u64,
        source_commitment: [u8; 32],
    },
}

/// A ZK proof (placeholder).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZkProof {
    /// Type of proof.
    pub proof_type: ZkProofType,
    /// Serialized proof data.
    pub proof_data: Vec<u8>,
}
