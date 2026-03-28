//! Prediction engine for WorldForge.
//!
//! Handles forward prediction of world states, multi-provider comparison,
//! and planning through world models.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::action::Action;
use crate::error::{Result, WorldForgeError};
use crate::guardrail::{Guardrail, GuardrailConfig, GuardrailResult, ViolationSeverity};
use crate::provider::CostEstimate;
use crate::scene::SpatialRelationship;
use crate::state::WorldState;
use crate::types::{ObjectId, PredictionId, VideoClip};

/// Execution provenance for a prediction.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PredictionProvenance {
    /// Deterministic hash of the executed model or inspected asset bundle.
    pub model_hash: [u8; 32],
    /// Asset fingerprint captured while inspecting local weights, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asset_fingerprint: Option<u64>,
    /// Backend or execution engine used for the prediction, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
}

/// Result of a single forward prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    /// Unique identifier for this prediction.
    pub id: PredictionId,
    /// Provider that generated this prediction.
    pub provider: String,
    /// Model identifier used.
    pub model: String,
    /// Input world state.
    pub input_state: WorldState,
    /// Action that was applied.
    pub action: Action,
    /// Predicted output world state.
    pub output_state: WorldState,
    /// Generated video of the transition (if requested).
    pub video: Option<VideoClip>,
    /// Confidence score (0.0–1.0).
    pub confidence: f32,
    /// Physics plausibility scores.
    pub physics_scores: PhysicsScores,
    /// Latency of the prediction in milliseconds.
    pub latency_ms: u64,
    /// Cost of the prediction.
    pub cost: CostEstimate,
    /// Execution provenance for the model run, if the provider surfaced it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<PredictionProvenance>,
    /// Guardrail evaluation results.
    pub guardrail_results: Vec<GuardrailResult>,
    /// When the prediction was generated.
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Physics plausibility scores for a prediction.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PhysicsScores {
    /// Overall physics plausibility (0.0–1.0).
    pub overall: f32,
    /// Object permanence score.
    pub object_permanence: f32,
    /// Gravity compliance score.
    pub gravity_compliance: f32,
    /// Collision accuracy score.
    pub collision_accuracy: f32,
    /// Spatial consistency score.
    pub spatial_consistency: f32,
    /// Temporal consistency score.
    pub temporal_consistency: f32,
}

/// Configuration for a prediction request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PredictionConfig {
    /// Number of future steps to predict.
    pub steps: u32,
    /// Output resolution `(width, height)`.
    pub resolution: (u32, u32),
    /// Output frames per second.
    pub fps: f32,
    /// Whether to return the generated video.
    pub return_video: bool,
    /// Whether to return depth maps.
    pub return_depth: bool,
    /// Whether to return segmentation maps.
    pub return_segmentation: bool,
    /// Guardrail configurations to apply.
    pub guardrails: Vec<crate::guardrail::GuardrailConfig>,
    /// Maximum latency before timeout (in milliseconds).
    pub max_latency_ms: Option<u64>,
    /// Fallback provider if primary fails.
    pub fallback_provider: Option<String>,
    /// Number of samples for uncertainty estimation.
    pub num_samples: u32,
    /// Sampling temperature.
    pub temperature: f32,
}

/// Result of multi-provider prediction comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiPrediction {
    /// Individual predictions from each provider.
    pub predictions: Vec<Prediction>,
    /// Agreement score between providers (0.0–1.0).
    pub agreement_score: f32,
    /// Index of the highest-quality prediction.
    pub best_prediction: usize,
    /// Detailed comparison report.
    pub comparison: ComparisonReport,
}

/// Comparison report across multiple provider predictions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ComparisonReport {
    /// Per-provider scores.
    pub scores: Vec<ProviderScore>,
    /// Pairwise agreement metrics across provider outputs.
    #[serde(default)]
    pub pairwise_agreements: Vec<PairwiseAgreement>,
    /// Consensus signals shared across all compared providers.
    #[serde(default)]
    pub consensus: ComparisonConsensus,
    /// Summary text.
    pub summary: String,
}

/// Score for a single provider in a comparison.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ProviderScore {
    /// Provider name.
    pub provider: String,
    /// Provider-reported confidence.
    #[serde(default)]
    pub confidence: f32,
    /// Physics scores.
    pub physics_scores: PhysicsScores,
    /// Latency in milliseconds.
    pub latency_ms: u64,
    /// Cost estimate.
    pub cost: CostEstimate,
    /// Guardrail pass/fail diagnostics for this prediction.
    #[serde(default)]
    pub guardrails: GuardrailDiagnostics,
    /// State-level diagnostics derived from the input/output states.
    #[serde(default)]
    pub state: StateDiagnostics,
    /// Composite quality score used to rank the best prediction.
    #[serde(default)]
    pub quality_score: f32,
}

/// Guardrail diagnostics for a provider prediction.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct GuardrailDiagnostics {
    /// Number of guardrails evaluated for the prediction.
    pub evaluated_count: usize,
    /// Number of passing guardrails.
    pub passed_count: usize,
    /// Number of failing guardrails.
    pub failed_count: usize,
    /// Number of blocking guardrail violations.
    pub blocking_failures: usize,
    /// Fraction of guardrails that passed.
    pub pass_rate: f32,
}

/// State-level diagnostics for a provider prediction.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct StateDiagnostics {
    /// Number of input-state objects before prediction.
    pub input_object_count: usize,
    /// Number of output-state objects after prediction.
    pub output_object_count: usize,
    /// Signed object-count delta (`output - input`).
    pub object_count_delta: i64,
    /// Number of input objects preserved in the output by stable ID.
    pub preserved_object_count: usize,
    /// Number of input objects dropped by the provider.
    pub dropped_object_count: usize,
    /// Number of output objects newly introduced by the provider.
    pub novel_object_count: usize,
    /// Fraction of input objects preserved in the output.
    pub object_preservation_rate: f32,
    /// Number of input relationships before prediction.
    pub input_relationship_count: usize,
    /// Number of output relationships after prediction.
    pub output_relationship_count: usize,
    /// Signed relationship-count delta (`output - input`).
    pub relationship_count_delta: i64,
    /// Number of input relationships preserved in the output.
    pub preserved_relationship_count: usize,
    /// Fraction of input relationships preserved in the output.
    pub relationship_preservation_rate: f32,
    /// Mean positional drift across preserved objects.
    pub average_position_shift: f32,
    /// Maximum positional drift across preserved objects.
    pub max_position_shift: f32,
    /// Stable identities of dropped objects.
    #[serde(default)]
    pub dropped_objects: Vec<ObjectDiagnostic>,
    /// Stable identities of newly introduced objects.
    #[serde(default)]
    pub novel_objects: Vec<ObjectDiagnostic>,
    /// Drift details for preserved objects, sorted by decreasing distance.
    #[serde(default)]
    pub object_drifts: Vec<ObjectDrift>,
}

/// Stable object identity surfaced in comparison diagnostics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ObjectDiagnostic {
    /// Stable object identifier.
    pub object_id: ObjectId,
    /// Human-readable object name.
    pub object_name: String,
}

/// Positional drift for a single object.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ObjectDrift {
    /// Stable object identifier.
    pub object_id: ObjectId,
    /// Human-readable object name.
    pub object_name: String,
    /// Euclidean drift distance in world units.
    pub distance: f32,
}

/// Pairwise agreement diagnostics between two provider outputs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PairwiseAgreement {
    /// First provider name.
    pub provider_a: String,
    /// Second provider name.
    pub provider_b: String,
    /// Aggregate agreement score for the pair (0.0–1.0).
    pub agreement_score: f32,
    /// Number of shared output objects by stable ID.
    pub common_object_count: usize,
    /// Jaccard overlap across output object IDs.
    pub object_overlap_rate: f32,
    /// Signed output object-count delta (`provider_a - provider_b`).
    pub object_count_delta: i64,
    /// Jaccard overlap across output spatial relationships.
    pub relationship_overlap_rate: f32,
    /// Signed output relationship-count delta (`provider_a - provider_b`).
    pub relationship_count_delta: i64,
    /// Agreement score derived from mean positional deltas.
    pub position_agreement: f32,
    /// Mean positional delta across shared objects.
    pub average_position_delta: f32,
    /// Maximum positional delta across shared objects.
    pub max_position_delta: f32,
    /// Fraction of guardrail outcomes that agree by name.
    pub guardrail_agreement_rate: f32,
    /// Absolute physics-score delta between providers.
    pub physics_score_delta: f32,
    /// Absolute confidence delta between providers.
    pub confidence_delta: f32,
    /// Absolute latency delta in milliseconds.
    pub latency_delta_ms: u64,
    /// Highest positional disagreements across shared objects.
    #[serde(default)]
    pub positional_disagreements: Vec<ObjectDrift>,
}

/// Consensus-level diagnostics shared across all compared providers.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ComparisonConsensus {
    /// Number of object identities shared by every provider output.
    pub shared_object_count: usize,
    /// Shared object identities sorted by name then ID.
    #[serde(default)]
    pub shared_objects: Vec<ObjectDiagnostic>,
    /// Number of relationships shared by every provider output.
    pub shared_relationship_count: usize,
    /// Mean provider confidence across the comparison set.
    pub average_confidence: f32,
    /// Mean guardrail pass rate across the comparison set.
    pub average_guardrail_pass_rate: f32,
    /// Mean provider quality score across the comparison set.
    pub average_quality_score: f32,
    /// Mean provider latency across the comparison set.
    pub average_latency_ms: u64,
    /// Mean pairwise position delta across all provider pairs.
    pub average_pairwise_position_delta: f32,
}

impl MultiPrediction {
    /// Build a comparison from a set of previously generated predictions.
    ///
    /// # Errors
    ///
    /// Returns `WorldForgeError::InvalidState` if no predictions are supplied.
    pub fn try_from_predictions(predictions: Vec<Prediction>) -> Result<Self> {
        if predictions.is_empty() {
            return Err(WorldForgeError::InvalidState(
                "multi-provider comparison requires at least one prediction".to_string(),
            ));
        }

        let scores: Vec<_> = predictions.iter().map(build_provider_score).collect();
        let best_prediction = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.quality_score
                    .partial_cmp(&b.quality_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        a.physics_scores
                            .overall
                            .partial_cmp(&b.physics_scores.overall)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .then_with(|| b.latency_ms.cmp(&a.latency_ms))
            })
            .map(|(index, _)| index)
            .unwrap_or(0);
        let pairwise_agreements = build_pairwise_agreements(&predictions);
        let agreement_score = compute_agreement_score(&pairwise_agreements);
        let consensus = build_consensus(&predictions, &scores, &pairwise_agreements);
        let summary = build_comparison_summary(
            predictions.len(),
            &predictions[best_prediction].provider,
            scores[best_prediction].quality_score,
            consensus.shared_object_count,
            agreement_score,
            &pairwise_agreements,
        );

        Ok(Self {
            predictions,
            agreement_score,
            best_prediction,
            comparison: ComparisonReport {
                scores,
                pairwise_agreements,
                consensus,
                summary,
            },
        })
    }
}

fn build_provider_score(prediction: &Prediction) -> ProviderScore {
    ProviderScore {
        provider: prediction.provider.clone(),
        confidence: prediction.confidence,
        physics_scores: prediction.physics_scores,
        latency_ms: prediction.latency_ms,
        cost: prediction.cost.clone(),
        guardrails: build_guardrail_diagnostics(&prediction.guardrail_results),
        state: build_state_diagnostics(&prediction.input_state, &prediction.output_state),
        quality_score: 0.0,
    }
    .with_quality_score()
}

trait ProviderScoreExt {
    fn with_quality_score(self) -> Self;
}

impl ProviderScoreExt for ProviderScore {
    fn with_quality_score(mut self) -> Self {
        let latency_quality = 1.0 / (1.0 + self.latency_ms as f32 / 1_000.0);
        self.quality_score = 0.30 * self.physics_scores.overall.clamp(0.0, 1.0)
            + 0.20 * self.confidence.clamp(0.0, 1.0)
            + 0.20 * self.guardrails.pass_rate.clamp(0.0, 1.0)
            + 0.15 * self.state.object_preservation_rate.clamp(0.0, 1.0)
            + 0.10 * self.state.relationship_preservation_rate.clamp(0.0, 1.0)
            + 0.05 * latency_quality.clamp(0.0, 1.0);
        self
    }
}

fn build_guardrail_diagnostics(results: &[GuardrailResult]) -> GuardrailDiagnostics {
    let evaluated_count = results.len();
    let passed_count = results.iter().filter(|result| result.passed).count();
    let failed_count = evaluated_count.saturating_sub(passed_count);
    let blocking_failures = results
        .iter()
        .filter(|result| !result.passed && result.severity == ViolationSeverity::Blocking)
        .count();
    let pass_rate = if evaluated_count == 0 {
        1.0
    } else {
        passed_count as f32 / evaluated_count as f32
    };

    GuardrailDiagnostics {
        evaluated_count,
        passed_count,
        failed_count,
        blocking_failures,
        pass_rate,
    }
}

fn build_state_diagnostics(
    input_state: &WorldState,
    output_state: &WorldState,
) -> StateDiagnostics {
    let input_objects = &input_state.scene.objects;
    let output_objects = &output_state.scene.objects;

    let preserved_ids: Vec<_> = input_objects
        .keys()
        .copied()
        .filter(|object_id| output_objects.contains_key(object_id))
        .collect();
    let dropped_objects = collect_object_diagnostics(
        input_objects
            .iter()
            .filter(|(object_id, _)| !output_objects.contains_key(*object_id))
            .map(|(_, object)| object),
    );
    let novel_objects = collect_object_diagnostics(
        output_objects
            .iter()
            .filter(|(object_id, _)| !input_objects.contains_key(*object_id))
            .map(|(_, object)| object),
    );
    let object_drifts = collect_object_drifts(preserved_ids.iter().filter_map(|object_id| {
        let input_object = input_objects.get(object_id)?;
        let output_object = output_objects.get(object_id)?;
        Some((
            *object_id,
            output_object.name.clone(),
            input_object
                .pose
                .position
                .distance(output_object.pose.position),
        ))
    }));
    let average_position_shift = average_distance(&object_drifts);
    let max_position_shift = max_distance(&object_drifts);
    let input_relationship_count = input_state.scene.relationships.len();
    let output_relationship_count = output_state.scene.relationships.len();
    let input_relationships: HashSet<_> = input_state
        .scene
        .relationships
        .iter()
        .map(RelationshipSignature::from)
        .collect();
    let output_relationships: HashSet<_> = output_state
        .scene
        .relationships
        .iter()
        .map(RelationshipSignature::from)
        .collect();
    let preserved_relationship_count = input_relationships
        .intersection(&output_relationships)
        .count();
    let object_preservation_rate = if input_objects.is_empty() {
        1.0
    } else {
        preserved_ids.len() as f32 / input_objects.len() as f32
    };
    let relationship_preservation_rate = if input_relationships.is_empty() {
        1.0
    } else {
        preserved_relationship_count as f32 / input_relationships.len() as f32
    };

    StateDiagnostics {
        input_object_count: input_objects.len(),
        output_object_count: output_objects.len(),
        object_count_delta: output_objects.len() as i64 - input_objects.len() as i64,
        preserved_object_count: preserved_ids.len(),
        dropped_object_count: dropped_objects.len(),
        novel_object_count: novel_objects.len(),
        object_preservation_rate,
        input_relationship_count,
        output_relationship_count,
        relationship_count_delta: output_relationship_count as i64
            - input_relationship_count as i64,
        preserved_relationship_count,
        relationship_preservation_rate,
        average_position_shift,
        max_position_shift,
        dropped_objects,
        novel_objects,
        object_drifts,
    }
}

fn build_pairwise_agreements(predictions: &[Prediction]) -> Vec<PairwiseAgreement> {
    let mut pairwise_agreements = Vec::new();

    for i in 0..predictions.len() {
        for j in (i + 1)..predictions.len() {
            pairwise_agreements.push(build_pairwise_agreement(&predictions[i], &predictions[j]));
        }
    }

    pairwise_agreements
}

fn build_pairwise_agreement(left: &Prediction, right: &Prediction) -> PairwiseAgreement {
    let left_objects = &left.output_state.scene.objects;
    let right_objects = &right.output_state.scene.objects;
    let left_ids: HashSet<_> = left_objects.keys().copied().collect();
    let right_ids: HashSet<_> = right_objects.keys().copied().collect();
    let common_ids: Vec<_> = left_ids.intersection(&right_ids).copied().collect();
    let object_overlap_rate = overlap_rate(left_ids.len(), right_ids.len(), common_ids.len());
    let positional_disagreements =
        collect_object_drifts(common_ids.iter().filter_map(|object_id| {
            let left_object = left_objects.get(object_id)?;
            let right_object = right_objects.get(object_id)?;
            Some((
                *object_id,
                left_object.name.clone(),
                left_object
                    .pose
                    .position
                    .distance(right_object.pose.position),
            ))
        }));
    let average_position_delta = average_distance(&positional_disagreements);
    let max_position_delta = max_distance(&positional_disagreements);
    let position_agreement = if common_ids.is_empty() {
        1.0
    } else {
        1.0 / (1.0 + average_position_delta)
    };

    let left_relationships: HashSet<_> = left
        .output_state
        .scene
        .relationships
        .iter()
        .map(RelationshipSignature::from)
        .collect();
    let right_relationships: HashSet<_> = right
        .output_state
        .scene
        .relationships
        .iter()
        .map(RelationshipSignature::from)
        .collect();
    let shared_relationships = left_relationships
        .intersection(&right_relationships)
        .count();
    let relationship_overlap_rate = overlap_rate(
        left_relationships.len(),
        right_relationships.len(),
        shared_relationships,
    );
    let guardrail_agreement_rate =
        build_guardrail_agreement_rate(&left.guardrail_results, &right.guardrail_results);
    let physics_score_delta = (left.physics_scores.overall - right.physics_scores.overall).abs();
    let confidence_delta = (left.confidence - right.confidence).abs();
    let latency_delta_ms = left.latency_ms.abs_diff(right.latency_ms);
    let agreement_score = 0.30 * object_overlap_rate
        + 0.20 * relationship_overlap_rate
        + 0.20 * position_agreement
        + 0.15 * guardrail_agreement_rate
        + 0.10 * (1.0 - physics_score_delta.clamp(0.0, 1.0))
        + 0.05 * (1.0 - confidence_delta.clamp(0.0, 1.0));

    PairwiseAgreement {
        provider_a: left.provider.clone(),
        provider_b: right.provider.clone(),
        agreement_score,
        common_object_count: common_ids.len(),
        object_overlap_rate,
        object_count_delta: left_objects.len() as i64 - right_objects.len() as i64,
        relationship_overlap_rate,
        relationship_count_delta: left_relationships.len() as i64
            - right_relationships.len() as i64,
        position_agreement,
        average_position_delta,
        max_position_delta,
        guardrail_agreement_rate,
        physics_score_delta,
        confidence_delta,
        latency_delta_ms,
        positional_disagreements,
    }
}

fn build_consensus(
    predictions: &[Prediction],
    scores: &[ProviderScore],
    pairwise_agreements: &[PairwiseAgreement],
) -> ComparisonConsensus {
    let shared_objects = shared_output_objects(predictions);
    let shared_relationship_count = shared_output_relationship_count(predictions);
    let average_confidence = average_f32(scores.iter().map(|score| score.confidence));
    let average_guardrail_pass_rate =
        average_f32(scores.iter().map(|score| score.guardrails.pass_rate));
    let average_quality_score = average_f32(scores.iter().map(|score| score.quality_score));
    let average_latency_ms = average_u64(scores.iter().map(|score| score.latency_ms));
    let average_pairwise_position_delta = average_f32(
        pairwise_agreements
            .iter()
            .map(|pair| pair.average_position_delta),
    );

    ComparisonConsensus {
        shared_object_count: shared_objects.len(),
        shared_objects,
        shared_relationship_count,
        average_confidence,
        average_guardrail_pass_rate,
        average_quality_score,
        average_latency_ms,
        average_pairwise_position_delta,
    }
}

fn build_guardrail_agreement_rate(
    left_results: &[GuardrailResult],
    right_results: &[GuardrailResult],
) -> f32 {
    let left_by_name: HashMap<_, _> = left_results
        .iter()
        .map(|result| (result.guardrail_name.as_str(), result.passed))
        .collect();
    let right_by_name: HashMap<_, _> = right_results
        .iter()
        .map(|result| (result.guardrail_name.as_str(), result.passed))
        .collect();
    let mut guardrail_names: HashSet<&str> = left_by_name.keys().copied().collect();
    guardrail_names.extend(right_by_name.keys().copied());
    if guardrail_names.is_empty() {
        return 1.0;
    }

    let matches = guardrail_names
        .iter()
        .filter(|name| left_by_name.get(**name) == right_by_name.get(**name))
        .count();
    matches as f32 / guardrail_names.len() as f32
}

fn build_comparison_summary(
    provider_count: usize,
    best_provider: &str,
    best_quality_score: f32,
    shared_object_count: usize,
    agreement_score: f32,
    pairwise_agreements: &[PairwiseAgreement],
) -> String {
    match pairwise_agreements.iter().max_by(|left, right| {
        left.agreement_score
            .partial_cmp(&right.agreement_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        Some(strongest_pair) => format!(
            "Compared {provider_count} providers; best: {best_provider} (quality {best_quality_score:.2}); shared objects: {shared_object_count}; average agreement: {agreement_score:.2}; strongest agreement: {} vs {} ({:.2})",
            strongest_pair.provider_a, strongest_pair.provider_b, strongest_pair.agreement_score
        ),
        None => format!(
            "Compared {provider_count} providers; best: {best_provider} (quality {best_quality_score:.2}); shared objects: {shared_object_count}; average agreement: {agreement_score:.2}"
        ),
    }
}

fn compute_agreement_score(pairwise_agreements: &[PairwiseAgreement]) -> f32 {
    if pairwise_agreements.is_empty() {
        return 1.0;
    }

    pairwise_agreements
        .iter()
        .map(|pair| pair.agreement_score)
        .sum::<f32>()
        / pairwise_agreements.len() as f32
}

fn collect_object_diagnostics<'a>(
    objects: impl Iterator<Item = &'a crate::scene::SceneObject>,
) -> Vec<ObjectDiagnostic> {
    let mut diagnostics: Vec<_> = objects
        .map(|object| ObjectDiagnostic {
            object_id: object.id,
            object_name: object.name.clone(),
        })
        .collect();
    diagnostics.sort_by(|left, right| {
        left.object_name
            .cmp(&right.object_name)
            .then_with(|| left.object_id.as_bytes().cmp(right.object_id.as_bytes()))
    });
    diagnostics
}

fn collect_object_drifts(
    objects: impl Iterator<Item = (ObjectId, String, f32)>,
) -> Vec<ObjectDrift> {
    let mut drifts: Vec<_> = objects
        .filter(|(_, _, distance)| *distance > f32::EPSILON)
        .map(|(object_id, object_name, distance)| ObjectDrift {
            object_id,
            object_name,
            distance,
        })
        .collect();
    drifts.sort_by(|left, right| {
        right
            .distance
            .partial_cmp(&left.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.object_name.cmp(&right.object_name))
            .then_with(|| left.object_id.as_bytes().cmp(right.object_id.as_bytes()))
    });
    drifts
}

fn shared_output_objects(predictions: &[Prediction]) -> Vec<ObjectDiagnostic> {
    let mut shared_ids = predictions
        .first()
        .map(|prediction| {
            prediction
                .output_state
                .scene
                .objects
                .keys()
                .copied()
                .collect::<HashSet<_>>()
        })
        .unwrap_or_default();

    for prediction in predictions.iter().skip(1) {
        let output_ids: HashSet<_> = prediction
            .output_state
            .scene
            .objects
            .keys()
            .copied()
            .collect();
        shared_ids.retain(|object_id| output_ids.contains(object_id));
    }

    let Some(reference_prediction) = predictions.first() else {
        return Vec::new();
    };
    collect_object_diagnostics(
        reference_prediction
            .output_state
            .scene
            .objects
            .iter()
            .filter(|(object_id, _)| shared_ids.contains(*object_id))
            .map(|(_, object)| object),
    )
}

fn shared_output_relationship_count(predictions: &[Prediction]) -> usize {
    let mut shared_relationships = predictions
        .first()
        .map(|prediction| {
            prediction
                .output_state
                .scene
                .relationships
                .iter()
                .map(RelationshipSignature::from)
                .collect::<HashSet<_>>()
        })
        .unwrap_or_default();

    for prediction in predictions.iter().skip(1) {
        let relationships: HashSet<_> = prediction
            .output_state
            .scene
            .relationships
            .iter()
            .map(RelationshipSignature::from)
            .collect();
        shared_relationships.retain(|relationship| relationships.contains(relationship));
    }

    shared_relationships.len()
}

fn average_distance(drifts: &[ObjectDrift]) -> f32 {
    if drifts.is_empty() {
        0.0
    } else {
        drifts.iter().map(|drift| drift.distance).sum::<f32>() / drifts.len() as f32
    }
}

fn max_distance(drifts: &[ObjectDrift]) -> f32 {
    drifts
        .iter()
        .map(|drift| drift.distance)
        .fold(0.0, f32::max)
}

fn overlap_rate(left_count: usize, right_count: usize, intersection_count: usize) -> f32 {
    let union_count = left_count + right_count - intersection_count;
    if union_count == 0 {
        1.0
    } else {
        intersection_count as f32 / union_count as f32
    }
}

fn average_f32(values: impl Iterator<Item = f32>) -> f32 {
    let mut total = 0.0f32;
    let mut count = 0usize;
    for value in values {
        total += value;
        count += 1;
    }

    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

fn average_u64(values: impl Iterator<Item = u64>) -> u64 {
    let mut total = 0u128;
    let mut count = 0u128;
    for value in values {
        total += u128::from(value);
        count += 1;
    }

    if count == 0 {
        0
    } else {
        (total / count) as u64
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum RelationshipSignature {
    On {
        subject: ObjectId,
        surface: ObjectId,
    },
    In {
        subject: ObjectId,
        container: ObjectId,
    },
    Near {
        a: ObjectId,
        b: ObjectId,
    },
    Touching {
        a: ObjectId,
        b: ObjectId,
    },
    Above {
        subject: ObjectId,
        reference: ObjectId,
    },
    Below {
        subject: ObjectId,
        reference: ObjectId,
    },
}

impl From<&SpatialRelationship> for RelationshipSignature {
    fn from(value: &SpatialRelationship) -> Self {
        match value {
            SpatialRelationship::On { subject, surface } => Self::On {
                subject: *subject,
                surface: *surface,
            },
            SpatialRelationship::In { subject, container } => Self::In {
                subject: *subject,
                container: *container,
            },
            SpatialRelationship::Near { a, b, .. } => {
                let (a, b) = ordered_pair(*a, *b);
                Self::Near { a, b }
            }
            SpatialRelationship::Touching { a, b } => {
                let (a, b) = ordered_pair(*a, *b);
                Self::Touching { a, b }
            }
            SpatialRelationship::Above { subject, reference } => Self::Above {
                subject: *subject,
                reference: *reference,
            },
            SpatialRelationship::Below { subject, reference } => Self::Below {
                subject: *subject,
                reference: *reference,
            },
        }
    }
}

fn ordered_pair(left: ObjectId, right: ObjectId) -> (ObjectId, ObjectId) {
    if left.as_bytes() <= right.as_bytes() {
        (left, right)
    } else {
        (right, left)
    }
}

// ---------------------------------------------------------------------------
// Planning types
// ---------------------------------------------------------------------------

/// Request for planning a sequence of actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanRequest {
    /// Current world state.
    pub current_state: WorldState,
    /// Goal to achieve.
    pub goal: PlanGoal,
    /// Maximum number of planning steps.
    pub max_steps: u32,
    /// Guardrails to enforce during planning.
    pub guardrails: Vec<crate::guardrail::GuardrailConfig>,
    /// Planning algorithm to use.
    pub planner: PlannerType,
    /// Maximum planning time in seconds.
    pub timeout_seconds: f64,
    /// Optional fallback provider used if the primary planning provider fails.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_provider: Option<String>,
}

/// Goal specification for planning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlanGoal {
    /// A condition that must be satisfied.
    Condition(crate::action::Condition),
    /// A target world state to reach.
    TargetState(Box<WorldState>),
    /// An image depicting the goal state.
    GoalImage(crate::types::Tensor),
    /// A natural language description of the goal.
    Description(String),
}

/// Serde-friendly input shape for plan goals across APIs.
///
/// This preserves backward compatibility for callers that still send a bare
/// string while allowing richer structured goals to flow through the CLI,
/// server, and Python bindings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PlanGoalInput {
    /// Backward-compatible natural-language goal.
    Description(String),
    /// Structured goal payload.
    Structured(PlanGoalSpec),
}

/// Structured plan-goal payload for external APIs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PlanGoalSpec {
    /// Natural-language goal payload.
    Description { description: String },
    /// Boolean condition goal.
    Condition {
        /// Condition that must evaluate to true.
        condition: crate::action::Condition,
    },
    /// Explicit target world-state goal.
    TargetState {
        /// State that planning should approximate.
        state: Box<WorldState>,
    },
    /// Goal image tensor for image-conditioned planning.
    GoalImage {
        /// Serialized tensor describing the desired image.
        image: crate::types::Tensor,
    },
}

/// Planning algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlannerType {
    /// Gradient-based optimization.
    Gradient {
        learning_rate: f32,
        num_iterations: u32,
    },
    /// Random sampling.
    Sampling { num_samples: u32, top_k: u32 },
    /// Cross-entropy method.
    CEM {
        population_size: u32,
        elite_fraction: f32,
        num_iterations: u32,
    },
    /// Model predictive control.
    MPC {
        horizon: u32,
        num_samples: u32,
        replanning_interval: u32,
    },
    /// Use the provider's native planner.
    ProviderNative,
}

impl PredictionConfig {
    /// Disable guardrail evaluation for this request.
    ///
    /// This sets an explicit sentinel so core can distinguish opt-out from
    /// the default safety path used when the list is empty.
    pub fn disable_guardrails(mut self) -> Self {
        self.guardrails = vec![GuardrailConfig {
            guardrail: Guardrail::Disabled,
            blocking: false,
        }];
        self
    }
}

impl PlanRequest {
    /// Disable guardrail evaluation for this planning request.
    ///
    /// This sets an explicit sentinel so core can distinguish opt-out from
    /// the default safety path used when the list is empty.
    pub fn disable_guardrails(mut self) -> Self {
        self.guardrails = vec![GuardrailConfig {
            guardrail: Guardrail::Disabled,
            blocking: false,
        }];
        self
    }
}

impl From<PlanGoalInput> for PlanGoal {
    fn from(value: PlanGoalInput) -> Self {
        match value {
            PlanGoalInput::Description(description) => Self::Description(description),
            PlanGoalInput::Structured(spec) => spec.into(),
        }
    }
}

impl From<PlanGoalSpec> for PlanGoal {
    fn from(value: PlanGoalSpec) -> Self {
        match value {
            PlanGoalSpec::Description { description } => Self::Description(description),
            PlanGoalSpec::Condition { condition } => Self::Condition(condition),
            PlanGoalSpec::TargetState { state } => Self::TargetState(state),
            PlanGoalSpec::GoalImage { image } => Self::GoalImage(image),
        }
    }
}

/// Result of a planning operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    /// Planned action sequence.
    pub actions: Vec<Action>,
    /// Predicted world states at each step.
    pub predicted_states: Vec<WorldState>,
    /// Predicted videos for each step (if requested).
    pub predicted_videos: Option<Vec<VideoClip>>,
    /// Total estimated cost.
    pub total_cost: f32,
    /// Probability of success (0.0–1.0).
    pub success_probability: f32,
    /// Guardrail compliance at each step.
    pub guardrail_compliance: Vec<Vec<GuardrailResult>>,
    /// Time spent planning in milliseconds.
    pub planning_time_ms: u64,
    /// Number of iterations used.
    pub iterations_used: u32,
    /// Attached verification proof when planning was augmented with proof generation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verification_proof: Option<crate::proof::ZkProof>,
}

/// Result of executing a materialized plan against a live world.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanExecution {
    /// Step-by-step predictions produced while replaying the plan.
    pub predictions: Vec<Prediction>,
    /// Final committed world state after all actions succeeded.
    pub final_state: WorldState,
    /// Aggregate cost across all executed prediction steps.
    pub total_cost: CostEstimate,
    /// End-to-end execution time in milliseconds.
    pub execution_time_ms: u64,
}

impl Default for PredictionConfig {
    fn default() -> Self {
        Self {
            steps: 1,
            resolution: (640, 480),
            fps: 24.0,
            return_video: false,
            return_depth: false,
            return_segmentation: false,
            guardrails: Vec::new(),
            max_latency_ms: None,
            fallback_provider: None,
            num_samples: 1,
            temperature: 1.0,
        }
    }
}

impl Default for PhysicsScores {
    fn default() -> Self {
        Self {
            overall: 0.0,
            object_permanence: 0.0,
            gravity_compliance: 0.0,
            collision_accuracy: 0.0,
            spatial_consistency: 0.0,
            temporal_consistency: 0.0,
        }
    }
}

impl PlanExecution {
    /// Build an execution report from a completed sequence of predictions.
    pub fn from_predictions(
        predictions: Vec<Prediction>,
        final_state: WorldState,
        execution_time_ms: u64,
    ) -> Self {
        let total_cost = predictions.iter().fold(
            CostEstimate {
                usd: 0.0,
                credits: 0.0,
                estimated_latency_ms: 0,
            },
            |mut total, prediction| {
                total.usd += prediction.cost.usd;
                total.credits += prediction.cost.credits;
                total.estimated_latency_ms += prediction.cost.estimated_latency_ms;
                total
            },
        );

        Self {
            predictions,
            final_state,
            total_cost,
            execution_time_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::SceneObject;
    use crate::state::WorldState;
    use crate::types::{BBox, Pose, Position};

    fn sample_prediction(provider: &str, physics_score: f32, latency_ms: u64) -> Prediction {
        let state = WorldState::new(format!("{provider}-state"), provider);
        Prediction {
            id: uuid::Uuid::new_v4(),
            provider: provider.to_string(),
            model: format!("{provider}-model"),
            input_state: state.clone(),
            action: Action::Move {
                target: crate::types::Position::default(),
                speed: 1.0,
            },
            output_state: state,
            video: None,
            confidence: physics_score,
            physics_scores: PhysicsScores {
                overall: physics_score,
                object_permanence: physics_score,
                gravity_compliance: physics_score,
                collision_accuracy: physics_score,
                spatial_consistency: physics_score,
                temporal_consistency: physics_score,
            },
            latency_ms,
            cost: CostEstimate {
                usd: latency_ms as f64 / 1_000.0,
                credits: 1.0,
                estimated_latency_ms: latency_ms,
            },
            guardrail_results: Vec::new(),
            timestamp: chrono::Utc::now(),
        }
    }

    fn diagnostic_prediction(
        provider: &str,
        physics_score: f32,
        latency_ms: u64,
        mug_x: f32,
        keep_mug: bool,
        add_cube: bool,
        guardrail_passed: bool,
    ) -> Prediction {
        let mut input_state = WorldState::new(format!("{provider}-state"), provider);
        let mut counter = SceneObject::new(
            "counter",
            Pose {
                position: Position {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                ..Pose::default()
            },
            BBox {
                min: Position {
                    x: -1.0,
                    y: -0.1,
                    z: -1.0,
                },
                max: Position {
                    x: 1.0,
                    y: 0.0,
                    z: 1.0,
                },
            },
        );
        counter.id = uuid::Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap();
        let mut mug = SceneObject::new(
            "mug",
            Pose {
                position: Position {
                    x: 0.0,
                    y: 0.1,
                    z: 0.0,
                },
                ..Pose::default()
            },
            BBox {
                min: Position {
                    x: -0.05,
                    y: 0.0,
                    z: -0.05,
                },
                max: Position {
                    x: 0.05,
                    y: 0.2,
                    z: 0.05,
                },
            },
        );
        mug.id = uuid::Uuid::parse_str("00000000-0000-0000-0000-000000000002").unwrap();
        let mug_id = mug.id;
        input_state.scene.add_object(counter);
        input_state.scene.add_object(mug);

        let mut output_state = input_state.clone();
        if keep_mug {
            output_state.scene.set_object_position(
                &mug_id,
                Position {
                    x: mug_x,
                    y: 0.1,
                    z: 0.0,
                },
            );
        } else {
            output_state.scene.remove_object(&mug_id);
        }

        if add_cube {
            let mut cube = SceneObject::new(
                "cube",
                Pose {
                    position: Position {
                        x: 0.4,
                        y: 0.1,
                        z: 0.0,
                    },
                    ..Pose::default()
                },
                BBox {
                    min: Position {
                        x: 0.3,
                        y: 0.0,
                        z: -0.1,
                    },
                    max: Position {
                        x: 0.5,
                        y: 0.2,
                        z: 0.1,
                    },
                },
            );
            cube.id = uuid::Uuid::parse_str("00000000-0000-0000-0000-000000000003").unwrap();
            output_state.scene.add_object(cube);
        }
        output_state.scene.refresh_relationships();

        Prediction {
            id: uuid::Uuid::new_v4(),
            provider: provider.to_string(),
            model: format!("{provider}-model"),
            input_state,
            action: Action::Move {
                target: Position {
                    x: mug_x,
                    y: 0.1,
                    z: 0.0,
                },
                speed: 1.0,
            },
            output_state,
            video: None,
            confidence: physics_score,
            physics_scores: PhysicsScores {
                overall: physics_score,
                object_permanence: physics_score,
                gravity_compliance: physics_score,
                collision_accuracy: physics_score,
                spatial_consistency: physics_score,
                temporal_consistency: physics_score,
            },
            latency_ms,
            cost: CostEstimate {
                usd: latency_ms as f64 / 1_000.0,
                credits: if add_cube { 1.5 } else { 1.0 },
                estimated_latency_ms: latency_ms,
            },
            guardrail_results: vec![GuardrailResult {
                guardrail_name: "NoCollisions".to_string(),
                passed: guardrail_passed,
                violation_details: (!guardrail_passed)
                    .then_some("collision between mug and counter".to_string()),
                severity: if guardrail_passed {
                    ViolationSeverity::Info
                } else {
                    ViolationSeverity::Blocking
                },
            }],
            timestamp: chrono::Utc::now(),
        }
    }

    #[test]
    fn test_prediction_config_default() {
        let config = PredictionConfig::default();
        assert_eq!(config.steps, 1);
        assert_eq!(config.resolution, (640, 480));
        assert!(!config.return_video);
    }

    #[test]
    fn test_physics_scores_default() {
        let scores = PhysicsScores::default();
        assert_eq!(scores.overall, 0.0);
    }

    #[test]
    fn test_prediction_config_deserializes_partial_json() {
        let config: PredictionConfig =
            serde_json::from_str(r#"{"fallback_provider":"mock"}"#).unwrap();

        assert_eq!(config.fallback_provider.as_deref(), Some("mock"));
        assert_eq!(config.steps, 1);
        assert_eq!(config.resolution, (640, 480));
    }

    #[test]
    fn test_planner_type_serialization() {
        let planner = PlannerType::CEM {
            population_size: 100,
            elite_fraction: 0.1,
            num_iterations: 50,
        };
        let json = serde_json::to_string(&planner).unwrap();
        let planner2: PlannerType = serde_json::from_str(&json).unwrap();
        match planner2 {
            PlannerType::CEM {
                population_size, ..
            } => assert_eq!(population_size, 100),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_plan_goal_description() {
        let goal = PlanGoal::Description("stack the blocks".to_string());
        let json = serde_json::to_string(&goal).unwrap();
        let goal2: PlanGoal = serde_json::from_str(&json).unwrap();
        match goal2 {
            PlanGoal::Description(s) => assert_eq!(s, "stack the blocks"),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_plan_goal_input_accepts_string_description() {
        let input: PlanGoalInput = serde_json::from_str(r#""stack the blocks""#).unwrap();
        let goal: PlanGoal = input.into();

        match goal {
            PlanGoal::Description(description) => assert_eq!(description, "stack the blocks"),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_plan_goal_input_accepts_structured_condition() {
        let input: PlanGoalInput = serde_json::from_str(
            r#"{
                "type":"condition",
                "condition":{
                    "ObjectExists":{"object":"00000000-0000-0000-0000-000000000123"}
                }
            }"#,
        )
        .unwrap();
        let goal: PlanGoal = input.into();

        match goal {
            PlanGoal::Condition(crate::action::Condition::ObjectExists { object }) => {
                assert_eq!(
                    object,
                    uuid::Uuid::parse_str("00000000-0000-0000-0000-000000000123").unwrap()
                );
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_plan_goal_input_accepts_structured_target_state() {
        let state = WorldState::new("goal-state", "mock");
        let json = serde_json::json!({
            "type": "target_state",
            "state": state,
        });
        let input: PlanGoalInput = serde_json::from_value(json).unwrap();
        let goal: PlanGoal = input.into();

        match goal {
            PlanGoal::TargetState(target) => {
                assert_eq!(target.metadata.name, "goal-state");
                assert_eq!(target.metadata.created_by, "mock");
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_plan_goal_input_accepts_structured_goal_image() {
        let state = WorldState::new("goal-image", "mock");
        let image = crate::goal_image::render_scene_goal_image(&state, (12, 8));
        let json = serde_json::json!({
            "type": "goal_image",
            "image": image,
        });
        let input: PlanGoalInput = serde_json::from_value(json).unwrap();
        let goal: PlanGoal = input.into();

        match goal {
            PlanGoal::GoalImage(image) => assert_eq!(image.shape, vec![8, 12]),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_plan_serialization_defaults_missing_verification_proof() {
        let plan_json = serde_json::json!({
            "actions": [],
            "predicted_states": [],
            "predicted_videos": null,
            "total_cost": 0.0,
            "success_probability": 0.0,
            "guardrail_compliance": [],
            "planning_time_ms": 0,
            "iterations_used": 0
        });

        let plan: Plan = serde_json::from_value(plan_json).unwrap();
        assert!(plan.verification_proof.is_none());
    }

    #[test]
    fn test_plan_serialization_roundtrip_preserves_verification_proof() {
        let proof = crate::proof::ZkProof {
            proof_type: crate::proof::ZkProofType::GuardrailCompliance {
                plan_hash: [1; 32],
                guardrail_hashes: vec![[2; 32]],
                all_passed: true,
            },
            proof_data: vec![1, 2, 3],
            backend: crate::proof::VerificationBackend::Mock,
            generation_time_ms: 5,
        };
        let plan = Plan {
            actions: Vec::new(),
            predicted_states: Vec::new(),
            predicted_videos: None,
            total_cost: 0.0,
            success_probability: 1.0,
            guardrail_compliance: Vec::new(),
            planning_time_ms: 1,
            iterations_used: 1,
            verification_proof: Some(proof.clone()),
        };

        let json = serde_json::to_string(&plan).unwrap();
        let restored: Plan = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.verification_proof, Some(proof));
    }

    #[test]
    fn test_multi_prediction_try_from_predictions_picks_best_provider() {
        let multi = MultiPrediction::try_from_predictions(vec![
            sample_prediction("provider-a", 0.52, 90),
            sample_prediction("provider-b", 0.81, 110),
            sample_prediction("provider-c", 0.73, 70),
        ])
        .unwrap();

        assert_eq!(multi.predictions.len(), 3);
        assert_eq!(multi.best_prediction, 1);
        assert_eq!(multi.comparison.scores.len(), 3);
        assert!(multi
            .comparison
            .summary
            .contains("Compared 3 providers; best: provider-b"));
        assert!(multi.agreement_score > 0.0);
        assert!(multi.agreement_score <= 1.0);
        assert!(
            multi.comparison.scores[1].quality_score >= multi.comparison.scores[0].quality_score
        );
    }

    #[test]
    fn test_multi_prediction_collects_state_and_pairwise_diagnostics() {
        let multi = MultiPrediction::try_from_predictions(vec![
            diagnostic_prediction("provider-a", 0.52, 90, 0.20, true, false, true),
            diagnostic_prediction("provider-b", 0.81, 110, 0.25, true, false, true),
            diagnostic_prediction("provider-c", 0.73, 70, 0.20, false, true, false),
        ])
        .unwrap();

        assert_eq!(multi.comparison.scores.len(), 3);
        assert_eq!(multi.comparison.pairwise_agreements.len(), 3);
        assert!(multi.comparison.summary.contains("strongest agreement"));
        assert_eq!(multi.comparison.consensus.shared_object_count, 1);
        assert_eq!(
            multi.comparison.consensus.shared_objects[0].object_name,
            "counter"
        );
        assert!(multi.comparison.consensus.average_quality_score > 0.0);

        let provider_b = multi
            .comparison
            .scores
            .iter()
            .find(|score| score.provider == "provider-b")
            .unwrap();
        assert_eq!(provider_b.state.output_object_count, 2);
        assert_eq!(provider_b.state.preserved_object_count, 2);
        assert!(provider_b.state.preserved_relationship_count > 0);
        assert!(provider_b.state.relationship_preservation_rate > 0.0);
        assert!(provider_b.state.average_position_shift > 0.0);
        assert_eq!(provider_b.guardrails.evaluated_count, 1);
        assert_eq!(provider_b.guardrails.passed_count, 1);
        assert!(provider_b.quality_score > 0.0);

        let provider_c = multi
            .comparison
            .scores
            .iter()
            .find(|score| score.provider == "provider-c")
            .unwrap();
        assert_eq!(provider_c.state.dropped_object_count, 1);
        assert_eq!(provider_c.state.novel_object_count, 1);
        assert_eq!(provider_c.state.dropped_objects[0].object_name, "mug");
        assert_eq!(provider_c.state.novel_objects[0].object_name, "cube");
        assert_eq!(provider_c.guardrails.blocking_failures, 1);

        let pair = multi
            .comparison
            .pairwise_agreements
            .iter()
            .find(|pair| pair.provider_a == "provider-a" && pair.provider_b == "provider-b")
            .unwrap();
        assert_eq!(pair.common_object_count, 2);
        assert!(pair.object_overlap_rate > 0.99);
        assert!(pair.average_position_delta > 0.0);
        assert_eq!(pair.guardrail_agreement_rate, 1.0);

        let json = serde_json::to_value(&multi).unwrap();
        assert!(json["comparison"]["scores"][2]["state"]["dropped_objects"].is_array());
        assert!(json["comparison"]["pairwise_agreements"][0]["agreement_score"].is_number());
        assert!(json["comparison"]["consensus"]["average_quality_score"].is_number());
    }

    #[test]
    fn test_multi_prediction_try_from_predictions_rejects_empty_input() {
        let error = MultiPrediction::try_from_predictions(Vec::new()).unwrap_err();

        assert!(matches!(error, WorldForgeError::InvalidState(_)));
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn arb_physics_scores() -> impl Strategy<Value = PhysicsScores> {
            (
                0.0f32..=1.0,
                0.0f32..=1.0,
                0.0f32..=1.0,
                0.0f32..=1.0,
                0.0f32..=1.0,
                0.0f32..=1.0,
            )
                .prop_map(|(overall, op, gc, ca, sc, tc)| PhysicsScores {
                    overall,
                    object_permanence: op,
                    gravity_compliance: gc,
                    collision_accuracy: ca,
                    spatial_consistency: sc,
                    temporal_consistency: tc,
                })
        }

        fn arb_planner_type() -> impl Strategy<Value = PlannerType> {
            prop_oneof![
                (
                    any::<f32>().prop_filter("finite", |v| v.is_finite()),
                    any::<u32>()
                )
                    .prop_map(|(lr, ni)| PlannerType::Gradient {
                        learning_rate: lr,
                        num_iterations: ni,
                    }),
                (any::<u32>(), any::<u32>()).prop_map(|(ns, tk)| PlannerType::Sampling {
                    num_samples: ns,
                    top_k: tk,
                }),
                Just(PlannerType::ProviderNative),
            ]
        }

        proptest! {
            #[test]
            fn physics_scores_roundtrip(scores in arb_physics_scores()) {
                let json = serde_json::to_string(&scores).unwrap();
                let scores2: PhysicsScores = serde_json::from_str(&json).unwrap();
                prop_assert!((scores.overall - scores2.overall).abs() < f32::EPSILON);
                prop_assert!((scores.object_permanence - scores2.object_permanence).abs() < f32::EPSILON);
            }

            #[test]
            fn prediction_config_roundtrip(
                steps in 1u32..100,
                w in 1u32..4096,
                h in 1u32..4096,
                fps in 1.0f32..120.0,
            ) {
                let config = PredictionConfig {
                    steps,
                    resolution: (w, h),
                    fps,
                    ..PredictionConfig::default()
                };
                let json = serde_json::to_string(&config).unwrap();
                let config2: PredictionConfig = serde_json::from_str(&json).unwrap();
                prop_assert_eq!(config2.steps, steps);
                prop_assert_eq!(config2.resolution, (w, h));
            }

            #[test]
            fn planner_type_roundtrip(pt in arb_planner_type()) {
                let json = serde_json::to_string(&pt).unwrap();
                let _pt2: PlannerType = serde_json::from_str(&json).unwrap();
            }

            #[test]
            fn plan_goal_description_roundtrip(desc in ".*") {
                let goal = PlanGoal::Description(desc.clone());
                let json = serde_json::to_string(&goal).unwrap();
                let goal2: PlanGoal = serde_json::from_str(&json).unwrap();
                match goal2 {
                    PlanGoal::Description(s) => prop_assert_eq!(s, desc),
                    _ => prop_assert!(false, "wrong variant"),
                }
            }
        }
    }
}
