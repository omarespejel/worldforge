//! WR-Arena evaluation dimensions for world foundation models.
//!
//! Implements evaluation metrics from the WR-Arena benchmark (arXiv 2603.25887):
//!
//! - **Action Simulation Fidelity**: LLM-as-judge scoring for instruction following
//! - **Transition Smoothness (MRS)**: Optical-flow-based temporal smoothness metric
//! - **Generation Consistency**: Multi-aspect consistency scoring (WorldScore-based)
//!
//! These dimensions complement WorldForge's existing physics-based evaluation
//! by testing instruction-following, temporal stability, and visual coherence.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use worldforge_core::types::VideoClip;

use crate::datasets::SimulationType;

// ---------------------------------------------------------------------------
// Action Simulation Fidelity (LLM-as-judge)
// ---------------------------------------------------------------------------

/// Scoring rubric for action simulation fidelity (0–3 integer scale).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum FidelityScore {
    /// Sequence does not follow the instruction at all.
    NoCompliance = 0,
    /// Correct object but wrong action, or vice versa.
    PartialMatch = 1,
    /// Follows instruction and shows tendency toward the intended goal.
    TendencyTowardGoal = 2,
    /// Follows instruction precisely and achieves the goal.
    FullCompliance = 3,
}

impl FidelityScore {
    /// Convert from a raw integer score (clamped to 0–3).
    pub fn from_raw(value: i32) -> Self {
        match value {
            ..=0 => Self::NoCompliance,
            1 => Self::PartialMatch,
            2 => Self::TendencyTowardGoal,
            3.. => Self::FullCompliance,
        }
    }

    /// The numeric value (0–3).
    pub fn value(self) -> f32 {
        self as u8 as f32
    }

    /// Normalized score (0.0–1.0).
    pub fn normalized(self) -> f32 {
        self.value() / 3.0
    }
}

/// The evaluation prompt template for action simulation fidelity.
///
/// This is sent to a multimodal LLM (GPT-4o, Claude, etc.) along with
/// extracted frames from the generated video.
pub const ACTION_FIDELITY_PROMPT: &str = r#"You are given a sequence of frames sampled in chronological order from a video.
Evaluate whether the sequence follows the instruction: "{instruction}".
Use the following scoring criteria:
- 0: The sequence does not follow the instruction at all.
- 1: The sequence includes the correct object but performs the wrong action, or the action is correct but on the wrong object.
- 2: The sequence follows the instruction and shows a tendency toward the intended goal.
- 3: The sequence follows the instruction precisely and successfully achieves the goal.
Return ONLY one integer: 0, 1, 2, or 3. Do not output any other text."#;

/// Build the evaluation prompt with the instruction filled in.
pub fn build_fidelity_prompt(instruction: &str) -> String {
    ACTION_FIDELITY_PROMPT.replace("{instruction}", instruction)
}

/// Result of evaluating action simulation fidelity for one instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionFidelityResult {
    /// Instance identifier.
    pub instance_id: String,
    /// Simulation type (agent or environment).
    pub simulation_type: SimulationType,
    /// Per-round scores.
    pub round_scores: Vec<FidelityScore>,
    /// Average score across all rounds (0.0–3.0).
    pub mean_score: f32,
    /// Normalized average score (0.0–1.0).
    pub normalized_score: f32,
}

impl ActionFidelityResult {
    /// Compute from a list of per-round scores.
    pub fn from_scores(
        instance_id: impl Into<String>,
        simulation_type: SimulationType,
        round_scores: Vec<FidelityScore>,
    ) -> Self {
        let mean_score = if round_scores.is_empty() {
            0.0
        } else {
            round_scores.iter().map(|s| s.value()).sum::<f32>() / round_scores.len() as f32
        };
        Self {
            instance_id: instance_id.into(),
            simulation_type,
            round_scores,
            mean_score,
            normalized_score: mean_score / 3.0,
        }
    }
}

/// Aggregate action fidelity results across multiple instances.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionFidelityReport {
    /// Per-instance results.
    pub results: Vec<ActionFidelityResult>,
    /// Mean score across all agent-type instances.
    pub agent_mean: f32,
    /// Mean score across all environment-type instances.
    pub environment_mean: f32,
    /// Overall mean across all instances.
    pub overall_mean: f32,
}

impl ActionFidelityReport {
    /// Build a report from individual results.
    pub fn from_results(results: Vec<ActionFidelityResult>) -> Self {
        let agent_scores: Vec<f32> = results
            .iter()
            .filter(|r| r.simulation_type == SimulationType::Agent)
            .map(|r| r.mean_score)
            .collect();
        let env_scores: Vec<f32> = results
            .iter()
            .filter(|r| r.simulation_type == SimulationType::Environment)
            .map(|r| r.mean_score)
            .collect();
        let all_scores: Vec<f32> = results.iter().map(|r| r.mean_score).collect();

        Self {
            results,
            agent_mean: mean_or_zero(&agent_scores),
            environment_mean: mean_or_zero(&env_scores),
            overall_mean: mean_or_zero(&all_scores),
        }
    }
}

// ---------------------------------------------------------------------------
// Transition Smoothness (MRS metric)
// ---------------------------------------------------------------------------

/// Configuration for the MRS smoothness metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmoothnessConfig {
    /// Smoothness penalty weight λ in the MRS formula.
    /// Higher values penalize acceleration more.
    pub lambda: f32,
}

impl Default for SmoothnessConfig {
    fn default() -> Self {
        Self { lambda: 1.0 }
    }
}

/// Result of smoothness evaluation for one video.
///
/// The MRS (Motion Regularity Score) formula is:
/// ```text
/// MRS = vmag_median * exp(-λ * amag_median)
/// ```
/// where `vmag` is the per-pixel velocity magnitude from optical flow
/// and `amag` is the per-pixel acceleration magnitude.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmoothnessResult {
    /// Instance identifier.
    pub instance_id: String,
    /// Per-round MRS scores.
    pub round_scores: Vec<f32>,
    /// Overall MRS score for the full video.
    pub overall_mrs: f32,
    /// MRS computed only at round boundaries.
    pub boundary_mrs: Option<f32>,
    /// MRS computed only within rounds (excluding boundaries).
    pub intra_round_mrs: Option<f32>,
    /// Median velocity magnitude.
    pub vmag_median: f32,
    /// Median acceleration magnitude.
    pub amag_median: f32,
}

/// Compute the MRS score from velocity and acceleration medians.
///
/// # Formula
///
/// `MRS = vmag_median * exp(-λ * amag_median)`
///
/// Higher MRS = smoother motion (high velocity without sudden acceleration).
pub fn compute_mrs(vmag_median: f32, amag_median: f32, lambda: f32) -> f32 {
    vmag_median * (-lambda * amag_median).exp()
}

/// Compute velocity magnitudes from a sequence of frame pixel differences.
///
/// This is a simplified optical flow proxy: it computes per-pixel differences
/// between consecutive frames and takes the magnitude. For production use,
/// a proper optical flow estimator (RAFT, SEA-RAFT) should be used.
///
/// Returns a vector of velocity magnitude medians, one per frame transition.
pub fn compute_velocity_magnitudes(clip: &VideoClip) -> Vec<f32> {
    if clip.frames.len() < 2 {
        return Vec::new();
    }

    clip.frames
        .windows(2)
        .map(|pair| {
            let a = &pair[0].data;
            let b = &pair[1].data;
            frame_difference_magnitude(a, b)
        })
        .collect()
}

/// Compute acceleration magnitudes from velocity magnitudes.
///
/// Acceleration is the absolute change in velocity between consecutive
/// frame pairs.
pub fn compute_acceleration_magnitudes(velocities: &[f32]) -> Vec<f32> {
    if velocities.len() < 2 {
        return Vec::new();
    }

    velocities
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).abs())
        .collect()
}

/// Evaluate smoothness of a video clip using the MRS metric.
pub fn evaluate_smoothness(
    instance_id: impl Into<String>,
    clip: &VideoClip,
    config: &SmoothnessConfig,
) -> SmoothnessResult {
    let velocities = compute_velocity_magnitudes(clip);
    let accelerations = compute_acceleration_magnitudes(&velocities);

    let vmag_median = median(&velocities);
    let amag_median = median(&accelerations);
    let overall_mrs = compute_mrs(vmag_median, amag_median, config.lambda);

    SmoothnessResult {
        instance_id: instance_id.into(),
        round_scores: Vec::new(), // Per-round requires round boundary info
        overall_mrs,
        boundary_mrs: None,
        intra_round_mrs: None,
        vmag_median,
        amag_median,
    }
}

/// Aggregate smoothness report across multiple instances.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmoothnessReport {
    /// Per-instance results.
    pub results: Vec<SmoothnessResult>,
    /// Mean MRS across all instances.
    pub mean_mrs: f32,
    /// Standard deviation of MRS.
    pub std_mrs: f32,
    /// Minimum MRS.
    pub min_mrs: f32,
    /// Maximum MRS.
    pub max_mrs: f32,
}

impl SmoothnessReport {
    /// Build a report from individual results.
    pub fn from_results(results: Vec<SmoothnessResult>) -> Self {
        let scores: Vec<f32> = results.iter().map(|r| r.overall_mrs).collect();
        Self {
            results,
            mean_mrs: mean_or_zero(&scores),
            std_mrs: std_dev(&scores),
            min_mrs: scores.iter().copied().reduce(f32::min).unwrap_or(0.0),
            max_mrs: scores.iter().copied().reduce(f32::max).unwrap_or(0.0),
        }
    }
}

// ---------------------------------------------------------------------------
// Generation Consistency (WorldScore-based)
// ---------------------------------------------------------------------------

/// Aspects of generation consistency to evaluate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsistencyAspect {
    /// Does the camera go where instructed?
    CameraControl,
    /// Are specified objects present and correctly positioned?
    ObjectControl,
    /// Does the output match the text prompt? (CLIP score)
    ContentAlignment,
    /// Is the 3D structure stable across rounds?
    ThreeDConsistency,
    /// Are pixel intensities consistent across rounds?
    PhotometricConsistency,
    /// Is the visual style stable across rounds?
    StyleConsistency,
    /// How good does the output look? (IQA metrics)
    SubjectiveQuality,
}

impl ConsistencyAspect {
    /// All available consistency aspects.
    pub fn all() -> &'static [ConsistencyAspect] {
        &[
            Self::CameraControl,
            Self::ObjectControl,
            Self::ContentAlignment,
            Self::ThreeDConsistency,
            Self::PhotometricConsistency,
            Self::StyleConsistency,
            Self::SubjectiveQuality,
        ]
    }

    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::CameraControl => "Camera Control",
            Self::ObjectControl => "Object Control",
            Self::ContentAlignment => "Content Alignment",
            Self::ThreeDConsistency => "3D Consistency",
            Self::PhotometricConsistency => "Photometric Consistency",
            Self::StyleConsistency => "Style Consistency",
            Self::SubjectiveQuality => "Subjective Quality",
        }
    }
}

/// Per-round consistency scores for a single instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyResult {
    /// Instance identifier.
    pub instance_id: String,
    /// Scores per aspect per round.
    pub aspect_scores: HashMap<ConsistencyAspect, Vec<f32>>,
    /// Composite WorldScore-Static score per round.
    pub composite_per_round: Vec<f32>,
    /// Overall composite score.
    pub composite_overall: f32,
    /// Degradation metric (AP): average of score(r)/score(1) across rounds.
    /// Values below 1.0 indicate degradation over rounds.
    pub degradation_ap: f32,
}

impl ConsistencyResult {
    /// Compute the AP (average persistence) degradation metric.
    ///
    /// AP = mean over rounds r of: score(r) / score(1)
    /// Lower AP means faster degradation.
    pub fn compute_degradation(per_round: &[f32]) -> f32 {
        if per_round.is_empty() {
            return 1.0;
        }
        let baseline = per_round[0];
        if baseline <= 0.0 {
            return 0.0;
        }
        let ratios: Vec<f32> = per_round.iter().map(|s| s / baseline).collect();
        mean_or_zero(&ratios)
    }
}

/// Aggregate consistency report across multiple instances.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyReport {
    /// Per-instance results.
    pub results: Vec<ConsistencyResult>,
    /// Mean composite score across all instances.
    pub mean_composite: f32,
    /// Mean degradation AP across all instances.
    pub mean_degradation_ap: f32,
    /// Per-aspect mean scores.
    pub aspect_means: HashMap<ConsistencyAspect, f32>,
}

impl ConsistencyReport {
    /// Build a report from individual results.
    pub fn from_results(results: Vec<ConsistencyResult>) -> Self {
        let composites: Vec<f32> = results.iter().map(|r| r.composite_overall).collect();
        let degradations: Vec<f32> = results.iter().map(|r| r.degradation_ap).collect();

        let mut aspect_means = HashMap::new();
        for aspect in ConsistencyAspect::all() {
            let scores: Vec<f32> = results
                .iter()
                .filter_map(|r| {
                    r.aspect_scores
                        .get(aspect)
                        .map(|per_round| mean_or_zero(per_round))
                })
                .collect();
            if !scores.is_empty() {
                aspect_means.insert(*aspect, mean_or_zero(&scores));
            }
        }

        Self {
            results,
            mean_composite: mean_or_zero(&composites),
            mean_degradation_ap: mean_or_zero(&degradations),
            aspect_means,
        }
    }
}

// ---------------------------------------------------------------------------
// Shared negative prompt (quality guard)
// ---------------------------------------------------------------------------

/// Shared negative prompt used by Cosmos, KLING, and other providers
/// as a quality guard during video generation.
pub const QUALITY_GUARD_NEGATIVE_PROMPT: &str = "The video captures a series of frames \
showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, \
low resolution, grainy texture, poor lighting, washed out colors, lens distortion, \
unnatural movements, clipping artifacts, rendering errors, text overlays, watermarks";

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

fn mean_or_zero(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn std_dev(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = mean_or_zero(values);
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
    variance.sqrt()
}

fn median(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

/// Compute per-pixel difference magnitude between two frame tensors.
///
/// This is a simplified proxy for optical flow — it computes the mean
/// absolute difference across all channels. For production use, a proper
/// optical flow estimator should be used.
fn frame_difference_magnitude(
    a: &worldforge_core::types::Tensor,
    b: &worldforge_core::types::Tensor,
) -> f32 {
    let a_values = a.data.to_f32_values();
    let b_values = b.data.to_f32_values();

    if a_values.len() != b_values.len() || a_values.is_empty() {
        return 0.0;
    }

    let sum: f32 = a_values
        .iter()
        .zip(b_values.iter())
        .map(|(av, bv)| (*av - *bv).abs())
        .sum();

    sum / a_values.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fidelity_score_from_raw() {
        assert_eq!(FidelityScore::from_raw(0), FidelityScore::NoCompliance);
        assert_eq!(FidelityScore::from_raw(1), FidelityScore::PartialMatch);
        assert_eq!(
            FidelityScore::from_raw(2),
            FidelityScore::TendencyTowardGoal
        );
        assert_eq!(FidelityScore::from_raw(3), FidelityScore::FullCompliance);
        assert_eq!(FidelityScore::from_raw(5), FidelityScore::FullCompliance);
        assert_eq!(FidelityScore::from_raw(-1), FidelityScore::NoCompliance);
    }

    #[test]
    fn test_fidelity_score_normalized() {
        assert_eq!(FidelityScore::NoCompliance.normalized(), 0.0);
        assert!((FidelityScore::PartialMatch.normalized() - 1.0 / 3.0).abs() < 0.001);
        assert!((FidelityScore::TendencyTowardGoal.normalized() - 2.0 / 3.0).abs() < 0.001);
        assert_eq!(FidelityScore::FullCompliance.normalized(), 1.0);
    }

    #[test]
    fn test_action_fidelity_result() {
        let result = ActionFidelityResult::from_scores(
            "agent_001_1",
            SimulationType::Agent,
            vec![
                FidelityScore::FullCompliance,
                FidelityScore::TendencyTowardGoal,
                FidelityScore::PartialMatch,
            ],
        );
        assert_eq!(result.mean_score, 2.0);
        assert!((result.normalized_score - 2.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_action_fidelity_report() {
        let results = vec![
            ActionFidelityResult::from_scores(
                "agent_001",
                SimulationType::Agent,
                vec![FidelityScore::FullCompliance],
            ),
            ActionFidelityResult::from_scores(
                "env_001",
                SimulationType::Environment,
                vec![FidelityScore::NoCompliance],
            ),
        ];
        let report = ActionFidelityReport::from_results(results);
        assert_eq!(report.agent_mean, 3.0);
        assert_eq!(report.environment_mean, 0.0);
        assert_eq!(report.overall_mean, 1.5);
    }

    #[test]
    fn test_build_fidelity_prompt() {
        let prompt = build_fidelity_prompt("robot picks up cup");
        assert!(prompt.contains("robot picks up cup"));
        assert!(prompt.contains("Return ONLY one integer"));
    }

    #[test]
    fn test_compute_mrs() {
        // High velocity, low acceleration = smooth
        let smooth = compute_mrs(10.0, 0.1, 1.0);
        // High velocity, high acceleration = jerky
        let jerky = compute_mrs(10.0, 5.0, 1.0);
        assert!(smooth > jerky);

        // Zero velocity = zero MRS regardless of acceleration
        assert_eq!(compute_mrs(0.0, 1.0, 1.0), 0.0);
    }

    #[test]
    fn test_compute_mrs_lambda_effect() {
        let low_lambda = compute_mrs(10.0, 1.0, 0.5);
        let high_lambda = compute_mrs(10.0, 1.0, 2.0);
        // Higher lambda penalizes acceleration more
        assert!(low_lambda > high_lambda);
    }

    #[test]
    fn test_compute_velocity_magnitudes_empty() {
        let clip = VideoClip {
            frames: Vec::new(),
            fps: 24.0,
            resolution: (640, 480),
            duration: 0.0,
        };
        assert!(compute_velocity_magnitudes(&clip).is_empty());
    }

    #[test]
    fn test_compute_acceleration_from_velocities() {
        let velocities = vec![1.0, 3.0, 2.0, 5.0];
        let accelerations = compute_acceleration_magnitudes(&velocities);
        assert_eq!(accelerations.len(), 3);
        assert_eq!(accelerations[0], 2.0); // |3-1|
        assert_eq!(accelerations[1], 1.0); // |2-3|
        assert_eq!(accelerations[2], 3.0); // |5-2|
    }

    #[test]
    fn test_consistency_degradation_ap() {
        // Constant quality → AP = 1.0
        assert_eq!(
            ConsistencyResult::compute_degradation(&[0.8, 0.8, 0.8]),
            1.0
        );

        // Degrading quality → AP < 1.0
        let ap = ConsistencyResult::compute_degradation(&[1.0, 0.5, 0.25]);
        assert!(ap < 1.0);
        // 1.0/1.0=1.0, 0.5/1.0=0.5, 0.25/1.0=0.25 → mean = 0.583...
        assert!((ap - 0.5833).abs() < 0.01);

        // Empty → 1.0
        assert_eq!(ConsistencyResult::compute_degradation(&[]), 1.0);
    }

    #[test]
    fn test_consistency_aspect_all() {
        assert_eq!(ConsistencyAspect::all().len(), 7);
    }

    #[test]
    fn test_consistency_report() {
        let results = vec![ConsistencyResult {
            instance_id: "s1".to_string(),
            aspect_scores: HashMap::from([(
                ConsistencyAspect::ContentAlignment,
                vec![0.9, 0.8, 0.7],
            )]),
            composite_per_round: vec![0.9, 0.8, 0.7],
            composite_overall: 0.8,
            degradation_ap: 0.89,
        }];
        let report = ConsistencyReport::from_results(results);
        assert_eq!(report.mean_composite, 0.8);
        assert!((report.mean_degradation_ap - 0.89).abs() < 0.001);
        assert!(report
            .aspect_means
            .contains_key(&ConsistencyAspect::ContentAlignment));
    }

    #[test]
    fn test_median() {
        assert_eq!(median(&[]), 0.0);
        assert_eq!(median(&[5.0]), 5.0);
        assert_eq!(median(&[1.0, 3.0, 2.0]), 2.0);
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    }

    #[test]
    fn test_std_dev() {
        assert_eq!(std_dev(&[]), 0.0);
        assert_eq!(std_dev(&[5.0]), 0.0);
        // std of [2.0, 4.0, 6.0] = sqrt(8/3) ≈ 1.633
        let s = std_dev(&[2.0, 4.0, 6.0]);
        assert!((s - 1.633).abs() < 0.01);
    }

    #[test]
    fn test_quality_guard_negative_prompt() {
        assert!(QUALITY_GUARD_NEGATIVE_PROMPT.contains("ugly scenes"));
        assert!(QUALITY_GUARD_NEGATIVE_PROMPT.contains("motion blur"));
    }

    #[test]
    fn test_fidelity_score_serde_roundtrip() {
        let score = FidelityScore::TendencyTowardGoal;
        let json = serde_json::to_string(&score).unwrap();
        let parsed: FidelityScore = serde_json::from_str(&json).unwrap();
        assert_eq!(score, parsed);
    }

    #[test]
    fn test_consistency_aspect_serde_roundtrip() {
        let aspect = ConsistencyAspect::ThreeDConsistency;
        let json = serde_json::to_string(&aspect).unwrap();
        assert_eq!(json, r#""three_d_consistency""#);
        let parsed: ConsistencyAspect = serde_json::from_str(&json).unwrap();
        assert_eq!(aspect, parsed);
    }
}
