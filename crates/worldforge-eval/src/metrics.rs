//! Real evaluation metrics for WorldForge (RFC-0008).
//!
//! Implements:
//! - SSIM (Structural Similarity Index) with 11x11 Gaussian window
//! - Cosine similarity for embedding vectors
//! - Physics evaluation suite (object permanence, gravity, collision, conservation)
//! - Cross-provider comparison with pairwise SSIM and ranking

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use worldforge_core::scene::SceneObject;
use worldforge_core::state::WorldState;
use worldforge_core::types::{ObjectId, Tensor, TensorData};

// ---------------------------------------------------------------------------
// SSIM (Structural Similarity Index)
// ---------------------------------------------------------------------------

/// Constants for SSIM calculation (following Wang et al. 2004).
const SSIM_K1: f64 = 0.01;
const SSIM_K2: f64 = 0.03;
/// Default dynamic range for 8-bit images.
const SSIM_L: f64 = 255.0;
/// Gaussian window radius (11x11 kernel).
const SSIM_WINDOW_SIZE: usize = 11;
/// Gaussian sigma for the SSIM window.
const SSIM_SIGMA: f64 = 1.5;

/// Result of an SSIM comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SsimResult {
    /// Overall mean SSIM in [0, 1].
    pub mean_ssim: f32,
    /// Per-channel SSIM values when applicable.
    pub channel_ssim: Vec<f32>,
    /// Width of the compared images.
    pub width: usize,
    /// Height of the compared images.
    pub height: usize,
}

/// Generate a 2D Gaussian kernel of the given size and sigma.
fn gaussian_kernel(size: usize, sigma: f64) -> Vec<f64> {
    let mut kernel = vec![0.0f64; size * size];
    let center = size as f64 / 2.0;
    let mut sum = 0.0;

    for row in 0..size {
        for col in 0..size {
            let dx = col as f64 + 0.5 - center;
            let dy = row as f64 + 0.5 - center;
            let value = (-((dx * dx + dy * dy) / (2.0 * sigma * sigma))).exp();
            kernel[row * size + col] = value;
            sum += value;
        }
    }

    // Normalize
    for value in &mut kernel {
        *value /= sum;
    }

    kernel
}

/// Extract a single channel as f64 values from a tensor with shape [H, W, C] or [H, W].
fn extract_channel(data: &[f64], height: usize, width: usize, channels: usize, channel: usize) -> Vec<f64> {
    let mut result = vec![0.0f64; height * width];
    for y in 0..height {
        for x in 0..width {
            let idx = if channels > 1 {
                (y * width + x) * channels + channel
            } else {
                y * width + x
            };
            result[y * width + x] = if idx < data.len() { data[idx] } else { 0.0 };
        }
    }
    result
}

/// Compute the windowed weighted sum of a*b using a Gaussian kernel.
fn windowed_product(
    a: &[f64],
    b: &[f64],
    width: usize,
    height: usize,
    kernel: &[f64],
    window_size: usize,
) -> Vec<f64> {
    let half = window_size / 2;
    let out_h = height.saturating_sub(window_size - 1);
    let out_w = width.saturating_sub(window_size - 1);
    let mut result = vec![0.0f64; out_h * out_w];

    for oy in 0..out_h {
        for ox in 0..out_w {
            let mut sum = 0.0f64;
            for ky in 0..window_size {
                for kx in 0..window_size {
                    let py = oy + ky;
                    let px = ox + kx;
                    let w = kernel[ky * window_size + kx];
                    sum += w * a[py * width + px] * b[py * width + px];
                }
            }
            result[oy * out_w + ox] = sum;
            let _ = half; // suppress unused warning
        }
    }
    result
}

/// Compute windowed mean of a single channel.
fn windowed_mean(
    data: &[f64],
    width: usize,
    height: usize,
    kernel: &[f64],
    window_size: usize,
) -> Vec<f64> {
    let out_h = height.saturating_sub(window_size - 1);
    let out_w = width.saturating_sub(window_size - 1);
    let mut result = vec![0.0f64; out_h * out_w];

    for oy in 0..out_h {
        for ox in 0..out_w {
            let mut sum = 0.0f64;
            for ky in 0..window_size {
                for kx in 0..window_size {
                    let py = oy + ky;
                    let px = ox + kx;
                    sum += kernel[ky * window_size + kx] * data[py * width + px];
                }
            }
            result[oy * out_w + ox] = sum;
        }
    }
    result
}

/// Compute SSIM for a single channel image pair.
fn ssim_single_channel(
    img_a: &[f64],
    img_b: &[f64],
    width: usize,
    height: usize,
    dynamic_range: f64,
) -> f32 {
    let c1 = (SSIM_K1 * dynamic_range) * (SSIM_K1 * dynamic_range);
    let c2 = (SSIM_K2 * dynamic_range) * (SSIM_K2 * dynamic_range);

    let kernel = gaussian_kernel(SSIM_WINDOW_SIZE, SSIM_SIGMA);
    let ws = SSIM_WINDOW_SIZE;

    let mu_a = windowed_mean(img_a, width, height, &kernel, ws);
    let mu_b = windowed_mean(img_b, width, height, &kernel, ws);

    let sigma_aa = windowed_product(img_a, img_a, width, height, &kernel, ws);
    let sigma_bb = windowed_product(img_b, img_b, width, height, &kernel, ws);
    let sigma_ab = windowed_product(img_a, img_b, width, height, &kernel, ws);

    let n = mu_a.len();
    if n == 0 {
        return 1.0;
    }

    let mut ssim_sum = 0.0f64;
    for i in 0..n {
        let mu_a_sq = mu_a[i] * mu_a[i];
        let mu_b_sq = mu_b[i] * mu_b[i];
        let mu_ab = mu_a[i] * mu_b[i];

        let var_a = sigma_aa[i] - mu_a_sq;
        let var_b = sigma_bb[i] - mu_b_sq;
        let cov_ab = sigma_ab[i] - mu_ab;

        let numerator = (2.0 * mu_ab + c1) * (2.0 * cov_ab + c2);
        let denominator = (mu_a_sq + mu_b_sq + c1) * (var_a + var_b + c2);

        ssim_sum += numerator / denominator;
    }

    ((ssim_sum / n as f64) as f32).clamp(0.0, 1.0)
}

/// Compute SSIM between two tensors representing images.
///
/// Tensors should have shape `[H, W, C]` or `[H, W]`.
/// Returns an `SsimResult` with mean SSIM in [0, 1].
pub fn compute_ssim(tensor_a: &Tensor, tensor_b: &Tensor) -> SsimResult {
    let (height_a, width_a, channels_a) = parse_image_shape(&tensor_a.shape);
    let (height_b, width_b, channels_b) = parse_image_shape(&tensor_b.shape);

    // If shapes don't match, return 0
    if height_a != height_b || width_a != width_b || channels_a != channels_b {
        return SsimResult {
            mean_ssim: 0.0,
            channel_ssim: vec![],
            width: width_a,
            height: height_a,
        };
    }

    let height = height_a;
    let width = width_a;
    let channels = channels_a;

    // Images too small for the window
    if height < SSIM_WINDOW_SIZE || width < SSIM_WINDOW_SIZE {
        return SsimResult {
            mean_ssim: 1.0,
            channel_ssim: vec![1.0; channels],
            width,
            height,
        };
    }

    let data_a = tensor_to_f64(&tensor_a.data);
    let data_b = tensor_to_f64(&tensor_b.data);

    let dynamic_range = infer_dynamic_range(&tensor_a.data);

    let mut channel_ssim = Vec::with_capacity(channels);
    for ch in 0..channels {
        let ch_a = extract_channel(&data_a, height, width, channels, ch);
        let ch_b = extract_channel(&data_b, height, width, channels, ch);
        let ssim = ssim_single_channel(&ch_a, &ch_b, width, height, dynamic_range);
        channel_ssim.push(ssim);
    }

    let mean_ssim = if channel_ssim.is_empty() {
        1.0
    } else {
        channel_ssim.iter().copied().sum::<f32>() / channel_ssim.len() as f32
    };

    SsimResult {
        mean_ssim,
        channel_ssim,
        width,
        height,
    }
}

/// Compute SSIM between two raw grayscale image buffers (u8).
///
/// Convenience function for simple image comparison.
pub fn compute_ssim_grayscale(
    img_a: &[u8],
    img_b: &[u8],
    width: usize,
    height: usize,
) -> f32 {
    if img_a.len() != width * height || img_b.len() != width * height {
        return 0.0;
    }
    if height < SSIM_WINDOW_SIZE || width < SSIM_WINDOW_SIZE {
        return 1.0;
    }

    let a: Vec<f64> = img_a.iter().map(|&v| v as f64).collect();
    let b: Vec<f64> = img_b.iter().map(|&v| v as f64).collect();

    ssim_single_channel(&a, &b, width, height, SSIM_L)
}

fn parse_image_shape(shape: &[usize]) -> (usize, usize, usize) {
    match shape.len() {
        2 => (shape[0], shape[1], 1),
        3 => (shape[0], shape[1], shape[2]),
        _ => (0, 0, 0),
    }
}

fn tensor_to_f64(data: &TensorData) -> Vec<f64> {
    match data {
        TensorData::UInt8(v) => v.iter().map(|&x| x as f64).collect(),
        TensorData::Float32(v) => v.iter().map(|&x| x as f64).collect(),
        TensorData::Float64(v) => v.clone(),
        TensorData::Int32(v) => v.iter().map(|&x| x as f64).collect(),
        TensorData::Int64(v) => v.iter().map(|&x| x as f64).collect(),
        TensorData::Float16(v) => v.iter().map(|&bits| super::half_bits_to_f32(bits) as f64).collect(),
        TensorData::BFloat16(v) => v.iter().map(|&bits| super::bfloat16_bits_to_f32(bits) as f64).collect(),
    }
}

fn infer_dynamic_range(data: &TensorData) -> f64 {
    match data {
        TensorData::UInt8(_) => 255.0,
        TensorData::Float32(v) => {
            let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if max <= 1.0 { 1.0 } else { max as f64 }
        }
        TensorData::Float64(v) => {
            let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if max <= 1.0 { 1.0 } else { max }
        }
        _ => 255.0,
    }
}

// ---------------------------------------------------------------------------
// Cosine Similarity
// ---------------------------------------------------------------------------

/// Compute cosine similarity between two embedding vectors.
///
/// Returns a value in [-1, 1] where 1 means identical direction.
/// For normalized embeddings this is equivalent to dot product.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a < f64::EPSILON || norm_b < f64::EPSILON {
        return 0.0;
    }

    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Compute cosine similarity between two tensors treated as flat embedding vectors.
pub fn tensor_cosine_similarity(a: &Tensor, b: &Tensor) -> f64 {
    let va = tensor_to_f64(&a.data);
    let vb = tensor_to_f64(&b.data);
    cosine_similarity(&va, &vb)
}

// ---------------------------------------------------------------------------
// Physics Evaluation Suite
// ---------------------------------------------------------------------------

/// Result of a physics evaluation test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsTestResult {
    /// Name of the test.
    pub name: String,
    /// Whether the test passed.
    pub passed: bool,
    /// Score in [0, 1].
    pub score: f32,
    /// Human-readable explanation.
    pub details: String,
}

/// Aggregate result of the physics evaluation suite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsEvalResult {
    /// Individual test results.
    pub tests: Vec<PhysicsTestResult>,
    /// Overall physics score (mean of test scores).
    pub overall_score: f32,
    /// Object permanence sub-score.
    pub object_permanence: f32,
    /// Gravity compliance sub-score.
    pub gravity: f32,
    /// Collision accuracy sub-score.
    pub collision: f32,
    /// Energy conservation sub-score.
    pub conservation: f32,
}

/// Run the full physics evaluation suite comparing initial and final world states.
///
/// Tests:
/// 1. Object permanence: all initial objects still exist in the final state
/// 2. Gravity: unsupported objects with non-zero mass have moved downward (negative Y)
/// 3. Collision: no two objects' bounding boxes overlap in the final state
/// 4. Conservation: total kinetic energy is roughly conserved
pub fn evaluate_physics(initial: &WorldState, final_state: &WorldState) -> PhysicsEvalResult {
    let permanence = test_object_permanence(initial, final_state);
    let gravity = test_gravity(initial, final_state);
    let collision = test_collision(final_state);
    let conservation = test_energy_conservation(initial, final_state);

    let tests = vec![permanence.clone(), gravity.clone(), collision.clone(), conservation.clone()];
    let overall_score = tests.iter().map(|t| t.score).sum::<f32>() / tests.len() as f32;

    PhysicsEvalResult {
        tests,
        overall_score,
        object_permanence: permanence.score,
        gravity: gravity.score,
        collision: collision.score,
        conservation: conservation.score,
    }
}

/// Test that all objects from the initial state still exist in the final state.
fn test_object_permanence(initial: &WorldState, final_state: &WorldState) -> PhysicsTestResult {
    let initial_ids: Vec<&ObjectId> = initial.scene.objects.keys().collect();
    let total = initial_ids.len();

    if total == 0 {
        return PhysicsTestResult {
            name: "object_permanence".to_string(),
            passed: true,
            score: 1.0,
            details: "No objects to track".to_string(),
        };
    }

    let mut present = 0usize;
    let mut missing = Vec::new();
    for id in &initial_ids {
        if final_state.scene.objects.contains_key(*id) {
            present += 1;
        } else {
            missing.push(id.to_string());
        }
    }

    let score = present as f32 / total as f32;
    let passed = score >= 1.0;
    let details = if passed {
        format!("All {} objects persisted", total)
    } else {
        format!("{} of {} objects persisted; missing: {}", present, total, missing.join(", "))
    };

    PhysicsTestResult {
        name: "object_permanence".to_string(),
        passed,
        score,
        details,
    }
}

/// Test that unsupported non-static objects have moved downward (negative Y direction).
fn test_gravity(initial: &WorldState, final_state: &WorldState) -> PhysicsTestResult {
    let mut tested = 0usize;
    let mut compliant = 0usize;
    let mut violations = Vec::new();

    for (id, obj) in &initial.scene.objects {
        // Only test non-static objects with mass
        if obj.physics.is_static {
            continue;
        }
        let mass = obj.physics.mass.unwrap_or(0.0);
        if mass <= 0.0 {
            continue;
        }

        // Check if object is supported (has an "On" relationship)
        let is_supported = initial.scene.relationships.iter().any(|rel| {
            matches!(rel, worldforge_core::scene::SpatialRelationship::On { subject, .. } if subject == id)
        });

        if is_supported {
            continue;
        }

        tested += 1;
        if let Some(final_obj) = final_state.scene.objects.get(id) {
            // Gravity means Y should decrease or stay same (falling down)
            if final_obj.pose.position.y <= obj.pose.position.y {
                compliant += 1;
            } else {
                violations.push(format!(
                    "{}: moved up from y={:.2} to y={:.2}",
                    obj.name, obj.pose.position.y, final_obj.pose.position.y
                ));
            }
        }
    }

    if tested == 0 {
        return PhysicsTestResult {
            name: "gravity".to_string(),
            passed: true,
            score: 1.0,
            details: "No unsupported dynamic objects to test".to_string(),
        };
    }

    let score = compliant as f32 / tested as f32;
    let passed = score >= 0.9;
    let details = if violations.is_empty() {
        format!("All {} unsupported objects moved downward", tested)
    } else {
        format!(
            "{} of {} compliant; violations: {}",
            compliant, tested, violations.join("; ")
        )
    };

    PhysicsTestResult {
        name: "gravity".to_string(),
        passed,
        score,
        details,
    }
}

/// Test that no two objects' bounding boxes overlap in the final state.
fn test_collision(state: &WorldState) -> PhysicsTestResult {
    let objects: Vec<&SceneObject> = state.scene.objects.values().collect();
    let total_pairs = if objects.len() > 1 {
        objects.len() * (objects.len() - 1) / 2
    } else {
        0
    };

    if total_pairs == 0 {
        return PhysicsTestResult {
            name: "collision".to_string(),
            passed: true,
            score: 1.0,
            details: "Fewer than 2 objects, no collision test needed".to_string(),
        };
    }

    let mut overlapping = 0usize;
    let mut overlap_details = Vec::new();

    for i in 0..objects.len() {
        for j in (i + 1)..objects.len() {
            if bboxes_overlap(&objects[i].bbox, &objects[j].bbox) {
                overlapping += 1;
                overlap_details.push(format!("{} <-> {}", objects[i].name, objects[j].name));
            }
        }
    }

    let non_overlapping = total_pairs - overlapping;
    let score = non_overlapping as f32 / total_pairs as f32;
    let passed = overlapping == 0;
    let details = if passed {
        format!("No overlaps among {} object pairs", total_pairs)
    } else {
        format!(
            "{} overlapping pairs out of {}: {}",
            overlapping,
            total_pairs,
            overlap_details.join("; ")
        )
    };

    PhysicsTestResult {
        name: "collision".to_string(),
        passed,
        score,
        details,
    }
}

fn bboxes_overlap(
    a: &worldforge_core::types::BBox,
    b: &worldforge_core::types::BBox,
) -> bool {
    a.min.x < b.max.x
        && a.max.x > b.min.x
        && a.min.y < b.max.y
        && a.max.y > b.min.y
        && a.min.z < b.max.z
        && a.max.z > b.min.z
}

/// Test that total kinetic energy is roughly conserved between states.
fn test_energy_conservation(initial: &WorldState, final_state: &WorldState) -> PhysicsTestResult {
    let initial_ke = compute_total_kinetic_energy(initial);
    let final_ke = compute_total_kinetic_energy(final_state);

    // Allow 20% tolerance for energy conservation (accounting for friction, etc.)
    let tolerance = 0.2;
    let max_ke = initial_ke.max(final_ke).max(f64::EPSILON);
    let ratio = (initial_ke - final_ke).abs() / max_ke;

    let score = (1.0 - ratio / tolerance).clamp(0.0, 1.0) as f32;
    let passed = ratio <= tolerance;

    let details = format!(
        "Initial KE: {:.4}, Final KE: {:.4}, ratio change: {:.2}%",
        initial_ke,
        final_ke,
        ratio * 100.0
    );

    PhysicsTestResult {
        name: "energy_conservation".to_string(),
        passed,
        score,
        details,
    }
}

fn compute_total_kinetic_energy(state: &WorldState) -> f64 {
    let mut total = 0.0f64;
    for obj in state.scene.objects.values() {
        let mass = obj.physics.mass.unwrap_or(0.0) as f64;
        let v = &obj.velocity;
        let speed_sq = (v.x * v.x + v.y * v.y + v.z * v.z) as f64;
        total += 0.5 * mass * speed_sq;
    }
    total
}

// ---------------------------------------------------------------------------
// Cross-Provider Comparison
// ---------------------------------------------------------------------------

/// Pairwise comparison entry between two providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseComparison {
    /// First provider name.
    pub provider_a: String,
    /// Second provider name.
    pub provider_b: String,
    /// SSIM score between outputs (if visual output available).
    pub ssim: Option<f32>,
    /// Physics score difference (a - b).
    pub physics_diff: f32,
}

/// Ranked provider entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderRanking {
    /// Provider name.
    pub provider: String,
    /// Aggregate physics score.
    pub physics_score: f32,
    /// Rank (1-indexed, 1 = best).
    pub rank: usize,
}

/// Full cross-provider comparison report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossProviderReport {
    /// Pairwise comparisons between all provider pairs.
    pub pairwise: Vec<PairwiseComparison>,
    /// Providers ranked by physics score.
    pub rankings: Vec<ProviderRanking>,
    /// Summary statistics.
    pub summary: HashMap<String, f32>,
}

/// Input for cross-provider comparison: provider name -> (physics eval, optional output tensor).
pub struct ProviderOutput {
    /// Provider name.
    pub name: String,
    /// Physics evaluation result.
    pub physics: PhysicsEvalResult,
    /// Optional output frame/image tensor for visual comparison.
    pub output_frame: Option<Tensor>,
}

/// Run cross-provider comparison given outputs from multiple providers.
///
/// Computes pairwise SSIM between output frames, ranks providers by physics score,
/// and generates a comparison report.
pub fn cross_provider_comparison(outputs: &[ProviderOutput]) -> CrossProviderReport {
    let mut pairwise = Vec::new();

    // Compute pairwise comparisons
    for i in 0..outputs.len() {
        for j in (i + 1)..outputs.len() {
            let ssim = match (&outputs[i].output_frame, &outputs[j].output_frame) {
                (Some(a), Some(b)) => Some(compute_ssim(a, b).mean_ssim),
                _ => None,
            };

            let physics_diff = outputs[i].physics.overall_score - outputs[j].physics.overall_score;

            pairwise.push(PairwiseComparison {
                provider_a: outputs[i].name.clone(),
                provider_b: outputs[j].name.clone(),
                ssim,
                physics_diff,
            });
        }
    }

    // Rank providers by physics score (descending)
    let mut rankings: Vec<ProviderRanking> = outputs
        .iter()
        .map(|o| ProviderRanking {
            provider: o.name.clone(),
            physics_score: o.physics.overall_score,
            rank: 0,
        })
        .collect();

    rankings.sort_by(|a, b| b.physics_score.partial_cmp(&a.physics_score).unwrap_or(std::cmp::Ordering::Equal));
    for (i, r) in rankings.iter_mut().enumerate() {
        r.rank = i + 1;
    }

    // Summary stats
    let mut summary = HashMap::new();
    if !outputs.is_empty() {
        let mean_physics: f32 = outputs.iter().map(|o| o.physics.overall_score).sum::<f32>() / outputs.len() as f32;
        let max_physics = outputs
            .iter()
            .map(|o| o.physics.overall_score)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_physics = outputs
            .iter()
            .map(|o| o.physics.overall_score)
            .fold(f32::INFINITY, f32::min);

        summary.insert("mean_physics_score".to_string(), mean_physics);
        summary.insert("max_physics_score".to_string(), max_physics);
        summary.insert("min_physics_score".to_string(), min_physics);
        summary.insert("provider_count".to_string(), outputs.len() as f32);

        // Mean pairwise SSIM
        let ssim_values: Vec<f32> = pairwise.iter().filter_map(|p| p.ssim).collect();
        if !ssim_values.is_empty() {
            let mean_ssim: f32 = ssim_values.iter().sum::<f32>() / ssim_values.len() as f32;
            summary.insert("mean_pairwise_ssim".to_string(), mean_ssim);
        }
    }

    CrossProviderReport {
        pairwise,
        rankings,
        summary,
    }
}

/// Generate a JSON report string from a cross-provider comparison.
pub fn report_to_json(report: &CrossProviderReport) -> String {
    serde_json::to_string_pretty(report).unwrap_or_else(|_| "{}".to_string())
}

/// Generate a markdown report from a cross-provider comparison.
pub fn report_to_markdown(report: &CrossProviderReport) -> String {
    let mut md = String::new();
    md.push_str("# Cross-Provider Comparison Report\n\n");

    // Rankings table
    md.push_str("## Provider Rankings\n\n");
    md.push_str("| Rank | Provider | Physics Score |\n");
    md.push_str("|------|----------|---------------|\n");
    for r in &report.rankings {
        md.push_str(&format!("| {} | {} | {:.4} |\n", r.rank, r.provider, r.physics_score));
    }
    md.push('\n');

    // Pairwise comparisons
    md.push_str("## Pairwise Comparisons\n\n");
    md.push_str("| Provider A | Provider B | SSIM | Physics Diff |\n");
    md.push_str("|------------|------------|------|--------------|\n");
    for p in &report.pairwise {
        let ssim_str = p.ssim.map_or("N/A".to_string(), |v| format!("{:.4}", v));
        md.push_str(&format!(
            "| {} | {} | {} | {:.4} |\n",
            p.provider_a, p.provider_b, ssim_str, p.physics_diff
        ));
    }
    md.push('\n');

    // Summary
    md.push_str("## Summary\n\n");
    let mut keys: Vec<&String> = report.summary.keys().collect();
    keys.sort();
    for key in keys {
        md.push_str(&format!("- **{}**: {:.4}\n", key, report.summary[key]));
    }

    md
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use worldforge_core::scene::{PhysicsProperties, SceneGraph, SceneNode, SpatialRelationship};
    use worldforge_core::state::{Compression, StateHistory, WorldMetadata, WorldState};
    use worldforge_core::types::{BBox, DType, Device, ObjectId, Pose, Position, Rotation, Tensor, TensorData, Velocity};
    use std::collections::{BTreeMap, HashMap, VecDeque};

    fn make_tensor_u8(data: Vec<u8>, shape: Vec<usize>) -> Tensor {
        Tensor {
            data: TensorData::UInt8(data),
            shape,
            dtype: DType::UInt8,
            device: Device::Cpu,
        }
    }

    fn make_tensor_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        Tensor {
            data: TensorData::Float32(data),
            shape,
            dtype: DType::Float32,
            device: Device::Cpu,
        }
    }

    fn make_world_state(objects: Vec<SceneObject>) -> WorldState {
        let mut obj_map = HashMap::new();
        for obj in &objects {
            obj_map.insert(obj.id.clone(), obj.clone());
        }

        WorldState {
            id: uuid::Uuid::new_v4(),
            time: worldforge_core::types::SimTime::default(),
            scene: SceneGraph {
                root: SceneNode {
                    name: "root".to_string(),
                    children: vec![],
                    object_id: None,
                },
                objects: obj_map,
                relationships: vec![],
            },
            history: StateHistory {
                states: VecDeque::new(),
                max_entries: 10,
                compression: Compression::None,
            },
            metadata: WorldMetadata {
                name: "test".to_string(),
                description: "test world".to_string(),
                created_by: "test".to_string(),
                created_at: chrono::Utc::now(),
                tags: vec![],
            },
            stored_plans: BTreeMap::new(),
            schema_version: worldforge_core::state::WORLD_STATE_SCHEMA_VERSION,
            version: 0,
        }
    }

    fn make_object(
        id: &str,
        name: &str,
        pos: (f32, f32, f32),
        bbox_min: (f32, f32, f32),
        bbox_max: (f32, f32, f32),
        mass: Option<f32>,
        is_static: bool,
        velocity: (f32, f32, f32),
    ) -> SceneObject {
        SceneObject {
            id: uuid::Uuid::from_bytes({
                let mut b = [0u8; 16];
                let src = id.as_bytes();
                for i in 0..src.len().min(16) {
                    b[i] = src[i];
                }
                b
            }),
            name: name.to_string(),
            pose: Pose {
                position: Position {
                    x: pos.0,
                    y: pos.1,
                    z: pos.2,
                },
                rotation: Rotation {
                    w: 1.0,
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
            },
            bbox: BBox {
                min: Position {
                    x: bbox_min.0,
                    y: bbox_min.1,
                    z: bbox_min.2,
                },
                max: Position {
                    x: bbox_max.0,
                    y: bbox_max.1,
                    z: bbox_max.2,
                },
            },
            mesh: None,
            physics: PhysicsProperties {
                mass,
                friction: None,
                restitution: None,
                is_static,
                is_graspable: false,
                material: None,
            },
            velocity: Velocity {
                x: velocity.0,
                y: velocity.1,
                z: velocity.2,
            },
            semantic_label: None,
            visual_embedding: None,
        }
    }

    // --- SSIM tests ---

    #[test]
    fn test_ssim_identical_images() {
        let data: Vec<u8> = (0..64 * 64).map(|i| (i % 256) as u8).collect();
        let tensor = make_tensor_u8(data, vec![64, 64]);
        let result = compute_ssim(&tensor, &tensor);
        assert!(
            (result.mean_ssim - 1.0).abs() < 0.001,
            "SSIM of identical images should be ~1.0, got {}",
            result.mean_ssim
        );
    }

    #[test]
    fn test_ssim_different_images() {
        let data_a: Vec<u8> = vec![0; 32 * 32];
        let data_b: Vec<u8> = vec![255; 32 * 32];
        let a = make_tensor_u8(data_a, vec![32, 32]);
        let b = make_tensor_u8(data_b, vec![32, 32]);
        let result = compute_ssim(&a, &b);
        assert!(
            result.mean_ssim < 0.1,
            "SSIM of very different images should be low, got {}",
            result.mean_ssim
        );
    }

    #[test]
    fn test_ssim_similar_images() {
        let data_a: Vec<u8> = (0..32 * 32).map(|i| (i % 256) as u8).collect();
        let data_b: Vec<u8> = (0..32 * 32).map(|i| ((i + 1) % 256) as u8).collect();
        let a = make_tensor_u8(data_a, vec![32, 32]);
        let b = make_tensor_u8(data_b, vec![32, 32]);
        let result = compute_ssim(&a, &b);
        assert!(
            result.mean_ssim > 0.8,
            "SSIM of similar images should be high, got {}",
            result.mean_ssim
        );
    }

    #[test]
    fn test_ssim_multichannel() {
        let data: Vec<u8> = (0..32 * 32 * 3).map(|i| (i % 256) as u8).collect();
        let tensor = make_tensor_u8(data, vec![32, 32, 3]);
        let result = compute_ssim(&tensor, &tensor);
        assert_eq!(result.channel_ssim.len(), 3);
        assert!((result.mean_ssim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_ssim_shape_mismatch() {
        let a = make_tensor_u8(vec![0; 32 * 32], vec![32, 32]);
        let b = make_tensor_u8(vec![0; 16 * 16], vec![16, 16]);
        let result = compute_ssim(&a, &b);
        assert_eq!(result.mean_ssim, 0.0);
    }

    #[test]
    fn test_ssim_grayscale_convenience() {
        let img: Vec<u8> = (0..32 * 32).map(|i| (i % 256) as u8).collect();
        let ssim = compute_ssim_grayscale(&img, &img, 32, 32);
        assert!((ssim - 1.0).abs() < 0.001);
    }

    // --- Cosine similarity tests ---

    #[test]
    fn test_cosine_identical() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_empty() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn test_cosine_length_mismatch() {
        assert_eq!(cosine_similarity(&[1.0, 2.0], &[1.0]), 0.0);
    }

    #[test]
    fn test_tensor_cosine_similarity() {
        let a = make_tensor_f32(vec![1.0, 0.0, 0.0], vec![3]);
        let b = make_tensor_f32(vec![0.0, 1.0, 0.0], vec![3]);
        let sim = tensor_cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);

        let c = make_tensor_f32(vec![1.0, 0.0, 0.0], vec![3]);
        let sim2 = tensor_cosine_similarity(&a, &c);
        assert!((sim2 - 1.0).abs() < 1e-6);
    }

    // --- Physics evaluation tests ---

    #[test]
    fn test_object_permanence_all_present() {
        let obj_a = make_object("a", "ball", (0.0, 5.0, 0.0), (-1.0, 4.0, -1.0), (1.0, 6.0, 1.0), Some(1.0), false, (0.0, 0.0, 0.0));
        let obj_b = make_object("b", "cube", (3.0, 5.0, 0.0), (2.0, 4.0, -1.0), (4.0, 6.0, 1.0), Some(2.0), false, (0.0, 0.0, 0.0));

        let initial = make_world_state(vec![obj_a.clone(), obj_b.clone()]);
        let final_state = make_world_state(vec![obj_a, obj_b]);

        let result = evaluate_physics(&initial, &final_state);
        assert!(result.object_permanence >= 1.0);
    }

    #[test]
    fn test_object_permanence_missing() {
        let obj_a = make_object("a", "ball", (0.0, 5.0, 0.0), (-1.0, 4.0, -1.0), (1.0, 6.0, 1.0), Some(1.0), false, (0.0, 0.0, 0.0));
        let obj_b = make_object("b", "cube", (3.0, 5.0, 0.0), (2.0, 4.0, -1.0), (4.0, 6.0, 1.0), Some(2.0), false, (0.0, 0.0, 0.0));

        let initial = make_world_state(vec![obj_a.clone(), obj_b]);
        let final_state = make_world_state(vec![obj_a]);

        let result = evaluate_physics(&initial, &final_state);
        assert!(result.object_permanence < 1.0);
        assert!((result.object_permanence - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_gravity_compliance() {
        // Initial: unsupported object at y=5
        let obj = make_object("a", "ball", (0.0, 5.0, 0.0), (-1.0, 4.0, -1.0), (1.0, 6.0, 1.0), Some(1.0), false, (0.0, 0.0, 0.0));
        let initial = make_world_state(vec![obj]);

        // Final: object fell to y=2
        let obj_fallen = make_object("a", "ball", (0.0, 2.0, 0.0), (-1.0, 1.0, -1.0), (1.0, 3.0, 1.0), Some(1.0), false, (0.0, -3.0, 0.0));
        let final_state = make_world_state(vec![obj_fallen]);

        let result = evaluate_physics(&initial, &final_state);
        assert!(result.gravity >= 1.0, "Falling down should be gravity compliant");
    }

    #[test]
    fn test_gravity_violation() {
        // Object floats upward
        let obj = make_object("a", "ball", (0.0, 2.0, 0.0), (-1.0, 1.0, -1.0), (1.0, 3.0, 1.0), Some(1.0), false, (0.0, 0.0, 0.0));
        let initial = make_world_state(vec![obj]);

        let obj_up = make_object("a", "ball", (0.0, 10.0, 0.0), (-1.0, 9.0, -1.0), (1.0, 11.0, 1.0), Some(1.0), false, (0.0, 0.0, 0.0));
        let final_state = make_world_state(vec![obj_up]);

        let result = evaluate_physics(&initial, &final_state);
        assert!(result.gravity < 1.0, "Floating up should be gravity non-compliant");
    }

    #[test]
    fn test_collision_no_overlap() {
        let obj_a = make_object("a", "ball", (0.0, 0.0, 0.0), (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), Some(1.0), false, (0.0, 0.0, 0.0));
        let obj_b = make_object("b", "cube", (5.0, 0.0, 0.0), (4.0, -1.0, -1.0), (6.0, 1.0, 1.0), Some(2.0), false, (0.0, 0.0, 0.0));

        let state = make_world_state(vec![obj_a, obj_b]);
        let result = test_collision(&state);
        assert!(result.passed);
        assert!((result.score - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_collision_overlap() {
        let obj_a = make_object("a", "ball", (0.0, 0.0, 0.0), (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), Some(1.0), false, (0.0, 0.0, 0.0));
        let obj_b = make_object("b", "cube", (0.5, 0.0, 0.0), (-0.5, -1.0, -1.0), (1.5, 1.0, 1.0), Some(2.0), false, (0.0, 0.0, 0.0));

        let state = make_world_state(vec![obj_a, obj_b]);
        let result = test_collision(&state);
        assert!(!result.passed);
        assert!(result.score < 1.0);
    }

    #[test]
    fn test_energy_conservation_preserved() {
        let obj = make_object("a", "ball", (0.0, 0.0, 0.0), (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), Some(2.0), false, (1.0, 0.0, 0.0));
        let initial = make_world_state(vec![obj]);

        let obj_final = make_object("a", "ball", (1.0, 0.0, 0.0), (0.0, -1.0, -1.0), (2.0, 1.0, 1.0), Some(2.0), false, (1.0, 0.0, 0.0));
        let final_state = make_world_state(vec![obj_final]);

        let result = evaluate_physics(&initial, &final_state);
        assert!((result.conservation - 1.0).abs() < 0.01, "Same KE should give conservation score ~1.0");
    }

    #[test]
    fn test_energy_conservation_violated() {
        let obj = make_object("a", "ball", (0.0, 0.0, 0.0), (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), Some(2.0), false, (10.0, 0.0, 0.0));
        let initial = make_world_state(vec![obj]);

        // Final: velocity dropped to 0 (all KE lost)
        let obj_final = make_object("a", "ball", (1.0, 0.0, 0.0), (0.0, -1.0, -1.0), (2.0, 1.0, 1.0), Some(2.0), false, (0.0, 0.0, 0.0));
        let final_state = make_world_state(vec![obj_final]);

        let result = evaluate_physics(&initial, &final_state);
        assert!(result.conservation < 0.5, "Total KE loss should violate conservation, got {}", result.conservation);
    }

    // --- Cross-provider comparison tests ---

    #[test]
    fn test_cross_provider_ranking() {
        let outputs = vec![
            ProviderOutput {
                name: "provider_a".to_string(),
                physics: PhysicsEvalResult {
                    tests: vec![],
                    overall_score: 0.9,
                    object_permanence: 1.0,
                    gravity: 0.8,
                    collision: 1.0,
                    conservation: 0.8,
                },
                output_frame: None,
            },
            ProviderOutput {
                name: "provider_b".to_string(),
                physics: PhysicsEvalResult {
                    tests: vec![],
                    overall_score: 0.7,
                    object_permanence: 0.5,
                    gravity: 0.8,
                    collision: 0.7,
                    conservation: 0.8,
                },
                output_frame: None,
            },
        ];

        let report = cross_provider_comparison(&outputs);
        assert_eq!(report.rankings.len(), 2);
        assert_eq!(report.rankings[0].provider, "provider_a");
        assert_eq!(report.rankings[0].rank, 1);
        assert_eq!(report.rankings[1].provider, "provider_b");
        assert_eq!(report.rankings[1].rank, 2);
    }

    #[test]
    fn test_cross_provider_pairwise_ssim() {
        let frame_a = make_tensor_u8((0..32 * 32).map(|i| (i % 256) as u8).collect(), vec![32, 32]);
        let frame_b = frame_a.clone();

        let outputs = vec![
            ProviderOutput {
                name: "a".to_string(),
                physics: PhysicsEvalResult {
                    tests: vec![],
                    overall_score: 0.8,
                    object_permanence: 0.8,
                    gravity: 0.8,
                    collision: 0.8,
                    conservation: 0.8,
                },
                output_frame: Some(frame_a),
            },
            ProviderOutput {
                name: "b".to_string(),
                physics: PhysicsEvalResult {
                    tests: vec![],
                    overall_score: 0.8,
                    object_permanence: 0.8,
                    gravity: 0.8,
                    collision: 0.8,
                    conservation: 0.8,
                },
                output_frame: Some(frame_b),
            },
        ];

        let report = cross_provider_comparison(&outputs);
        assert_eq!(report.pairwise.len(), 1);
        let ssim = report.pairwise[0].ssim.unwrap();
        assert!((ssim - 1.0).abs() < 0.01, "Identical frames should have SSIM ~1.0, got {}", ssim);
    }

    #[test]
    fn test_report_generation() {
        let report = CrossProviderReport {
            pairwise: vec![PairwiseComparison {
                provider_a: "a".to_string(),
                provider_b: "b".to_string(),
                ssim: Some(0.95),
                physics_diff: 0.1,
            }],
            rankings: vec![
                ProviderRanking {
                    provider: "a".to_string(),
                    physics_score: 0.9,
                    rank: 1,
                },
                ProviderRanking {
                    provider: "b".to_string(),
                    physics_score: 0.8,
                    rank: 2,
                },
            ],
            summary: {
                let mut m = HashMap::new();
                m.insert("mean_physics_score".to_string(), 0.85);
                m
            },
        };

        let json = report_to_json(&report);
        assert!(json.contains("provider_a"));
        assert!(json.contains("0.95"));

        let md = report_to_markdown(&report);
        assert!(md.contains("# Cross-Provider Comparison Report"));
        assert!(md.contains("| 1 | a |"));
    }

    #[test]
    fn test_gaussian_kernel_sums_to_one() {
        let kernel = gaussian_kernel(SSIM_WINDOW_SIZE, SSIM_SIGMA);
        let sum: f64 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Gaussian kernel should sum to 1.0, got {}", sum);
    }
}
