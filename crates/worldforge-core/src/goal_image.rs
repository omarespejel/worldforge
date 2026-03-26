//! Goal-image helpers for image-conditioned planning.
//!
//! This module provides a lightweight scene renderer and comparison helpers so
//! planning code can treat image goals as spatial targets without depending on
//! provider-specific image tooling.

use crate::state::WorldState;
use crate::types::{DType, Device, Position, Tensor, TensorData};

/// A derived target extracted from a goal image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GoalImageTarget {
    /// Suggested world-space position for the primary movable object.
    pub position: Position,
    /// Confidence in the target estimate, normalized to 0.0..=1.0.
    pub confidence: f32,
}

/// Render a coarse top-down grayscale signature of the current scene.
///
/// The output tensor uses the requested resolution and a single float channel,
/// making it suitable for deterministic comparison against goal images.
pub fn render_scene_goal_image(state: &WorldState, resolution: (u32, u32)) -> Tensor {
    let width = resolution.0.max(1);
    let height = resolution.1.max(1);
    let mut pixels = vec![0.05f32; (width * height) as usize];

    let bounds = scene_bounds(state);
    let objects = sorted_objects(state);
    for (index, object) in objects.into_iter().enumerate() {
        let intensity = if object.physics.is_static { 0.45 } else { 0.92 };
        let rect = bbox_to_rect(object, bounds, width, height);
        fill_rect(
            &mut pixels,
            width,
            rect,
            intensity * object_intensity_modifier(index),
        );
    }

    Tensor {
        data: TensorData::Float32(pixels),
        shape: vec![height as usize, width as usize],
        dtype: DType::Float32,
        device: Device::Cpu,
    }
}

/// Estimate where the primary object in the scene should move based on a goal image.
///
/// The image is interpreted as a coarse heat map. Bright regions map to the
/// world-space area that should contain the primary movable object.
pub fn goal_image_target(goal_image: &Tensor, state: &WorldState) -> Option<GoalImageTarget> {
    let (width, height, values) = tensor_grayscale(goal_image)?;
    let peak = values
        .iter()
        .copied()
        .fold(0.0f32, |best, value| best.max(value));
    let activation_threshold = (peak * 0.75).clamp(0.35, 0.85);
    let mut weight_sum = 0.0f32;
    let mut x_sum = 0.0f32;
    let mut y_sum = 0.0f32;
    let mut active_pixels = 0usize;

    for y in 0..height {
        for x in 0..width {
            let value = values[(y * width + x) as usize];
            let weight = (value - activation_threshold).max(0.0);
            if weight > 0.0 {
                active_pixels += 1;
                weight_sum += weight;
                x_sum += weight * (x as f32 + 0.5);
                y_sum += weight * (y as f32 + 0.5);
            }
        }
    }

    let (centroid_x, centroid_y, confidence) = if weight_sum > 0.0 {
        let active_ratio = active_pixels as f32 / (width as f32 * height as f32);
        (
            x_sum / weight_sum,
            y_sum / weight_sum,
            ((peak * 0.65) + (active_ratio * 0.35)).clamp(0.0, 1.0),
        )
    } else {
        (width as f32 * 0.5, height as f32 * 0.5, 0.0)
    };

    let bounds = scene_bounds(state);
    let world_x = lerp(bounds.min_x, bounds.max_x, centroid_x / width as f32);
    let world_z = lerp(bounds.max_z, bounds.min_z, centroid_y / height as f32);
    let world_y = primary_object_height(state);

    Some(GoalImageTarget {
        position: Position {
            x: world_x,
            y: world_y,
            z: world_z,
        },
        confidence,
    })
}

/// Compute a similarity score between a goal image and the current scene.
///
/// The scene is rendered using [`render_scene_goal_image`] at the goal image's
/// resolution. Returns `None` if the image tensor cannot be interpreted as a
/// 2D or 3D image-like tensor.
pub fn goal_image_similarity(goal_image: &Tensor, state: &WorldState) -> Option<f32> {
    let (width, height, goal_values) = tensor_grayscale(goal_image)?;
    let rendered = render_scene_goal_image(state, (width, height));
    let (_, _, rendered_values) = tensor_grayscale(&rendered)?;

    let mut total_difference = 0.0f32;
    let mut count = 0usize;
    for (left, right) in goal_values.iter().zip(rendered_values.iter()) {
        total_difference += (left - right).abs();
        count += 1;
    }

    if count == 0 {
        None
    } else {
        Some((1.0 - total_difference / count as f32).clamp(0.0, 1.0))
    }
}

fn tensor_grayscale(tensor: &Tensor) -> Option<(u32, u32, Vec<f32>)> {
    let (width, height, channels) = image_dimensions(tensor)?;
    let values = tensor_values(tensor)?;
    let mut grayscale = vec![0.0f32; (width * height) as usize];

    match channels {
        1 => {
            for (index, value) in values.into_iter().enumerate() {
                grayscale[index] = value;
            }
        }
        _ => {
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let base = (y * width as usize + x) * channels;
                    let pixel = &values[base..base + channels];
                    let sum = pixel.iter().copied().sum::<f32>();
                    grayscale[y * width as usize + x] = sum / channels as f32;
                }
            }
        }
    }

    Some((width, height, grayscale))
}

fn tensor_values(tensor: &Tensor) -> Option<Vec<f32>> {
    let values = tensor.data.to_f32_values();
    Some(match tensor.data.storage_kind() {
        crate::types::TensorStorageKind::Float16
        | crate::types::TensorStorageKind::Float32
        | crate::types::TensorStorageKind::Float64
        | crate::types::TensorStorageKind::BFloat16 => {
            let scale = values
                .iter()
                .copied()
                .fold(1.0f32, |acc, value| acc.max(value.abs()).max(1.0));
            values
                .into_iter()
                .map(|value| (value / scale).clamp(0.0, 1.0))
                .collect()
        }
        crate::types::TensorStorageKind::UInt8 => {
            values.into_iter().map(|value| value / 255.0).collect()
        }
        crate::types::TensorStorageKind::Int32 | crate::types::TensorStorageKind::Int64 => {
            let scale = values
                .iter()
                .copied()
                .map(|value| value.abs().max(1.0))
                .fold(1.0f32, f32::max);
            values
                .into_iter()
                .map(|value| (value / scale).clamp(0.0, 1.0))
                .collect()
        }
    })
}

fn image_dimensions(tensor: &Tensor) -> Option<(u32, u32, usize)> {
    let len = tensor_element_count(tensor);
    match tensor.shape.as_slice() {
        [height, width] => {
            let width_u32 = u32::try_from(*width).ok()?;
            let height_u32 = u32::try_from(*height).ok()?;
            if len != (*width * *height) {
                return None;
            }
            Some((width_u32, height_u32, 1))
        }
        [height, width, channels] => {
            let width_u32 = u32::try_from(*width).ok()?;
            let height_u32 = u32::try_from(*height).ok()?;
            let channels = *channels;
            if channels == 0 || len != (*width * *height * channels) {
                return None;
            }
            Some((width_u32, height_u32, channels))
        }
        _ => None,
    }
}

fn tensor_element_count(tensor: &Tensor) -> usize {
    tensor.element_count()
}

fn scene_bounds(state: &WorldState) -> SceneBounds {
    let mut bounds = SceneBounds::default();
    let mut saw_object = false;
    for object in sorted_objects(state) {
        saw_object = true;
        bounds.min_x = bounds.min_x.min(object.bbox.min.x);
        bounds.max_x = bounds.max_x.max(object.bbox.max.x);
        bounds.min_z = bounds.min_z.min(object.bbox.min.z);
        bounds.max_z = bounds.max_z.max(object.bbox.max.z);
    }

    if !saw_object {
        bounds.min_x = -1.0;
        bounds.max_x = 1.0;
        bounds.min_z = -1.0;
        bounds.max_z = 1.0;
    }

    let extent_x = bounds.min_x.abs().max(bounds.max_x.abs()).max(1.5) + 0.5;
    let extent_z = bounds.min_z.abs().max(bounds.max_z.abs()).max(1.5) + 0.5;
    bounds.min_x = -extent_x;
    bounds.max_x = extent_x;
    bounds.min_z = -extent_z;
    bounds.max_z = extent_z;

    bounds
}

fn primary_object_height(state: &WorldState) -> f32 {
    sorted_objects(state)
        .into_iter()
        .find(|object| !object.physics.is_static)
        .or_else(|| sorted_objects(state).into_iter().next())
        .map(|object| object.pose.position.y)
        .unwrap_or(0.0)
}

fn sorted_objects(state: &WorldState) -> Vec<&crate::scene::SceneObject> {
    let mut objects: Vec<_> = state.scene.objects.values().collect();
    objects.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then_with(|| left.id.as_bytes().cmp(right.id.as_bytes()))
    });
    objects
}

fn bbox_to_rect(
    object: &crate::scene::SceneObject,
    bounds: SceneBounds,
    width: u32,
    height: u32,
) -> GoalImageRect {
    let x0 = project(object.bbox.min.x, bounds.min_x, bounds.max_x, width);
    let x1 = project(object.bbox.max.x, bounds.min_x, bounds.max_x, width);
    let z0 = project(object.bbox.min.z, bounds.min_z, bounds.max_z, height);
    let z1 = project(object.bbox.max.z, bounds.min_z, bounds.max_z, height);

    GoalImageRect {
        x: x0.min(x1),
        y: height
            .saturating_sub(z0.max(z1).saturating_add(1))
            .min(height.saturating_sub(1)),
        width: x1.saturating_sub(x0).max(1),
        height: z1.saturating_sub(z0).max(1),
    }
}

fn project(value: f32, min: f32, max: f32, pixels: u32) -> u32 {
    if pixels <= 1 {
        return 0;
    }

    let span = (max - min).max(f32::EPSILON);
    let ratio = ((value - min) / span).clamp(0.0, 1.0);
    (ratio * (pixels as f32 - 1.0)).round() as u32
}

fn fill_rect(pixels: &mut [f32], width: u32, rect: GoalImageRect, intensity: f32) {
    let width = width as usize;
    let x_end = rect.x.saturating_add(rect.width).min(width as u32);
    let y_end = rect
        .y
        .saturating_add(rect.height)
        .min((pixels.len() / width).max(1) as u32);

    for y in rect.y..y_end {
        let row = y as usize * width;
        for x in rect.x..x_end {
            pixels[row + x as usize] = pixels[row + x as usize].max(intensity.clamp(0.0, 1.0));
        }
    }
}

fn object_intensity_modifier(index: usize) -> f32 {
    let base = 1.0 - (index as f32 * 0.03);
    base.clamp(0.75, 1.0)
}

fn lerp(min: f32, max: f32, t: f32) -> f32 {
    min + (max - min) * t.clamp(0.0, 1.0)
}

#[derive(Debug, Clone, Copy)]
struct SceneBounds {
    min_x: f32,
    max_x: f32,
    min_z: f32,
    max_z: f32,
}

impl Default for SceneBounds {
    fn default() -> Self {
        Self {
            min_x: f32::INFINITY,
            max_x: f32::NEG_INFINITY,
            min_z: f32::INFINITY,
            max_z: f32::NEG_INFINITY,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct GoalImageRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::scene::SceneObject;
    use crate::types::{BBox, DType, Device, Pose, SimTime, Tensor, TensorData};

    fn sample_state() -> WorldState {
        let mut state = WorldState::new("goal-image", "mock");
        let object = SceneObject::new(
            "cube",
            Pose {
                position: Position {
                    x: -0.6,
                    y: 0.75,
                    z: -0.6,
                },
                ..Pose::default()
            },
            BBox {
                min: Position {
                    x: -0.7,
                    y: 0.65,
                    z: -0.7,
                },
                max: Position {
                    x: -0.5,
                    y: 0.85,
                    z: -0.5,
                },
            },
        );
        state.scene.add_object(object);
        state.time = SimTime {
            step: 1,
            seconds: 0.5,
            dt: 0.5,
        };
        state
    }

    #[test]
    fn test_render_scene_goal_image_builds_grayscale_tensor() {
        let tensor = render_scene_goal_image(&sample_state(), (16, 12));
        assert_eq!(tensor.shape, vec![12, 16]);
        match tensor.data {
            TensorData::Float32(values) => assert_eq!(values.len(), 192),
            _ => panic!("expected float32 tensor"),
        }
    }

    #[test]
    fn test_goal_image_tensor_values_accept_half_precision_storage() {
        let tensor = Tensor {
            data: TensorData::Float16(vec![0x3c00, 0xbc00, 0x0000, 0x3800]),
            shape: vec![2, 2],
            dtype: DType::Float16,
            device: Device::Cpu,
        };

        let values = tensor_values(&tensor).expect("tensor values");
        assert_eq!(values.len(), 4);
        assert!(values.iter().all(|value| (0.0..=1.0).contains(value)));
    }

    #[test]
    fn test_goal_image_tensor_values_accept_bfloat16_storage() {
        let tensor = Tensor {
            data: TensorData::BFloat16(vec![0x3f80, 0xbf80, 0x0000, 0x3f00]),
            shape: vec![2, 2],
            dtype: DType::BFloat16,
            device: Device::Cpu,
        };

        let values = tensor_values(&tensor).expect("tensor values");
        assert_eq!(values.len(), 4);
        assert!(values.iter().all(|value| (0.0..=1.0).contains(value)));
    }

    #[test]
    fn test_goal_image_target_uses_bright_centroid() {
        let state = sample_state();
        let current_position = state.scene.objects.values().next().unwrap().pose.position;
        let goal = Tensor {
            data: TensorData::UInt8({
                let mut values = vec![0u8; 16 * 12];
                for y in 2..5 {
                    for x in 11..14 {
                        values[y * 16 + x] = 255;
                    }
                }
                values
            }),
            shape: vec![12, 16],
            dtype: DType::UInt8,
            device: Device::Cpu,
        };

        let target = goal_image_target(&goal, &state).expect("target");
        assert!(target.position.x > current_position.x);
        assert!(target.confidence > 0.0);
    }

    #[test]
    fn test_goal_image_similarity_increases_when_object_moves_toward_target() {
        let mut state = sample_state();
        let mut target_state = state.clone();
        target_state
            .scene
            .objects
            .values_mut()
            .next()
            .unwrap()
            .set_position(Position {
                x: 0.6,
                y: 0.75,
                z: 0.6,
            });
        let goal = render_scene_goal_image(&target_state, (16, 12));

        let before = goal_image_similarity(&goal, &state).unwrap();
        if let Some(object) = state.scene.objects.values_mut().next() {
            object.set_position(Position {
                x: 0.6,
                y: object.pose.position.y,
                z: 0.6,
            });
        }
        let after = goal_image_similarity(&goal, &state).unwrap();

        assert!(after > before);
    }
}
