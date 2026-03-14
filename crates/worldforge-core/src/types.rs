//! Core type definitions for WorldForge.
//!
//! Includes tensor, spatial, temporal, and media types used throughout
//! the WorldForge ecosystem.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Tensor types
// ---------------------------------------------------------------------------

/// N-dimensional tensor for model inputs/outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    /// Raw tensor data.
    pub data: TensorData,
    /// Shape of the tensor (e.g. `[3, 224, 224]`).
    pub shape: Vec<usize>,
    /// Element data type.
    pub dtype: DType,
    /// Device where the tensor resides.
    pub device: Device,
}

/// Raw storage for tensor data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TensorData {
    /// 32-bit floating point values.
    Float32(Vec<f32>),
    /// 64-bit floating point values.
    Float64(Vec<f64>),
    /// 8-bit unsigned integer values.
    UInt8(Vec<u8>),
    /// 32-bit signed integer values.
    Int32(Vec<i32>),
    /// 64-bit signed integer values.
    Int64(Vec<i64>),
}

/// Tensor element data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    /// IEEE 754 half-precision (16-bit).
    Float16,
    /// IEEE 754 single-precision (32-bit).
    Float32,
    /// Brain floating point (16-bit).
    BFloat16,
    /// Unsigned 8-bit integer.
    UInt8,
    /// Signed 32-bit integer.
    Int32,
    /// Signed 64-bit integer.
    Int64,
}

/// Compute device for tensor operations.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Device {
    /// CPU execution.
    #[default]
    Cpu,
    /// NVIDIA CUDA GPU with device index.
    Cuda(u32),
    /// WebAssembly runtime.
    Wasm,
    /// Remote provider endpoint.
    Remote(String),
}

// ---------------------------------------------------------------------------
// Spatial types
// ---------------------------------------------------------------------------

/// 3D position in world coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Position {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// 3D vector.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Quaternion rotation (w, x, y, z).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rotation {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Combined position and rotation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Pose {
    pub position: Position,
    pub rotation: Rotation,
}

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BBox {
    pub min: Position,
    pub max: Position,
}

/// 3D triangle mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mesh {
    /// Vertex positions.
    pub vertices: Vec<Position>,
    /// Triangle face indices.
    pub faces: Vec<[u32; 3]>,
    /// Per-vertex normals.
    pub normals: Option<Vec<Position>>,
    /// Per-vertex UV coordinates.
    pub uvs: Option<Vec<[f32; 2]>>,
}

/// 3D velocity vector.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Velocity {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

// ---------------------------------------------------------------------------
// Temporal types
// ---------------------------------------------------------------------------

/// Simulation time combining discrete steps and continuous seconds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SimTime {
    /// Discrete step index.
    pub step: u64,
    /// Continuous time in seconds.
    pub seconds: f64,
    /// Time delta since last step.
    pub dt: f64,
}

/// A range of simulation time.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: SimTime,
    pub end: SimTime,
}

/// A sequence of poses over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    /// Timestamped poses.
    pub poses: Vec<(SimTime, Pose)>,
    /// Optional timestamped velocities.
    pub velocities: Option<Vec<(SimTime, Velocity)>>,
}

// ---------------------------------------------------------------------------
// Media types
// ---------------------------------------------------------------------------

/// A single image frame with optional depth and segmentation maps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frame {
    /// Image tensor with shape `[H, W, C]`.
    pub data: Tensor,
    /// Timestamp of the frame.
    pub timestamp: SimTime,
    /// Camera pose when the frame was captured.
    pub camera: Option<CameraPose>,
    /// Depth map with shape `[H, W]`.
    pub depth: Option<Tensor>,
    /// Semantic segmentation labels with shape `[H, W]`.
    pub segmentation: Option<Tensor>,
}

/// A sequence of frames forming a video.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoClip {
    /// Ordered sequence of frames.
    pub frames: Vec<Frame>,
    /// Frames per second.
    pub fps: f32,
    /// Resolution as `(width, height)`.
    pub resolution: (u32, u32),
    /// Duration in seconds.
    pub duration: f64,
}

/// Camera extrinsics and intrinsics.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CameraPose {
    /// Camera extrinsic pose (position + orientation).
    pub extrinsics: Pose,
    /// Field of view in degrees.
    pub fov: f32,
    /// Near clipping plane distance.
    pub near_clip: f32,
    /// Far clipping plane distance.
    pub far_clip: f32,
}

// ---------------------------------------------------------------------------
// Identifier types
// ---------------------------------------------------------------------------

/// Unique identifier for a world instance.
pub type WorldId = uuid::Uuid;

/// Unique identifier for a scene object.
pub type ObjectId = uuid::Uuid;

/// Unique identifier for a prediction.
pub type PredictionId = uuid::Uuid;

// ---------------------------------------------------------------------------
// Default impls
// ---------------------------------------------------------------------------

impl Default for SimTime {
    fn default() -> Self {
        Self {
            step: 0,
            seconds: 0.0,
            dt: 0.0,
        }
    }
}

impl Default for Position {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
}

impl Default for Vec3 {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
}

impl Default for Rotation {
    fn default() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
}

impl Tensor {
    /// Create a new tensor filled with zeros.
    pub fn zeros(shape: Vec<usize>, dtype: DType) -> Self {
        let size: usize = shape.iter().product();
        let data = match dtype {
            DType::Float32 | DType::Float16 | DType::BFloat16 => {
                TensorData::Float32(vec![0.0; size])
            }
            DType::UInt8 => TensorData::UInt8(vec![0; size]),
            DType::Int32 => TensorData::Int32(vec![0; size]),
            DType::Int64 => TensorData::Int64(vec![0; size]),
        };
        Self {
            data,
            shape,
            dtype,
            device: Device::Cpu,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_zeros() {
        let t = Tensor::zeros(vec![2, 3], DType::Float32);
        assert_eq!(t.shape, vec![2, 3]);
        match &t.data {
            TensorData::Float32(v) => assert_eq!(v.len(), 6),
            _ => panic!("expected Float32"),
        }
    }

    #[test]
    fn test_position_default() {
        let p = Position::default();
        assert_eq!(p.x, 0.0);
        assert_eq!(p.y, 0.0);
        assert_eq!(p.z, 0.0);
    }

    #[test]
    fn test_rotation_default_is_identity() {
        let r = Rotation::default();
        assert_eq!(r.w, 1.0);
        assert_eq!(r.x, 0.0);
    }

    #[test]
    fn test_simtime_default() {
        let t = SimTime::default();
        assert_eq!(t.step, 0);
        assert_eq!(t.seconds, 0.0);
    }

    #[test]
    fn test_serialization_roundtrip_position() {
        let p = Position {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        };
        let json = serde_json::to_string(&p).unwrap();
        let p2: Position = serde_json::from_str(&json).unwrap();
        assert_eq!(p, p2);
    }

    #[test]
    fn test_serialization_roundtrip_pose() {
        let pose = Pose::default();
        let json = serde_json::to_string(&pose).unwrap();
        let pose2: Pose = serde_json::from_str(&json).unwrap();
        assert_eq!(pose, pose2);
    }

    #[test]
    fn test_serialization_roundtrip_device() {
        let devices = vec![
            Device::Cpu,
            Device::Cuda(0),
            Device::Wasm,
            Device::Remote("http://localhost".to_string()),
        ];
        for d in devices {
            let json = serde_json::to_string(&d).unwrap();
            let d2: Device = serde_json::from_str(&json).unwrap();
            assert_eq!(d, d2);
        }
    }

    #[test]
    fn test_bbox_serialization() {
        let bbox = BBox {
            min: Position {
                x: -1.0,
                y: -1.0,
                z: -1.0,
            },
            max: Position {
                x: 1.0,
                y: 1.0,
                z: 1.0,
            },
        };
        let json = serde_json::to_string(&bbox).unwrap();
        let bbox2: BBox = serde_json::from_str(&json).unwrap();
        assert_eq!(bbox, bbox2);
    }
}
