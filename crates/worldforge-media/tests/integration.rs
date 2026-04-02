//! Integration tests for worldforge-media.

use std::collections::HashMap;
use worldforge_core::types::*;
use worldforge_media::{clip, codec, frame_io, tensor_io};

fn make_test_frame(w: u32, h: u32, val: u8) -> Frame {
    Frame {
        data: Tensor {
            data: TensorData::UInt8(vec![val; (w * h * 3) as usize]),
            shape: vec![h as usize, w as usize, 3],
            dtype: DType::UInt8,
            device: Device::Cpu,
        },
        timestamp: SimTime::default(),
        camera: None,
        depth: None,
        segmentation: None,
    }
}

// ---------------------------------------------------------------------------
// Frame I/O
// ---------------------------------------------------------------------------

#[test]
fn test_frame_encode_decode_png() {
    let frame = make_test_frame(8, 8, 42);
    let bytes = frame_io::encode_frame(&frame, frame_io::ImageOutputFormat::Png).unwrap();
    assert!(!bytes.is_empty());
    let decoded = frame_io::decode_frame(&bytes).unwrap();
    assert_eq!(decoded.data.shape, vec![8, 8, 3]);
}

#[test]
fn test_frame_encode_decode_jpeg() {
    let frame = make_test_frame(16, 16, 128);
    let bytes = frame_io::encode_frame(&frame, frame_io::ImageOutputFormat::Jpeg).unwrap();
    let decoded = frame_io::decode_frame(&bytes).unwrap();
    assert_eq!(decoded.data.shape, vec![16, 16, 3]);
}

#[test]
fn test_frame_encode_decode_webp() {
    let frame = make_test_frame(8, 8, 200);
    let bytes = frame_io::encode_frame(&frame, frame_io::ImageOutputFormat::WebP).unwrap();
    let decoded = frame_io::decode_frame(&bytes).unwrap();
    assert_eq!(decoded.data.shape, vec![8, 8, 3]);
}

#[test]
fn test_frame_save_load() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.png");
    let frame = make_test_frame(4, 4, 100);
    frame_io::save_frame(&frame, &path).unwrap();
    let loaded = frame_io::load_frame(&path).unwrap();
    assert_eq!(loaded.data.shape, vec![4, 4, 3]);
}

#[test]
fn test_dynamic_image_conversion() {
    let frame = make_test_frame(4, 4, 55);
    let img = frame_io::frame_to_dynamic_image(&frame).unwrap();
    let roundtrip = frame_io::dynamic_image_to_frame(&img);
    assert_eq!(roundtrip.data.shape, frame.data.shape);
}

// ---------------------------------------------------------------------------
// Tensor I/O — SafeTensors
// ---------------------------------------------------------------------------

#[test]
fn test_safetensors_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tensors.safetensors");

    let mut tensors = HashMap::new();
    tensors.insert(
        "weights".to_string(),
        Tensor {
            data: TensorData::Float32(vec![1.0, 2.0, 3.0, 4.0]),
            shape: vec![2, 2],
            dtype: DType::Float32,
            device: Device::Cpu,
        },
    );
    tensors.insert(
        "bias".to_string(),
        Tensor {
            data: TensorData::Float32(vec![0.1, 0.2]),
            shape: vec![2],
            dtype: DType::Float32,
            device: Device::Cpu,
        },
    );

    tensor_io::save_safetensors(&tensors, &path).unwrap();
    let loaded = tensor_io::load_safetensors(&path).unwrap();

    assert_eq!(loaded.len(), 2);
    assert!(loaded.contains_key("weights"));
    assert!(loaded.contains_key("bias"));
    assert_eq!(loaded["weights"].shape, vec![2, 2]);
}

// ---------------------------------------------------------------------------
// Tensor I/O — NumPy .npy
// ---------------------------------------------------------------------------

#[test]
fn test_npy_roundtrip_f32() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tensor.npy");

    let tensor = Tensor {
        data: TensorData::Float32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        shape: vec![2, 3],
        dtype: DType::Float32,
        device: Device::Cpu,
    };

    tensor_io::save_npy(&tensor, &path).unwrap();
    let loaded = tensor_io::load_npy(&path).unwrap();
    assert_eq!(loaded.shape, vec![2, 3]);
    match loaded.data {
        TensorData::Float32(v) => assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        _ => panic!("expected Float32"),
    }
}

#[test]
fn test_npy_roundtrip_uint8() {
    let tensor = Tensor {
        data: TensorData::UInt8(vec![0, 127, 255]),
        shape: vec![3],
        dtype: DType::UInt8,
        device: Device::Cpu,
    };
    let bytes = tensor_io::tensor_to_npy_bytes(&tensor).unwrap();
    let loaded = tensor_io::npy_bytes_to_tensor(&bytes).unwrap();
    assert_eq!(loaded.shape, vec![3]);
    match loaded.data {
        TensorData::UInt8(v) => assert_eq!(v, vec![0, 127, 255]),
        _ => panic!("expected UInt8"),
    }
}

// ---------------------------------------------------------------------------
// Tensor I/O — Raw bytes
// ---------------------------------------------------------------------------

#[test]
fn test_raw_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tensor.bin");

    let tensor = Tensor {
        data: TensorData::Int32(vec![10, 20, 30]),
        shape: vec![3],
        dtype: DType::Int32,
        device: Device::Cpu,
    };

    tensor_io::save_raw(&tensor, &path).unwrap();
    let loaded = tensor_io::load_raw(&path).unwrap();
    assert_eq!(loaded.shape, vec![3]);
    match loaded.data {
        TensorData::Int32(v) => assert_eq!(v, vec![10, 20, 30]),
        _ => panic!("expected Int32"),
    }
}

// ---------------------------------------------------------------------------
// Clip assembly
// ---------------------------------------------------------------------------

#[test]
fn test_clip_assemble_and_trim() {
    let frames: Vec<Frame> = (0..10).map(|i| make_test_frame(4, 4, i)).collect();
    let full = clip::assemble_clip(frames, 10.0).unwrap();
    assert_eq!(full.frames.len(), 10);
    assert_eq!(full.resolution, (4, 4));

    let trimmed = clip::trim(&full, 2, 5).unwrap();
    assert_eq!(trimmed.frames.len(), 3);
}

#[test]
fn test_clip_concatenate() {
    let a = clip::assemble_clip(
        vec![make_test_frame(4, 4, 0), make_test_frame(4, 4, 1)],
        10.0,
    )
    .unwrap();
    let b = clip::assemble_clip(
        vec![make_test_frame(4, 4, 2), make_test_frame(4, 4, 3)],
        10.0,
    )
    .unwrap();
    let c = clip::concatenate(&a, &b).unwrap();
    assert_eq!(c.frames.len(), 4);
}

#[test]
fn test_clip_resample() {
    let frames: Vec<Frame> = (0..10).map(|i| make_test_frame(4, 4, i)).collect();
    let clip = clip::assemble_clip(frames, 10.0).unwrap();

    let upsampled = clip::resample(&clip, 20.0).unwrap();
    assert_eq!(upsampled.frames.len(), 20);

    let downsampled = clip::resample(&clip, 5.0).unwrap();
    assert_eq!(downsampled.frames.len(), 5);
}

// ---------------------------------------------------------------------------
// Codec — Image sequence
// ---------------------------------------------------------------------------

#[test]
fn test_image_sequence_codec() {
    use codec::{FrameDecoder, FrameEncoder, ImageSequenceDecoder, ImageSequenceEncoder};

    let dir = tempfile::tempdir().unwrap();
    let frames = vec![
        make_test_frame(4, 4, 10),
        make_test_frame(4, 4, 20),
        make_test_frame(4, 4, 30),
    ];
    let clip = clip::assemble_clip(frames, 24.0).unwrap();

    let encoder = ImageSequenceEncoder::new(dir.path(), frame_io::ImageOutputFormat::Png);
    encoder.encode(&clip).unwrap();

    let decoder = ImageSequenceDecoder::new(dir.path());
    let loaded = decoder.decode(24.0).unwrap();
    assert_eq!(loaded.frames.len(), 3);
    assert_eq!(loaded.fps, 24.0);
}
