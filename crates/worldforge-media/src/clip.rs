//! Video clip assembly and temporal operations.

use worldforge_core::types::{Frame, SimTime, VideoClip};

use crate::error::{MediaError, Result};

/// Build a [`VideoClip`] from a sequence of frames.
///
/// Frames are assigned timestamps based on the provided `fps`.
/// Resolution is derived from the first frame.
pub fn assemble_clip(frames: Vec<Frame>, fps: f32) -> Result<VideoClip> {
    if frames.is_empty() {
        return Err(MediaError::ClipError("cannot assemble empty clip".to_string()));
    }
    if fps <= 0.0 {
        return Err(MediaError::ClipError("fps must be positive".to_string()));
    }

    let shape = &frames[0].data.shape;
    if shape.len() < 2 {
        return Err(MediaError::InvalidFrame(
            "frame shape must have at least H and W".to_string(),
        ));
    }
    let height = shape[0] as u32;
    let width = shape[1] as u32;
    let duration = frames.len() as f64 / fps as f64;

    // Assign timestamps
    let mut timestamped_frames = frames;
    for (i, frame) in timestamped_frames.iter_mut().enumerate() {
        frame.timestamp = SimTime {
            step: i as u64,
            seconds: i as f64 / fps as f64,
            dt: 1.0 / fps as f64,
        };
    }

    Ok(VideoClip {
        frames: timestamped_frames,
        fps,
        resolution: (width, height),
        duration,
    })
}

/// Extract individual frames from a clip.
pub fn extract_frames(clip: &VideoClip) -> Vec<&Frame> {
    clip.frames.iter().collect()
}

/// Get a single frame by index.
pub fn get_frame(clip: &VideoClip, index: usize) -> Result<&Frame> {
    clip.frames
        .get(index)
        .ok_or_else(|| MediaError::ClipError(format!("frame index {index} out of bounds")))
}

/// Trim a clip to the specified frame range `[start, end)`.
pub fn trim(clip: &VideoClip, start: usize, end: usize) -> Result<VideoClip> {
    if start >= end {
        return Err(MediaError::ClipError(
            "start must be less than end".to_string(),
        ));
    }
    if end > clip.frames.len() {
        return Err(MediaError::ClipError(format!(
            "end index {end} exceeds frame count {}",
            clip.frames.len()
        )));
    }

    let frames: Vec<Frame> = clip.frames[start..end].to_vec();
    let duration = frames.len() as f64 / clip.fps as f64;

    Ok(VideoClip {
        frames,
        fps: clip.fps,
        resolution: clip.resolution,
        duration,
    })
}

/// Concatenate two clips. They must share the same resolution and fps.
pub fn concatenate(a: &VideoClip, b: &VideoClip) -> Result<VideoClip> {
    if a.resolution != b.resolution {
        return Err(MediaError::ClipError(format!(
            "resolution mismatch: {:?} vs {:?}",
            a.resolution, b.resolution
        )));
    }
    if (a.fps - b.fps).abs() > 0.001 {
        return Err(MediaError::ClipError(format!(
            "fps mismatch: {} vs {}",
            a.fps, b.fps
        )));
    }

    let mut frames = a.frames.clone();
    frames.extend(b.frames.clone());
    let duration = frames.len() as f64 / a.fps as f64;

    Ok(VideoClip {
        frames,
        fps: a.fps,
        resolution: a.resolution,
        duration,
    })
}

/// Resample a clip to a new frame rate using nearest-neighbor frame selection.
///
/// If `target_fps` is lower than the source, frames are dropped.
/// If higher, frames are duplicated.
pub fn resample(clip: &VideoClip, target_fps: f32) -> Result<VideoClip> {
    if target_fps <= 0.0 {
        return Err(MediaError::ClipError("target_fps must be positive".to_string()));
    }
    if clip.frames.is_empty() {
        return Err(MediaError::ClipError("cannot resample empty clip".to_string()));
    }

    let source_duration = clip.duration;
    let target_frame_count = (source_duration * target_fps as f64).round() as usize;
    if target_frame_count == 0 {
        return Err(MediaError::ClipError("resampled clip would have 0 frames".to_string()));
    }

    let source_len = clip.frames.len();
    let mut frames = Vec::with_capacity(target_frame_count);

    for i in 0..target_frame_count {
        let t = i as f64 / target_fps as f64;
        let source_idx = ((t * clip.fps as f64).round() as usize).min(source_len - 1);
        let mut frame = clip.frames[source_idx].clone();
        frame.timestamp = SimTime {
            step: i as u64,
            seconds: t,
            dt: 1.0 / target_fps as f64,
        };
        frames.push(frame);
    }

    Ok(VideoClip {
        frames,
        fps: target_fps,
        resolution: clip.resolution,
        duration: source_duration,
    })
}

/// Return the number of frames in a clip.
pub fn frame_count(clip: &VideoClip) -> usize {
    clip.frames.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use worldforge_core::types::{DType, Device, Tensor, TensorData};

    fn make_frame(idx: u8) -> Frame {
        Frame {
            data: Tensor {
                data: TensorData::UInt8(vec![idx; 12]),
                shape: vec![2, 2, 3],
                dtype: DType::UInt8,
                device: Device::Cpu,
            },
            timestamp: SimTime::default(),
            camera: None,
            depth: None,
            segmentation: None,
        }
    }

    #[test]
    fn test_assemble_and_extract() {
        let frames = vec![make_frame(0), make_frame(1), make_frame(2)];
        let clip = assemble_clip(frames, 30.0).unwrap();
        assert_eq!(clip.frames.len(), 3);
        assert_eq!(clip.resolution, (2, 2));
        assert!((clip.duration - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_trim() {
        let frames = (0..10).map(make_frame).collect();
        let clip = assemble_clip(frames, 10.0).unwrap();
        let trimmed = trim(&clip, 2, 5).unwrap();
        assert_eq!(trimmed.frames.len(), 3);
    }

    #[test]
    fn test_concatenate() {
        let a = assemble_clip(vec![make_frame(0), make_frame(1)], 10.0).unwrap();
        let b = assemble_clip(vec![make_frame(2), make_frame(3)], 10.0).unwrap();
        let c = concatenate(&a, &b).unwrap();
        assert_eq!(c.frames.len(), 4);
    }

    #[test]
    fn test_resample_downsample() {
        let frames = (0..10).map(make_frame).collect();
        let clip = assemble_clip(frames, 10.0).unwrap();
        let resampled = resample(&clip, 5.0).unwrap();
        assert_eq!(resampled.frames.len(), 5);
    }
}
