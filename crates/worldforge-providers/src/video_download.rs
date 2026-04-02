//! Download and decode video/image content from provider URLs.
//!
//! Provides helpers to download media from provider result URLs and convert
//! them into WorldForge core types (`Frame`, `VideoClip`).

use worldforge_core::error::{Result, WorldForgeError};
use worldforge_core::types::{DType, Device, Frame, SimTime, Tensor, TensorData, VideoClip};

/// Download raw bytes from a URL using the given `reqwest::Client`.
///
/// Reports progress via `tracing` spans.
///
/// # Errors
///
/// Returns `WorldForgeError::ProviderUnavailable` on network or HTTP errors.
pub async fn download_bytes(
    client: &reqwest::Client,
    url: &str,
    provider_name: &str,
) -> Result<Vec<u8>> {
    tracing::info!(
        provider = provider_name,
        url = url,
        "starting media download"
    );

    let response = client.get(url).send().await.map_err(|e| {
        WorldForgeError::ProviderUnavailable {
            provider: provider_name.to_string(),
            reason: format!("download request failed: {e}"),
        }
    })?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(WorldForgeError::ProviderUnavailable {
            provider: provider_name.to_string(),
            reason: format!("download HTTP {status}: {body}"),
        });
    }

    let content_length = response.content_length();
    tracing::debug!(
        provider = provider_name,
        content_length = ?content_length,
        "downloading media content"
    );

    let bytes = response.bytes().await.map_err(|e| {
        WorldForgeError::ProviderUnavailable {
            provider: provider_name.to_string(),
            reason: format!("failed to read response body: {e}"),
        }
    })?;

    tracing::info!(
        provider = provider_name,
        size_bytes = bytes.len(),
        "media download complete"
    );

    Ok(bytes.to_vec())
}

/// Create a single `Frame` from raw RGB pixel data.
///
/// The `data` must contain exactly `width * height * 3` bytes (RGB8).
pub fn frame_from_rgb(
    data: Vec<u8>,
    width: u32,
    height: u32,
    timestamp_seconds: f64,
    frame_index: u64,
    fps: f32,
) -> Result<Frame> {
    let expected = (width * height * 3) as usize;
    if data.len() != expected {
        return Err(WorldForgeError::ProviderUnavailable {
            provider: "video_download".to_string(),
            reason: format!(
                "frame data size mismatch: expected {expected} bytes, got {}",
                data.len()
            ),
        });
    }

    Ok(Frame {
        data: Tensor {
            data: TensorData::UInt8(data),
            shape: vec![height as usize, width as usize, 3],
            dtype: DType::UInt8,
            device: Device::Cpu,
        },
        timestamp: SimTime {
            step: frame_index,
            seconds: timestamp_seconds,
            dt: 1.0 / fps as f64,
        },
        camera: None,
        depth: None,
        segmentation: None,
    })
}

/// Create a `VideoClip` from a list of raw RGB frames.
///
/// Each element in `frame_data` must be exactly `width * height * 3` bytes.
pub fn video_clip_from_rgb_frames(
    frame_data: Vec<Vec<u8>>,
    width: u32,
    height: u32,
    fps: f32,
) -> Result<VideoClip> {
    let num_frames = frame_data.len();
    let mut frames = Vec::with_capacity(num_frames);

    for (i, data) in frame_data.into_iter().enumerate() {
        let t = i as f64 / fps as f64;
        frames.push(frame_from_rgb(data, width, height, t, i as u64, fps)?);
    }

    let duration = if num_frames > 0 {
        num_frames as f64 / fps as f64
    } else {
        0.0
    };

    Ok(VideoClip {
        frames,
        fps,
        resolution: (width, height),
        duration,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_from_rgb_valid() {
        let data = vec![128u8; 4 * 4 * 3]; // 4x4 RGB
        let frame = frame_from_rgb(data, 4, 4, 0.0, 0, 30.0).unwrap();
        assert_eq!(frame.data.shape, vec![4, 4, 3]);
        assert_eq!(frame.timestamp.step, 0);
    }

    #[test]
    fn test_frame_from_rgb_size_mismatch() {
        let data = vec![128u8; 10]; // Wrong size
        let result = frame_from_rgb(data, 4, 4, 0.0, 0, 30.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_video_clip_from_frames() {
        let frame_size = 2 * 2 * 3;
        let frames = vec![vec![64u8; frame_size]; 5];
        let clip = video_clip_from_rgb_frames(frames, 2, 2, 10.0).unwrap();
        assert_eq!(clip.frames.len(), 5);
        assert_eq!(clip.resolution, (2, 2));
        assert_eq!(clip.fps, 10.0);
        assert!((clip.duration - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_video_clip_empty() {
        let clip = video_clip_from_rgb_frames(vec![], 640, 480, 24.0).unwrap();
        assert!(clip.frames.is_empty());
        assert_eq!(clip.duration, 0.0);
    }
}
