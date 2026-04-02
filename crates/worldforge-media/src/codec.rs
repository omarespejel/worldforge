//! Codec traits and image-sequence backend.
//!
//! Defines [`FrameDecoder`] and [`FrameEncoder`] traits with an
//! `ImageSequenceDecoder`/`ImageSequenceEncoder` implementation that works
//! with directories of image files.

use std::path::{Path, PathBuf};

use worldforge_core::types::{Frame, VideoClip};

use crate::clip;
use crate::error::{MediaError, Result};
use crate::frame_io;

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// Decodes frames from a media source.
pub trait FrameDecoder {
    /// Decode all frames from the source and return them as a clip.
    fn decode(&self, fps: f32) -> Result<VideoClip>;
}

/// Encodes frames to a media destination.
pub trait FrameEncoder {
    /// Encode a clip to the destination.
    fn encode(&self, clip: &VideoClip) -> Result<()>;
}

// ---------------------------------------------------------------------------
// Image sequence backend
// ---------------------------------------------------------------------------

/// Decodes frames from a directory of image files.
///
/// Files are sorted lexicographically. Supported formats: PNG, JPEG, WebP.
pub struct ImageSequenceDecoder {
    /// Directory containing the image files.
    pub dir: PathBuf,
    /// Optional glob pattern (e.g., `"*.png"`). Defaults to all supported extensions.
    pub pattern: Option<String>,
}

impl ImageSequenceDecoder {
    /// Create a new decoder reading from the given directory.
    pub fn new<P: AsRef<Path>>(dir: P) -> Self {
        Self {
            dir: dir.as_ref().to_path_buf(),
            pattern: None,
        }
    }

    /// Set a file extension filter (e.g., `"png"`).
    pub fn with_extension(mut self, ext: &str) -> Self {
        self.pattern = Some(ext.to_string());
        self
    }

    fn collect_files(&self) -> Result<Vec<PathBuf>> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(&self.dir)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    let ext_lower = ext.to_lowercase();
                    if let Some(ref pattern) = self.pattern {
                        ext_lower == pattern.to_lowercase()
                    } else {
                        matches!(ext_lower.as_str(), "png" | "jpg" | "jpeg" | "webp")
                    }
                } else {
                    false
                }
            })
            .collect();
        files.sort();
        Ok(files)
    }
}

impl FrameDecoder for ImageSequenceDecoder {
    fn decode(&self, fps: f32) -> Result<VideoClip> {
        let files = self.collect_files()?;
        if files.is_empty() {
            return Err(MediaError::ClipError(
                "no image files found in directory".to_string(),
            ));
        }

        let frames: Result<Vec<Frame>> = files.iter().map(frame_io::load_frame).collect();
        clip::assemble_clip(frames?, fps)
    }
}

/// Encodes frames to a directory as numbered image files.
pub struct ImageSequenceEncoder {
    /// Output directory.
    pub dir: PathBuf,
    /// Output format.
    pub format: frame_io::ImageOutputFormat,
    /// File name prefix (default: `"frame"`).
    pub prefix: String,
}

impl ImageSequenceEncoder {
    /// Create a new encoder writing to the given directory.
    pub fn new<P: AsRef<Path>>(dir: P, format: frame_io::ImageOutputFormat) -> Self {
        Self {
            dir: dir.as_ref().to_path_buf(),
            format,
            prefix: "frame".to_string(),
        }
    }

    /// Set the file name prefix.
    pub fn with_prefix(mut self, prefix: &str) -> Self {
        self.prefix = prefix.to_string();
        self
    }

    fn extension(&self) -> &'static str {
        match self.format {
            frame_io::ImageOutputFormat::Png => "png",
            frame_io::ImageOutputFormat::Jpeg => "jpg",
            frame_io::ImageOutputFormat::WebP => "webp",
        }
    }
}

impl FrameEncoder for ImageSequenceEncoder {
    fn encode(&self, clip: &VideoClip) -> Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        let ext = self.extension();
        let pad_width = (clip.frames.len() as f64).log10().ceil() as usize;
        let pad_width = pad_width.max(4);

        for (i, frame) in clip.frames.iter().enumerate() {
            let filename = format!("{}_{:0>width$}.{}", self.prefix, i, ext, width = pad_width);
            let path = self.dir.join(filename);
            frame_io::save_frame(frame, &path)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FFmpeg stub (feature-gated)
// ---------------------------------------------------------------------------

/// Stub FFmpeg decoder — available only with the `ffmpeg` feature.
#[cfg(feature = "ffmpeg")]
pub struct FfmpegDecoder {
    pub path: PathBuf,
}

#[cfg(feature = "ffmpeg")]
impl FrameDecoder for FfmpegDecoder {
    fn decode(&self, _fps: f32) -> Result<VideoClip> {
        Err(MediaError::CodecUnavailable(
            "ffmpeg support is not yet implemented".to_string(),
        ))
    }
}

/// Stub FFmpeg encoder — available only with the `ffmpeg` feature.
#[cfg(feature = "ffmpeg")]
pub struct FfmpegEncoder {
    pub path: PathBuf,
    pub codec: String,
}

#[cfg(feature = "ffmpeg")]
impl FrameEncoder for FfmpegEncoder {
    fn encode(&self, _clip: &VideoClip) -> Result<()> {
        Err(MediaError::CodecUnavailable(
            "ffmpeg support is not yet implemented".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use worldforge_core::types::{DType, Device, SimTime, Tensor, TensorData};

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
    fn test_image_sequence_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let frames = vec![make_frame(100), make_frame(200)];
        let clip = clip::assemble_clip(frames, 10.0).unwrap();

        let encoder =
            ImageSequenceEncoder::new(dir.path(), frame_io::ImageOutputFormat::Png);
        encoder.encode(&clip).unwrap();

        let decoder = ImageSequenceDecoder::new(dir.path());
        let loaded = decoder.decode(10.0).unwrap();
        assert_eq!(loaded.frames.len(), 2);
    }
}
