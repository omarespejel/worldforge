//! Frame encode/decode for image formats (PNG, JPEG, WebP).
//!
//! Converts between [`worldforge_core::types::Frame`] and standard image
//! formats using the `image` crate.

use std::io::Cursor;
use std::path::Path;

use image::{DynamicImage, ImageFormat, RgbImage, RgbaImage};
use worldforge_core::types::{
    DType, Device, Frame, SimTime, Tensor, TensorData,
};

use crate::error::{MediaError, Result};

/// Supported image output formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageOutputFormat {
    Png,
    Jpeg,
    WebP,
}

impl ImageOutputFormat {
    fn to_image_format(self) -> ImageFormat {
        match self {
            Self::Png => ImageFormat::Png,
            Self::Jpeg => ImageFormat::Jpeg,
            Self::WebP => ImageFormat::WebP,
        }
    }

    /// Infer the output format from a file extension.
    pub fn from_extension(ext: &str) -> Result<Self> {
        match ext.to_lowercase().as_str() {
            "png" => Ok(Self::Png),
            "jpg" | "jpeg" => Ok(Self::Jpeg),
            "webp" => Ok(Self::WebP),
            other => Err(MediaError::UnsupportedFormat(other.to_string())),
        }
    }
}

/// Convert a [`Frame`] into an [`image::DynamicImage`].
///
/// The frame's data tensor must be UInt8 with shape `[H, W, C]` where C is 3 (RGB)
/// or 4 (RGBA).
pub fn frame_to_dynamic_image(frame: &Frame) -> Result<DynamicImage> {
    let shape = &frame.data.shape;
    if shape.len() != 3 {
        return Err(MediaError::InvalidFrame(format!(
            "expected 3D shape [H, W, C], got {:?}",
            shape
        )));
    }
    let (height, width, channels) = (shape[0] as u32, shape[1] as u32, shape[2]);

    let pixels = match &frame.data.data {
        TensorData::UInt8(data) => data.clone(),
        _ => {
            return Err(MediaError::InvalidFrame(
                "frame data must be UInt8".to_string(),
            ));
        }
    };

    match channels {
        3 => {
            let img = RgbImage::from_raw(width, height, pixels).ok_or_else(|| {
                MediaError::InvalidFrame("pixel data size mismatch for RGB".to_string())
            })?;
            Ok(DynamicImage::ImageRgb8(img))
        }
        4 => {
            let img = RgbaImage::from_raw(width, height, pixels).ok_or_else(|| {
                MediaError::InvalidFrame("pixel data size mismatch for RGBA".to_string())
            })?;
            Ok(DynamicImage::ImageRgba8(img))
        }
        _ => Err(MediaError::InvalidFrame(format!(
            "unsupported channel count: {channels}"
        ))),
    }
}

/// Convert an [`image::DynamicImage`] into a [`Frame`] with default timestamp.
pub fn dynamic_image_to_frame(img: &DynamicImage) -> Frame {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let pixels = rgb.into_raw();

    Frame {
        data: Tensor {
            data: TensorData::UInt8(pixels),
            shape: vec![height as usize, width as usize, 3],
            dtype: DType::UInt8,
            device: Device::Cpu,
        },
        timestamp: SimTime::default(),
        camera: None,
        depth: None,
        segmentation: None,
    }
}

/// Encode a [`Frame`] into bytes in the specified image format.
pub fn encode_frame(frame: &Frame, format: ImageOutputFormat) -> Result<Vec<u8>> {
    let img = frame_to_dynamic_image(frame)?;
    let mut buf = Cursor::new(Vec::new());
    img.write_to(&mut buf, format.to_image_format())?;
    Ok(buf.into_inner())
}

/// Decode a [`Frame`] from image bytes (format is auto-detected).
pub fn decode_frame(bytes: &[u8]) -> Result<Frame> {
    let img = image::load_from_memory(bytes)?;
    Ok(dynamic_image_to_frame(&img))
}

/// Load a [`Frame`] from a file path.
pub fn load_frame<P: AsRef<Path>>(path: P) -> Result<Frame> {
    let img = image::open(path)?;
    Ok(dynamic_image_to_frame(&img))
}

/// Save a [`Frame`] to a file path. Format is inferred from the extension.
pub fn save_frame<P: AsRef<Path>>(frame: &Frame, path: P) -> Result<()> {
    let path = path.as_ref();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .ok_or_else(|| MediaError::UnsupportedFormat("no file extension".to_string()))?;
    let format = ImageOutputFormat::from_extension(ext)?;
    let bytes = encode_frame(frame, format)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_frame(w: u32, h: u32) -> Frame {
        let pixels = vec![128u8; (w * h * 3) as usize];
        Frame {
            data: Tensor {
                data: TensorData::UInt8(pixels),
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

    #[test]
    fn test_roundtrip_png() {
        let frame = make_test_frame(4, 4);
        let bytes = encode_frame(&frame, ImageOutputFormat::Png).unwrap();
        let decoded = decode_frame(&bytes).unwrap();
        assert_eq!(decoded.data.shape, vec![4, 4, 3]);
    }

    #[test]
    fn test_roundtrip_jpeg() {
        let frame = make_test_frame(8, 8);
        let bytes = encode_frame(&frame, ImageOutputFormat::Jpeg).unwrap();
        let decoded = decode_frame(&bytes).unwrap();
        assert_eq!(decoded.data.shape, vec![8, 8, 3]);
    }

    #[test]
    fn test_frame_to_dynamic_image_and_back() {
        let frame = make_test_frame(4, 2);
        let img = frame_to_dynamic_image(&frame).unwrap();
        let roundtrip = dynamic_image_to_frame(&img);
        assert_eq!(roundtrip.data.shape, frame.data.shape);
    }
}
