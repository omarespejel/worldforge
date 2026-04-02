//! Error types for the media crate.

/// Errors returned by worldforge-media operations.
#[derive(Debug, thiserror::Error)]
pub enum MediaError {
    /// An I/O error occurred.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Image encoding/decoding failed.
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),

    /// Tensor serialization/deserialization failed.
    #[error("safetensors error: {0}")]
    SafeTensors(String),

    /// Invalid NumPy file format.
    #[error("invalid npy format: {0}")]
    InvalidNpy(String),

    /// Frame dimensions or channels are invalid.
    #[error("invalid frame: {0}")]
    InvalidFrame(String),

    /// Video clip operation error.
    #[error("clip error: {0}")]
    ClipError(String),

    /// Unsupported image format.
    #[error("unsupported format: {0}")]
    UnsupportedFormat(String),

    /// Codec not available (e.g., ffmpeg feature not enabled).
    #[error("codec unavailable: {0}")]
    CodecUnavailable(String),

    /// Shape mismatch or invalid tensor metadata.
    #[error("tensor error: {0}")]
    TensorError(String),
}

/// Convenience result type.
pub type Result<T> = std::result::Result<T, MediaError>;
