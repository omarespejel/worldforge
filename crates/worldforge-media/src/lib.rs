//! WorldForge Media — frame I/O, tensor serialization, and video clip assembly.
//!
//! This crate provides media encoding/decoding utilities for the WorldForge
//! ecosystem, built on top of the core types defined in `worldforge-core`.
//!
//! # Modules
//!
//! - [`frame_io`] — Encode/decode frames to/from image formats (PNG, JPEG, WebP)
//! - [`tensor_io`] — Serialize/deserialize tensors (SafeTensors, NumPy .npy, raw bytes)
//! - [`clip`] — Video clip assembly and temporal operations
//! - [`codec`] — Codec traits and image-sequence backend

pub mod clip;
pub mod codec;
pub mod error;
pub mod frame_io;
pub mod tensor_io;

pub use error::MediaError;
