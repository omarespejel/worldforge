//! Tensor serialization — SafeTensors, NumPy `.npy`, and raw bytes.

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use worldforge_core::types::{DType, Device, Tensor, TensorData};

use crate::error::{MediaError, Result};

// ---------------------------------------------------------------------------
// SafeTensors
// ---------------------------------------------------------------------------

/// Save a named collection of tensors in SafeTensors format.
pub fn save_safetensors<P: AsRef<Path>>(
    tensors: &HashMap<String, Tensor>,
    path: P,
) -> Result<()> {
    // Collect byte buffers first so they live long enough for TensorView refs
    let byte_buffers: Vec<(String, Vec<u8>, Vec<usize>, safetensors::Dtype)> = tensors
        .iter()
        .map(|(name, tensor)| {
            let bytes = tensor_data_to_bytes(&tensor.data);
            let st_dtype = dtype_to_safetensors(tensor.dtype);
            (name.clone(), bytes, tensor.shape.clone(), st_dtype)
        })
        .collect();

    let tensor_views: Vec<(String, safetensors::tensor::TensorView<'_>)> = byte_buffers
        .iter()
        .map(|(name, bytes, shape, dtype)| {
            (
                name.clone(),
                safetensors::tensor::TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        })
        .collect();

    let serialized = safetensors::serialize(tensor_views, &None)
        .map_err(|e| MediaError::SafeTensors(e.to_string()))?;
    std::fs::write(path, serialized)?;
    Ok(())
}

/// Load tensors from a SafeTensors file.
pub fn load_safetensors<P: AsRef<Path>>(path: P) -> Result<HashMap<String, Tensor>> {
    let data = std::fs::read(path)?;
    let st = safetensors::SafeTensors::deserialize(&data)
        .map_err(|e| MediaError::SafeTensors(e.to_string()))?;

    let mut result = HashMap::new();
    for (name, view) in st.tensors() {
        let dtype = safetensors_dtype_to_dtype(view.dtype());
        let shape: Vec<usize> = view.shape().to_vec();
        let tensor_data = bytes_to_tensor_data(view.data(), dtype);
        result.insert(
            name.to_string(),
            Tensor {
                data: tensor_data,
                shape,
                dtype,
                device: Device::Cpu,
            },
        );
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// NumPy .npy format
// ---------------------------------------------------------------------------

/// Save a tensor as a NumPy `.npy` file.
///
/// Format: magic `\x93NUMPY` + version 1.0 + header (with dtype, order, shape) + raw data.
pub fn save_npy<P: AsRef<Path>>(tensor: &Tensor, path: P) -> Result<()> {
    let mut file = std::fs::File::create(path)?;
    let npy_bytes = tensor_to_npy_bytes(tensor)?;
    file.write_all(&npy_bytes)?;
    Ok(())
}

/// Load a tensor from a NumPy `.npy` file.
pub fn load_npy<P: AsRef<Path>>(path: P) -> Result<Tensor> {
    let data = std::fs::read(path)?;
    npy_bytes_to_tensor(&data)
}

/// Serialize a tensor to `.npy` bytes in memory.
pub fn tensor_to_npy_bytes(tensor: &Tensor) -> Result<Vec<u8>> {
    let descr = dtype_to_npy_descr(tensor.dtype);
    let shape_str = if tensor.shape.is_empty() {
        "()".to_string()
    } else if tensor.shape.len() == 1 {
        format!("({},)", tensor.shape[0])
    } else {
        let parts: Vec<String> = tensor.shape.iter().map(|d| d.to_string()).collect();
        format!("({})", parts.join(", "))
    };

    let header = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        descr, shape_str
    );

    // Pad header to align data to 64 bytes
    let magic_and_version_len = 10; // 6 magic + 2 version + 2 header_len
    let total_before_data = magic_and_version_len + header.len() + 1; // +1 for \n
    let padding = (64 - (total_before_data % 64)) % 64;
    let padded_header = format!("{}{}\n", header, " ".repeat(padding));
    let header_len = padded_header.len() as u16;

    let mut buf = Vec::new();
    // Magic
    buf.extend_from_slice(&[0x93, b'N', b'U', b'M', b'P', b'Y']);
    // Version 1.0
    buf.push(1);
    buf.push(0);
    // Header length (little-endian u16)
    buf.extend_from_slice(&header_len.to_le_bytes());
    // Header
    buf.extend_from_slice(padded_header.as_bytes());
    // Raw data
    buf.extend_from_slice(&tensor_data_to_bytes(&tensor.data));
    Ok(buf)
}

/// Deserialize a tensor from `.npy` bytes.
pub fn npy_bytes_to_tensor(data: &[u8]) -> Result<Tensor> {
    if data.len() < 10 {
        return Err(MediaError::InvalidNpy("file too short".to_string()));
    }
    if &data[0..6] != b"\x93NUMPY" {
        return Err(MediaError::InvalidNpy("invalid magic bytes".to_string()));
    }
    let _major = data[6];
    let _minor = data[7];
    let header_len = u16::from_le_bytes([data[8], data[9]]) as usize;
    let header_end = 10 + header_len;
    if data.len() < header_end {
        return Err(MediaError::InvalidNpy("header extends past EOF".to_string()));
    }

    let header_str = std::str::from_utf8(&data[10..header_end])
        .map_err(|e| MediaError::InvalidNpy(format!("invalid header encoding: {e}")))?;

    let dtype = parse_npy_descr(header_str)?;
    let shape = parse_npy_shape(header_str)?;
    let raw_data = &data[header_end..];
    let tensor_data = bytes_to_tensor_data(raw_data, dtype);

    Ok(Tensor {
        data: tensor_data,
        shape,
        dtype,
        device: Device::Cpu,
    })
}

// ---------------------------------------------------------------------------
// Raw bytes with JSON metadata
// ---------------------------------------------------------------------------

/// Metadata for raw tensor bytes.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RawTensorMeta {
    pub shape: Vec<usize>,
    pub dtype: String,
}

/// Save a tensor as raw bytes + a `.meta.json` sidecar.
pub fn save_raw<P: AsRef<Path>>(tensor: &Tensor, path: P) -> Result<()> {
    let path = path.as_ref();
    let bytes = tensor_data_to_bytes(&tensor.data);
    std::fs::write(path, bytes)?;

    let meta = RawTensorMeta {
        shape: tensor.shape.clone(),
        dtype: dtype_to_string(tensor.dtype),
    };
    let meta_path = path.with_extension("meta.json");
    let meta_json = serde_json::to_string_pretty(&meta)
        .map_err(|e| MediaError::TensorError(e.to_string()))?;
    std::fs::write(meta_path, meta_json)?;
    Ok(())
}

/// Load a tensor from raw bytes + `.meta.json` sidecar.
pub fn load_raw<P: AsRef<Path>>(path: P) -> Result<Tensor> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)?;
    let meta_path = path.with_extension("meta.json");
    let meta_json = std::fs::read_to_string(meta_path)?;
    let meta: RawTensorMeta =
        serde_json::from_str(&meta_json).map_err(|e| MediaError::TensorError(e.to_string()))?;

    let dtype = string_to_dtype(&meta.dtype)?;
    let tensor_data = bytes_to_tensor_data(&bytes, dtype);

    Ok(Tensor {
        data: tensor_data,
        shape: meta.shape,
        dtype,
        device: Device::Cpu,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn tensor_data_to_bytes(data: &TensorData) -> Vec<u8> {
    match data {
        TensorData::Float16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        TensorData::Float32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        TensorData::Float64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        TensorData::BFloat16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        TensorData::UInt8(v) => v.clone(),
        TensorData::Int32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        TensorData::Int64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    }
}

fn bytes_to_tensor_data(bytes: &[u8], dtype: DType) -> TensorData {
    match dtype {
        DType::Float16 => {
            let vals: Vec<u16> = bytes
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            TensorData::Float16(vals)
        }
        DType::Float32 => {
            let vals: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            TensorData::Float32(vals)
        }
        DType::BFloat16 => {
            let vals: Vec<u16> = bytes
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            TensorData::BFloat16(vals)
        }
        DType::UInt8 => TensorData::UInt8(bytes.to_vec()),
        DType::Int32 => {
            let vals: Vec<i32> = bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            TensorData::Int32(vals)
        }
        DType::Int64 => {
            let vals: Vec<i64> = bytes
                .chunks_exact(8)
                .map(|c| {
                    i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                })
                .collect();
            TensorData::Int64(vals)
        }
    }
}

fn dtype_to_safetensors(dtype: DType) -> safetensors::Dtype {
    match dtype {
        DType::Float16 => safetensors::Dtype::F16,
        DType::Float32 => safetensors::Dtype::F32,
        DType::BFloat16 => safetensors::Dtype::BF16,
        DType::UInt8 => safetensors::Dtype::U8,
        DType::Int32 => safetensors::Dtype::I32,
        DType::Int64 => safetensors::Dtype::I64,
    }
}

fn safetensors_dtype_to_dtype(dtype: safetensors::Dtype) -> DType {
    match dtype {
        safetensors::Dtype::F16 => DType::Float16,
        safetensors::Dtype::F32 => DType::Float32,
        safetensors::Dtype::BF16 => DType::BFloat16,
        safetensors::Dtype::U8 => DType::UInt8,
        safetensors::Dtype::I32 => DType::Int32,
        safetensors::Dtype::I64 => DType::Int64,
        // Map unsupported types to Float32 as fallback
        _ => DType::Float32,
    }
}

fn dtype_to_npy_descr(dtype: DType) -> &'static str {
    match dtype {
        DType::Float16 => "<f2",
        DType::Float32 => "<f4",
        DType::BFloat16 => "<f2", // bfloat16 stored as f16 in npy (lossy)
        DType::UInt8 => "|u1",
        DType::Int32 => "<i4",
        DType::Int64 => "<i8",
    }
}

fn dtype_to_string(dtype: DType) -> String {
    match dtype {
        DType::Float16 => "float16".to_string(),
        DType::Float32 => "float32".to_string(),
        DType::BFloat16 => "bfloat16".to_string(),
        DType::UInt8 => "uint8".to_string(),
        DType::Int32 => "int32".to_string(),
        DType::Int64 => "int64".to_string(),
    }
}

fn string_to_dtype(s: &str) -> Result<DType> {
    match s {
        "float16" => Ok(DType::Float16),
        "float32" => Ok(DType::Float32),
        "bfloat16" => Ok(DType::BFloat16),
        "uint8" => Ok(DType::UInt8),
        "int32" => Ok(DType::Int32),
        "int64" => Ok(DType::Int64),
        other => Err(MediaError::TensorError(format!("unknown dtype: {other}"))),
    }
}

fn parse_npy_descr(header: &str) -> Result<DType> {
    // Find 'descr': '<XX'
    let descr_start = header
        .find("'descr'")
        .ok_or_else(|| MediaError::InvalidNpy("missing descr field".to_string()))?;
    let after = &header[descr_start..];
    let quote1 = after
        .find(": '")
        .ok_or_else(|| MediaError::InvalidNpy("malformed descr".to_string()))?;
    let rest = &after[quote1 + 3..];
    let quote2 = rest
        .find('\'')
        .ok_or_else(|| MediaError::InvalidNpy("unterminated descr".to_string()))?;
    let descr = &rest[..quote2];

    match descr {
        "<f2" => Ok(DType::Float16),
        "<f4" => Ok(DType::Float32),
        "|u1" | "<u1" => Ok(DType::UInt8),
        "<i4" => Ok(DType::Int32),
        "<i8" => Ok(DType::Int64),
        other => Err(MediaError::InvalidNpy(format!("unsupported dtype: {other}"))),
    }
}

fn parse_npy_shape(header: &str) -> Result<Vec<usize>> {
    let shape_start = header
        .find("'shape'")
        .ok_or_else(|| MediaError::InvalidNpy("missing shape field".to_string()))?;
    let after = &header[shape_start..];
    let paren_open = after
        .find('(')
        .ok_or_else(|| MediaError::InvalidNpy("missing shape parens".to_string()))?;
    let paren_close = after
        .find(')')
        .ok_or_else(|| MediaError::InvalidNpy("missing shape close paren".to_string()))?;
    let inner = after[paren_open + 1..paren_close].trim();

    if inner.is_empty() {
        return Ok(vec![]);
    }

    inner
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|e| MediaError::InvalidNpy(format!("invalid shape dim: {e}")))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_npy_roundtrip() {
        let tensor = Tensor {
            data: TensorData::Float32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            shape: vec![2, 3],
            dtype: DType::Float32,
            device: Device::Cpu,
        };
        let bytes = tensor_to_npy_bytes(&tensor).unwrap();
        let loaded = npy_bytes_to_tensor(&bytes).unwrap();
        assert_eq!(loaded.shape, vec![2, 3]);
        match loaded.data {
            TensorData::Float32(v) => assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            _ => panic!("wrong dtype"),
        }
    }

    #[test]
    fn test_npy_uint8() {
        let tensor = Tensor {
            data: TensorData::UInt8(vec![0, 127, 255]),
            shape: vec![3],
            dtype: DType::UInt8,
            device: Device::Cpu,
        };
        let bytes = tensor_to_npy_bytes(&tensor).unwrap();
        let loaded = npy_bytes_to_tensor(&bytes).unwrap();
        assert_eq!(loaded.shape, vec![3]);
        match loaded.data {
            TensorData::UInt8(v) => assert_eq!(v, vec![0, 127, 255]),
            _ => panic!("wrong dtype"),
        }
    }
}
