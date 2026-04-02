# RFC-0018: Video/Frame I/O Pipeline

| Field   | Value                          |
|---------|--------------------------------|
| Status  | Draft                          |
| Author  | WorldForge Core Team           |
| Created | 2026-04-02                     |
| Updated | 2026-04-02                     |

## Abstract

This RFC defines the video and frame I/O pipeline for WorldForge. The system
provides frame-level encoding/decoding via ffmpeg bindings, supports multiple
codecs (H.264, H.265, VP9, AV1), enables tensor serialization for ML pipelines,
handles video clip assembly from frame sequences, and supports real-time
streaming via GStreamer or WebRTC. GPU-accelerated processing, HDR/depth image
formats, memory-efficient large video handling, and frame rate conversion round
out the feature set.

## Motivation

WorldForge's type system already includes `Frame`, `VideoClip`, and `Tensor`
types in `types.rs`, but there is no actual encoding or decoding implementation.
World models fundamentally operate on visual data—they consume video frames as
input and produce predicted frames as output. Without a robust video I/O pipeline:

- Users cannot load video files for prediction input.
- Predicted frames cannot be assembled into playable video output.
- There is no path to real-time video streaming for interactive applications.
- Tensor data from ML models cannot be efficiently serialized or deserialized.
- Large video processing is impossible without memory-efficient chunked handling.
- GPU acceleration for encode/decode is unavailable, leaving performance on the
  table.

This RFC provides the I/O foundation that connects WorldForge's world models to
the real world of video data.

## Detailed Design

### 1. Frame Encoding/Decoding via FFmpeg

We use the `ffmpeg-next` crate (Rust bindings to libavcodec/libavformat) as the
primary codec backend. Direct FFI is available as a fallback.

```rust
use ffmpeg_next as ffmpeg;

/// Configuration for the video I/O subsystem.
pub struct VideoIoConfig {
    /// Default output codec.
    pub default_codec: VideoCodec,
    /// Hardware acceleration device (e.g., "/dev/dri/renderD128").
    pub hw_device: Option<String>,
    /// Maximum decode threads.
    pub decode_threads: usize,
    /// Maximum encode threads.
    pub encode_threads: usize,
    /// Temporary directory for intermediate files.
    pub temp_dir: PathBuf,
    /// Maximum memory for frame buffer pool (bytes).
    pub frame_buffer_pool_size: usize,
}

/// Supported video codecs.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum VideoCodec {
    H264,
    H265,
    VP9,
    AV1,
}

impl VideoCodec {
    pub fn ffmpeg_name(&self) -> &'static str {
        match self {
            VideoCodec::H264 => "libx264",
            VideoCodec::H265 => "libx265",
            VideoCodec::VP9 => "libvpx-vp9",
            VideoCodec::AV1 => "libsvtav1",
        }
    }

    pub fn hw_encoder_name(&self) -> Option<&'static str> {
        match self {
            VideoCodec::H264 => Some("h264_nvenc"),
            VideoCodec::H265 => Some("hevc_nvenc"),
            VideoCodec::VP9 => None,  // No widely available HW encoder
            VideoCodec::AV1 => Some("av1_nvenc"),
        }
    }

    pub fn container_format(&self) -> &'static str {
        match self {
            VideoCodec::H264 | VideoCodec::H265 => "mp4",
            VideoCodec::VP9 | VideoCodec::AV1 => "webm",
        }
    }
}
```

#### Frame Decoder

```rust
/// Decodes video files into individual frames.
pub struct FrameDecoder {
    input_ctx: ffmpeg::format::context::Input,
    decoder: ffmpeg::codec::decoder::Video,
    video_stream_index: usize,
    frame_count: u64,
    time_base: ffmpeg::Rational,
    scaler: Option<ffmpeg::software::scaling::Context>,
}

impl FrameDecoder {
    /// Open a video file for decoding.
    pub fn open(path: &Path, options: &DecodeOptions) -> Result<Self, VideoError> {
        ffmpeg::init()?;

        let input_ctx = ffmpeg::format::input(path)?;
        let video_stream = input_ctx
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or(VideoError::NoVideoStream)?;

        let video_stream_index = video_stream.index();
        let time_base = video_stream.time_base();

        let context_decoder = ffmpeg::codec::context::Context::from_parameters(
            video_stream.parameters()
        )?;
        let mut decoder = context_decoder.decoder().video()?;

        if let Some(threads) = options.threads {
            decoder.set_threading(ffmpeg::threading::Config {
                kind: ffmpeg::threading::Type::Frame,
                count: threads,
                ..Default::default()
            });
        }

        Ok(Self {
            input_ctx,
            decoder,
            video_stream_index,
            frame_count: 0,
            time_base,
            scaler: None,
        })
    }

    /// Decode the next frame as an RGB Frame.
    pub fn next_frame(&mut self) -> Result<Option<Frame>, VideoError> {
        let mut decoded_frame = ffmpeg::frame::Video::empty();

        loop {
            match self.input_ctx.packets().next() {
                Some((stream, packet)) => {
                    if stream.index() != self.video_stream_index {
                        continue;
                    }
                    self.decoder.send_packet(&packet)?;

                    if self.decoder.receive_frame(&mut decoded_frame).is_ok() {
                        let frame = self.convert_frame(&decoded_frame)?;
                        self.frame_count += 1;
                        return Ok(Some(frame));
                    }
                }
                None => {
                    // Flush decoder
                    self.decoder.send_eof()?;
                    if self.decoder.receive_frame(&mut decoded_frame).is_ok() {
                        let frame = self.convert_frame(&decoded_frame)?;
                        self.frame_count += 1;
                        return Ok(Some(frame));
                    }
                    return Ok(None);
                }
            }
        }
    }

    /// Seek to a specific timestamp (seconds).
    pub fn seek(&mut self, timestamp_secs: f64) -> Result<(), VideoError> {
        let ts = (timestamp_secs * self.time_base.denominator() as f64
            / self.time_base.numerator() as f64) as i64;
        self.input_ctx.seek(ts, ..ts)?;
        self.decoder.flush();
        Ok(())
    }

    /// Get video metadata.
    pub fn metadata(&self) -> VideoMetadata {
        VideoMetadata {
            width: self.decoder.width(),
            height: self.decoder.height(),
            frame_rate: self.decoder.frame_rate(),
            duration_secs: self.input_ctx.duration() as f64 / ffmpeg::ffi::AV_TIME_BASE as f64,
            codec: self.decoder.id().name().to_string(),
            pixel_format: format!("{:?}", self.decoder.format()),
            frame_count: self.estimate_frame_count(),
        }
    }

    /// Convert an ffmpeg frame to a WorldForge Frame.
    fn convert_frame(&mut self, src: &ffmpeg::frame::Video) -> Result<Frame, VideoError> {
        // Initialize or reuse scaler for pixel format conversion
        let scaler = self.scaler.get_or_insert_with(|| {
            ffmpeg::software::scaling::Context::get(
                src.format(), src.width(), src.height(),
                ffmpeg::format::Pixel::RGB24, src.width(), src.height(),
                ffmpeg::software::scaling::Flags::BILINEAR,
            ).unwrap()
        });

        let mut rgb_frame = ffmpeg::frame::Video::empty();
        scaler.run(src, &mut rgb_frame)?;

        Ok(Frame {
            width: rgb_frame.width() as usize,
            height: rgb_frame.height() as usize,
            channels: 3,
            data: rgb_frame.data(0).to_vec(),
            timestamp: src.timestamp().map(|t| {
                t as f64 * self.time_base.numerator() as f64
                    / self.time_base.denominator() as f64
            }),
            format: FrameFormat::Rgb8,
        })
    }
}

/// Iterator interface for frame decoding.
impl Iterator for FrameDecoder {
    type Item = Result<Frame, VideoError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_frame() {
            Ok(Some(frame)) => Some(Ok(frame)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}
```

#### Frame Encoder

```rust
/// Encodes individual frames into a video file.
pub struct FrameEncoder {
    output_ctx: ffmpeg::format::context::Output,
    encoder: ffmpeg::codec::encoder::Video,
    video_stream_index: usize,
    frame_index: i64,
    time_base: ffmpeg::Rational,
    scaler: Option<ffmpeg::software::scaling::Context>,
    config: EncodeConfig,
}

pub struct EncodeConfig {
    pub codec: VideoCodec,
    pub width: u32,
    pub height: u32,
    pub frame_rate: u32,
    pub bitrate: u64,
    pub quality: Option<u32>,  // CRF value (0-51 for x264, lower = better)
    pub preset: EncodePreset,
    pub pixel_format: ffmpeg::format::Pixel,
    pub use_hw_accel: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum EncodePreset {
    Ultrafast,
    Fast,
    Medium,
    Slow,
    Veryslow,
}

impl FrameEncoder {
    /// Create a new encoder writing to the specified path.
    pub fn create(path: &Path, config: EncodeConfig) -> Result<Self, VideoError> {
        ffmpeg::init()?;

        let mut output_ctx = ffmpeg::format::output(path)?;

        let codec = ffmpeg::encoder::find_by_name(config.codec.ffmpeg_name())
            .ok_or(VideoError::CodecNotFound(config.codec))?;

        let mut stream = output_ctx.add_stream(codec)?;
        let video_stream_index = stream.index();

        let mut encoder = ffmpeg::codec::context::Context::new_with_codec(codec)
            .encoder()
            .video()?;

        encoder.set_width(config.width);
        encoder.set_height(config.height);
        encoder.set_format(config.pixel_format);
        encoder.set_time_base(ffmpeg::Rational(1, config.frame_rate as i32));
        encoder.set_frame_rate(Some(ffmpeg::Rational(config.frame_rate as i32, 1)));

        if let Some(quality) = config.quality {
            // CRF mode
            encoder.set_parameters(/* CRF params */)?;
        } else {
            encoder.set_bit_rate(config.bitrate as usize);
        }

        let encoder = encoder.open_as(codec)?;
        stream.set_parameters(&encoder);

        output_ctx.write_header()?;

        Ok(Self {
            output_ctx,
            encoder,
            video_stream_index,
            frame_index: 0,
            time_base: ffmpeg::Rational(1, config.frame_rate as i32),
            scaler: None,
            config,
        })
    }

    /// Encode a single frame.
    pub fn encode_frame(&mut self, frame: &Frame) -> Result<(), VideoError> {
        let mut video_frame = self.frame_to_ffmpeg(frame)?;
        video_frame.set_pts(Some(self.frame_index));
        self.frame_index += 1;

        self.encoder.send_frame(&video_frame)?;
        self.flush_packets()?;
        Ok(())
    }

    /// Finalize the video file.
    pub fn finish(mut self) -> Result<(), VideoError> {
        self.encoder.send_eof()?;
        self.flush_packets()?;
        self.output_ctx.write_trailer()?;
        Ok(())
    }

    fn flush_packets(&mut self) -> Result<(), VideoError> {
        let mut encoded = ffmpeg::Packet::empty();
        while self.encoder.receive_packet(&mut encoded).is_ok() {
            encoded.set_stream(self.video_stream_index);
            encoded.rescale_ts(
                self.time_base,
                self.output_ctx.stream(self.video_stream_index).unwrap().time_base(),
            );
            encoded.write_interleaved(&mut self.output_ctx)?;
        }
        Ok(())
    }
}
```

### 2. Codec Support Matrix

| Codec | Decode | Encode | HW Decode | HW Encode | Container |
|-------|--------|--------|-----------|-----------|-----------|
| H.264 | Yes    | Yes    | NVDEC     | NVENC     | MP4, MKV  |
| H.265 | Yes    | Yes    | NVDEC     | NVENC     | MP4, MKV  |
| VP9   | Yes    | Yes    | NVDEC     | No        | WebM      |
| AV1   | Yes    | Yes    | NVDEC*    | NVENC*    | WebM, MP4 |

*AV1 HW support requires Ada Lovelace (RTX 40xx) or newer NVIDIA GPUs.

### 3. Tensor Serialization

For ML pipeline integration, frames and predictions must be serializable as
tensors.

#### SafeTensors Format

```rust
use safetensors::{SafeTensors, tensor::TensorView};

pub struct TensorSerializer;

impl TensorSerializer {
    /// Serialize a Frame as a safetensors file.
    pub fn frame_to_safetensors(
        frame: &Frame,
        path: &Path,
    ) -> Result<(), TensorError> {
        let shape = vec![frame.height, frame.width, frame.channels];
        let tensor = TensorView::new(
            safetensors::Dtype::U8,
            &shape.iter().map(|&s| s).collect::<Vec<_>>(),
            &frame.data,
        )?;

        let tensors = vec![("frame", tensor)];
        safetensors::serialize_to_file(&tensors, &None, path)?;
        Ok(())
    }

    /// Deserialize a Frame from a safetensors file.
    pub fn safetensors_to_frame(path: &Path) -> Result<Frame, TensorError> {
        let data = std::fs::read(path)?;
        let tensors = SafeTensors::deserialize(&data)?;
        let tensor = tensors.tensor("frame")?;

        let shape = tensor.shape();
        Ok(Frame {
            height: shape[0],
            width: shape[1],
            channels: shape[2],
            data: tensor.data().to_vec(),
            timestamp: None,
            format: FrameFormat::Rgb8,
        })
    }

    /// Serialize a batch of tensors (e.g., model predictions).
    pub fn serialize_tensor_batch(
        tensors: &HashMap<String, Tensor>,
        path: &Path,
    ) -> Result<(), TensorError> {
        let views: Vec<(&str, TensorView)> = tensors.iter()
            .map(|(name, tensor)| {
                let view = TensorView::new(
                    tensor.dtype.into(),
                    &tensor.shape,
                    &tensor.data,
                ).unwrap();
                (name.as_str(), view)
            })
            .collect();

        safetensors::serialize_to_file(&views, &None, path)?;
        Ok(())
    }
}
```

#### NumPy .npy Format

```rust
pub struct NpySerializer;

impl NpySerializer {
    /// Save a frame as a NumPy .npy file.
    pub fn frame_to_npy(frame: &Frame, path: &Path) -> Result<(), TensorError> {
        let shape = [frame.height, frame.width, frame.channels];
        let header = Self::build_npy_header(&shape, "<u1")?;

        let mut file = File::create(path)?;
        file.write_all(&header)?;
        file.write_all(&frame.data)?;
        Ok(())
    }

    /// Load a frame from a NumPy .npy file.
    pub fn npy_to_frame(path: &Path) -> Result<Frame, TensorError> {
        let data = std::fs::read(path)?;
        let (header, payload) = Self::parse_npy_header(&data)?;

        Ok(Frame {
            height: header.shape[0],
            width: header.shape[1],
            channels: if header.shape.len() > 2 { header.shape[2] } else { 1 },
            data: payload.to_vec(),
            timestamp: None,
            format: FrameFormat::Rgb8,
        })
    }

    fn build_npy_header(shape: &[usize], dtype: &str) -> Result<Vec<u8>, TensorError> {
        let header_str = format!(
            "{{'descr': '{}', 'fortran_order': False, 'shape': ({},)}}",
            dtype,
            shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ")
        );

        let mut header = Vec::new();
        // Magic number
        header.extend_from_slice(&[0x93, b'N', b'U', b'M', b'P', b'Y']);
        // Version 1.0
        header.push(1);
        header.push(0);
        // Header length (padded to 64-byte alignment)
        let padding = 64 - ((header_str.len() + 10) % 64);
        let header_len = (header_str.len() + padding) as u16;
        header.extend_from_slice(&header_len.to_le_bytes());
        header.extend_from_slice(header_str.as_bytes());
        header.extend(std::iter::repeat(b' ').take(padding - 1));
        header.push(b'\n');
        Ok(header)
    }
}
```

### 4. Video Clip Assembly

```rust
/// Assemble individual frames into a video clip.
pub struct ClipAssembler {
    encoder: FrameEncoder,
    frame_count: usize,
}

impl ClipAssembler {
    /// Create a new clip assembler.
    pub fn new(
        output_path: &Path,
        width: u32,
        height: u32,
        frame_rate: u32,
        codec: VideoCodec,
    ) -> Result<Self, VideoError> {
        let config = EncodeConfig {
            codec,
            width,
            height,
            frame_rate,
            bitrate: Self::auto_bitrate(width, height, frame_rate),
            quality: Some(23),  // CRF 23 default
            preset: EncodePreset::Medium,
            pixel_format: ffmpeg::format::Pixel::YUV420P,
            use_hw_accel: false,
        };

        Ok(Self {
            encoder: FrameEncoder::create(output_path, config)?,
            frame_count: 0,
        })
    }

    /// Add a frame to the clip.
    pub fn add_frame(&mut self, frame: &Frame) -> Result<(), VideoError> {
        self.encoder.encode_frame(frame)?;
        self.frame_count += 1;
        Ok(())
    }

    /// Add frames from an iterator.
    pub fn add_frames<I>(&mut self, frames: I) -> Result<usize, VideoError>
    where
        I: IntoIterator<Item = Frame>,
    {
        let mut count = 0;
        for frame in frames {
            self.add_frame(&frame)?;
            count += 1;
        }
        Ok(count)
    }

    /// Finalize and write the clip.
    pub fn finish(self) -> Result<ClipInfo, VideoError> {
        let frame_count = self.frame_count;
        self.encoder.finish()?;
        Ok(ClipInfo { frame_count })
    }

    fn auto_bitrate(width: u32, height: u32, fps: u32) -> u64 {
        // Rough heuristic: 0.1 bits per pixel per frame
        let pixels = width as u64 * height as u64;
        pixels * fps as u64 / 10
    }
}

/// Concatenate multiple video files into one.
pub struct ClipConcatenator;

impl ClipConcatenator {
    pub fn concatenate(
        input_paths: &[&Path],
        output_path: &Path,
        options: &ConcatOptions,
    ) -> Result<(), VideoError> {
        // Use ffmpeg's concat demuxer for lossless concatenation
        // when codecs match, or re-encode when they differ
        let all_same_codec = Self::check_codec_compatibility(input_paths)?;

        if all_same_codec && !options.force_reencode {
            Self::concat_demuxer(input_paths, output_path)
        } else {
            Self::concat_reencode(input_paths, output_path, options)
        }
    }
}
```

### 5. Streaming Video I/O

#### GStreamer Pipeline

```rust
pub struct GstreamerPipeline {
    pipeline: gstreamer::Pipeline,
    appsink: gstreamer_app::AppSink,
    appsrc: Option<gstreamer_app::AppSrc>,
}

impl GstreamerPipeline {
    /// Create an RTSP input pipeline.
    pub fn rtsp_input(url: &str) -> Result<Self, VideoError> {
        gstreamer::init()?;

        let pipeline_str = format!(
            "rtspsrc location={} latency=100 ! \
             rtph264depay ! h264parse ! avdec_h264 ! \
             videoconvert ! video/x-raw,format=RGB ! \
             appsink name=sink emit-signals=true sync=false",
            url
        );

        let pipeline = gstreamer::parse::launch(&pipeline_str)?
            .downcast::<gstreamer::Pipeline>()
            .unwrap();

        let appsink = pipeline
            .by_name("sink")
            .unwrap()
            .downcast::<gstreamer_app::AppSink>()
            .unwrap();

        pipeline.set_state(gstreamer::State::Playing)?;

        Ok(Self {
            pipeline,
            appsink,
            appsrc: None,
        })
    }

    /// Pull the next frame from the pipeline.
    pub fn pull_frame(&self) -> Result<Option<Frame>, VideoError> {
        match self.appsink.pull_sample() {
            Ok(sample) => {
                let buffer = sample.buffer().ok_or(VideoError::NoBuffer)?;
                let caps = sample.caps().ok_or(VideoError::NoCaps)?;
                let info = gstreamer_video::VideoInfo::from_caps(caps)?;

                let map = buffer.map_readable()?;
                Ok(Some(Frame {
                    width: info.width() as usize,
                    height: info.height() as usize,
                    channels: 3,
                    data: map.as_slice().to_vec(),
                    timestamp: buffer.pts().map(|pts| pts.nseconds() as f64 / 1e9),
                    format: FrameFormat::Rgb8,
                }))
            }
            Err(_) => Ok(None),
        }
    }
}
```

#### WebRTC Output

```rust
pub struct WebRtcOutput {
    peer_connection: Arc<RTCPeerConnection>,
    video_track: Arc<TrackLocalStaticSample>,
    encoder: Mutex<FrameEncoder>,
}

impl WebRtcOutput {
    /// Create a WebRTC output for streaming frames to a browser.
    pub async fn new(config: &WebRtcConfig) -> Result<Self, VideoError> {
        let api = webrtc::api::APIBuilder::new()
            .with_media_engine({
                let mut m = MediaEngine::default();
                m.register_default_codecs()?;
                m
            })
            .build();

        let pc = api.new_peer_connection(RTCConfiguration {
            ice_servers: config.ice_servers.clone(),
            ..Default::default()
        }).await?;

        let video_track = Arc::new(TrackLocalStaticSample::new(
            RTCRtpCodecCapability {
                mime_type: "video/h264".to_string(),
                ..Default::default()
            },
            "video".to_string(),
            "worldforge".to_string(),
        ));

        pc.add_track(video_track.clone()).await?;

        Ok(Self {
            peer_connection: Arc::new(pc),
            video_track,
            encoder: Mutex::new(/* in-memory H.264 encoder */),
        })
    }

    /// Send a frame to all connected peers.
    pub async fn send_frame(&self, frame: &Frame) -> Result<(), VideoError> {
        let mut encoder = self.encoder.lock().await;
        let encoded = encoder.encode_to_buffer(frame)?;

        self.video_track.write_sample(&webrtc::media::Sample {
            data: encoded.into(),
            duration: Duration::from_millis(33), // ~30fps
            ..Default::default()
        }).await?;

        Ok(())
    }
}
```

### 6. GPU-Accelerated Frame Processing

```rust
/// GPU-accelerated decode/encode using NVIDIA NVDEC/NVENC.
pub struct GpuVideoProcessor {
    device: CudaDevice,
    decoder: Option<NvDecoder>,
    encoder: Option<NvEncoder>,
}

impl GpuVideoProcessor {
    pub fn new(device_id: usize) -> Result<Self, VideoError> {
        let device = CudaDevice::new(device_id)?;
        Ok(Self {
            device,
            decoder: None,
            encoder: None,
        })
    }

    /// Decode a video frame on GPU, returning a GPU buffer.
    pub fn decode_frame_gpu(
        &mut self,
        packet: &[u8],
    ) -> Result<GpuFrame, VideoError> {
        let decoder = self.decoder.get_or_insert_with(|| {
            NvDecoder::new(&self.device, VideoCodec::H264).unwrap()
        });

        let gpu_buffer = decoder.decode(packet)?;
        Ok(GpuFrame {
            buffer: gpu_buffer,
            width: decoder.width(),
            height: decoder.height(),
            format: PixelFormat::NV12,
        })
    }

    /// Encode a GPU frame buffer to a compressed packet.
    pub fn encode_frame_gpu(
        &mut self,
        frame: &GpuFrame,
        config: &GpuEncodeConfig,
    ) -> Result<Vec<u8>, VideoError> {
        let encoder = self.encoder.get_or_insert_with(|| {
            NvEncoder::new(
                &self.device,
                config.codec,
                frame.width,
                frame.height,
                config.bitrate,
            ).unwrap()
        });

        encoder.encode(&frame.buffer)
    }

    /// Transfer a GPU frame to CPU memory.
    pub fn download_frame(&self, gpu_frame: &GpuFrame) -> Result<Frame, VideoError> {
        let data = gpu_frame.buffer.download()?;
        // Convert NV12 to RGB on CPU (or use GPU kernel)
        let rgb_data = nv12_to_rgb(&data, gpu_frame.width, gpu_frame.height);
        Ok(Frame {
            width: gpu_frame.width as usize,
            height: gpu_frame.height as usize,
            channels: 3,
            data: rgb_data,
            timestamp: None,
            format: FrameFormat::Rgb8,
        })
    }
}

/// A frame residing in GPU memory.
pub struct GpuFrame {
    pub buffer: CudaBuffer,
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
}
```

### 7. Image Format Support

```rust
pub struct ImageIo;

impl ImageIo {
    /// Load an image file as a Frame.
    pub fn load(path: &Path) -> Result<Frame, ImageError> {
        let extension = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        match extension.to_lowercase().as_str() {
            "png" | "jpg" | "jpeg" | "webp" => Self::load_standard(path),
            "exr" => Self::load_exr(path),
            _ => Err(ImageError::UnsupportedFormat(extension.to_string())),
        }
    }

    fn load_standard(path: &Path) -> Result<Frame, ImageError> {
        let img = image::open(path)?;
        let rgb = img.to_rgb8();
        Ok(Frame {
            width: rgb.width() as usize,
            height: rgb.height() as usize,
            channels: 3,
            data: rgb.into_raw(),
            timestamp: None,
            format: FrameFormat::Rgb8,
        })
    }

    /// Load an EXR file (HDR / depth maps).
    fn load_exr(path: &Path) -> Result<Frame, ImageError> {
        let reader = exr::prelude::read_first_rgba_layer_from_file(
            path,
            |resolution, _| {
                vec![0f32; resolution.width() * resolution.height() * 4]
            },
            |buffer, pos, (r, g, b, a): (f32, f32, f32, f32)| {
                let idx = (pos.y() * pos.width() + pos.x()) * 4;
                buffer[idx] = r;
                buffer[idx + 1] = g;
                buffer[idx + 2] = b;
                buffer[idx + 3] = a;
            },
        )?;

        let (width, height) = (
            reader.layer_data.size.width(),
            reader.layer_data.size.height(),
        );

        // Convert f32 RGBA to bytes for Frame
        let float_data = reader.layer_data.channel_data.pixels;
        let byte_data: Vec<u8> = float_data.iter()
            .map(|&f| (f.clamp(0.0, 1.0) * 255.0) as u8)
            .collect();

        Ok(Frame {
            width,
            height,
            channels: 4,
            data: byte_data,
            timestamp: None,
            format: FrameFormat::Rgba8,
        })
    }

    /// Save a Frame to an image file.
    pub fn save(frame: &Frame, path: &Path) -> Result<(), ImageError> {
        let extension = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("png");

        match extension.to_lowercase().as_str() {
            "png" => Self::save_png(frame, path),
            "jpg" | "jpeg" => Self::save_jpeg(frame, path, 95),
            "webp" => Self::save_webp(frame, path, 90),
            "exr" => Self::save_exr(frame, path),
            _ => Err(ImageError::UnsupportedFormat(extension.to_string())),
        }
    }

    fn save_png(frame: &Frame, path: &Path) -> Result<(), ImageError> {
        let img = image::RgbImage::from_raw(
            frame.width as u32,
            frame.height as u32,
            frame.data.clone(),
        ).ok_or(ImageError::InvalidDimensions)?;
        img.save(path)?;
        Ok(())
    }
}

/// Float32 frame for HDR and depth data.
pub struct FloatFrame {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub data: Vec<f32>,
    pub timestamp: Option<f64>,
}

impl FloatFrame {
    /// Convert to 8-bit Frame with tone mapping.
    pub fn to_ldr(&self, exposure: f32, gamma: f32) -> Frame {
        let data: Vec<u8> = self.data.iter()
            .map(|&v| {
                let mapped = (v * exposure).powf(1.0 / gamma);
                (mapped.clamp(0.0, 1.0) * 255.0) as u8
            })
            .collect();

        Frame {
            width: self.width,
            height: self.height,
            channels: self.channels,
            data,
            timestamp: self.timestamp,
            format: FrameFormat::Rgb8,
        }
    }
}
```

### 8. Memory-Efficient Large Video Handling

```rust
/// Memory-mapped video reader for large files.
pub struct MappedVideoReader {
    mmap: memmap2::Mmap,
    index: VideoIndex,
    decoder: FrameDecoder,
}

/// Pre-built index of frame positions in a video file.
pub struct VideoIndex {
    pub frame_offsets: Vec<u64>,
    pub keyframe_indices: Vec<usize>,
    pub frame_count: usize,
}

impl MappedVideoReader {
    pub fn open(path: &Path) -> Result<Self, VideoError> {
        let file = File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let index = Self::build_index(&mmap)?;
        let decoder = FrameDecoder::open(path, &DecodeOptions::default())?;

        Ok(Self { mmap, index, decoder })
    }

    /// Random access to any frame by index.
    pub fn frame_at(&mut self, index: usize) -> Result<Frame, VideoError> {
        if index >= self.index.frame_count {
            return Err(VideoError::FrameOutOfRange(index, self.index.frame_count));
        }

        // Find nearest keyframe before requested index
        let keyframe = self.index.keyframe_indices.iter()
            .rev()
            .find(|&&kf| kf <= index)
            .copied()
            .unwrap_or(0);

        // Seek to keyframe and decode forward
        self.decoder.seek_to_frame(keyframe)?;
        for _ in keyframe..index {
            self.decoder.next_frame()?;
        }
        self.decoder.next_frame()?.ok_or(VideoError::UnexpectedEof)
    }
}

/// Chunked video processor for batch operations.
pub struct ChunkedVideoProcessor {
    chunk_size: usize,  // frames per chunk
    max_memory: usize,  // maximum memory usage in bytes
}

impl ChunkedVideoProcessor {
    /// Process a video in chunks, applying a function to each chunk.
    pub fn process<F>(
        &self,
        input: &Path,
        output: &Path,
        codec: VideoCodec,
        process_fn: F,
    ) -> Result<(), VideoError>
    where
        F: Fn(&[Frame]) -> Vec<Frame>,
    {
        let mut decoder = FrameDecoder::open(input, &DecodeOptions::default())?;
        let metadata = decoder.metadata();

        let mut encoder = FrameEncoder::create(output, EncodeConfig {
            codec,
            width: metadata.width,
            height: metadata.height,
            frame_rate: metadata.frame_rate.numerator() as u32,
            ..Default::default()
        })?;

        let mut chunk = Vec::with_capacity(self.chunk_size);

        while let Some(frame) = decoder.next_frame()? {
            chunk.push(frame);

            if chunk.len() >= self.chunk_size {
                let processed = process_fn(&chunk);
                for processed_frame in &processed {
                    encoder.encode_frame(processed_frame)?;
                }
                chunk.clear();
            }
        }

        // Process remaining frames
        if !chunk.is_empty() {
            let processed = process_fn(&chunk);
            for processed_frame in &processed {
                encoder.encode_frame(processed_frame)?;
            }
        }

        encoder.finish()?;
        Ok(())
    }
}
```

### 9. Frame Rate Conversion and Temporal Interpolation

```rust
pub struct FrameRateConverter {
    source_fps: f64,
    target_fps: f64,
    method: InterpolationMethod,
}

#[derive(Debug, Clone, Copy)]
pub enum InterpolationMethod {
    /// Nearest neighbor (drop/duplicate frames).
    NearestNeighbor,
    /// Linear blend between adjacent frames.
    LinearBlend,
    /// Optical flow-based interpolation (higher quality, slower).
    OpticalFlow,
}

impl FrameRateConverter {
    pub fn new(source_fps: f64, target_fps: f64, method: InterpolationMethod) -> Self {
        Self { source_fps, target_fps, method }
    }

    /// Convert frame rate, yielding interpolated frames.
    pub fn convert<I>(
        &self,
        input_frames: I,
    ) -> impl Iterator<Item = Result<Frame, VideoError>>
    where
        I: Iterator<Item = Result<Frame, VideoError>>,
    {
        FrameRateIterator {
            input: input_frames.peekable(),
            source_fps: self.source_fps,
            target_fps: self.target_fps,
            method: self.method,
            source_time: 0.0,
            target_time: 0.0,
            prev_frame: None,
            curr_frame: None,
        }
    }

    /// Interpolate between two frames at a given blend factor (0.0 to 1.0).
    pub fn interpolate(
        frame_a: &Frame,
        frame_b: &Frame,
        t: f64,
        method: InterpolationMethod,
    ) -> Result<Frame, VideoError> {
        match method {
            InterpolationMethod::NearestNeighbor => {
                if t < 0.5 {
                    Ok(frame_a.clone())
                } else {
                    Ok(frame_b.clone())
                }
            }
            InterpolationMethod::LinearBlend => {
                let data: Vec<u8> = frame_a.data.iter()
                    .zip(frame_b.data.iter())
                    .map(|(&a, &b)| {
                        ((a as f64) * (1.0 - t) + (b as f64) * t) as u8
                    })
                    .collect();

                Ok(Frame {
                    width: frame_a.width,
                    height: frame_a.height,
                    channels: frame_a.channels,
                    data,
                    timestamp: Some(
                        frame_a.timestamp.unwrap_or(0.0) * (1.0 - t)
                            + frame_b.timestamp.unwrap_or(0.0) * t
                    ),
                    format: frame_a.format,
                })
            }
            InterpolationMethod::OpticalFlow => {
                // Compute dense optical flow between frames
                // Warp both frames toward intermediate time t
                // Blend warped frames
                todo!("Optical flow interpolation requires CV backend")
            }
        }
    }
}
```

## Implementation Plan

### Phase 1: Core Decode/Encode (3 weeks)
- Add `ffmpeg-next` dependency with proper feature flags.
- Implement `FrameDecoder` with seek and metadata support.
- Implement `FrameEncoder` with H.264 and H.265 support.
- Wire into existing `Frame` and `VideoClip` types in `types.rs`.
- Add VP9 and AV1 codec support.

### Phase 2: Image and Tensor I/O (2 weeks)
- Implement `ImageIo` for PNG, JPEG, WebP using the `image` crate.
- Add EXR support via `exr` crate for depth/HDR.
- Implement safetensors serialization for frames and tensors.
- Implement NumPy .npy serialization for Python interop.

### Phase 3: Video Assembly (2 weeks)
- Build `ClipAssembler` for frame-to-video conversion.
- Implement `ClipConcatenator` for joining video segments.
- Add clip trimming and segment extraction.
- Video metadata inspection API.

### Phase 4: Streaming (3 weeks)
- GStreamer integration for RTSP and pipeline-based input.
- WebRTC output for browser-based real-time viewing.
- Async frame delivery via channels.
- Backpressure handling for slow consumers.

### Phase 5: GPU Acceleration (2 weeks)
- NVIDIA NVDEC integration for GPU decode.
- NVIDIA NVENC integration for GPU encode.
- GPU<->CPU frame transfer utilities.
- Benchmark GPU vs CPU paths.

### Phase 6: Large Video & Frame Rate (1 week)
- Memory-mapped video reader.
- Chunked processing pipeline.
- Frame rate conversion with nearest-neighbor and linear blend.
- Optical flow interpolation (stretch goal).

## Testing Strategy

### Unit Tests
- Decode a known test video and verify frame dimensions, count, timestamps.
- Encode frames and decode them back, verify pixel data roundtrip.
- Tensor serialization/deserialization roundtrip for safetensors and npy.
- Image format load/save roundtrip for each supported format.
- Frame rate conversion produces correct number of output frames.
- Linear blend interpolation produces expected pixel values.

### Integration Tests
- Full pipeline: decode video -> process frames -> encode video.
- Large video (1GB+) processing with chunked pipeline.
- Concurrent decode of multiple videos.
- GStreamer pipeline creation and frame pulling (requires GStreamer installed).

### Codec Conformance Tests
- Verify each codec produces valid container files (ffprobe validation).
- Cross-decoder compatibility (encode with WorldForge, decode with ffmpeg CLI).
- Bitrate accuracy within 10% of target.

### Performance Benchmarks
- Decode throughput: frames/second for 1080p and 4K H.264.
- Encode throughput: frames/second for each codec and preset.
- GPU vs CPU decode/encode comparison.
- Memory usage during large video processing.
- Tensor serialization speed for batch operations.

### Fuzz Tests
- Feed random bytes to decoder (should return errors, not crash).
- Random frame dimensions and data to encoder.
- Malformed npy/safetensors files to deserializer.

## Open Questions

1. **FFmpeg linking strategy**: Should we statically link ffmpeg (larger binary,
   simpler deployment) or dynamically link (smaller binary, requires system
   ffmpeg)? Consider a feature flag for both.

2. **GStreamer vs pure ffmpeg for streaming**: GStreamer adds a large dependency
   tree. Could we achieve streaming with just ffmpeg's RTSP demuxer and
   fragmented MP4 output instead?

3. **WebRTC crate choice**: `webrtc-rs` is pure Rust but less mature than
   `libwebrtc` bindings. Which is more appropriate for production use?

4. **Optical flow implementation**: Should we bring in OpenCV bindings for
   optical flow, implement a simple block-matching algorithm in Rust, or
   delegate to a GPU compute shader?

5. **Color space handling**: Professional video uses BT.709, BT.2020, etc.
   How much color space awareness do we need? Start with sRGB only?

6. **Audio support**: This RFC is video-only. Should audio tracks be preserved
   during video processing? If so, when should we add audio passthrough?

7. **WASM compatibility**: Should the video I/O pipeline work in WASM
   environments (using browser MediaCodecs API)? This would require a completely
   different backend.
