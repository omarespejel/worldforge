# RFC-0004: Meta V-JEPA Local Provider

| Field   | Value                          |
|---------|--------------------------------|
| Title   | Meta V-JEPA Local Provider     |
| Status  | Draft                          |
| Author  | WorldForge Contributors        |
| Created | 2026-04-02                     |
| RFC     | 0004                           |

---

## Abstract

This RFC proposes the integration of Meta's V-JEPA (Video Joint Embedding
Predictive Architecture) as a local inference provider within WorldForge.
Unlike cloud-based providers, V-JEPA runs entirely on the user's hardware,
performing prediction in latent space without pixel-level reconstruction.
This document covers the V-JEPA architecture, model weight acquisition from
HuggingFace, inference backend selection (ONNX Runtime vs PyTorch/libtorch
via tch-rs), GPU memory management, implementation of `predict()` and
`embed()` trait methods, batch inference, model quantization strategies,
local model lifecycle management, caching, and performance targets.

---

## Motivation

### Why V-JEPA?

V-JEPA represents a fundamentally different approach to world modeling compared
to generative video models like Runway Gen-4 or OpenAI Sora. Rather than
generating pixel-level outputs, V-JEPA operates entirely in a learned latent
space, predicting future representations of video without reconstructing
individual pixels.

This architecture provides several unique advantages for WorldForge:

1. **Efficiency**: Latent-space prediction is orders of magnitude faster than
   pixel-level generation. A single prediction can complete in milliseconds
   rather than minutes.

2. **Privacy**: All computation is local. No data leaves the user's machine.
   This is critical for sensitive applications (medical imaging, security
   footage, proprietary environments).

3. **Cost**: After initial model download, inference is free. No per-request
   API costs. This enables bulk processing and experimentation.

4. **Latent Representations**: V-JEPA's embeddings capture high-level semantic
   understanding of video content—object relationships, motion patterns,
   scene dynamics—making them ideal for WorldForge's `embed()` method.

5. **Offline Operation**: No internet connection required after model download.

### V-JEPA Architecture Overview

V-JEPA is based on the Joint Embedding Predictive Architecture (JEPA) paradigm:

```
Input Video → Context Encoder → Context Embeddings
                                        ↓
                                   Predictor → Predicted Target Embeddings
                                        ↓
Target Video → Target Encoder → Target Embeddings (for training only)
```

Key architectural properties:

- **No Pixel Reconstruction**: The model never generates pixels. It predicts
  in a learned representation space.
- **Masking Strategy**: During training, portions of the input video are masked,
  and the predictor learns to fill in the missing representations.
- **Self-Supervised**: Trained without labels on large-scale video datasets.
- **Vision Transformer (ViT) Backbone**: Uses a ViT-H/16 or ViT-G/14 encoder.

### WorldForge Alignment

V-JEPA maps to the `WorldModelProvider` trait as follows:

- `predict()` → Latent-space next-state prediction (core capability)
- `embed()` → Extract latent representations from video (core capability)
- `plan()` → Multi-step rollout in latent space
- `reason()` → Latent similarity and comparison queries
- `generate()` → Not directly supported (no pixel output)
- `transfer()` → Not directly supported
- `health_check()` → Model loaded and GPU available check
- `cost_estimate()` → Always zero (local inference)

---

## Detailed Design

### 1. Model Weights and Acquisition

#### 1.1 Available Model Variants

| Model         | Parameters | Encoder    | Resolution | HuggingFace ID                     |
|---------------|-----------|------------|------------|-------------------------------------|
| V-JEPA Base   | 86M       | ViT-B/16   | 224×224    | facebook/vjepa-base                 |
| V-JEPA Large  | 307M      | ViT-L/16   | 224×224    | facebook/vjepa-large                |
| V-JEPA Huge   | 632M      | ViT-H/16   | 224×224    | facebook/vjepa-huge                 |
| V-JEPA Giant  | 1.1B      | ViT-G/14   | 224×224    | facebook/vjepa-giant (unreleased)   |

#### 1.2 Weight Download and Management

```rust
pub struct ModelManager {
    /// Local directory for model storage
    model_dir: PathBuf,
    /// HuggingFace Hub client
    hf_client: HfClient,
    /// Currently loaded models
    loaded_models: Arc<RwLock<HashMap<ModelVariant, LoadedModel>>>,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum ModelVariant {
    Base,
    Large,
    Huge,
    Giant,
}

impl ModelVariant {
    pub fn hf_repo_id(&self) -> &str {
        match self {
            Self::Base => "facebook/vjepa-base",
            Self::Large => "facebook/vjepa-large",
            Self::Huge => "facebook/vjepa-huge",
            Self::Giant => "facebook/vjepa-giant",
        }
    }

    pub fn file_size_bytes(&self) -> u64 {
        match self {
            Self::Base => 344_000_000,    // ~344 MB
            Self::Large => 1_228_000_000, // ~1.2 GB
            Self::Huge => 2_528_000_000,  // ~2.5 GB
            Self::Giant => 4_400_000_000, // ~4.4 GB
        }
    }

    pub fn embedding_dim(&self) -> usize {
        match self {
            Self::Base => 768,
            Self::Large => 1024,
            Self::Huge => 1280,
            Self::Giant => 1408,
        }
    }
}

impl ModelManager {
    pub async fn download_model(
        &self,
        variant: ModelVariant,
        progress_callback: Option<Box<dyn Fn(f64) + Send>>,
    ) -> Result<PathBuf, VjepaError> {
        let repo_id = variant.hf_repo_id();
        let target_dir = self.model_dir.join(variant.dir_name());

        if target_dir.join("model.safetensors").exists() {
            tracing::info!("Model {} already downloaded", repo_id);
            return Ok(target_dir);
        }

        tracing::info!("Downloading model {} from HuggingFace...", repo_id);

        // Download with progress tracking
        let files = self.hf_client
            .download_repo(repo_id, &target_dir, progress_callback)
            .await?;

        // Verify checksums
        self.verify_checksums(&target_dir, &files).await?;

        Ok(target_dir)
    }

    pub async fn verify_checksums(
        &self,
        model_dir: &Path,
        expected: &[FileInfo],
    ) -> Result<(), VjepaError> {
        for file_info in expected {
            let path = model_dir.join(&file_info.name);
            let actual_hash = sha256_file(&path).await?;
            if actual_hash != file_info.sha256 {
                return Err(VjepaError::ChecksumMismatch {
                    file: file_info.name.clone(),
                    expected: file_info.sha256.clone(),
                    actual: actual_hash,
                });
            }
        }
        Ok(())
    }
}
```

### 2. Inference Backend Selection

#### 2.1 ONNX Runtime Backend

ONNX Runtime provides the most portable inference option with strong GPU
acceleration via CUDA and TensorRT execution providers.

```rust
pub struct OnnxBackend {
    session: ort::Session,
    device: Device,
    model_variant: ModelVariant,
}

impl OnnxBackend {
    pub fn new(
        model_path: &Path,
        device: Device,
    ) -> Result<Self, VjepaError> {
        let environment = ort::Environment::builder()
            .with_name("worldforge-vjepa")
            .with_execution_providers([
                match device {
                    Device::Cuda(id) => {
                        ort::ExecutionProvider::CUDA(
                            ort::CUDAExecutionProviderOptions {
                                device_id: id as i32,
                                ..Default::default()
                            }
                        )
                    }
                    Device::Cpu => ort::ExecutionProvider::CPU(Default::default()),
                }
            ])
            .build()?;

        let session = ort::Session::builder()?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path.join("model.onnx"))?;

        Ok(Self {
            session,
            device,
            model_variant: ModelVariant::from_path(model_path)?,
        })
    }

    pub fn predict(
        &self,
        input: &ndarray::Array4<f32>,
        mask: &ndarray::Array2<bool>,
    ) -> Result<ndarray::Array3<f32>, VjepaError> {
        let input_tensor = ort::Value::from_array(input)?;
        let mask_tensor = ort::Value::from_array(mask)?;

        let outputs = self.session.run(
            ort::inputs!["video" => input_tensor, "mask" => mask_tensor]?
        )?;

        let predicted = outputs[0].try_extract_tensor::<f32>()?;
        Ok(predicted.into_owned().into_dimensionality()?)
    }
}
```

#### 2.2 PyTorch/libtorch Backend via tch-rs

For users with a PyTorch installation, `tch-rs` provides direct access to
libtorch operations, which may offer better compatibility with the original
model weights.

```rust
pub struct TchBackend {
    model: tch::CModule,
    device: tch::Device,
    model_variant: ModelVariant,
}

impl TchBackend {
    pub fn new(
        model_path: &Path,
        device: tch::Device,
    ) -> Result<Self, VjepaError> {
        // Load TorchScript model
        let model = tch::CModule::load_on_device(
            model_path.join("model.pt"),
            device,
        )?;

        Ok(Self {
            model,
            device,
            model_variant: ModelVariant::from_path(model_path)?,
        })
    }

    pub fn predict(
        &self,
        input: &tch::Tensor,
        mask: &tch::Tensor,
    ) -> Result<tch::Tensor, VjepaError> {
        let input = input.to(self.device);
        let mask = mask.to(self.device);

        let output = self.model.forward_ts(&[&input, &mask])?;
        Ok(output)
    }

    pub fn embed(
        &self,
        input: &tch::Tensor,
    ) -> Result<tch::Tensor, VjepaError> {
        let input = input.to(self.device);

        // Use only the encoder (no predictor)
        let output = self.model.method_ts("encode", &[&input])?;
        Ok(output)
    }
}
```

#### 2.3 Backend Abstraction

```rust
pub enum InferenceBackend {
    Onnx(OnnxBackend),
    Tch(TchBackend),
}

pub trait VjepaInference: Send + Sync {
    fn predict_latent(
        &self,
        frames: &VideoFrames,
        mask: &PredictionMask,
    ) -> Result<LatentPrediction, VjepaError>;

    fn extract_embedding(
        &self,
        frames: &VideoFrames,
    ) -> Result<Embedding, VjepaError>;

    fn device_info(&self) -> DeviceInfo;
    fn model_variant(&self) -> ModelVariant;
}
```

### 3. GPU Memory Management

GPU memory is a critical constraint for local inference. The provider
implements careful memory management:

```rust
pub struct GpuMemoryManager {
    /// Maximum GPU memory budget (bytes)
    memory_budget: u64,
    /// Current allocated memory tracking
    allocated: Arc<AtomicU64>,
    /// Allocation semaphore (limits concurrent inferences)
    inference_semaphore: Arc<Semaphore>,
}

impl GpuMemoryManager {
    pub fn new(config: &VjepaConfig) -> Self {
        let memory_budget = config.gpu_memory_budget_mb
            .map(|mb| mb * 1024 * 1024)
            .unwrap_or_else(|| Self::detect_available_gpu_memory());

        // Estimate concurrent inferences based on model size and budget
        let model_memory = config.model_variant.file_size_bytes() * 2; // rough estimate
        let max_concurrent = ((memory_budget - model_memory) / Self::inference_memory_per_batch())
            .max(1) as usize;

        Self {
            memory_budget,
            allocated: Arc::new(AtomicU64::new(0)),
            inference_semaphore: Arc::new(Semaphore::new(max_concurrent)),
        }
    }

    pub fn detect_available_gpu_memory() -> u64 {
        #[cfg(feature = "cuda")]
        {
            // Query CUDA for available GPU memory
            cuda_runtime::mem_get_info()
                .map(|(free, _total)| free)
                .unwrap_or(4 * 1024 * 1024 * 1024) // Default 4GB
        }
        #[cfg(not(feature = "cuda"))]
        {
            // CPU-only: use system RAM budget
            4 * 1024 * 1024 * 1024 // Default 4GB
        }
    }

    pub async fn acquire_inference_slot(&self) -> Result<InferenceGuard, VjepaError> {
        let permit = tokio::time::timeout(
            Duration::from_secs(30),
            self.inference_semaphore.acquire(),
        )
        .await
        .map_err(|_| VjepaError::GpuMemoryTimeout)?
        .map_err(|_| VjepaError::GpuMemoryExhausted)?;

        Ok(InferenceGuard {
            _permit: permit,
            allocated: self.allocated.clone(),
        })
    }
}
```

### 4. Model Lifecycle Management

```rust
pub enum ModelState {
    /// Model weights not downloaded
    NotDownloaded,
    /// Model downloaded but not loaded into memory
    Downloaded { path: PathBuf },
    /// Model loaded into CPU memory (warm state)
    Warm {
        path: PathBuf,
        backend: Arc<dyn VjepaInference>,
    },
    /// Model loaded onto GPU and ready for inference (hot state)
    Hot {
        path: PathBuf,
        backend: Arc<dyn VjepaInference>,
        last_inference: Instant,
    },
    /// Model is currently running an inference
    Inferring {
        path: PathBuf,
        backend: Arc<dyn VjepaInference>,
        active_count: Arc<AtomicU32>,
    },
    /// Model is being unloaded
    Unloading,
}

pub struct ModelLifecycle {
    state: Arc<RwLock<ModelState>>,
    config: VjepaConfig,
    memory_manager: GpuMemoryManager,
}

impl ModelLifecycle {
    /// Transition: NotDownloaded → Downloaded
    pub async fn download(&self) -> Result<(), VjepaError> {
        // Download from HuggingFace...
    }

    /// Transition: Downloaded → Warm (load into CPU memory)
    pub async fn warm_up(&self) -> Result<(), VjepaError> {
        let mut state = self.state.write().await;
        if let ModelState::Downloaded { path } = &*state {
            let backend = self.create_backend(path, Device::Cpu)?;
            *state = ModelState::Warm {
                path: path.clone(),
                backend: Arc::new(backend),
            };
        }
        Ok(())
    }

    /// Transition: Warm → Hot (move to GPU)
    pub async fn activate(&self) -> Result<(), VjepaError> {
        let mut state = self.state.write().await;
        if let ModelState::Warm { path, .. } = &*state {
            let backend = self.create_backend(path, Device::Cuda(0))?;
            *state = ModelState::Hot {
                path: path.clone(),
                backend: Arc::new(backend),
                last_inference: Instant::now(),
            };
        }
        Ok(())
    }

    /// Transition: Hot → Warm (offload from GPU to save memory)
    pub async fn deactivate(&self) -> Result<(), VjepaError> {
        // Move model back to CPU...
    }

    /// Transition: * → Unloading → Downloaded
    pub async fn unload(&self) -> Result<(), VjepaError> {
        // Release all resources...
    }

    /// Auto-deactivate after idle timeout
    pub async fn run_idle_monitor(&self) {
        let idle_timeout = self.config.idle_timeout;
        loop {
            tokio::time::sleep(Duration::from_secs(30)).await;
            let state = self.state.read().await;
            if let ModelState::Hot { last_inference, .. } = &*state {
                if last_inference.elapsed() > idle_timeout {
                    drop(state);
                    let _ = self.deactivate().await;
                }
            }
        }
    }
}
```

### 5. Implementing predict() and embed()

```rust
#[async_trait]
impl WorldModelProvider for VjepaProvider {
    async fn predict(
        &self,
        input: &WorldState,
        params: &PredictionParams,
    ) -> Result<WorldState, ProviderError> {
        // Ensure model is in Hot state
        self.lifecycle.ensure_hot().await?;

        // Acquire GPU memory slot
        let _guard = self.memory_manager.acquire_inference_slot().await?;

        // Extract video frames from input world state
        let frames = self.extract_frames(input)?;

        // Preprocess: resize to model resolution, normalize
        let preprocessed = self.preprocess_frames(&frames)?;

        // Create prediction mask (which future frames to predict)
        let mask = PredictionMask::future_frames(
            params.num_steps.unwrap_or(4),
            preprocessed.num_frames(),
        );

        // Run inference
        let backend = self.lifecycle.get_backend().await?;
        let prediction = backend.predict_latent(&preprocessed, &mask)?;

        // Convert latent prediction back to WorldState
        let predicted_state = WorldState {
            id: Uuid::new_v4().to_string(),
            parent_id: Some(input.id.clone()),
            timestamp: input.timestamp + params.time_delta(),
            latent_representation: Some(prediction.to_vec()),
            confidence: Some(prediction.confidence()),
            metadata: PredictionMetadata {
                model: "v-jepa".to_string(),
                variant: self.config.model_variant.to_string(),
                steps: params.num_steps.unwrap_or(4),
                latent_dim: self.config.model_variant.embedding_dim(),
            }.into(),
        };

        Ok(predicted_state)
    }

    async fn embed(
        &self,
        input: &WorldState,
    ) -> Result<Embedding, ProviderError> {
        self.lifecycle.ensure_hot().await?;
        let _guard = self.memory_manager.acquire_inference_slot().await?;

        let frames = self.extract_frames(input)?;
        let preprocessed = self.preprocess_frames(&frames)?;

        let backend = self.lifecycle.get_backend().await?;
        let embedding = backend.extract_embedding(&preprocessed)?;

        Ok(Embedding {
            vector: embedding.to_vec(),
            dimensions: self.config.model_variant.embedding_dim(),
            model: format!("v-jepa-{}", self.config.model_variant),
            normalized: true,
        })
    }

    async fn plan(
        &self,
        initial: &WorldState,
        goal: &WorldState,
        params: &PlanningParams,
    ) -> Result<Plan, ProviderError> {
        // Multi-step rollout in latent space
        let mut trajectory = Vec::new();
        let mut current = initial.clone();
        let max_steps = params.max_steps.unwrap_or(16);

        for step in 0..max_steps {
            let prediction_params = PredictionParams {
                num_steps: Some(1),
                ..Default::default()
            };

            let next = self.predict(&current, &prediction_params).await?;

            // Check if we've reached the goal (in latent space)
            if let (Some(pred_latent), Some(goal_latent)) =
                (&next.latent_representation, &goal.latent_representation)
            {
                let distance = cosine_distance(pred_latent, goal_latent);
                trajectory.push(next.clone());

                if distance < params.goal_threshold.unwrap_or(0.1) {
                    break;
                }
            }

            current = next;
        }

        Ok(Plan {
            steps: trajectory,
            total_steps: trajectory.len(),
            goal_reached: true, // Check actual distance
        })
    }

    async fn health_check(&self) -> Result<HealthStatus, ProviderError> {
        match &*self.lifecycle.state.read().await {
            ModelState::Hot { .. } | ModelState::Inferring { .. } => {
                Ok(HealthStatus::Healthy)
            }
            ModelState::Warm { .. } => {
                Ok(HealthStatus::Degraded("Model on CPU, not GPU".into()))
            }
            ModelState::Downloaded { .. } => {
                Ok(HealthStatus::Degraded("Model downloaded but not loaded".into()))
            }
            ModelState::NotDownloaded => {
                Ok(HealthStatus::Unhealthy("Model not downloaded".into()))
            }
            ModelState::Unloading => {
                Ok(HealthStatus::Degraded("Model is being unloaded".into()))
            }
        }
    }

    async fn cost_estimate(
        &self,
        _params: &GenerationParams,
    ) -> Result<CostEstimate, ProviderError> {
        Ok(CostEstimate {
            credits: 0,
            estimated_usd: 0.0,
            currency: "USD".to_string(),
            breakdown: vec![
                CostItem {
                    description: "Local inference (no cost)".to_string(),
                    amount: 0.0,
                },
            ],
        })
    }
}
```

### 6. Batch Inference

```rust
impl VjepaProvider {
    /// Process multiple inputs in a single batched inference call
    pub async fn batch_predict(
        &self,
        inputs: &[WorldState],
        params: &PredictionParams,
    ) -> Result<Vec<WorldState>, ProviderError> {
        self.lifecycle.ensure_hot().await?;
        let _guard = self.memory_manager.acquire_inference_slot().await?;

        let batch_size = inputs.len().min(self.config.max_batch_size);
        let mut results = Vec::with_capacity(inputs.len());

        for chunk in inputs.chunks(batch_size) {
            let frames_batch: Vec<_> = chunk.iter()
                .map(|input| self.extract_and_preprocess(input))
                .collect::<Result<_, _>>()?;

            let stacked = self.stack_batch(&frames_batch)?;
            let masks = PredictionMask::batch_future_frames(
                params.num_steps.unwrap_or(4),
                stacked.num_frames(),
                chunk.len(),
            );

            let backend = self.lifecycle.get_backend().await?;
            let predictions = backend.predict_latent(&stacked, &masks)?;

            for (i, pred) in predictions.split_batch().enumerate() {
                results.push(WorldState {
                    id: Uuid::new_v4().to_string(),
                    parent_id: Some(chunk[i].id.clone()),
                    latent_representation: Some(pred.to_vec()),
                    ..Default::default()
                });
            }
        }

        Ok(results)
    }

    /// Process multiple embedding requests in a batch
    pub async fn batch_embed(
        &self,
        inputs: &[WorldState],
    ) -> Result<Vec<Embedding>, ProviderError> {
        self.lifecycle.ensure_hot().await?;
        let _guard = self.memory_manager.acquire_inference_slot().await?;

        let batch_size = inputs.len().min(self.config.max_batch_size);
        let mut results = Vec::with_capacity(inputs.len());

        for chunk in inputs.chunks(batch_size) {
            let frames_batch: Vec<_> = chunk.iter()
                .map(|input| self.extract_and_preprocess(input))
                .collect::<Result<_, _>>()?;

            let stacked = self.stack_batch(&frames_batch)?;

            let backend = self.lifecycle.get_backend().await?;
            let embeddings = backend.extract_embedding(&stacked)?;

            for emb in embeddings.split_batch() {
                results.push(Embedding {
                    vector: emb.to_vec(),
                    dimensions: self.config.model_variant.embedding_dim(),
                    model: format!("v-jepa-{}", self.config.model_variant),
                    normalized: true,
                });
            }
        }

        Ok(results)
    }
}
```

### 7. Model Quantization

```rust
#[derive(Debug, Clone)]
pub enum QuantizationMode {
    /// Full precision (FP32) - maximum accuracy
    FP32,
    /// Half precision (FP16) - good balance of speed and accuracy
    FP16,
    /// Brain floating point (BF16) - better for certain GPU architectures
    BF16,
    /// 8-bit integer quantization - fastest, some accuracy loss
    INT8,
    /// Dynamic quantization - quantize at runtime
    DynamicINT8,
}

impl QuantizationMode {
    pub fn memory_multiplier(&self) -> f64 {
        match self {
            Self::FP32 => 1.0,
            Self::FP16 | Self::BF16 => 0.5,
            Self::INT8 | Self::DynamicINT8 => 0.25,
        }
    }

    pub fn accuracy_note(&self) -> &str {
        match self {
            Self::FP32 => "Full accuracy, highest memory usage",
            Self::FP16 => "Negligible accuracy loss, half memory",
            Self::BF16 => "Negligible accuracy loss, half memory, better on Ampere+",
            Self::INT8 => "Small accuracy loss (~1-2%), quarter memory",
            Self::DynamicINT8 => "Small accuracy loss, quarter memory, no calibration needed",
        }
    }
}

/// Memory estimates for different model + quantization combos
pub fn estimate_gpu_memory(variant: &ModelVariant, quant: &QuantizationMode) -> u64 {
    let base = variant.file_size_bytes() as f64 * 1.3; // overhead
    (base * quant.memory_multiplier()) as u64
}
```

### 8. Cache Strategy

```rust
pub struct LatentCache {
    /// LRU cache for embeddings, keyed by content hash
    embedding_cache: Arc<Mutex<LruCache<ContentHash, Embedding>>>,
    /// LRU cache for predictions, keyed by (input_hash, params_hash)
    prediction_cache: Arc<Mutex<LruCache<(ContentHash, ParamsHash), LatentPrediction>>>,
    /// Maximum cache memory budget
    max_memory_bytes: u64,
    /// Current estimated memory usage
    current_memory: Arc<AtomicU64>,
}

impl LatentCache {
    pub fn new(max_memory_mb: u64) -> Self {
        let max_entries = (max_memory_mb * 1024 * 1024 / 8192) as usize; // rough estimate
        Self {
            embedding_cache: Arc::new(Mutex::new(LruCache::new(
                NonZeroUsize::new(max_entries).unwrap()
            ))),
            prediction_cache: Arc::new(Mutex::new(LruCache::new(
                NonZeroUsize::new(max_entries / 2).unwrap()
            ))),
            max_memory_bytes: max_memory_mb * 1024 * 1024,
            current_memory: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn get_embedding(&self, hash: &ContentHash) -> Option<Embedding> {
        self.embedding_cache.lock().unwrap().get(hash).cloned()
    }

    pub fn put_embedding(&self, hash: ContentHash, embedding: Embedding) {
        let size = embedding.vector.len() * std::mem::size_of::<f32>();
        self.current_memory.fetch_add(size as u64, Ordering::Relaxed);
        self.embedding_cache.lock().unwrap().put(hash, embedding);
    }

    pub fn get_prediction(
        &self,
        input_hash: &ContentHash,
        params_hash: &ParamsHash,
    ) -> Option<LatentPrediction> {
        self.prediction_cache.lock().unwrap()
            .get(&(*input_hash, *params_hash))
            .cloned()
    }
}
```

### 9. Performance Targets

| Metric                  | Target (ViT-H, FP16, RTX 4090) | Target (ViT-H, FP16, CPU) |
|-------------------------|----------------------------------|----------------------------|
| Single prediction       | < 15 ms                         | < 500 ms                  |
| Single embedding        | < 10 ms                         | < 300 ms                  |
| Batch predict (32)      | < 100 ms                        | < 8,000 ms                |
| Batch embed (32)        | < 60 ms                         | < 5,000 ms                |
| Model load (cold)       | < 5 s                           | < 10 s                    |
| Model warm → hot        | < 2 s                           | N/A                       |
| GPU memory (FP16)       | < 3 GB                          | N/A                       |
| GPU memory (INT8)       | < 1.5 GB                        | N/A                       |

---

## Implementation Plan

### Phase 1: Model Management (Week 1-2)

1. Create `crates/worldforge-providers/src/vjepa/` module
2. Implement HuggingFace model download with progress reporting
3. Implement checksum verification for downloaded weights
4. Implement model variant selection and configuration
5. Add model storage directory management

### Phase 2: Inference Backends (Week 3-5)

1. Implement ONNX Runtime backend
2. Implement tch-rs/libtorch backend
3. Create backend abstraction trait
4. Add automatic backend selection based on available libraries
5. Implement FP16 and INT8 quantization support

### Phase 3: WorldModelProvider Implementation (Week 6-7)

1. Implement `predict()` with latent-space prediction
2. Implement `embed()` for embedding extraction
3. Implement `plan()` with multi-step rollout
4. Implement `health_check()` with model state reporting
5. Add preprocessing pipeline (resize, normalize, tensorize)

### Phase 4: Optimization (Week 8-10)

1. Implement batch inference
2. Add GPU memory management and monitoring
3. Implement LRU caching for embeddings and predictions
4. Add model lifecycle (load/warm/hot/unload) with idle timeout
5. Performance benchmarking and optimization
6. Memory profiling and leak detection

---

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_variant_properties() {
        assert_eq!(ModelVariant::Huge.embedding_dim(), 1280);
        assert!(ModelVariant::Huge.file_size_bytes() > 2_000_000_000);
    }

    #[test]
    fn test_quantization_memory_estimates() {
        let fp32 = estimate_gpu_memory(&ModelVariant::Huge, &QuantizationMode::FP32);
        let fp16 = estimate_gpu_memory(&ModelVariant::Huge, &QuantizationMode::FP16);
        let int8 = estimate_gpu_memory(&ModelVariant::Huge, &QuantizationMode::INT8);
        assert!(fp32 > fp16);
        assert!(fp16 > int8);
    }

    #[test]
    fn test_prediction_mask_creation() {
        let mask = PredictionMask::future_frames(4, 16);
        assert_eq!(mask.num_masked(), 4);
        assert_eq!(mask.num_visible(), 12);
    }

    #[tokio::test]
    async fn test_latent_cache_lru_eviction() {
        let cache = LatentCache::new(1); // 1MB budget
        // Fill cache and verify LRU eviction
    }

    #[tokio::test]
    async fn test_model_lifecycle_transitions() {
        // Test state machine transitions
    }
}
```

### Integration Tests (GPU Required)

```rust
#[cfg(feature = "vjepa-integration-tests")]
mod integration {
    #[tokio::test]
    #[ignore = "Requires GPU and model weights"]
    async fn test_real_prediction() {
        let config = VjepaConfig {
            model_variant: ModelVariant::Base,
            quantization: QuantizationMode::FP16,
            ..Default::default()
        };

        let provider = VjepaProvider::new(config).await.unwrap();
        provider.lifecycle.download().await.unwrap();
        provider.lifecycle.warm_up().await.unwrap();
        provider.lifecycle.activate().await.unwrap();

        let input = create_test_world_state_with_video();
        let params = PredictionParams::default();
        let result = provider.predict(&input, &params).await.unwrap();

        assert!(result.latent_representation.is_some());
        let latent = result.latent_representation.unwrap();
        assert_eq!(latent.len(), ModelVariant::Base.embedding_dim());
    }

    #[tokio::test]
    #[ignore = "Requires GPU and model weights"]
    async fn test_batch_embedding() {
        // Test batch embedding extraction
    }

    #[tokio::test]
    #[ignore = "Requires GPU and model weights"]
    async fn test_model_lifecycle_full_cycle() {
        // Test download -> warm -> hot -> infer -> deactivate -> unload
    }
}
```

---

## Open Questions

1. **ONNX Conversion**: The official V-JEPA weights are in PyTorch format.
   Should we ship pre-converted ONNX models, or convert at first load?

2. **Apple Silicon Support**: Should we add a CoreML or Metal backend for
   Mac users? The `candle` crate could be an alternative backend.

3. **Multi-GPU**: Should we support model parallelism across multiple GPUs
   for the Giant variant?

4. **Decoder Integration**: V-JEPA does not produce pixels. Should we bundle
   a separate decoder network that can convert latent predictions back to
   frames for visualization?

5. **Fine-tuning**: Should the provider support fine-tuning V-JEPA on custom
   video datasets? This would require training infrastructure.

6. **Model Updates**: How do we handle model weight updates? Versioned
   downloads with migration support?

7. **CPU-only Viability**: Is CPU-only inference practical for the Huge/Giant
   variants? Should we recommend minimum hardware requirements?

8. **Candle Backend**: Should we also consider the Hugging Face `candle`
   crate as a pure-Rust inference backend, avoiding the C++ dependency of
   libtorch?

9. **Embedding Normalization**: Should embeddings always be L2-normalized,
   or should we preserve the raw activations?

10. **Video Preprocessing**: What frame sampling strategy should we use for
    variable-length input videos? Uniform sampling, keyframe extraction, or
    adaptive sampling?
