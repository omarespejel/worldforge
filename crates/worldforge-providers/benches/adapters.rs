use std::fs;
use std::hint::black_box;
use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, Criterion};
use tokio::runtime::Runtime;

use worldforge_core::action::Action;
use worldforge_core::prediction::{PlanGoal, PlanRequest, PlannerType, PredictionConfig};
use worldforge_core::provider::{
    EmbeddingInput, GenerationConfig, GenerationPrompt, Operation, ReasoningInput, SpatialControls,
    TransferConfig, WorldModelProvider,
};
use worldforge_core::scene::SceneObject;
use worldforge_core::state::WorldState;
use worldforge_core::types::{
    BBox, CameraPose, DType, Device, Frame, Pose, Position, Rotation, SimTime, Tensor, TensorData,
    Trajectory, Vec3, VideoClip,
};
use worldforge_providers::{
    cosmos::{CosmosEndpoint, CosmosModel},
    genie::GenieModel,
    CosmosProvider, GenieProvider, JepaBackend, JepaProvider, MockProvider, RunwayProvider,
};

fn bench_provider_operations(c: &mut Criterion) {
    let rt = Runtime::new().expect("tokio runtime for provider benches");
    let base_state = sample_world_state("bench-provider");
    let action = sample_action();
    let prediction_config = sample_prediction_config();
    let generation_prompt = sample_generation_prompt();
    let generation_config = sample_generation_config();
    let reasoning_input = ReasoningInput {
        video: None,
        state: Some(base_state.clone()),
    };
    let embed_input = EmbeddingInput::from_text("bench embedding input");
    let source_clip = sample_video_clip();
    let transfer_controls = sample_transfer_controls();
    let transfer_config = sample_transfer_config();
    let plan_request = sample_plan_request(&base_state);

    let mut mock = MockProvider::with_name("mock-bench");
    mock.latency_ms = 0;

    let genie = GenieProvider::new(GenieModel::Genie3, "");
    let jepa_root = create_jepa_fixture();
    let jepa = JepaProvider::new(&jepa_root, JepaBackend::Safetensors);
    let cosmos = CosmosProvider::new(
        CosmosModel::Predict2_5,
        "bench-key",
        CosmosEndpoint::NimApi("https://example.invalid".to_string()),
    );
    let runway =
        RunwayProvider::full_stack_with_endpoint("bench-secret", "https://example.invalid");

    let mut group = c.benchmark_group("provider_offline_operations");

    group.bench_function("mock_predict", |b| {
        b.iter(|| {
            rt.block_on(mock.predict(
                black_box(&base_state),
                black_box(&action),
                black_box(&prediction_config),
            ))
            .expect("mock prediction")
        })
    });
    group.bench_function("mock_generate", |b| {
        b.iter(|| {
            rt.block_on(mock.generate(black_box(&generation_prompt), black_box(&generation_config)))
                .expect("mock generation")
        })
    });
    group.bench_function("mock_reason", |b| {
        b.iter(|| {
            rt.block_on(mock.reason(black_box(&reasoning_input), "what is here?"))
                .expect("mock reasoning")
        })
    });
    group.bench_function("mock_embed", |b| {
        b.iter(|| {
            rt.block_on(mock.embed(black_box(&embed_input)))
                .expect("mock embedding")
        })
    });
    group.bench_function("mock_transfer", |b| {
        b.iter(|| {
            rt.block_on(mock.transfer(
                black_box(&source_clip),
                black_box(&transfer_controls),
                black_box(&transfer_config),
            ))
            .expect("mock transfer")
        })
    });
    group.bench_function("mock_plan", |b| {
        b.iter(|| {
            rt.block_on(mock.plan(black_box(&plan_request)))
                .expect("mock planning")
        })
    });
    group.bench_function("mock_health", |b| {
        b.iter(|| rt.block_on(mock.health_check()).expect("mock health"))
    });

    group.bench_function("genie_predict", |b| {
        b.iter(|| {
            rt.block_on(genie.predict(
                black_box(&base_state),
                black_box(&action),
                black_box(&prediction_config),
            ))
            .expect("genie prediction")
        })
    });
    group.bench_function("genie_generate", |b| {
        b.iter(|| {
            rt.block_on(
                genie.generate(black_box(&generation_prompt), black_box(&generation_config)),
            )
            .expect("genie generation")
        })
    });
    group.bench_function("genie_reason", |b| {
        b.iter(|| {
            rt.block_on(genie.reason(black_box(&reasoning_input), "what is here?"))
                .expect("genie reasoning")
        })
    });
    group.bench_function("genie_transfer", |b| {
        b.iter(|| {
            rt.block_on(genie.transfer(
                black_box(&source_clip),
                black_box(&transfer_controls),
                black_box(&transfer_config),
            ))
            .expect("genie transfer")
        })
    });
    group.bench_function("genie_plan", |b| {
        b.iter(|| {
            rt.block_on(genie.plan(black_box(&plan_request)))
                .expect("genie planning")
        })
    });
    group.bench_function("genie_health", |b| {
        b.iter(|| rt.block_on(genie.health_check()).expect("genie health"))
    });

    group.bench_function("jepa_predict", |b| {
        b.iter(|| {
            rt.block_on(jepa.predict(
                black_box(&base_state),
                black_box(&action),
                black_box(&prediction_config),
            ))
            .expect("jepa prediction")
        })
    });
    group.bench_function("jepa_plan", |b| {
        b.iter(|| {
            rt.block_on(jepa.plan(black_box(&plan_request)))
                .expect("jepa planning")
        })
    });
    group.bench_function("jepa_health", |b| {
        b.iter(|| rt.block_on(jepa.health_check()).expect("jepa health"))
    });

    group.bench_function("cosmos_estimate_cost", |b| {
        let operation = Operation::Predict {
            steps: 4,
            resolution: (640, 360),
        };
        b.iter(|| black_box(cosmos.estimate_cost(black_box(&operation))))
    });
    group.bench_function("runway_estimate_cost", |b| {
        let operation = Operation::Transfer {
            duration_seconds: 3.0,
        };
        b.iter(|| black_box(runway.estimate_cost(black_box(&operation))))
    });
    group.bench_function("genie_estimate_cost", |b| {
        let operation = Operation::Generate {
            duration_seconds: 3.0,
            resolution: (1280, 720),
        };
        b.iter(|| black_box(genie.estimate_cost(black_box(&operation))))
    });
    group.bench_function("jepa_estimate_cost", |b| {
        let operation = Operation::Predict {
            steps: 4,
            resolution: (224, 224),
        };
        b.iter(|| black_box(jepa.estimate_cost(black_box(&operation))))
    });

    group.finish();
}

fn sample_world_state(provider: &str) -> WorldState {
    let mut state = WorldState::new("bench-scene", provider);
    let support = SceneObject::new(
        "table",
        Pose {
            position: Position {
                x: 0.0,
                y: 0.75,
                z: 0.0,
            },
            rotation: Rotation::default(),
        },
        BBox::from_center_half_extents(
            Position {
                x: 0.0,
                y: 0.75,
                z: 0.0,
            },
            Vec3 {
                x: 0.8,
                y: 0.05,
                z: 0.6,
            },
        ),
    );
    let mut mug = SceneObject::new(
        "mug",
        Pose {
            position: Position {
                x: -0.15,
                y: 0.84,
                z: 0.0,
            },
            rotation: Rotation::default(),
        },
        BBox::from_center_half_extents(
            Position {
                x: -0.15,
                y: 0.84,
                z: 0.0,
            },
            Vec3 {
                x: 0.05,
                y: 0.08,
                z: 0.05,
            },
        ),
    );
    mug.semantic_label = Some("mug".to_string());
    mug.physics.is_graspable = true;

    state.scene.add_object(support);
    state.scene.add_object(mug);
    state
        .ensure_history_initialized(provider)
        .expect("seed history");
    state
}

fn sample_action() -> Action {
    Action::Move {
        target: Position {
            x: 0.2,
            y: 0.84,
            z: 0.0,
        },
        speed: 0.5,
    }
}

fn sample_prediction_config() -> PredictionConfig {
    PredictionConfig {
        steps: 3,
        resolution: (128, 128),
        fps: 12.0,
        return_video: true,
        return_depth: true,
        return_segmentation: true,
        guardrails: Vec::new(),
        max_latency_ms: None,
        fallback_provider: None,
        num_samples: 4,
        temperature: 0.9,
    }
}

fn sample_generation_prompt() -> GenerationPrompt {
    GenerationPrompt {
        text: "A mug on a wooden table".to_string(),
        reference_image: None,
        negative_prompt: Some("blurry".to_string()),
    }
}

fn sample_generation_config() -> GenerationConfig {
    GenerationConfig {
        resolution: (320, 180),
        fps: 8.0,
        duration_seconds: 2.0,
        temperature: 0.7,
        seed: Some(42),
    }
}

fn sample_video_clip() -> VideoClip {
    let frame_tensor = Tensor {
        data: TensorData::Float32(vec![0.0; 12]),
        shape: vec![2, 2, 3],
        dtype: DType::Float32,
        device: Device::Cpu,
    };

    let frames = (0..4)
        .map(|index| Frame {
            data: frame_tensor.clone(),
            timestamp: SimTime {
                step: index,
                seconds: index as f64 * 0.25,
                dt: 0.25,
            },
            camera: Some(CameraPose {
                extrinsics: Pose::default(),
                fov: 60.0,
                near_clip: 0.1,
                far_clip: 10.0,
            }),
            depth: None,
            segmentation: None,
        })
        .collect();

    VideoClip {
        frames,
        fps: 8.0,
        resolution: (2, 2),
        duration: 0.5,
    }
}

fn sample_transfer_controls() -> SpatialControls {
    SpatialControls {
        camera_trajectory: Some(Trajectory {
            poses: vec![
                (
                    SimTime {
                        step: 0,
                        seconds: 0.0,
                        dt: 0.0,
                    },
                    Pose::default(),
                ),
                (
                    SimTime {
                        step: 1,
                        seconds: 0.25,
                        dt: 0.25,
                    },
                    Pose {
                        position: Position {
                            x: 0.1,
                            y: 0.9,
                            z: 0.2,
                        },
                        rotation: Rotation::default(),
                    },
                ),
            ],
            velocities: None,
        }),
        depth_map: Some(Tensor {
            data: TensorData::Float32(vec![0.0; 4]),
            shape: vec![2, 2],
            dtype: DType::Float32,
            device: Device::Cpu,
        }),
        segmentation_map: None,
    }
}

fn sample_transfer_config() -> TransferConfig {
    TransferConfig {
        resolution: (320, 180),
        fps: 8.0,
        control_strength: 0.8,
    }
}

fn sample_plan_request(state: &WorldState) -> PlanRequest {
    let current_state = state.clone();
    let mut target_state = state.clone();
    let mug_id = current_state
        .scene
        .find_object_by_name("mug")
        .expect("mug object")
        .id;

    target_state
        .scene
        .get_object_mut(&mug_id)
        .expect("mug target")
        .translate_by(Vec3 {
            x: 0.24,
            y: 0.0,
            z: 0.0,
        });

    PlanRequest {
        current_state,
        goal: PlanGoal::TargetState(Box::new(target_state)),
        max_steps: 4,
        guardrails: Vec::new(),
        planner: PlannerType::Sampling {
            num_samples: 12,
            top_k: 3,
        },
        timeout_seconds: 5.0,
        fallback_provider: None,
        return_video: false,
        return_depth: false,
        return_segmentation: false,
    }
}

fn create_jepa_fixture() -> PathBuf {
    let root = std::env::temp_dir().join(format!("worldforge-jepa-bench-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&root).expect("create jepa fixture root");
    fs::write(
        root.join("worldforge-jepa.json"),
        r#"{
  "model_name": "bench-jepa",
  "representation_dim": 256,
  "action_gain": 1.2,
  "temporal_smoothness": 0.88,
  "gravity_bias": 0.92,
  "collision_bias": 0.91,
  "confidence_bias": 0.03
}"#,
    )
    .expect("write jepa manifest");
    fs::write(
        root.join("weights.safetensors"),
        b"worldforge-jepa-benchmark-weights",
    )
    .expect("write jepa weights");
    root
}

criterion_group!(benches, bench_provider_operations);
criterion_main!(benches);
