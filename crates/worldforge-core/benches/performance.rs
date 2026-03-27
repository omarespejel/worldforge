use std::sync::Arc;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use worldforge_core::action::Action;
use worldforge_core::guardrail::{default_guardrails, evaluate_guardrails};
use worldforge_core::prediction::{PlanGoal, PlanRequest, PlannerType, PredictionConfig};
use worldforge_core::provider::ProviderRegistry;
use worldforge_core::state::WorldState;
use worldforge_core::state::{deserialize_world_state, serialize_world_state, StateFileFormat};
use worldforge_core::types::{BBox, Pose, Position, Rotation, Vec3};
use worldforge_core::world::World;
use worldforge_providers::MockProvider;

fn rich_world_state() -> WorldState {
    let mut state = WorldState::from_prompt(
        "A kitchen counter with a red mug, a blue block, a metal spoon, and a human nearby",
        "bench",
        Some("benchmark-kitchen"),
    )
    .expect("benchmark fixture should build");

    let human_pose = Pose {
        position: Position {
            x: 0.24,
            y: 1.68,
            z: 0.12,
        },
        rotation: Rotation::default(),
    };
    let human_bbox = BBox::from_center_half_extents(
        human_pose.position,
        Vec3 {
            x: 0.18,
            y: 0.48,
            z: 0.18,
        },
    );
    let mut human = worldforge_core::scene::SceneObject::new("human", human_pose, human_bbox);
    human.semantic_label = Some("human".to_string());
    human.physics.mass = Some(70.0);
    human.physics.is_static = false;
    human.physics.material = Some("organic".to_string());
    state.scene.add_object(human);

    state
}

fn rich_world() -> World {
    let state = rich_world_state();
    let mut provider = MockProvider::new();
    provider.latency_ms = 0;

    let mut registry = ProviderRegistry::new();
    registry.register(Box::new(provider));

    World::new(state, "mock", Arc::new(registry))
}

fn predict_action() -> Action {
    Action::Move {
        target: Position {
            x: 0.18,
            y: 0.82,
            z: 0.05,
        },
        speed: 0.25,
    }
}

fn plan_request() -> PlanRequest {
    PlanRequest {
        current_state: rich_world_state(),
        goal: PlanGoal::Description("move the red mug next to the blue block".to_string()),
        max_steps: 4,
        guardrails: default_guardrails(),
        planner: PlannerType::Sampling {
            num_samples: 12,
            top_k: 3,
        },
        timeout_seconds: 5.0,
    }
}

fn benchmark_bootstrap(c: &mut Criterion) {
    c.bench_function("core/bootstrap_from_prompt", |b| {
        b.iter(|| {
            let state = WorldState::from_prompt(
                "A workbench with a red mug, a blue block, a wrench, and a human nearby",
                "bench",
                Some("bootstrap-bench"),
            )
            .expect("fixture should be valid");
            std::hint::black_box(state);
        })
    });
}

fn benchmark_state_roundtrip(c: &mut Criterion) {
    let state = rich_world_state();

    c.bench_function("core/state_roundtrip_json", |b| {
        b.iter(|| {
            let bytes = serialize_world_state(StateFileFormat::Json, &state)
                .expect("json serialization should succeed");
            let restored = deserialize_world_state(StateFileFormat::Json, &bytes)
                .expect("json deserialization should succeed");
            std::hint::black_box(restored);
        })
    });

    c.bench_function("core/state_roundtrip_msgpack", |b| {
        b.iter(|| {
            let bytes = serialize_world_state(StateFileFormat::MessagePack, &state)
                .expect("msgpack serialization should succeed");
            let restored = deserialize_world_state(StateFileFormat::MessagePack, &bytes)
                .expect("msgpack deserialization should succeed");
            std::hint::black_box(restored);
        })
    });
}

fn benchmark_guardrails(c: &mut Criterion) {
    let state = rich_world_state();
    let guardrails = default_guardrails();

    c.bench_function("core/guardrail_evaluation", |b| {
        b.iter(|| {
            let results = evaluate_guardrails(&guardrails, &state);
            std::hint::black_box(results);
        })
    });
}

fn benchmark_predict(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().expect("runtime should build");
    let config = PredictionConfig {
        steps: 3,
        ..PredictionConfig::default()
    };
    let action = predict_action();

    c.bench_function("core/world_predict_mock", |b| {
        b.iter_batched(
            rich_world,
            |mut world| {
                let prediction = rt
                    .block_on(async { world.predict(&action, &config).await })
                    .expect("prediction should succeed");
                std::hint::black_box(prediction);
            },
            BatchSize::SmallInput,
        )
    });
}

fn benchmark_plan(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().expect("runtime should build");
    let request = plan_request();

    c.bench_function("core/world_plan_sampling_mock", |b| {
        b.iter(|| {
            let world = rich_world();
            let plan = rt
                .block_on(async { world.plan(&request).await })
                .expect("planning should succeed");
            std::hint::black_box(plan);
        })
    });

    let native_request = PlanRequest {
        planner: PlannerType::ProviderNative,
        ..plan_request()
    };

    c.bench_function("core/world_plan_native_mock", |b| {
        b.iter(|| {
            let world = rich_world();
            let plan = rt
                .block_on(async { world.plan(&native_request).await })
                .expect("native planning should succeed");
            std::hint::black_box(plan);
        })
    });
}

criterion_group!(
    benches,
    benchmark_bootstrap,
    benchmark_state_roundtrip,
    benchmark_guardrails,
    benchmark_predict,
    benchmark_plan
);
criterion_main!(benches);
