use criterion::{black_box, criterion_group, criterion_main, Criterion};

use worldforge_core::state::WorldState;
use worldforge_verify::{sha256_hash, state_hash};

fn bench_sha256_hash(c: &mut Criterion) {
    let payload = vec![0x5au8; 4096];
    c.bench_function("sha256_hash_4k", |b| {
        b.iter(|| sha256_hash(black_box(&payload)))
    });
}

fn bench_state_hash(c: &mut Criterion) {
    let state = WorldState::new("bench-world", "mock");
    c.bench_function("state_hash_empty_world", |b| {
        b.iter(|| state_hash(black_box(&state)).expect("state hash should serialize"))
    });
}

criterion_group!(proof_benches, bench_sha256_hash, bench_state_hash);
criterion_main!(proof_benches);
