use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use tokio::runtime::Runtime;

use worldforge_core::provider::WorldModelProvider;
use worldforge_eval::{EvalReportFormat, EvalSuite};
use worldforge_providers::MockProvider;

fn bench_eval_suite(c: &mut Criterion) {
    let rt = Runtime::new().expect("tokio runtime for eval benches");
    let provider = MockProvider::with_name("mock-eval");
    let providers: [&dyn WorldModelProvider; 1] = [&provider];
    let suite = EvalSuite::physics_standard();
    let report = rt
        .block_on(suite.run(&providers))
        .expect("physics suite report");

    let mut group = c.benchmark_group("evaluation_suite");

    group.bench_function("physics_standard_run", |b| {
        b.iter(|| {
            rt.block_on(suite.run(&providers))
                .expect("physics suite execution")
        })
    });

    group.bench_function("report_to_markdown", |b| {
        b.iter(|| black_box(report.to_markdown().expect("markdown report")))
    });

    group.bench_function("report_to_csv", |b| {
        b.iter(|| black_box(report.to_csv().expect("csv report")))
    });

    group.bench_function("report_render_json", |b| {
        b.iter(|| black_box(report.render(EvalReportFormat::Json).expect("json report")))
    });

    group.finish();
}

criterion_group!(benches, bench_eval_suite);
criterion_main!(benches);
