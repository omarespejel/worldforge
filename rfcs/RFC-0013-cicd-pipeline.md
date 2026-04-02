# RFC-0013: CI/CD & Release Pipeline

| Field   | Value                              |
|---------|------------------------------------|
| Title   | CI/CD & Release Pipeline           |
| Status  | Draft                              |
| Author  | WorldForge Core Team               |
| Created | 2026-04-02                         |
| Updated | 2026-04-02                         |

---

## Abstract

This RFC defines a comprehensive CI/CD and release pipeline for WorldForge,
covering GitHub Actions workflows with multi-platform build matrices, tiered
testing (unit, mock integration, live API), crate publishing to crates.io,
Python wheel builds via maturin, Docker image builds, semantic versioning
policy, changelog generation, security scanning, code coverage, and benchmark
regression detection. The pipeline automates the entire path from commit to
published release across all distribution channels.

---

## Motivation

WorldForge currently has no CI/CD pipeline. This means:

1. **No automated testing**: Tests are run manually by individual developers.
   Regressions are discovered late, often by users. There is no guarantee that
   the main branch builds or passes tests at any given time.

2. **No cross-platform verification**: The project targets Linux, macOS, and
   Windows, but builds are only tested on developers' machines (primarily
   Linux). Platform-specific bugs go undetected.

3. **No published artifacts**: The Rust crates are not published to crates.io,
   the Python wheels are not on PyPI, and there is no Docker image. Users must
   build from source.

4. **No security scanning**: Dependencies are not audited for known
   vulnerabilities. License compliance is not verified.

5. **No performance regression detection**: There are no automated benchmarks.
   Performance regressions are discovered in production.

6. **No release process**: Releases are ad-hoc, with no versioning policy,
   no changelog, and no coordination across the Rust crates and Python package.

This RFC establishes a professional-grade CI/CD pipeline that addresses all
of these gaps.

---

## Detailed Design

### 1. GitHub Actions Workflow Architecture

```
Workflows:
├── ci.yml              # Primary CI (every push/PR)
├── release.yml         # Release pipeline (on version tag)
├── nightly.yml         # Nightly builds & extended tests
├── security.yml        # Security scanning (weekly + on PR)
├── benchmark.yml       # Performance benchmarks (on PR to main)
├── docs.yml            # Documentation build & deploy
└── dependabot.yml      # Dependency update configuration
```

### 2. Primary CI Workflow (ci.yml)

#### 2.1 Build Matrix

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  CARGO_TERM_COLOR: always
  RUSTFLAGS: "-D warnings"

jobs:
  # Tier 1: Fast checks (< 2 min)
  check:
    name: Check & Lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy, rustfmt
      - uses: Swatinem/rust-cache@v2

      - name: Check formatting
        run: cargo fmt --all -- --check

      - name: Clippy lints
        run: cargo clippy --all-targets --all-features -- -D warnings

      - name: Check documentation
        run: cargo doc --no-deps --all-features
        env:
          RUSTDOCFLAGS: "-D warnings"

  # Tier 2: Unit tests across platforms (< 10 min)
  test:
    name: Test (${{ matrix.os }}, ${{ matrix.rust }})
    needs: check
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        rust: [stable, nightly]
        exclude:
          # Only test nightly on Linux (save CI minutes)
          - os: macos-latest
            rust: nightly
          - os: windows-latest
            rust: nightly
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@master
        with:
          toolchain: ${{ matrix.rust }}
      - uses: Swatinem/rust-cache@v2

      - name: Run unit tests
        run: cargo test --workspace --lib --bins

      - name: Run doc tests
        run: cargo test --workspace --doc

  # Tier 3: Integration tests with mock providers (< 15 min)
  integration-mock:
    name: Integration Tests (Mock)
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2

      - name: Run mock integration tests
        run: cargo test --workspace --test '*' --features mock-providers

      - name: Run server integration tests
        run: |
          cargo build --release -p worldforge-server
          ./target/release/worldforge-server &
          SERVER_PID=$!
          sleep 2
          cargo test --package worldforge-server --test api_integration
          kill $SERVER_PID

  # Tier 4: Integration tests with live APIs (manual trigger / nightly)
  integration-live:
    name: Integration Tests (Live API)
    needs: integration-mock
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: live-api-testing
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2

      - name: Run live API tests
        run: cargo test --workspace --test '*' --features live-api-tests
        env:
          COSMOS_API_KEY: ${{ secrets.COSMOS_API_KEY }}
          GENIE2_API_KEY: ${{ secrets.GENIE2_API_KEY }}
          WAYVE_API_KEY: ${{ secrets.WAYVE_API_KEY }}
        timeout-minutes: 30

  # Python SDK tests
  python:
    name: Python SDK (${{ matrix.python-version }})
    needs: check
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11', '3.12', '3.13']
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install maturin
        run: pip install maturin[patchelf] pytest pytest-asyncio

      - name: Build and install Python package
        run: |
          cd crates/worldforge-py
          maturin develop --release

      - name: Run Python tests
        run: |
          cd crates/worldforge-py
          pytest tests/ -v
```

#### 2.2 Test Tiers

| Tier | Name           | Trigger            | Duration | Dependencies          |
|------|----------------|--------------------|---------|-----------------------|
| 1    | Check & Lint   | Every push/PR      | < 2 min | None                  |
| 2    | Unit Tests     | Every push/PR      | < 10 min| Tier 1 passes         |
| 3    | Mock Integration| Every push/PR     | < 15 min| Tier 2 passes         |
| 4    | Live API       | Push to main only  | < 30 min| Tier 3 passes + secrets|

### 3. Crate Publishing (cargo-release)

#### 3.1 Crate Dependency Order

The crates must be published in dependency order:

```
1. worldforge-core          (no internal dependencies)
2. worldforge-verify         (depends on worldforge-core)
3. worldforge-providers      (depends on worldforge-core)
4. worldforge-server         (depends on core, providers, verify)
5. worldforge                (facade crate, depends on all)
```

#### 3.2 cargo-release Configuration

```toml
# Release.toml (workspace root)
[workspace]
allow-branch = ["main"]
sign-commit = true
sign-tag = true
push-remote = "origin"
pre-release-commit-message = "chore: release {{version}}"
tag-message = "Release {{crate_name}} v{{version}}"
tag-prefix = ""
tag-name = "{{crate_name}}-v{{version}}"

# Shared hooks for all crates
pre-release-hook = ["git-cliff", "--output", "CHANGELOG.md", "--tag", "{{version}}"]
```

```toml
# crates/worldforge-core/release.toml
pre-release-replacements = [
    { file = "README.md", search = "worldforge-core = \".*\"", replace = "worldforge-core = \"{{version}}\"" },
]
```

#### 3.3 Release Workflow

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

permissions:
  contents: write

jobs:
  publish-crates:
    name: Publish to crates.io
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: dtolnay/rust-toolchain@stable

      - name: Publish worldforge-core
        run: cargo publish -p worldforge-core
        env:
          CARGO_REGISTRY_TOKEN: ${{ secrets.CRATES_IO_TOKEN }}

      - name: Wait for crates.io index
        run: sleep 30

      - name: Publish worldforge-verify
        run: cargo publish -p worldforge-verify
        env:
          CARGO_REGISTRY_TOKEN: ${{ secrets.CRATES_IO_TOKEN }}

      - name: Wait for crates.io index
        run: sleep 30

      - name: Publish worldforge-providers
        run: cargo publish -p worldforge-providers
        env:
          CARGO_REGISTRY_TOKEN: ${{ secrets.CRATES_IO_TOKEN }}

      - name: Wait for crates.io index
        run: sleep 30

      - name: Publish worldforge-server
        run: cargo publish -p worldforge-server
        env:
          CARGO_REGISTRY_TOKEN: ${{ secrets.CRATES_IO_TOKEN }}

      - name: Wait for crates.io index
        run: sleep 30

      - name: Publish worldforge (facade)
        run: cargo publish -p worldforge
        env:
          CARGO_REGISTRY_TOKEN: ${{ secrets.CRATES_IO_TOKEN }}

  # Generate changelog and create GitHub release
  github-release:
    name: Create GitHub Release
    needs: publish-crates
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Generate changelog
        uses: orhun/git-cliff-action@v3
        with:
          config: cliff.toml
          args: --latest --strip header
        env:
          OUTPUT: CHANGES.md

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v2
        with:
          body_path: CHANGES.md
          draft: false
          prerelease: ${{ contains(github.ref, '-alpha') || contains(github.ref, '-beta') || contains(github.ref, '-rc') }}
```

### 4. Python Wheel Builds (maturin)

#### 4.1 Multi-Platform Wheel Matrix

```yaml
  publish-python:
    name: Python Wheels (${{ matrix.target }})
    needs: publish-crates
    strategy:
      matrix:
        include:
          # Linux x86_64
          - os: ubuntu-latest
            target: x86_64-unknown-linux-gnu
            manylinux: manylinux2014
          # Linux aarch64
          - os: ubuntu-latest
            target: aarch64-unknown-linux-gnu
            manylinux: manylinux2014
          # macOS x86_64
          - os: macos-13
            target: x86_64-apple-darwin
          # macOS aarch64 (Apple Silicon)
          - os: macos-14
            target: aarch64-apple-darwin
          # Windows x64
          - os: windows-latest
            target: x86_64-pc-windows-msvc
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Build wheels
        uses: PyO3/maturin-action@v1
        with:
          target: ${{ matrix.target }}
          args: --release --out dist -m crates/worldforge-py/Cargo.toml
          manylinux: ${{ matrix.manylinux || 'auto' }}

      - name: Upload wheels
        uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.target }}
          path: dist/

  # Build source distribution
  sdist:
    name: Python Source Distribution
    needs: publish-crates
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build sdist
        uses: PyO3/maturin-action@v1
        with:
          command: sdist
          args: --out dist -m crates/worldforge-py/Cargo.toml
      - name: Upload sdist
        uses: actions/upload-artifact@v4
        with:
          name: wheels-sdist
          path: dist/

  # Publish to PyPI
  publish-pypi:
    name: Publish to PyPI
    needs: [publish-python, sdist]
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write  # Trusted publisher
    steps:
      - uses: actions/download-artifact@v4
        with:
          pattern: wheels-*
          merge-multiple: true
          path: dist/

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
```

#### 4.2 macOS Universal2 Build

For macOS, we also produce a universal2 (fat binary) wheel that contains both
x86_64 and aarch64 binaries:

```yaml
  macos-universal:
    name: macOS Universal2
    runs-on: macos-14
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Build universal2 wheel
        uses: PyO3/maturin-action@v1
        with:
          target: universal2-apple-darwin
          args: --release --out dist -m crates/worldforge-py/Cargo.toml

      - name: Upload wheel
        uses: actions/upload-artifact@v4
        with:
          name: wheels-macos-universal2
          path: dist/
```

### 5. Docker Image (Multi-Stage Build)

#### 5.1 Dockerfile

```dockerfile
# Stage 1: Build
FROM rust:1.82-bookworm AS builder

WORKDIR /build

# Cache dependency build
COPY Cargo.toml Cargo.lock ./
COPY crates/worldforge-core/Cargo.toml crates/worldforge-core/
COPY crates/worldforge-providers/Cargo.toml crates/worldforge-providers/
COPY crates/worldforge-server/Cargo.toml crates/worldforge-server/
COPY crates/worldforge-verify/Cargo.toml crates/worldforge-verify/

# Create dummy source files to build dependencies
RUN mkdir -p crates/worldforge-core/src && echo "pub fn dummy() {}" > crates/worldforge-core/src/lib.rs && \
    mkdir -p crates/worldforge-providers/src && echo "pub fn dummy() {}" > crates/worldforge-providers/src/lib.rs && \
    mkdir -p crates/worldforge-server/src && echo "fn main() {}" > crates/worldforge-server/src/main.rs && \
    echo "pub fn dummy() {}" > crates/worldforge-server/src/lib.rs && \
    mkdir -p crates/worldforge-verify/src && echo "pub fn dummy() {}" > crates/worldforge-verify/src/lib.rs

RUN cargo build --release -p worldforge-server 2>/dev/null || true

# Copy real source and build
COPY crates/ crates/
RUN touch crates/*/src/*.rs && \
    cargo build --release -p worldforge-server

# Stage 2: Runtime
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash worldforge

COPY --from=builder /build/target/release/worldforge-server /usr/local/bin/

USER worldforge
WORKDIR /home/worldforge

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8080/v1/health || exit 1

ENTRYPOINT ["worldforge-server"]
CMD ["--bind", "0.0.0.0:8080"]
```

#### 5.2 Docker Build & Push Workflow

```yaml
  docker:
    name: Docker Image
    needs: [publish-crates]
    runs-on: ubuntu-latest
    permissions:
      packages: write
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract version from tag
        id: version
        run: echo "VERSION=${GITHUB_REF#refs/tags/v}" >> $GITHUB_OUTPUT

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          platforms: linux/amd64,linux/arm64
          tags: |
            ghcr.io/worldforge/worldforge-server:${{ steps.version.outputs.VERSION }}
            ghcr.io/worldforge/worldforge-server:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

### 6. Semantic Versioning Policy

#### 6.1 Version Format

All crates follow Semantic Versioning 2.0.0: `MAJOR.MINOR.PATCH[-PRE]`

- **MAJOR**: Breaking changes to public API
- **MINOR**: New features, backward-compatible
- **PATCH**: Bug fixes, backward-compatible
- **PRE**: Pre-release identifiers (`alpha.1`, `beta.1`, `rc.1`)

#### 6.2 Versioning Rules

1. All workspace crates share the same version number (unified versioning)
2. The Python package version matches the Rust crate version
3. The Docker image tag matches the version
4. Pre-release versions are published as pre-release on all channels:
   - crates.io: `0.2.0-alpha.1`
   - PyPI: `0.2.0a1` (PEP 440 format)
   - Docker: `ghcr.io/worldforge/worldforge-server:0.2.0-alpha.1`

#### 6.3 Version Progression

```
0.1.0-alpha.1 -> 0.1.0-alpha.2 -> 0.1.0-beta.1 -> 0.1.0-rc.1 -> 0.1.0
0.1.1 (patch)
0.2.0-alpha.1 -> ... -> 0.2.0
1.0.0-rc.1 -> 1.0.0 (first stable release)
```

#### 6.4 Pre-1.0 Policy

While WorldForge is pre-1.0:
- MINOR version bumps may include breaking changes
- PATCH version bumps are always backward-compatible
- Breaking changes are called out in the changelog with **BREAKING** prefix

### 7. Changelog Generation (git-cliff)

#### 7.1 Conventional Commits

All commits must follow the Conventional Commits specification:

```
feat: add WebSocket streaming for predictions
fix: handle empty response from Cosmos API
perf: cache compiled ONNX circuits for repeated inference
docs: add provider integration guide
refactor!: rename PredictionResult to PredictionResponse

BREAKING CHANGE: PredictionResult has been renamed to PredictionResponse.
Update all usages accordingly.
```

#### 7.2 git-cliff Configuration

See RFC-0012 for the full git-cliff configuration. Key features:

- Commits grouped by type (Features, Bug Fixes, Performance, etc.)
- Breaking changes highlighted with **BREAKING** prefix
- Each commit linked to its GitHub commit URL
- Unreleased changes shown at the top
- Release-specific changelogs for GitHub Releases

#### 7.3 Commit Message Enforcement

```yaml
# .github/workflows/ci.yml (added job)
  commit-lint:
    name: Commit Message Lint
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Check commit messages
        uses: wagoid/commitlint-github-action@v5
        with:
          configFile: .commitlintrc.yml
```

```yaml
# .commitlintrc.yml
extends:
  - '@commitlint/config-conventional'
rules:
  type-enum:
    - 2
    - always
    - [feat, fix, perf, refactor, docs, test, ci, chore, style, build]
  subject-case:
    - 2
    - never
    - [upper-case, pascal-case, start-case]
  body-max-line-length:
    - 0  # Disable body line length limit
```

### 8. Security Scanning

#### 8.1 cargo audit

```yaml
  security:
    name: Security Audit
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install cargo-audit
        run: cargo install cargo-audit

      - name: Run cargo audit
        run: cargo audit --deny warnings

      - name: Install cargo-deny
        run: cargo install cargo-deny

      - name: Run cargo deny (licenses)
        run: cargo deny check licenses

      - name: Run cargo deny (advisories)
        run: cargo deny check advisories

      - name: Run cargo deny (bans)
        run: cargo deny check bans

      - name: Run cargo deny (sources)
        run: cargo deny check sources
```

#### 8.2 cargo deny Configuration

```toml
# deny.toml
[advisories]
vulnerability = "deny"
unmaintained = "warn"
yanked = "deny"
notice = "warn"

[licenses]
unlicensed = "deny"
allow = [
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "Unicode-DFS-2016",
    "Zlib",
]
copyleft = "deny"

[bans]
multiple-versions = "warn"
wildcards = "deny"
deny = [
    # Banned crates (security or quality concerns)
    { name = "openssl", wrappers = ["openssl-sys"] },  # Prefer rustls
]

[sources]
unknown-registry = "deny"
unknown-git = "deny"
allow-registry = ["https://github.com/rust-lang/crates.io-index"]
```

#### 8.3 Scheduled Security Scan

```yaml
name: Security Scan

on:
  schedule:
    - cron: '0 6 * * 1'  # Every Monday at 6 AM UTC
  workflow_dispatch:

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Audit dependencies
        run: |
          cargo install cargo-audit
          cargo audit --json | tee audit-report.json
      - name: Create issue on vulnerability
        if: failure()
        uses: JasonEtco/create-an-issue@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          filename: .github/SECURITY_ISSUE_TEMPLATE.md
```

### 9. Code Coverage (cargo-tarpaulin)

```yaml
  coverage:
    name: Code Coverage
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable

      - name: Install cargo-tarpaulin
        run: cargo install cargo-tarpaulin

      - name: Generate coverage
        run: |
          cargo tarpaulin \
            --workspace \
            --out xml \
            --out html \
            --exclude-files "crates/worldforge-py/*" \
            --timeout 300 \
            --engine llvm

      - name: Upload to Codecov
        uses: codecov/codecov-action@v4
        with:
          file: cobertura.xml
          token: ${{ secrets.CODECOV_TOKEN }}
          fail_ci_if_error: true

      - name: Upload HTML report
        uses: actions/upload-artifact@v4
        with:
          name: coverage-report
          path: tarpaulin-report.html
```

Coverage targets:
- worldforge-core: > 80%
- worldforge-providers: > 70% (limited by external API dependencies)
- worldforge-server: > 75%
- worldforge-verify: > 80%
- Overall workspace: > 75%

### 10. Benchmark Regression Detection

#### 10.1 Benchmark Suite

```rust
// benches/prediction_benchmarks.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn prediction_pipeline_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("prediction_pipeline");

    for size in [1, 4, 16, 64].iter() {
        group.bench_with_input(
            BenchmarkId::new("frames", size),
            size,
            |b, &size| {
                let runtime = tokio::runtime::Runtime::new().unwrap();
                let engine = create_test_engine();
                let request = create_test_request(size);

                b.iter(|| {
                    runtime.block_on(engine.predict(request.clone()))
                });
            },
        );
    }
    group.finish();
}

fn guardrail_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("guardrails");

    group.bench_function("physical_plausibility", |b| {
        let runner = create_test_guardrail_runner();
        let output = create_test_output(16);

        b.iter(|| {
            runner.check_physical_plausibility(&output)
        });
    });

    group.bench_function("all_checks", |b| {
        let runner = create_test_guardrail_runner();
        let output = create_test_output(16);

        b.iter(|| {
            runner.run_all_checks(&output)
        });
    });

    group.finish();
}

criterion_group!(benches, prediction_pipeline_benchmark, guardrail_benchmark);
criterion_main!(benches);
```

#### 10.2 Benchmark CI Workflow

```yaml
  benchmarks:
    name: Benchmark Regression Check
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2

      - name: Run benchmarks
        run: cargo bench --workspace -- --output-format bencher | tee bench-output.txt

      - name: Compare with baseline
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: 'cargo'
          output-file-path: bench-output.txt
          github-token: ${{ secrets.GITHUB_TOKEN }}
          alert-threshold: '120%'  # Alert if 20% slower
          comment-on-alert: true
          fail-on-alert: true
          alert-comment-cc-users: '@worldforge/core-team'
          auto-push: ${{ github.ref == 'refs/heads/main' }}
```

### 11. Dependabot Configuration

```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: cargo
    directory: "/"
    schedule:
      interval: weekly
      day: monday
    open-pull-requests-limit: 10
    reviewers:
      - worldforge/core-team
    labels:
      - dependencies
      - rust
    groups:
      tokio:
        patterns: ["tokio*", "hyper*", "tower*", "axum*"]
      serde:
        patterns: ["serde*"]
      crypto:
        patterns: ["sha2", "blake3", "ed25519*"]

  - package-ecosystem: pip
    directory: "/crates/worldforge-py"
    schedule:
      interval: weekly
    open-pull-requests-limit: 5
    labels:
      - dependencies
      - python

  - package-ecosystem: github-actions
    directory: "/"
    schedule:
      interval: weekly
    open-pull-requests-limit: 5
    labels:
      - dependencies
      - ci
```

### 12. CI Performance Optimization

#### 12.1 Caching Strategy

- **Rust compilation cache**: Swatinem/rust-cache for cargo build artifacts
- **Docker layer cache**: GitHub Actions cache for Docker build layers
- **sccache**: Optional shared compilation cache for larger teams

#### 12.2 Job Dependencies and Parallelism

```
check ──────────► test (3x3 matrix) ──► integration-mock ──► integration-live
   │                                            │
   └──► python (5 versions)                     │
   │                                            │
   └──► security                                │
   │                                            │
   └──► coverage                                │
   │                                            │
   └──► benchmarks (PR only)                    │
```

- Jobs run in parallel where possible
- Expensive jobs (live API, benchmarks) gated behind cheaper ones
- Total CI time target: < 20 minutes for PRs, < 40 minutes for main branch

#### 12.3 CI Cost Management

- Use GitHub-hosted runners (free for public repos, 2,000 min/month for private)
- macOS and Windows jobs only run on the test tier (most expensive)
- Nightly builds use spot/preemptible runners where possible
- Cache aggressively to reduce compilation time
- Cancel in-progress CI runs when new commits are pushed

---

## Implementation Plan

### Phase 1: Core CI (Week 1)

1. Create `.github/workflows/ci.yml` with check + unit test jobs
2. Configure rust-cache for fast builds
3. Add clippy and rustfmt enforcement
4. Set up commit message linting
5. Add branch protection rules on `main`

### Phase 2: Extended CI (Week 2)

6. Add cross-platform test matrix (macOS, Windows)
7. Add Rust nightly testing
8. Add mock integration test job
9. Add Python SDK test job
10. Configure code coverage with cargo-tarpaulin + Codecov

### Phase 3: Security & Quality (Week 3)

11. Configure cargo-audit and cargo-deny
12. Set up weekly security scanning
13. Add benchmark suite and regression detection
14. Configure Dependabot
15. Add documentation build verification

### Phase 4: Release Pipeline (Week 4)

16. Configure cargo-release for workspace
17. Set up crate publishing workflow
18. Set up Python wheel build matrix (maturin)
19. Configure PyPI publishing (trusted publisher)
20. Set up Docker build and push to GHCR

### Phase 5: Polish (Week 5)

21. Configure git-cliff for changelog generation
22. Set up GitHub Release creation
23. Add live API integration test environment
24. Optimize CI performance (caching, parallelism)
25. Document CI/CD pipeline for contributors
26. Create release runbook

---

## Testing Strategy

### CI Pipeline Tests

- Verify CI workflows run correctly on a test repository
- Test all matrix combinations (OS × Rust version)
- Verify caching works correctly (second run faster than first)
- Test failure modes (failing test, failing lint, security advisory)

### Release Pipeline Tests

- Dry-run crate publishing (`cargo publish --dry-run`)
- Test Python wheel installation on each platform
- Test Docker image builds and runs correctly
- Verify changelog generation produces correct output
- Verify version numbers are consistent across all artifacts

### Security Scanning Tests

- Introduce a known-vulnerable dependency and verify CI catches it
- Test license compliance with a copyleft dependency
- Verify Dependabot PRs are created for outdated dependencies

### Performance Regression Tests

- Introduce a deliberate performance regression and verify CI detects it
- Verify benchmark baselines are updated on main branch merges
- Test alerting for significant regressions (> 20% slower)

---

## Open Questions

1. **Self-Hosted Runners**: Should we use self-hosted runners for GPU-dependent
   tests (EZKL proof generation, CUDA-accelerated inference)? This adds
   maintenance burden but enables testing that's impossible on GitHub-hosted
   runners.

2. **Release Cadence**: Should we release on a fixed schedule (e.g., every 2
   weeks) or on-demand when features are ready? Fixed cadence is more
   predictable but may lead to empty releases.

3. **Monorepo vs. Poly-repo CI**: The current monorepo structure means every
   PR triggers CI for all crates. Should we add path-based filtering to only
   test affected crates, or is the full-workspace test worth the CI cost?

4. **Pre-release Channels**: Should we publish nightly pre-release versions
   to crates.io and PyPI automatically, or only on manual trigger?

5. **MSRV Policy**: What is the Minimum Supported Rust Version? Should we
   test against it in CI? Options: stable-2 (conservative), stable-1
   (reasonable), stable-only (aggressive).

6. **Code Signing**: Should release artifacts (crates, wheels, Docker images)
   be cryptographically signed? This adds trust but complicates the release
   process.

7. **Feature Flags in CI**: Some features (live-api-tests, gpu-acceleration)
   are behind feature flags. Should CI test all feature flag combinations,
   or just the default and all-features?

8. **Flaky Test Policy**: How should we handle flaky tests? Options: automatic
   retry (masks real issues), quarantine (removes from CI), fix immediately
   (blocks all PRs until fixed).
