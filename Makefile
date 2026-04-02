.PHONY: fmt lint test test-all bench doc build-python clean check audit deny

# Format all Rust code
fmt:
	cargo fmt --all

# Run clippy lints
lint:
	cargo clippy --workspace --all-targets -- -D warnings

# Run tests (default features)
test:
	cargo test --workspace

# Run all tests including ignored / long-running
test-all:
	cargo test --workspace -- --include-ignored

# Run benchmarks
bench:
	cargo bench --workspace

# Build documentation
doc:
	cargo doc --workspace --no-deps --open

# Build Python wheel (dev mode)
build-python:
	maturin develop --release

# Full CI check (mirrors the CI pipeline)
check: fmt lint test doc

# Security: cargo audit
audit:
	cargo audit

# Security: cargo deny
deny:
	cargo deny check

# Clean build artifacts
clean:
	cargo clean
	rm -rf dist/ target/wheels/
