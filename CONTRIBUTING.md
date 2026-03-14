# Contributing to WorldForge

Thank you for your interest in contributing to WorldForge!

## Development Setup

1. Install Rust (1.80+): https://rustup.rs/
2. Clone the repository: `git clone https://github.com/AbdelStark/worldforge`
3. Build: `cargo build`
4. Test: `cargo test`
5. Lint: `cargo clippy`
6. Format: `cargo fmt`

## Project Structure

- `crates/worldforge-core/` - Core library (types, traits, state management)
- `crates/worldforge-providers/` - Provider adapters (Cosmos, GWM, JEPA, etc.)
- `crates/worldforge-eval/` - Evaluation framework
- `crates/worldforge-verify/` - ZK verification (optional)
- `crates/worldforge-server/` - REST API server
- `crates/worldforge-cli/` - Command-line tool

## How to Contribute

1. Check open issues for "good first issue" or "help wanted" labels
2. Fork the repository
3. Create a feature branch: `git checkout -b feature/my-feature`
4. Write code + tests
5. Run `cargo test && cargo clippy && cargo fmt`
6. Submit a pull request

## Adding a New Provider

1. Create a new module in `crates/worldforge-providers/src/`
2. Implement the `WorldModelProvider` trait from `worldforge-core`
3. Add integration tests
4. Add documentation with examples
5. Update README with provider capabilities

## Code Style

- Follow Rust conventions (rustfmt enforced)
- All public types must have doc comments
- All functions must have error handling (no unwrap in library code)
- Property-based tests for core types (proptest)
- Integration tests for provider adapters
