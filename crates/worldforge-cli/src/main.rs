//! WorldForge CLI entry point.

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    worldforge_cli::run().await
}
