//! Binary entrypoint for the WorldForge REST API server.

use std::sync::Arc;

use anyhow::{bail, Result};

use worldforge_server::{serve, ServerConfig};

const USAGE: &str = "\
WorldForge REST API server

Usage:
  worldforge-server [--bind <addr>] [--state-dir <path>] [--state-backend <file|sqlite>] [--state-db-path <path>]

Options:
  --bind <addr>             Address to bind to (default: 127.0.0.1:8080)
  --state-dir <path>        Directory for file-backed state or default SQLite location (default: .worldforge)
  --state-backend <kind>    Persistence backend: file or sqlite (default: file)
  --state-db-path <path>    SQLite database path override
  -h, --help                Show this help text";

fn parse_args<I>(args: I) -> Result<ServerConfig>
where
    I: IntoIterator<Item = String>,
{
    let mut config = ServerConfig::default();
    let mut args = args.into_iter();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bind" => {
                config.bind_address = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("missing value for --bind"))?;
            }
            "--state-dir" => {
                config.state_dir = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("missing value for --state-dir"))?;
            }
            "--state-backend" => {
                config.state_backend = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("missing value for --state-backend"))?;
            }
            "--state-db-path" => {
                config.state_db_path = Some(
                    args.next()
                        .ok_or_else(|| anyhow::anyhow!("missing value for --state-db-path"))?,
                );
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => bail!("unknown argument: {other}\n\n{USAGE}"),
        }
    }

    Ok(config)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().init();

    let config = parse_args(std::env::args().skip(1))?;
    let registry = Arc::new(worldforge_providers::auto_detect());

    serve(config, registry).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_args_uses_defaults() {
        let config = parse_args(Vec::<String>::new()).unwrap();
        assert_eq!(config.bind_address, "127.0.0.1:8080");
        assert_eq!(config.state_dir, ".worldforge");
        assert_eq!(config.state_backend, "file");
        assert_eq!(config.state_db_path, None);
    }

    #[test]
    fn parse_args_supports_overrides() {
        let config = parse_args([
            "--bind".to_string(),
            "127.0.0.1:9001".to_string(),
            "--state-dir".to_string(),
            "/tmp/worldforge".to_string(),
            "--state-backend".to_string(),
            "sqlite".to_string(),
            "--state-db-path".to_string(),
            "/tmp/worldforge/state.db".to_string(),
        ])
        .unwrap();

        assert_eq!(config.bind_address, "127.0.0.1:9001");
        assert_eq!(config.state_dir, "/tmp/worldforge");
        assert_eq!(config.state_backend, "sqlite");
        assert_eq!(
            config.state_db_path.as_deref(),
            Some("/tmp/worldforge/state.db")
        );
    }

    #[test]
    fn parse_args_rejects_unknown_flags() {
        let error = parse_args(["--bogus".to_string()]).unwrap_err();
        assert!(error.to_string().contains("unknown argument"));
    }
}
