//! Binary entrypoint for the WorldForge REST API server.

use std::sync::Arc;

use anyhow::{bail, Result};

use worldforge_server::{serve, ServerConfig};

const USAGE: &str = "\
WorldForge REST API server

Usage:
  worldforge-server [--bind <addr>] [--state-dir <path>] [--state-backend <file|sqlite|redis|s3>] [--state-file-format <json|msgpack>] [--state-db-path <path>] [--state-redis-url <url>] [--state-s3-bucket <name>] [--state-s3-region <region>] [--state-s3-access-key-id <id>] [--state-s3-secret-access-key <secret>] [--state-s3-endpoint <url>] [--state-s3-session-token <token>] [--state-s3-prefix <path>]

Options:
  --bind <addr>             Address to bind to (default: 127.0.0.1:8080)
  --state-dir <path>        Directory for file-backed state or default SQLite location (default: .worldforge)
  --state-backend <kind>    Persistence backend: file, sqlite, redis, or s3 (default: file)
  --state-file-format <fmt> File-store serialization format: json or msgpack (default: json)
  --state-db-path <path>    SQLite database path override
  --state-redis-url <url>   Redis connection URL override
  --state-s3-bucket <name>  S3 bucket override
  --state-s3-region <name>  S3 region override (or AWS_REGION / AWS_DEFAULT_REGION)
  --state-s3-access-key-id <id>
                           S3 access key override (or AWS_ACCESS_KEY_ID)
  --state-s3-secret-access-key <secret>
                           S3 secret override (or AWS_SECRET_ACCESS_KEY)
  --state-s3-endpoint <url> S3-compatible endpoint override
  --state-s3-session-token <token>
                           S3 session token override (or AWS_SESSION_TOKEN)
  --state-s3-prefix <path>  S3 object-key prefix override
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
            "--state-file-format" => {
                config.state_file_format = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("missing value for --state-file-format"))?;
            }
            "--state-db-path" => {
                config.state_db_path = Some(
                    args.next()
                        .ok_or_else(|| anyhow::anyhow!("missing value for --state-db-path"))?,
                );
            }
            "--state-redis-url" => {
                config.state_redis_url = Some(
                    args.next()
                        .ok_or_else(|| anyhow::anyhow!("missing value for --state-redis-url"))?,
                );
            }
            "--state-s3-bucket" => {
                config.state_s3_bucket = Some(
                    args.next()
                        .ok_or_else(|| anyhow::anyhow!("missing value for --state-s3-bucket"))?,
                );
            }
            "--state-s3-region" => {
                config.state_s3_region = Some(
                    args.next()
                        .ok_or_else(|| anyhow::anyhow!("missing value for --state-s3-region"))?,
                );
            }
            "--state-s3-access-key-id" => {
                config.state_s3_access_key_id = Some(args.next().ok_or_else(|| {
                    anyhow::anyhow!("missing value for --state-s3-access-key-id")
                })?);
            }
            "--state-s3-secret-access-key" => {
                config.state_s3_secret_access_key = Some(args.next().ok_or_else(|| {
                    anyhow::anyhow!("missing value for --state-s3-secret-access-key")
                })?);
            }
            "--state-s3-endpoint" => {
                config.state_s3_endpoint = Some(
                    args.next()
                        .ok_or_else(|| anyhow::anyhow!("missing value for --state-s3-endpoint"))?,
                );
            }
            "--state-s3-session-token" => {
                config.state_s3_session_token = Some(args.next().ok_or_else(|| {
                    anyhow::anyhow!("missing value for --state-s3-session-token")
                })?);
            }
            "--state-s3-prefix" => {
                config.state_s3_prefix = Some(
                    args.next()
                        .ok_or_else(|| anyhow::anyhow!("missing value for --state-s3-prefix"))?,
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
        assert_eq!(config.state_file_format, "json");
        assert_eq!(config.state_db_path, None);
        assert_eq!(config.state_redis_url, None);
        assert_eq!(config.state_s3_bucket, None);
        assert_eq!(config.state_s3_region, None);
        assert_eq!(config.state_s3_access_key_id, None);
        assert_eq!(config.state_s3_secret_access_key, None);
        assert_eq!(config.state_s3_endpoint, None);
        assert_eq!(config.state_s3_session_token, None);
        assert_eq!(config.state_s3_prefix, None);
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
            "--state-file-format".to_string(),
            "msgpack".to_string(),
            "--state-db-path".to_string(),
            "/tmp/worldforge/state.db".to_string(),
        ])
        .unwrap();

        assert_eq!(config.bind_address, "127.0.0.1:9001");
        assert_eq!(config.state_dir, "/tmp/worldforge");
        assert_eq!(config.state_backend, "sqlite");
        assert_eq!(config.state_file_format, "msgpack");
        assert_eq!(
            config.state_db_path.as_deref(),
            Some("/tmp/worldforge/state.db")
        );
        assert_eq!(config.state_redis_url, None);
    }

    #[test]
    fn parse_args_supports_redis_url() {
        let config = parse_args([
            "--state-backend".to_string(),
            "redis".to_string(),
            "--state-redis-url".to_string(),
            "redis://127.0.0.1:6379/3".to_string(),
        ])
        .unwrap();

        assert_eq!(config.state_backend, "redis");
        assert_eq!(
            config.state_redis_url.as_deref(),
            Some("redis://127.0.0.1:6379/3")
        );
    }

    #[test]
    fn parse_args_supports_s3_overrides() {
        let config = parse_args([
            "--state-backend".to_string(),
            "s3".to_string(),
            "--state-s3-bucket".to_string(),
            "worldforge-states".to_string(),
            "--state-s3-region".to_string(),
            "us-east-1".to_string(),
            "--state-s3-access-key-id".to_string(),
            "test-access".to_string(),
            "--state-s3-secret-access-key".to_string(),
            "test-secret".to_string(),
            "--state-s3-endpoint".to_string(),
            "http://127.0.0.1:9000".to_string(),
            "--state-s3-session-token".to_string(),
            "test-session".to_string(),
            "--state-s3-prefix".to_string(),
            "snapshots".to_string(),
        ])
        .unwrap();

        assert_eq!(config.state_backend, "s3");
        assert_eq!(config.state_s3_bucket.as_deref(), Some("worldforge-states"));
        assert_eq!(config.state_s3_region.as_deref(), Some("us-east-1"));
        assert_eq!(
            config.state_s3_access_key_id.as_deref(),
            Some("test-access")
        );
        assert_eq!(
            config.state_s3_secret_access_key.as_deref(),
            Some("test-secret")
        );
        assert_eq!(
            config.state_s3_endpoint.as_deref(),
            Some("http://127.0.0.1:9000")
        );
        assert_eq!(
            config.state_s3_session_token.as_deref(),
            Some("test-session")
        );
        assert_eq!(config.state_s3_prefix.as_deref(), Some("snapshots"));
    }

    #[test]
    fn parse_args_rejects_unknown_flags() {
        let error = parse_args(["--bogus".to_string()]).unwrap_err();
        assert!(error.to_string().contains("unknown argument"));
    }
}
