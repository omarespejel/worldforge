# RFC-0019: Security & Authentication

| Field   | Value                          |
|---------|--------------------------------|
| Status  | Draft                          |
| Author  | WorldForge Core Team           |
| Created | 2026-04-02                     |
| Updated | 2026-04-02                     |

## Abstract

This RFC introduces a comprehensive security and authentication layer for
WorldForge. The system provides API key authentication for REST endpoints, JWT
tokens for service-to-service communication, role-based access control (RBAC)
with three roles (admin, developer, viewer), secure provider credential storage
with rotation support, TLS configuration, input validation, audit logging,
security headers, and dependency vulnerability scanning. The goal is to bring
WorldForge from its current unauthenticated state to production-grade security.

## Motivation

The WorldForge server currently has no authentication or authorization. Any
client can access any endpoint, create or delete worlds, and invoke provider
APIs with the server's credentials. This is unacceptable for any deployment
beyond local development:

- **No access control**: Any network-accessible instance is fully open.
- **No credential isolation**: Provider API keys (OpenAI, Anthropic, etc.) are
  stored in plain environment variables with no rotation mechanism.
- **No audit trail**: There is no record of who did what and when.
- **No input validation**: Provider API calls could be vulnerable to injection
  if user input is passed through unsanitized.
- **No TLS guidance**: The server runs plain HTTP by default.
- **No dependency scanning**: Vulnerable dependencies could go unnoticed.

Security is not optional for a system that handles API keys, makes billable
provider calls, and stores user data. This RFC addresses all critical security
gaps.

## Detailed Design

### 1. API Key Authentication

API keys are the primary authentication mechanism for REST API access. Keys are
transmitted via the `X-API-Key` header.

```rust
use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier};
use rand::Rng;

/// An API key with metadata.
#[derive(Debug, Clone)]
pub struct ApiKey {
    /// Unique key ID (prefix of the key, e.g., "wf_live_abc123").
    pub key_id: String,
    /// Argon2 hash of the full key (never store plaintext).
    pub key_hash: String,
    /// Account this key belongs to.
    pub account_id: AccountId,
    /// Human-readable name for the key.
    pub name: String,
    /// Role assigned to this key.
    pub role: Role,
    /// Scopes: which API endpoints this key can access.
    pub scopes: Vec<ApiScope>,
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
    /// Expiration (None = never expires).
    pub expires_at: Option<DateTime<Utc>>,
    /// Last used timestamp.
    pub last_used_at: Option<DateTime<Utc>>,
    /// Whether the key is active.
    pub active: bool,
}

/// API key format: wf_{environment}_{random_bytes}
/// Example: wf_live_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
pub struct ApiKeyGenerator;

impl ApiKeyGenerator {
    const PREFIX: &'static str = "wf";
    const KEY_BYTES: usize = 32;

    pub fn generate(environment: &str) -> (String, String) {
        let random_bytes: [u8; Self::KEY_BYTES] = rand::thread_rng().gen();
        let encoded = base62::encode(&random_bytes);
        let full_key = format!("{}_{}_{}",
            Self::PREFIX, environment, encoded
        );
        let key_id = format!("{}_{}_{}", Self::PREFIX, environment, &encoded[..8]);
        (full_key, key_id)
    }

    pub fn hash_key(key: &str) -> Result<String, AuthError> {
        let salt = argon2::password_hash::SaltString::generate(&mut rand::thread_rng());
        let argon2 = Argon2::default();
        let hash = argon2
            .hash_password(key.as_bytes(), &salt)
            .map_err(|e| AuthError::HashError(e.to_string()))?;
        Ok(hash.to_string())
    }

    pub fn verify_key(key: &str, hash: &str) -> Result<bool, AuthError> {
        let parsed_hash = PasswordHash::new(hash)
            .map_err(|e| AuthError::HashError(e.to_string()))?;
        Ok(Argon2::default()
            .verify_password(key.as_bytes(), &parsed_hash)
            .is_ok())
    }
}
```

#### API Key Store

```rust
#[async_trait]
pub trait ApiKeyStore: Send + Sync {
    /// Look up a key by its ID prefix.
    async fn find_by_id(&self, key_id: &str) -> Result<Option<ApiKey>, AuthError>;
    /// Look up a key by its hash (for validation).
    async fn find_by_prefix(&self, prefix: &str) -> Result<Vec<ApiKey>, AuthError>;
    /// Store a new API key.
    async fn create(&self, key: &ApiKey) -> Result<(), AuthError>;
    /// Deactivate a key.
    async fn revoke(&self, key_id: &str) -> Result<(), AuthError>;
    /// Update last_used_at.
    async fn touch(&self, key_id: &str) -> Result<(), AuthError>;
    /// List all keys for an account.
    async fn list_for_account(&self, account_id: &AccountId) -> Result<Vec<ApiKey>, AuthError>;
}

pub struct SqliteApiKeyStore {
    pool: SqlitePool,
}

impl SqliteApiKeyStore {
    pub async fn initialize(&self) -> Result<(), AuthError> {
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS api_keys (
                key_id TEXT PRIMARY KEY,
                key_hash TEXT NOT NULL,
                account_id TEXT NOT NULL,
                name TEXT NOT NULL,
                role TEXT NOT NULL,
                scopes TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                last_used_at TEXT,
                active BOOLEAN NOT NULL DEFAULT TRUE
            )"
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }
}
```

#### Authentication Middleware

```rust
pub struct AuthMiddleware {
    key_store: Arc<dyn ApiKeyStore>,
    jwt_validator: JwtValidator,
    config: AuthConfig,
}

pub struct AuthConfig {
    /// Whether authentication is required (false for local dev).
    pub enabled: bool,
    /// Paths that don't require authentication.
    pub public_paths: Vec<String>,
    /// Maximum API key age before requiring rotation.
    pub max_key_age_days: Option<u64>,
}

impl AuthMiddleware {
    pub async fn authenticate(
        &self,
        request: &Request,
    ) -> Result<AuthContext, AuthError> {
        if !self.config.enabled {
            return Ok(AuthContext::anonymous());
        }

        // Check if path is public
        if self.is_public_path(request.uri().path()) {
            return Ok(AuthContext::anonymous());
        }

        // Try API key first
        if let Some(api_key) = request.headers().get("X-API-Key") {
            return self.authenticate_api_key(api_key.to_str()?).await;
        }

        // Try JWT bearer token
        if let Some(auth_header) = request.headers().get("Authorization") {
            let header = auth_header.to_str()?;
            if header.starts_with("Bearer ") {
                let token = &header[7..];
                return self.authenticate_jwt(token).await;
            }
        }

        Err(AuthError::NoCredentials)
    }

    async fn authenticate_api_key(
        &self,
        key: &str,
    ) -> Result<AuthContext, AuthError> {
        // Extract key_id from the key prefix
        let parts: Vec<&str> = key.split('_').collect();
        if parts.len() < 3 {
            return Err(AuthError::InvalidKeyFormat);
        }
        let key_prefix = format!("{}_{}_{}", parts[0], parts[1], &parts[2][..8.min(parts[2].len())]);

        // Find candidate keys by prefix
        let candidates = self.key_store.find_by_prefix(&key_prefix).await?;

        for candidate in &candidates {
            if !candidate.active {
                continue;
            }

            // Check expiration
            if let Some(expires_at) = candidate.expires_at {
                if Utc::now() > expires_at {
                    continue;
                }
            }

            // Verify hash
            if ApiKeyGenerator::verify_key(key, &candidate.key_hash)? {
                // Update last used
                self.key_store.touch(&candidate.key_id).await?;

                return Ok(AuthContext {
                    account_id: candidate.account_id.clone(),
                    role: candidate.role.clone(),
                    scopes: candidate.scopes.clone(),
                    auth_method: AuthMethod::ApiKey,
                    key_id: Some(candidate.key_id.clone()),
                });
            }
        }

        Err(AuthError::InvalidApiKey)
    }
}

#[derive(Debug, Clone)]
pub struct AuthContext {
    pub account_id: AccountId,
    pub role: Role,
    pub scopes: Vec<ApiScope>,
    pub auth_method: AuthMethod,
    pub key_id: Option<String>,
}

impl AuthContext {
    pub fn anonymous() -> Self {
        Self {
            account_id: AccountId::anonymous(),
            role: Role::Viewer,
            scopes: vec![],
            auth_method: AuthMethod::Anonymous,
            key_id: None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum AuthMethod {
    ApiKey,
    Jwt,
    Anonymous,
}
```

### 2. JWT Tokens for Service-to-Service Auth

For internal service communication and temporary access tokens, WorldForge uses
RS256 JWTs.

```rust
use jsonwebtoken::{encode, decode, Header, Algorithm, Validation,
    EncodingKey, DecodingKey};

#[derive(Debug, Serialize, Deserialize)]
pub struct WorldForgeClaims {
    /// Subject (account ID or service name).
    pub sub: String,
    /// Issuer (always "worldforge").
    pub iss: String,
    /// Audience (target service).
    pub aud: Vec<String>,
    /// Expiration (Unix timestamp).
    pub exp: u64,
    /// Issued at (Unix timestamp).
    pub iat: u64,
    /// JWT ID (unique identifier for this token).
    pub jti: String,
    /// Role.
    pub role: String,
    /// Scopes.
    pub scopes: Vec<String>,
}

pub struct JwtIssuer {
    encoding_key: EncodingKey,
    issuer: String,
    default_ttl: Duration,
}

impl JwtIssuer {
    pub fn new(private_key_pem: &[u8], issuer: String) -> Result<Self, AuthError> {
        let encoding_key = EncodingKey::from_rsa_pem(private_key_pem)?;
        Ok(Self {
            encoding_key,
            issuer,
            default_ttl: Duration::from_secs(3600), // 1 hour
        })
    }

    pub fn issue(
        &self,
        subject: &str,
        role: Role,
        scopes: Vec<ApiScope>,
        audience: Vec<String>,
    ) -> Result<String, AuthError> {
        let now = Utc::now();
        let claims = WorldForgeClaims {
            sub: subject.to_string(),
            iss: self.issuer.clone(),
            aud: audience,
            exp: (now + self.default_ttl).timestamp() as u64,
            iat: now.timestamp() as u64,
            jti: Uuid::new_v4().to_string(),
            role: role.to_string(),
            scopes: scopes.iter().map(|s| s.to_string()).collect(),
        };

        let header = Header::new(Algorithm::RS256);
        encode(&header, &claims, &self.encoding_key)
            .map_err(|e| AuthError::JwtError(e.to_string()))
    }
}

pub struct JwtValidator {
    decoding_key: DecodingKey,
    validation: Validation,
}

impl JwtValidator {
    pub fn new(
        public_key_pem: &[u8],
        expected_issuer: &str,
        expected_audience: &[String],
    ) -> Result<Self, AuthError> {
        let decoding_key = DecodingKey::from_rsa_pem(public_key_pem)?;
        let mut validation = Validation::new(Algorithm::RS256);
        validation.set_issuer(&[expected_issuer]);
        validation.set_audience(expected_audience);
        validation.set_required_spec_claims(&["sub", "iss", "aud", "exp", "iat"]);

        Ok(Self { decoding_key, validation })
    }

    pub fn validate(&self, token: &str) -> Result<WorldForgeClaims, AuthError> {
        let token_data = decode::<WorldForgeClaims>(
            token, &self.decoding_key, &self.validation
        ).map_err(|e| AuthError::JwtError(e.to_string()))?;

        Ok(token_data.claims)
    }
}
```

### 3. Role-Based Access Control (RBAC)

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    /// Full access: manage accounts, keys, configuration.
    Admin,
    /// Create/update/delete worlds, run predictions, manage own resources.
    Developer,
    /// Read-only access to worlds and predictions.
    Viewer,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApiScope {
    WorldsRead,
    WorldsWrite,
    WorldsDelete,
    PredictionsRun,
    PredictionsRead,
    BillingRead,
    BillingManage,
    AccountManage,
    ApiKeysManage,
    AdminAll,
}

impl Role {
    /// Default scopes for each role.
    pub fn default_scopes(&self) -> Vec<ApiScope> {
        match self {
            Role::Admin => vec![ApiScope::AdminAll],
            Role::Developer => vec![
                ApiScope::WorldsRead,
                ApiScope::WorldsWrite,
                ApiScope::WorldsDelete,
                ApiScope::PredictionsRun,
                ApiScope::PredictionsRead,
                ApiScope::BillingRead,
                ApiScope::ApiKeysManage,
            ],
            Role::Viewer => vec![
                ApiScope::WorldsRead,
                ApiScope::PredictionsRead,
            ],
        }
    }
}

/// Authorization check: does the auth context allow the requested action?
pub struct Authorizer;

impl Authorizer {
    pub fn check(
        context: &AuthContext,
        required_scope: ApiScope,
    ) -> Result<(), AuthError> {
        // Admin has access to everything
        if context.role == Role::Admin || context.scopes.contains(&ApiScope::AdminAll) {
            return Ok(());
        }

        if context.scopes.contains(&required_scope) {
            return Ok(());
        }

        Err(AuthError::InsufficientPermissions {
            required: required_scope,
            role: context.role.clone(),
            scopes: context.scopes.clone(),
        })
    }

    /// Resource-level authorization: check ownership.
    pub fn check_resource_access(
        context: &AuthContext,
        resource_owner: &AccountId,
        required_scope: ApiScope,
    ) -> Result<(), AuthError> {
        Self::check(context, required_scope)?;

        // Non-admin users can only access their own resources
        if context.role != Role::Admin && context.account_id != *resource_owner {
            return Err(AuthError::ResourceNotOwned {
                resource_owner: resource_owner.clone(),
                requester: context.account_id.clone(),
            });
        }

        Ok(())
    }
}

/// Per-endpoint authorization requirements.
pub fn endpoint_scopes(method: &str, path: &str) -> Option<ApiScope> {
    match (method, path) {
        ("GET", p) if p.starts_with("/api/v1/worlds") => Some(ApiScope::WorldsRead),
        ("POST", "/api/v1/worlds") => Some(ApiScope::WorldsWrite),
        ("PUT", p) if p.starts_with("/api/v1/worlds") => Some(ApiScope::WorldsWrite),
        ("DELETE", p) if p.starts_with("/api/v1/worlds") => Some(ApiScope::WorldsDelete),
        ("POST", p) if p.starts_with("/api/v1/predict") => Some(ApiScope::PredictionsRun),
        ("GET", p) if p.starts_with("/api/v1/predictions") => Some(ApiScope::PredictionsRead),
        ("GET", p) if p.starts_with("/api/v1/billing") => Some(ApiScope::BillingRead),
        ("POST", p) if p.starts_with("/api/v1/billing") => Some(ApiScope::BillingManage),
        ("GET", "/api/v1/health") => None,  // Public
        _ => Some(ApiScope::AdminAll),  // Default: admin only
    }
}
```

### 4. Provider Credential Storage

Secure storage for external provider API keys with support for multiple backends.

```rust
#[async_trait]
pub trait CredentialStore: Send + Sync {
    /// Retrieve a credential by name.
    async fn get(&self, name: &str) -> Result<SecretString, CredentialError>;
    /// Store or update a credential.
    async fn set(&self, name: &str, value: &SecretString) -> Result<(), CredentialError>;
    /// Delete a credential.
    async fn delete(&self, name: &str) -> Result<(), CredentialError>;
    /// List credential names (not values).
    async fn list(&self) -> Result<Vec<String>, CredentialError>;
}

/// Environment variable backend (simplest, for development).
pub struct EnvCredentialStore;

#[async_trait]
impl CredentialStore for EnvCredentialStore {
    async fn get(&self, name: &str) -> Result<SecretString, CredentialError> {
        let env_name = format!("WORLDFORGE_CRED_{}", name.to_uppercase());
        std::env::var(&env_name)
            .map(SecretString::new)
            .map_err(|_| CredentialError::NotFound(name.to_string()))
    }

    async fn set(&self, _name: &str, _value: &SecretString) -> Result<(), CredentialError> {
        Err(CredentialError::ReadOnly(
            "Environment variable store is read-only at runtime".to_string()
        ))
    }

    async fn delete(&self, _name: &str) -> Result<(), CredentialError> {
        Err(CredentialError::ReadOnly(
            "Environment variable store is read-only at runtime".to_string()
        ))
    }

    async fn list(&self) -> Result<Vec<String>, CredentialError> {
        Ok(std::env::vars()
            .filter_map(|(k, _)| k.strip_prefix("WORLDFORGE_CRED_").map(|s| s.to_lowercase()))
            .collect())
    }
}

/// HashiCorp Vault backend (production).
pub struct VaultCredentialStore {
    client: reqwest::Client,
    vault_addr: String,
    vault_token: SecretString,
    mount_path: String,
    secret_path: String,
}

#[async_trait]
impl CredentialStore for VaultCredentialStore {
    async fn get(&self, name: &str) -> Result<SecretString, CredentialError> {
        let url = format!(
            "{}/v1/{}/data/{}",
            self.vault_addr, self.mount_path, self.secret_path
        );

        let response = self.client
            .get(&url)
            .header("X-Vault-Token", self.vault_token.expose_secret())
            .send()
            .await?;

        let body: VaultResponse = response.json().await?;
        body.data.data.get(name)
            .map(|v| SecretString::new(v.clone()))
            .ok_or(CredentialError::NotFound(name.to_string()))
    }

    async fn set(&self, name: &str, value: &SecretString) -> Result<(), CredentialError> {
        let url = format!(
            "{}/v1/{}/data/{}",
            self.vault_addr, self.mount_path, self.secret_path
        );

        let mut data = HashMap::new();
        data.insert(name.to_string(), value.expose_secret().to_string());

        self.client
            .post(&url)
            .header("X-Vault-Token", self.vault_token.expose_secret())
            .json(&serde_json::json!({ "data": data }))
            .send()
            .await?;

        Ok(())
    }

    async fn delete(&self, name: &str) -> Result<(), CredentialError> {
        // Vault KV v2: update without the key
        let existing = self.list().await?;
        if !existing.contains(&name.to_string()) {
            return Err(CredentialError::NotFound(name.to_string()));
        }
        // Implementation omitted for brevity
        Ok(())
    }

    async fn list(&self) -> Result<Vec<String>, CredentialError> {
        let url = format!(
            "{}/v1/{}/metadata/{}",
            self.vault_addr, self.mount_path, self.secret_path
        );
        let response = self.client
            .get(&url)
            .header("X-Vault-Token", self.vault_token.expose_secret())
            .send()
            .await?;
        let body: VaultListResponse = response.json().await?;
        Ok(body.data.keys)
    }
}

/// AWS Secrets Manager backend.
pub struct AwsSecretsManagerStore {
    client: aws_sdk_secretsmanager::Client,
    prefix: String,
}

#[async_trait]
impl CredentialStore for AwsSecretsManagerStore {
    async fn get(&self, name: &str) -> Result<SecretString, CredentialError> {
        let secret_name = format!("{}/{}", self.prefix, name);
        let result = self.client
            .get_secret_value()
            .secret_id(&secret_name)
            .send()
            .await?;

        result.secret_string()
            .map(|s| SecretString::new(s.to_string()))
            .ok_or(CredentialError::NotFound(name.to_string()))
    }

    // ... set, delete, list implementations
}
```

### 5. Credential Rotation

```rust
pub struct CredentialRotator {
    store: Arc<dyn CredentialStore>,
    rotation_policies: HashMap<String, RotationPolicy>,
}

pub struct RotationPolicy {
    pub max_age: Duration,
    pub auto_rotate: bool,
    pub notify_before: Duration,
    pub rotation_fn: Option<Box<dyn Fn() -> SecretString + Send + Sync>>,
}

impl CredentialRotator {
    /// Check all credentials for rotation needs.
    pub async fn check_rotations(&self) -> Vec<RotationNeeded> {
        let mut needs_rotation = Vec::new();

        for (name, policy) in &self.rotation_policies {
            if let Ok(metadata) = self.store.get_metadata(name).await {
                let age = Utc::now() - metadata.created_at;
                if age > chrono::Duration::from_std(policy.max_age).unwrap() {
                    needs_rotation.push(RotationNeeded {
                        credential_name: name.clone(),
                        age,
                        policy: policy.clone(),
                    });
                }
            }
        }

        needs_rotation
    }

    /// Rotate a specific credential.
    pub async fn rotate(&self, name: &str) -> Result<(), CredentialError> {
        let policy = self.rotation_policies.get(name)
            .ok_or(CredentialError::NoPolicyFound(name.to_string()))?;

        if let Some(ref rotation_fn) = policy.rotation_fn {
            let new_value = rotation_fn();
            self.store.set(name, &new_value).await?;
            tracing::info!(credential = name, "Credential rotated successfully");
        }

        Ok(())
    }
}
```

### 6. TLS Configuration

WorldForge supports TLS via `rustls` natively, with guidance for reverse proxy
setups.

```rust
pub struct TlsConfig {
    /// Path to PEM certificate file.
    pub cert_path: PathBuf,
    /// Path to PEM private key file.
    pub key_path: PathBuf,
    /// Minimum TLS version (default: 1.2).
    pub min_version: TlsVersion,
    /// OCSP stapling.
    pub ocsp_stapling: bool,
}

pub enum TlsVersion {
    Tls12,
    Tls13,
}

impl TlsConfig {
    pub fn build_rustls_config(&self) -> Result<rustls::ServerConfig, TlsError> {
        let cert_chain = Self::load_certs(&self.cert_path)?;
        let key = Self::load_private_key(&self.key_path)?;

        let mut config = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(cert_chain, key)?;

        config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];

        Ok(config)
    }

    fn load_certs(path: &Path) -> Result<Vec<rustls::Certificate>, TlsError> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let certs = rustls_pemfile::certs(&mut reader)?;
        Ok(certs.into_iter().map(rustls::Certificate).collect())
    }

    fn load_private_key(path: &Path) -> Result<rustls::PrivateKey, TlsError> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let keys = rustls_pemfile::pkcs8_private_keys(&mut reader)?;
        keys.into_iter()
            .next()
            .map(rustls::PrivateKey)
            .ok_or(TlsError::NoPrivateKey)
    }
}
```

Reverse proxy configuration examples are provided in documentation:

```nginx
# nginx reverse proxy for WorldForge
server {
    listen 443 ssl http2;
    server_name worldforge.example.com;

    ssl_certificate /etc/letsencrypt/live/worldforge.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/worldforge.example.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 7. Input Validation and Sanitization

```rust
pub struct InputValidator;

impl InputValidator {
    /// Validate and sanitize a world name.
    pub fn validate_world_name(name: &str) -> Result<String, ValidationError> {
        let trimmed = name.trim();

        if trimmed.is_empty() || trimmed.len() > 256 {
            return Err(ValidationError::InvalidLength {
                field: "name".to_string(),
                min: 1,
                max: 256,
                actual: trimmed.len(),
            });
        }

        // Only allow alphanumeric, spaces, hyphens, underscores
        if !trimmed.chars().all(|c| c.is_alphanumeric() || c == ' ' || c == '-' || c == '_') {
            return Err(ValidationError::InvalidCharacters {
                field: "name".to_string(),
                allowed: "alphanumeric, spaces, hyphens, underscores",
            });
        }

        Ok(trimmed.to_string())
    }

    /// Sanitize input that will be included in provider API calls.
    /// Prevents prompt injection and ensures safe content.
    pub fn sanitize_provider_input(input: &str) -> Result<String, ValidationError> {
        // Maximum length for any single input field
        const MAX_INPUT_LENGTH: usize = 100_000;

        if input.len() > MAX_INPUT_LENGTH {
            return Err(ValidationError::InputTooLarge {
                max_bytes: MAX_INPUT_LENGTH,
                actual_bytes: input.len(),
            });
        }

        // Strip null bytes
        let sanitized = input.replace('\0', "");

        // Log if input contains suspicious patterns (but don't block)
        if Self::contains_injection_patterns(&sanitized) {
            tracing::warn!(
                input_prefix = &sanitized[..sanitized.len().min(100)],
                "Input contains potential injection patterns"
            );
        }

        Ok(sanitized)
    }

    fn contains_injection_patterns(input: &str) -> bool {
        let patterns = [
            "ignore previous instructions",
            "ignore all instructions",
            "system prompt",
            "you are now",
            "<|im_start|>",
            "[INST]",
        ];
        let lower = input.to_lowercase();
        patterns.iter().any(|p| lower.contains(p))
    }

    /// Validate a URL for webhook delivery.
    pub fn validate_webhook_url(url: &str) -> Result<url::Url, ValidationError> {
        let parsed = url::Url::parse(url)
            .map_err(|e| ValidationError::InvalidUrl(e.to_string()))?;

        // Only allow HTTPS
        if parsed.scheme() != "https" {
            return Err(ValidationError::InsecureUrl);
        }

        // Block private/internal IPs (SSRF prevention)
        if let Some(host) = parsed.host_str() {
            if Self::is_private_host(host) {
                return Err(ValidationError::PrivateUrl);
            }
        }

        Ok(parsed)
    }

    fn is_private_host(host: &str) -> bool {
        if let Ok(ip) = host.parse::<std::net::IpAddr>() {
            match ip {
                std::net::IpAddr::V4(ipv4) => {
                    ipv4.is_private() || ipv4.is_loopback() || ipv4.is_link_local()
                }
                std::net::IpAddr::V6(ipv6) => {
                    ipv6.is_loopback()
                }
            }
        } else {
            host == "localhost" || host.ends_with(".local")
        }
    }
}
```

### 8. Audit Logging

```rust
#[derive(Debug, Serialize)]
pub struct AuditEvent {
    pub timestamp: DateTime<Utc>,
    pub event_type: AuditEventType,
    pub actor: AuditActor,
    pub resource: Option<AuditResource>,
    pub action: String,
    pub outcome: AuditOutcome,
    pub details: HashMap<String, serde_json::Value>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub request_id: String,
}

#[derive(Debug, Serialize)]
pub enum AuditEventType {
    Authentication,
    Authorization,
    ResourceAccess,
    ResourceModification,
    ResourceDeletion,
    ConfigurationChange,
    CredentialAccess,
    BillingEvent,
    SecurityEvent,
}

#[derive(Debug, Serialize)]
pub struct AuditActor {
    pub account_id: String,
    pub key_id: Option<String>,
    pub role: String,
    pub auth_method: String,
}

#[derive(Debug, Serialize)]
pub enum AuditOutcome {
    Success,
    Failure { reason: String },
    Denied { reason: String },
}

#[async_trait]
pub trait AuditLogger: Send + Sync {
    async fn log(&self, event: AuditEvent);
}

/// Structured audit logger using the tracing crate.
pub struct TracingAuditLogger;

#[async_trait]
impl AuditLogger for TracingAuditLogger {
    async fn log(&self, event: AuditEvent) {
        tracing::info!(
            target: "audit",
            event_type = ?event.event_type,
            actor.account_id = %event.actor.account_id,
            actor.role = %event.actor.role,
            action = %event.action,
            outcome = ?event.outcome,
            resource = ?event.resource,
            ip_address = ?event.ip_address,
            request_id = %event.request_id,
            "audit_event"
        );
    }
}

/// Database-backed audit logger for queryable audit trail.
pub struct DatabaseAuditLogger {
    pool: SqlitePool,
}

#[async_trait]
impl AuditLogger for DatabaseAuditLogger {
    async fn log(&self, event: AuditEvent) {
        let details_json = serde_json::to_string(&event.details).unwrap_or_default();
        sqlx::query(
            "INSERT INTO audit_log (timestamp, event_type, actor_account_id,
             actor_role, action, outcome, details, ip_address, request_id)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(event.timestamp.to_rfc3339())
        .bind(format!("{:?}", event.event_type))
        .bind(&event.actor.account_id)
        .bind(&event.actor.role)
        .bind(&event.action)
        .bind(format!("{:?}", event.outcome))
        .bind(&details_json)
        .bind(&event.ip_address)
        .bind(&event.request_id)
        .execute(&self.pool)
        .await
        .ok();  // Audit logging failures should not break requests
    }
}
```

### 9. Security Headers

```rust
pub fn security_headers_middleware(
    mut response: Response,
    config: &SecurityHeadersConfig,
) -> Response {
    let headers = response.headers_mut();

    // HSTS: enforce HTTPS
    if config.hsts_enabled {
        headers.insert(
            "Strict-Transport-Security",
            format!("max-age={}; includeSubDomains", config.hsts_max_age)
                .parse().unwrap(),
        );
    }

    // Content Security Policy
    headers.insert(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'"
            .parse().unwrap(),
    );

    // Prevent MIME type sniffing
    headers.insert("X-Content-Type-Options", "nosniff".parse().unwrap());

    // Prevent clickjacking
    headers.insert("X-Frame-Options", "DENY".parse().unwrap());

    // XSS protection
    headers.insert("X-XSS-Protection", "1; mode=block".parse().unwrap());

    // Referrer policy
    headers.insert("Referrer-Policy", "strict-origin-when-cross-origin".parse().unwrap());

    // Permissions policy
    headers.insert(
        "Permissions-Policy",
        "camera=(), microphone=(), geolocation=()".parse().unwrap(),
    );

    response
}
```

### 10. Dependency Vulnerability Scanning

Integration with `cargo-audit` and `cargo-deny` in CI:

```yaml
# .github/workflows/security.yml
name: Security Audit
on:
  push:
    branches: [main]
  pull_request:
  schedule:
    - cron: '0 6 * * 1'  # Weekly Monday 6 AM UTC

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: rustsec/audit-check@v2
        with:
          token: ${{ secrets.GITHUB_TOKEN }}

  deny:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: EmbarkStudios/cargo-deny-action@v2
        with:
          arguments: --all-features
          command: check advisories sources licenses
```

```toml
# deny.toml
[advisories]
vulnerability = "deny"
unmaintained = "warn"
yanked = "deny"

[licenses]
allow = ["MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "Zlib"]
confidence-threshold = 0.8

[sources]
unknown-registry = "deny"
unknown-git = "deny"
allow-registry = ["https://github.com/rust-lang/crates.io-index"]
```

## Implementation Plan

### Phase 1: API Key Authentication (2 weeks)
- Implement `ApiKeyGenerator` with secure random generation.
- Build `SqliteApiKeyStore` with Argon2 hashing.
- Create `AuthMiddleware` for request authentication.
- Add API key management endpoints (create, list, revoke).
- CLI command for initial admin key creation.

### Phase 2: RBAC (1 week)
- Define roles and scopes.
- Implement `Authorizer` with scope checking.
- Add authorization checks to all existing endpoints.
- Resource-level ownership validation.

### Phase 3: JWT & Service Auth (2 weeks)
- RSA key pair generation and management.
- Implement `JwtIssuer` and `JwtValidator`.
- Add token exchange endpoint.
- Service-to-service authentication for internal APIs.

### Phase 4: Credential Storage (2 weeks)
- Abstract `CredentialStore` trait.
- Implement `EnvCredentialStore` (already partially exists).
- Implement `VaultCredentialStore`.
- Implement `AwsSecretsManagerStore`.
- Credential rotation framework.

### Phase 5: TLS & Headers (1 week)
- Native rustls TLS support.
- Security headers middleware.
- Reverse proxy configuration documentation.
- HTTPS redirect middleware.

### Phase 6: Audit & Validation (2 weeks)
- Input validation for all API endpoints.
- Audit logging infrastructure.
- Database-backed audit store.
- Audit log query API (admin only).

### Phase 7: CI Security (1 week)
- Set up cargo-audit in CI.
- Configure cargo-deny.
- SAST scanning integration.
- Dependency update automation (Dependabot/Renovate).

## Testing Strategy

### Unit Tests
- API key generation format validation.
- Argon2 hash and verify roundtrip.
- JWT issue and validate roundtrip.
- JWT rejection: expired, wrong issuer, wrong audience.
- Role scope resolution.
- Authorization checks for each role/scope combination.
- Input validation: valid inputs, boundary cases, injection patterns.
- SSRF prevention for webhook URLs.

### Integration Tests
- Full authentication flow: create key, use key, revoke key.
- JWT service-to-service authentication between components.
- Vault credential store operations (requires Vault dev server).
- Audit log write and query.
- TLS handshake with rustls.

### Security Tests
- Timing attack resistance for key comparison (constant-time).
- Rate limiting on authentication failures.
- Invalid/malformed API keys don't leak information.
- JWT with tampered signature is rejected.
- SQL injection in audit log queries.
- SSRF via webhook URL.

### Penetration Testing Checklist
- Authentication bypass attempts.
- Privilege escalation (viewer -> developer -> admin).
- API key enumeration.
- JWT confusion attacks (alg:none, HMAC/RSA confusion).
- Header injection.
- Path traversal in resource IDs.

## Open Questions

1. **OAuth2/OIDC support**: Should we support external identity providers
   (Google, GitHub, Okta) for user authentication? This would complement API
   keys for human users vs. programmatic access.

2. **Multi-tenancy isolation**: How strong should tenant isolation be? Separate
   databases? Separate processes? Row-level security?

3. **API key rate limiting on auth failures**: How many failed authentication
   attempts before temporary lockout? Per IP? Per key prefix?

4. **Audit log retention**: How long should audit logs be retained? Should there
   be different retention for different event types?

5. **Secret zero problem**: How does WorldForge itself authenticate to Vault or
   AWS Secrets Manager? Environment variables? Instance IAM roles?

6. **Client certificate authentication**: Should we support mTLS for high-security
   deployments?

7. **CORS configuration**: What should the default CORS policy be? Should it be
   configurable per account?
