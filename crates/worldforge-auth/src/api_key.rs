use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use sha2::{Digest, Sha256};

use crate::{AuthError, Result};

/// Identity extracted from a valid API key.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ApiKeyIdentity {
    pub key_id: String,
    pub owner: String,
    pub scopes: Vec<String>,
}

/// Metadata stored for each API key.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ApiKeyMetadata {
    pub key_id: String,
    pub owner: String,
    pub scopes: Vec<String>,
    pub created_at: u64,
    pub revoked: bool,
}

/// Trait for API key storage backends.
pub trait ApiKeyStore: Send + Sync {
    fn validate_key(&self, key: &str) -> Result<ApiKeyIdentity>;
    fn create_key(&self, owner: &str, scopes: Vec<String>) -> Result<String>;
    fn revoke_key(&self, key_hash: &str) -> Result<()>;
}

/// In-memory API key store backed by a HashMap of key_hash -> metadata.
#[derive(Debug, Clone)]
pub struct InMemoryApiKeyStore {
    keys: Arc<RwLock<HashMap<String, ApiKeyMetadata>>>,
    counter: Arc<RwLock<u64>>,
}

impl InMemoryApiKeyStore {
    pub fn new() -> Self {
        Self {
            keys: Arc::new(RwLock::new(HashMap::new())),
            counter: Arc::new(RwLock::new(0)),
        }
    }
}

impl Default for InMemoryApiKeyStore {
    fn default() -> Self {
        Self::new()
    }
}

fn hash_key(key: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(key.as_bytes());
    hex::encode(hasher.finalize())
}

fn generate_raw_key(id: u64) -> String {
    use sha2::Digest;
    let mut hasher = Sha256::new();
    let seed = format!("{}-{}", id, std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos());
    hasher.update(seed.as_bytes());
    let hash = hex::encode(hasher.finalize());
    format!("wf_{}", &hash[..32])
}

impl ApiKeyStore for InMemoryApiKeyStore {
    fn validate_key(&self, key: &str) -> Result<ApiKeyIdentity> {
        if !key.starts_with("wf_") {
            return Err(AuthError::InvalidApiKey);
        }
        let key_hash = hash_key(key);
        let keys = self.keys.read().map_err(|e| AuthError::Internal(e.to_string()))?;
        match keys.get(&key_hash) {
            Some(meta) if meta.revoked => Err(AuthError::RevokedApiKey),
            Some(meta) => Ok(ApiKeyIdentity {
                key_id: meta.key_id.clone(),
                owner: meta.owner.clone(),
                scopes: meta.scopes.clone(),
            }),
            None => Err(AuthError::InvalidApiKey),
        }
    }

    fn create_key(&self, owner: &str, scopes: Vec<String>) -> Result<String> {
        let mut counter = self.counter.write().map_err(|e| AuthError::Internal(e.to_string()))?;
        *counter += 1;
        let raw_key = generate_raw_key(*counter);
        let key_hash = hash_key(&raw_key);

        let metadata = ApiKeyMetadata {
            key_id: format!("key_{}", *counter),
            owner: owner.to_string(),
            scopes,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            revoked: false,
        };

        let mut keys = self.keys.write().map_err(|e| AuthError::Internal(e.to_string()))?;
        keys.insert(key_hash, metadata);
        Ok(raw_key)
    }

    fn revoke_key(&self, key_hash: &str) -> Result<()> {
        let mut keys = self.keys.write().map_err(|e| AuthError::Internal(e.to_string()))?;
        match keys.get_mut(key_hash) {
            Some(meta) => {
                meta.revoked = true;
                Ok(())
            }
            None => Err(AuthError::InvalidApiKey),
        }
    }
}

/// High-level API key authenticator wrapping any ApiKeyStore.
pub struct ApiKeyAuth<S: ApiKeyStore> {
    store: S,
}

impl<S: ApiKeyStore> ApiKeyAuth<S> {
    pub fn new(store: S) -> Self {
        Self { store }
    }

    pub fn validate(&self, key: &str) -> Result<ApiKeyIdentity> {
        tracing::debug!("Validating API key");
        self.store.validate_key(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_validate_key() {
        let store = InMemoryApiKeyStore::new();
        let key = store.create_key("alice", vec!["read".into(), "write".into()]).unwrap();
        assert!(key.starts_with("wf_"));

        let identity = store.validate_key(&key).unwrap();
        assert_eq!(identity.owner, "alice");
        assert_eq!(identity.scopes, vec!["read", "write"]);
    }

    #[test]
    fn test_invalid_key_rejected() {
        let store = InMemoryApiKeyStore::new();
        assert!(store.validate_key("invalid_key").is_err());
        assert!(store.validate_key("wf_nonexistent").is_err());
    }

    #[test]
    fn test_revoke_key() {
        let store = InMemoryApiKeyStore::new();
        let key = store.create_key("bob", vec![]).unwrap();
        let key_hash = hash_key(&key);

        assert!(store.validate_key(&key).is_ok());
        store.revoke_key(&key_hash).unwrap();

        match store.validate_key(&key) {
            Err(AuthError::RevokedApiKey) => {}
            other => panic!("Expected RevokedApiKey, got {:?}", other),
        }
    }

    #[test]
    fn test_api_key_auth_wrapper() {
        let store = InMemoryApiKeyStore::new();
        let key = store.create_key("charlie", vec!["admin".into()]).unwrap();
        let auth = ApiKeyAuth::new(store);
        let identity = auth.validate(&key).unwrap();
        assert_eq!(identity.owner, "charlie");
    }

    #[test]
    fn test_key_hash_is_sha256() {
        let h = hash_key("wf_test123");
        assert_eq!(h.len(), 64); // SHA-256 hex = 64 chars
    }
}
