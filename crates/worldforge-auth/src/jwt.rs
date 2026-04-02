use jsonwebtoken::{Algorithm, DecodingKey, EncodingKey, Header, Validation};

use crate::{AuthError, Result};

/// JWT claims.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Claims {
    pub sub: String,
    pub exp: u64,
    pub iat: u64,
    pub iss: String,
    pub scopes: Vec<String>,
}

/// JWT authenticator using RS256.
pub struct JwtAuth {
    encoding_key: Option<EncodingKey>,
    decoding_key: DecodingKey,
    issuer: String,
}

impl JwtAuth {
    /// Create from RSA PEM keys. `private_pem` is optional (only needed for signing).
    pub fn new(
        public_pem: &[u8],
        private_pem: Option<&[u8]>,
        issuer: &str,
    ) -> Result<Self> {
        let decoding_key = DecodingKey::from_rsa_pem(public_pem)
            .map_err(|e| AuthError::Internal(format!("Invalid public key: {}", e)))?;

        let encoding_key = match private_pem {
            Some(pem) => Some(
                EncodingKey::from_rsa_pem(pem)
                    .map_err(|e| AuthError::Internal(format!("Invalid private key: {}", e)))?,
            ),
            None => None,
        };

        Ok(Self {
            encoding_key,
            decoding_key,
            issuer: issuer.to_string(),
        })
    }

    /// Validate a JWT token and return claims.
    pub fn validate_token(&self, token: &str) -> Result<Claims> {
        let mut validation = Validation::new(Algorithm::RS256);
        validation.set_issuer(&[&self.issuer]);
        validation.set_required_spec_claims(&["exp", "sub", "iss"]);

        let token_data = jsonwebtoken::decode::<Claims>(token, &self.decoding_key, &validation)
            .map_err(|e| match e.kind() {
                jsonwebtoken::errors::ErrorKind::ExpiredSignature => AuthError::TokenExpired,
                _ => AuthError::InvalidToken(e.to_string()),
            })?;

        Ok(token_data.claims)
    }

    /// Generate a signed JWT token from claims.
    pub fn generate_token(&self, claims: &Claims) -> Result<String> {
        let encoding_key = self
            .encoding_key
            .as_ref()
            .ok_or_else(|| AuthError::Internal("No private key configured for signing".into()))?;

        let header = Header::new(Algorithm::RS256);
        jsonwebtoken::encode(&header, claims, encoding_key)
            .map_err(|e| AuthError::Internal(format!("Failed to encode JWT: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Generate RSA keys for testing
    fn test_keys() -> (Vec<u8>, Vec<u8>) {
        // Use pre-generated test RSA keys (2048-bit)
        let private_key = include_str!("../test_data/private_key.pem");
        let public_key = include_str!("../test_data/public_key.pem");
        (private_key.as_bytes().to_vec(), public_key.as_bytes().to_vec())
    }

    #[test]
    fn test_generate_and_validate_token() {
        let (private_pem, public_pem) = test_keys();
        let auth = JwtAuth::new(&public_pem, Some(&private_pem), "worldforge").unwrap();

        let now = jsonwebtoken::get_current_timestamp();
        let claims = Claims {
            sub: "user123".into(),
            exp: now + 3600,
            iat: now,
            iss: "worldforge".into(),
            scopes: vec!["read".into(), "write".into()],
        };

        let token = auth.generate_token(&claims).unwrap();
        let validated = auth.validate_token(&token).unwrap();
        assert_eq!(validated.sub, "user123");
        assert_eq!(validated.scopes, vec!["read", "write"]);
    }

    #[test]
    fn test_expired_token_rejected() {
        let (private_pem, public_pem) = test_keys();
        let auth = JwtAuth::new(&public_pem, Some(&private_pem), "worldforge").unwrap();

        let claims = Claims {
            sub: "user123".into(),
            exp: 1000, // long expired
            iat: 900,
            iss: "worldforge".into(),
            scopes: vec![],
        };

        let token = auth.generate_token(&claims).unwrap();
        match auth.validate_token(&token) {
            Err(AuthError::TokenExpired) => {}
            other => panic!("Expected TokenExpired, got {:?}", other),
        }
    }

    #[test]
    fn test_invalid_token_rejected() {
        let (_private_pem, public_pem) = test_keys();
        let auth = JwtAuth::new(&public_pem, None, "worldforge").unwrap();
        assert!(auth.validate_token("not.a.valid.token").is_err());
    }

    #[test]
    fn test_no_private_key_cannot_sign() {
        let (_private_pem, public_pem) = test_keys();
        let auth = JwtAuth::new(&public_pem, None, "worldforge").unwrap();

        let claims = Claims {
            sub: "user".into(),
            exp: 9999999999,
            iat: 0,
            iss: "worldforge".into(),
            scopes: vec![],
        };

        match auth.generate_token(&claims) {
            Err(AuthError::Internal(_)) => {}
            other => panic!("Expected Internal error, got {:?}", other),
        }
    }
}
