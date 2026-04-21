use crate::error::VididxError;
use sha2::{Digest, Sha256};
use std::path::Path;

/// Compute SHA-256 hash of bytes and return as "sha256:<hex>" format.
pub fn sha256_bytes(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let hash = hasher.finalize();
    format!("sha256:{}", hex::encode(hash))
}

/// Compute SHA-256 hash of a string and return as "sha256:<hex>" format.
pub fn sha256_str(s: &str) -> String {
    sha256_bytes(s.as_bytes())
}

/// Compute SHA-256 hash of a file and return as "sha256:<hex>" format.
pub fn sha256_file(path: &Path) -> Result<String, VididxError> {
    let data = std::fs::read(path)?;
    Ok(sha256_bytes(&data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_bytes_deterministic() {
        let data = b"test data";
        let hash1 = sha256_bytes(data);
        let hash2 = sha256_bytes(data);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_sha256_bytes_different_for_different_input() {
        let data1 = b"test data 1";
        let data2 = b"test data 2";
        let hash1 = sha256_bytes(data1);
        let hash2 = sha256_bytes(data2);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_sha256_format() {
        let hash = sha256_bytes(b"test");
        assert!(hash.starts_with("sha256:"));
        assert_eq!(hash.len(), 7 + 64); // "sha256:" + 64 hex chars
    }

    #[test]
    fn test_sha256_str() {
        let hash = sha256_str("test");
        assert!(hash.starts_with("sha256:"));
        assert_eq!(hash.len(), 7 + 64);
    }

    #[test]
    fn test_sha256_str_matches_bytes() {
        let hash_str = sha256_str("test");
        let hash_bytes = sha256_bytes(b"test");
        assert_eq!(hash_str, hash_bytes);
    }

    #[test]
    fn test_sha256_file() {
        let path = std::path::Path::new("Cargo.toml");
        if path.exists() {
            let hash1 = sha256_file(path).unwrap();
            let hash2 = sha256_file(path).unwrap();
            assert_eq!(hash1, hash2);
            assert!(hash1.starts_with("sha256:"));
        }
    }
}
