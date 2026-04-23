/// Core error type for vididx.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum VididxError {
    #[error("media error: {0}")]
    Media(String),
    #[error("asr error: {0}")]
    Asr(String),
    #[error("llm error: {0}")]
    Llm(String),
    #[error("vision error: {0}")]
    Vision(String),
    #[error("segment error: {0}")]
    Segment(String),
    #[error("config error: {0}")]
    Config(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("external tool not found: {0}")]
    ToolNotFound(String),
}

impl VididxError {
    /// Returns true if this error is retryable (rate limit, server error, timeout).
    /// Non-retryable errors include auth failures, bad requests, and config errors.
    pub fn is_retryable(&self) -> bool {
        match self {
            VididxError::Llm(msg) => {
                msg.contains("429")
                    || msg.contains("5")
                    || msg.contains("timeout")
                    || msg.contains("rate")
            }
            VididxError::Io(_) => true,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vididx_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "test");
        let vididx_err: VididxError = io_err.into();
        assert!(matches!(vididx_err, VididxError::Io(_)));
    }

    #[test]
    fn test_vididx_error_from_serde() {
        let serde_err = serde_json::from_str::<serde_json::Value>("invalid").unwrap_err();
        let vididx_err: VididxError = serde_err.into();
        assert!(matches!(vididx_err, VididxError::Serde(_)));
    }

    #[test]
    fn test_vididx_error_display() {
        let err = VididxError::Config("test config error".to_string());
        assert_eq!(err.to_string(), "config error: test config error");
    }
}
