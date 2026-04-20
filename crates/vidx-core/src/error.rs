/// Core error type for vidx.
#[derive(Debug, thiserror::Error)]
pub enum VidxError {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vidx_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "test");
        let vidx_err: VidxError = io_err.into();
        assert!(matches!(vidx_err, VidxError::Io(_)));
    }

    #[test]
    fn test_vidx_error_from_serde() {
        let serde_err = serde_json::from_str::<serde_json::Value>("invalid").unwrap_err();
        let vidx_err: VidxError = serde_err.into();
        assert!(matches!(vidx_err, VidxError::Serde(_)));
    }

    #[test]
    fn test_vidx_error_display() {
        let err = VidxError::Config("test config error".to_string());
        assert_eq!(err.to_string(), "config error: test config error");
    }
}
