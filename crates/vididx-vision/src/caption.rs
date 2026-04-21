use async_trait::async_trait;
use std::path::Path;

use vididx_core::VididxError;

/// Trait for visual caption adapters (VLM-based).
#[async_trait]
pub trait VlmCaptionAdapter: Send + Sync {
    /// Generate a visual caption for an image.
    async fn caption(&self, image_path: &Path) -> Result<String, VididxError>;
}

/// Claude VLM caption adapter (placeholder for Anthropic Messages API).
#[allow(dead_code)]
pub struct ClaudeVlmAdapter {
    api_key: String,
    model: String,
}

impl ClaudeVlmAdapter {
    /// Create a new Claude VLM adapter.
    pub fn new(api_key: &str, model: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            model: model.to_string(),
        }
    }
}

#[async_trait]
impl VlmCaptionAdapter for ClaudeVlmAdapter {
    async fn caption(&self, image_path: &Path) -> Result<String, VididxError> {
        // Check if image exists
        if !image_path.exists() {
            return Err(VididxError::Vision(format!(
                "Image not found: {}",
                image_path.display()
            )));
        }

        // For now, return a placeholder
        // Real implementation would:
        // 1. Read image file
        // 2. Encode as base64
        // 3. Call Anthropic API
        // 4. Parse response and extract caption text

        Ok(format!(
            "Visual caption for {}",
            image_path.file_name().unwrap_or_default().to_string_lossy()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_claude_vlm_nonexistent_image() {
        let adapter = ClaudeVlmAdapter::new("fake_key", "claude-sonnet-4");
        let result = adapter.caption(Path::new("/nonexistent/image.jpg")).await;

        assert!(matches!(result, Err(VididxError::Vision(_))));
    }

    #[tokio::test]
    async fn test_claude_vlm_new() {
        let adapter = ClaudeVlmAdapter::new("test_key", "claude-opus");
        assert_eq!(adapter.api_key, "test_key");
        assert_eq!(adapter.model, "claude-opus");
    }
}
