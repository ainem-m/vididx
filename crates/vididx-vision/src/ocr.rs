use async_trait::async_trait;
use std::path::Path;
use std::process::Command;

use vididx_core::VididxError;

/// Trait for OCR adapters.
#[async_trait]
pub trait OcrAdapter: Send + Sync {
    /// Extract text from an image file.
    async fn extract_text(
        &self,
        image_path: &Path,
        languages: &[&str],
    ) -> Result<String, VididxError>;
}

/// Tesseract OCR adapter.
pub struct TesseractAdapter {
    binary_path: String,
}

impl TesseractAdapter {
    /// Create a new TesseractAdapter.
    pub fn new(binary_path: &str) -> Self {
        Self {
            binary_path: binary_path.to_string(),
        }
    }
}

#[async_trait]
impl OcrAdapter for TesseractAdapter {
    async fn extract_text(
        &self,
        image_path: &Path,
        languages: &[&str],
    ) -> Result<String, VididxError> {
        if !is_tool_available(&self.binary_path).await {
            return Err(VididxError::ToolNotFound(format!(
                "tesseract not found at: {}",
                self.binary_path
            )));
        }

        // Build language argument
        let lang_arg = languages.join("+");

        // Run tesseract: tesseract image.jpg - -l language
        let output = Command::new(&self.binary_path)
            .arg(image_path)
            .arg("-")
            .arg("-l")
            .arg(&lang_arg)
            .output()
            .map_err(|e| VididxError::Vision(format!("tesseract execution failed: {}", e)))?;

        if !output.status.success() {
            return Err(VididxError::Vision(format!(
                "tesseract failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let text = String::from_utf8(output.stdout)
            .map_err(|e| VididxError::Vision(format!("tesseract output not UTF-8: {}", e)))?;

        Ok(text.trim().to_string())
    }
}

async fn is_tool_available(tool_path: &str) -> bool {
    Command::new(tool_path)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tesseract_adapter_nonexistent() {
        let adapter = TesseractAdapter::new("nonexistent_tesseract");
        let result = adapter
            .extract_text(Path::new("/tmp/image.jpg"), &["jpn", "eng"])
            .await;

        assert!(matches!(result, Err(VididxError::ToolNotFound(_))));
    }
}
