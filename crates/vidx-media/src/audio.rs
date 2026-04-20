use std::path::Path;
use std::process::Command;

use vidx_core::VidxError;

/// Extract audio from video and convert to 16kHz mono WAV.
pub async fn extract_audio(
    ffmpeg_path: &str,
    mp4_path: &Path,
    output_wav_path: &Path,
) -> Result<(), VidxError> {
    // Check if ffmpeg exists
    if !is_tool_available(ffmpeg_path).await {
        return Err(VidxError::ToolNotFound(format!(
            "ffmpeg not found at: {}",
            ffmpeg_path
        )));
    }

    // Create parent directory if needed
    if let Some(parent) = output_wav_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| VidxError::Media(format!("Failed to create output directory: {}", e)))?;
    }

    // Run ffmpeg to extract and resample audio
    let output = Command::new(ffmpeg_path)
        .arg("-i")
        .arg(mp4_path)
        .arg("-ac")
        .arg("1")
        .arg("-ar")
        .arg("16000")
        .arg("-y")
        .arg(output_wav_path)
        .output()
        .map_err(|e| VidxError::Media(format!("ffmpeg execution failed: {}", e)))?;

    if !output.status.success() {
        return Err(VidxError::Media(format!(
            "ffmpeg audio extraction failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }

    Ok(())
}

/// Check if a tool is available on the system.
async fn is_tool_available(tool_path: &str) -> bool {
    Command::new(tool_path)
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_tool_not_available() {
        // This test just verifies the function doesn't crash on a nonexistent tool
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(is_tool_available("nonexistent_tool_xyz_123"));
        assert!(!result);
    }

    #[tokio::test]
    async fn test_extract_audio_nonexistent_ffmpeg() {
        let temp_dir = tempfile::tempdir().unwrap();
        let output_path = temp_dir.path().join("output.wav");

        let result = extract_audio(
            "nonexistent_ffmpeg",
            Path::new("/tmp/video.mp4"),
            &output_path,
        )
        .await;

        assert!(matches!(result, Err(VidxError::ToolNotFound(_))));
    }
}
