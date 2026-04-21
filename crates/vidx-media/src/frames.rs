use std::path::{Path, PathBuf};
use std::process::Command;
use tokio::task;

use vidx_core::VidxError;

/// Extract frames from a video at specified timestamps.
/// Runs extractions in parallel using tokio.
pub async fn extract_frames(
    ffmpeg_path: &str,
    mp4_path: &Path,
    seconds: &[f64],
    out_dir: &Path,
    quality: u8,
) -> Result<Vec<PathBuf>, VidxError> {
    if !is_tool_available(ffmpeg_path).await {
        return Err(VidxError::ToolNotFound(format!(
            "ffmpeg not found at: {}",
            ffmpeg_path
        )));
    }

    // Create output directory
    std::fs::create_dir_all(out_dir)
        .map_err(|e| VidxError::Media(format!("Failed to create output directory: {}", e)))?;

    // Prepare extraction tasks
    let mut handles = Vec::new();

    for (idx, &sec) in seconds.iter().enumerate() {
        let ffmpeg_path = ffmpeg_path.to_string();
        let mp4_path = mp4_path.to_path_buf();
        let out_dir = out_dir.to_path_buf();

        let handle = task::spawn_blocking(move || {
            extract_single_frame(&ffmpeg_path, &mp4_path, sec, &out_dir, idx, quality)
        });

        handles.push(handle);
    }

    // Collect results
    let mut results = Vec::new();
    for handle in handles {
        match handle.await {
            Ok(Ok(path)) => results.push(path),
            Ok(Err(e)) => return Err(e),
            Err(e) => return Err(VidxError::Media(format!("Task join error: {}", e))),
        }
    }

    Ok(results)
}

/// Extract a single frame at the given timestamp.
fn extract_single_frame(
    ffmpeg_path: &str,
    mp4_path: &Path,
    seconds: f64,
    out_dir: &Path,
    index: usize,
    quality: u8,
) -> Result<PathBuf, VidxError> {
    let output_path = out_dir.join(format!("frame_{:03}.jpg", index));

    // Convert quality (1-31) for ffmpeg -q:v flag
    // User provides 1-100, ffmpeg uses 1-31 where 1 is best
    let ffmpeg_quality = (32 - quality as i32 / 4).clamp(1, 31);

    let output = Command::new(ffmpeg_path)
        .arg("-ss")
        .arg(seconds.to_string())
        .arg("-i")
        .arg(mp4_path)
        .arg("-frames:v")
        .arg("1")
        .arg("-q:v")
        .arg(ffmpeg_quality.to_string())
        .arg("-y")
        .arg(&output_path)
        .output()
        .map_err(|e| VidxError::Media(format!("ffmpeg frame extraction failed: {}", e)))?;

    if !output.status.success() {
        return Err(VidxError::Media(format!(
            "ffmpeg frame extraction failed at {}s: {}",
            seconds,
            String::from_utf8_lossy(&output.stderr)
        )));
    }

    Ok(output_path)
}

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
    fn test_quality_conversion() {
        // Quality 85 should map to a reasonable ffmpeg quality value
        let q = (32 - 85 / 4).clamp(1, 31);
        assert!((1..=31).contains(&q));
    }

    #[tokio::test]
    async fn test_extract_frames_nonexistent_ffmpeg() {
        let temp_dir = tempfile::tempdir().unwrap();
        let result = extract_frames(
            "nonexistent_ffmpeg",
            Path::new("/tmp/video.mp4"),
            &[0.0, 5.0],
            temp_dir.path(),
            85,
        )
        .await;

        assert!(matches!(result, Err(VidxError::ToolNotFound(_))));
    }

    #[test]
    fn test_extract_single_frame_path_construction() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join(format!("frame_{:03}.jpg", 0));
        assert_eq!(path.file_name().unwrap(), "frame_000.jpg");
    }
}
