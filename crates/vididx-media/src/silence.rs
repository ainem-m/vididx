use regex::Regex;
use std::path::Path;
use std::process::Command;

use vididx_core::{SilenceInterval, VididxError};

/// Detect silence intervals in a video using ffmpeg silencedetect filter.
pub async fn detect_silence(
    ffmpeg_path: &str,
    mp4_path: &Path,
    noise_db: f32,
    duration: f32,
) -> Result<Vec<SilenceInterval>, VididxError> {
    if !is_tool_available(ffmpeg_path).await {
        return Err(VididxError::ToolNotFound(format!(
            "ffmpeg not found at: {}",
            ffmpeg_path
        )));
    }

    let output = Command::new(ffmpeg_path)
        .arg("-i")
        .arg(mp4_path)
        .arg("-af")
        .arg(format!("silencedetect=noise={}dB:d={}", noise_db, duration))
        .arg("-f")
        .arg("null")
        .arg("-")
        .output()
        .map_err(|e| VididxError::Media(format!("ffmpeg silence detection failed: {}", e)))?;

    let stderr = String::from_utf8(output.stderr).unwrap_or_default();
    parse_silence_output(&stderr)
}

/// Parse ffmpeg silencedetect stderr output.
#[allow(clippy::collapsible_if)]
fn parse_silence_output(output: &str) -> Result<Vec<SilenceInterval>, VididxError> {
    let mut intervals = Vec::new();

    // Pattern: [silencedetect @ ...] silence_start: 0.5
    //          [silencedetect @ ...] silence_end: 2.5 | silence_duration: 2
    let silence_start_regex = Regex::new(r"silence_start:\s+([\d.]+)")
        .map_err(|e| VididxError::Vision(format!("Regex compilation error: {}", e)))?;

    let silence_end_regex = Regex::new(r"silence_end:\s+([\d.]+)")
        .map_err(|e| VididxError::Vision(format!("Regex compilation error: {}", e)))?;

    let mut current_start: Option<f64> = None;

    for line in output.lines() {
        if let Some(caps) = silence_start_regex.captures(line) {
            if let Ok(start) = caps[1].parse::<f64>() {
                current_start = Some(start);
            }
        }

        if let Some(caps) = silence_end_regex.captures(line) {
            if let (Some(start), Ok(end)) = (current_start, caps[1].parse::<f64>()) {
                intervals.push(SilenceInterval {
                    start_sec: start,
                    end_sec: end,
                });
                current_start = None;
            }
        }
    }

    Ok(intervals)
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
    fn test_parse_silence_output() {
        let output = r#"
[silencedetect @ 0x...] silence_start: 0.5
[silencedetect @ 0x...] silence_end: 2.5 | silence_duration: 2
[silencedetect @ 0x...] silence_start: 5.0
[silencedetect @ 0x...] silence_end: 7.5 | silence_duration: 2.5
"#;

        let intervals = parse_silence_output(output).unwrap();
        assert_eq!(intervals.len(), 2);
        assert_eq!(intervals[0].start_sec, 0.5);
        assert_eq!(intervals[0].end_sec, 2.5);
        assert_eq!(intervals[1].start_sec, 5.0);
        assert_eq!(intervals[1].end_sec, 7.5);
    }

    #[test]
    fn test_parse_silence_empty() {
        let output = "";
        let intervals = parse_silence_output(output).unwrap();
        assert!(intervals.is_empty());
    }

    #[test]
    fn test_parse_silence_no_silence() {
        let output = "No silence detected";
        let intervals = parse_silence_output(output).unwrap();
        assert!(intervals.is_empty());
    }
}
