use regex::Regex;
use std::path::Path;
use std::process::Command;

use vidx_core::{SceneChange, VidxError};

/// Detect scene changes in a video using ffmpeg scene detection filter.
pub async fn detect_scene_changes(
    ffmpeg_path: &str,
    mp4_path: &Path,
    threshold: f32,
) -> Result<Vec<SceneChange>, VidxError> {
    if !is_tool_available(ffmpeg_path).await {
        return Err(VidxError::ToolNotFound(format!(
            "ffmpeg not found at: {}",
            ffmpeg_path
        )));
    }

    let output = Command::new(ffmpeg_path)
        .arg("-i")
        .arg(mp4_path)
        .arg("-vf")
        .arg(format!("select='gt(scene,{})',showinfo", threshold))
        .arg("-f")
        .arg("null")
        .arg("-")
        .output()
        .map_err(|e| VidxError::Media(format!("ffmpeg scene detection failed: {}", e)))?;

    let stderr = String::from_utf8(output.stderr).unwrap_or_default();
    parse_scene_output(&stderr)
}

/// Parse ffmpeg scene detection output (showinfo).
#[allow(clippy::collapsible_if)]
fn parse_scene_output(output: &str) -> Result<Vec<SceneChange>, VidxError> {
    let mut changes = Vec::new();

    // Pattern: [Parsed_showinfo_... pts:value time:value
    // We extract the time value which is in format HH:MM:SS.mmm
    let timecode_regex = Regex::new(r"time=([\d:.]+)")
        .map_err(|e| VidxError::Vision(format!("Regex compilation error: {}", e)))?;
    let pts_time_regex = Regex::new(r"pts_time:([\d.]+)")
        .map_err(|e| VidxError::Vision(format!("Regex compilation error: {}", e)))?;

    for line in output.lines() {
        if let Some(caps) = timecode_regex.captures(line) {
            if let Ok(at_sec) = parse_timecode(&caps[1]) {
                changes.push(SceneChange {
                    at_sec,
                    score: 0.0, // ffmpeg doesn't easily expose the actual score
                });
            }
        } else if let Some(caps) = pts_time_regex.captures(line) {
            if let Ok(at_sec) = caps[1].parse::<f64>() {
                changes.push(SceneChange {
                    at_sec,
                    score: 0.0,
                });
            }
        }
    }

    Ok(changes)
}

/// Parse timecode string "HH:MM:SS.mmm" to seconds.
fn parse_timecode(tc: &str) -> Result<f64, VidxError> {
    let parts: Vec<&str> = tc.split(':').collect();
    if parts.len() != 3 {
        return Err(VidxError::Media(format!("Invalid timecode: {}", tc)));
    }

    let hours: f64 = parts[0]
        .parse()
        .map_err(|_| VidxError::Media(format!("Invalid hours in timecode: {}", tc)))?;
    let minutes: f64 = parts[1]
        .parse()
        .map_err(|_| VidxError::Media(format!("Invalid minutes in timecode: {}", tc)))?;
    let seconds: f64 = parts[2]
        .parse()
        .map_err(|_| VidxError::Media(format!("Invalid seconds in timecode: {}", tc)))?;

    Ok(hours * 3600.0 + minutes * 60.0 + seconds)
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
    fn test_parse_timecode() {
        assert_eq!(parse_timecode("00:00:00.000").unwrap(), 0.0);
        assert_eq!(parse_timecode("00:01:00.000").unwrap(), 60.0);
        assert_eq!(parse_timecode("01:00:00.000").unwrap(), 3600.0);
        assert_eq!(parse_timecode("00:00:30.500").unwrap(), 30.5);
        assert!(parse_timecode("invalid").is_err());
    }

    #[test]
    fn test_parse_scene_output() {
        let output = r#"
[Parsed_showinfo_0 @ ...] n:0 pts:0 pts_time:0 time=00:00:01.234
[Parsed_showinfo_0 @ ...] n:1 pts:150 pts_time:5 time=00:00:05.000
[Parsed_showinfo_0 @ ...] n:2 pts:600 pts_time:20 time=00:00:20.500
"#;

        let changes = parse_scene_output(output).unwrap();
        assert_eq!(changes.len(), 3);
        assert_eq!(changes[0].at_sec, 1.234);
        assert_eq!(changes[1].at_sec, 5.0);
        assert_eq!(changes[2].at_sec, 20.5);
    }

    #[test]
    fn test_parse_scene_empty() {
        let output = "";
        let changes = parse_scene_output(output).unwrap();
        assert!(changes.is_empty());
    }
}
