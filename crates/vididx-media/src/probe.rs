use serde_json::Value;
use std::path::Path;
use std::process::Command;

use vididx_core::{MediaProbe, VididxError};

/// Probe a video file using ffprobe to extract metadata.
pub async fn probe(ffprobe_path: &str, mp4_path: &Path) -> Result<MediaProbe, VididxError> {
    // Check if ffprobe exists
    if !is_tool_available(ffprobe_path).await {
        return Err(VididxError::ToolNotFound(format!(
            "ffprobe not found at: {}",
            ffprobe_path
        )));
    }

    // Run ffprobe with JSON output
    let output = Command::new(ffprobe_path)
        .arg("-v")
        .arg("error")
        .arg("-show_format")
        .arg("-show_streams")
        .arg("-of")
        .arg("json")
        .arg(mp4_path)
        .output()
        .map_err(|e| VididxError::Media(format!("ffprobe execution failed: {}", e)))?;

    if !output.status.success() {
        return Err(VididxError::Media(format!(
            "ffprobe failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }

    let stdout = String::from_utf8(output.stdout)
        .map_err(|e| VididxError::Media(format!("ffprobe output not UTF-8: {}", e)))?;

    parse_ffprobe_output(&stdout)
}

/// Parse ffprobe JSON output into MediaProbe.
fn parse_ffprobe_output(json_str: &str) -> Result<MediaProbe, VididxError> {
    let probe: Value = serde_json::from_str(json_str)
        .map_err(|e| VididxError::Media(format!("Invalid ffprobe JSON: {}", e)))?;

    // Extract duration from format
    let duration_sec = probe
        .get("format")
        .and_then(|f| f.get("duration"))
        .and_then(|d| d.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);

    // Find video and audio streams
    let empty_vec = vec![];
    let streams = probe
        .get("streams")
        .and_then(|s| s.as_array())
        .unwrap_or(&empty_vec);

    let mut has_video = false;
    let mut has_audio = false;
    let mut fps = 0.0;
    let mut width = 0u32;
    let mut height = 0u32;

    for stream in streams {
        if let Some(codec_type) = stream.get("codec_type").and_then(|ct| ct.as_str()) {
            match codec_type {
                "video" => {
                    has_video = true;
                    // Extract fps from r_frame_rate
                    if let Some(r_frame_rate) = stream.get("r_frame_rate").and_then(|r| r.as_str())
                    {
                        fps = parse_fraction(r_frame_rate).unwrap_or(0.0);
                    }
                    // Extract resolution
                    if let Some(w) = stream.get("width").and_then(|v| v.as_u64()) {
                        width = w as u32;
                    }
                    if let Some(h) = stream.get("height").and_then(|v| v.as_u64()) {
                        height = h as u32;
                    }
                }
                "audio" => {
                    has_audio = true;
                }
                _ => {}
            }
        }
    }

    Ok(MediaProbe {
        duration_sec,
        fps,
        resolution: (width, height),
        has_video,
        has_audio,
    })
}

/// Parse fraction string like "30000/1001" to f64.
fn parse_fraction(frac: &str) -> Option<f64> {
    let parts: Vec<&str> = frac.split('/').collect();
    if parts.len() != 2 {
        return None;
    }
    let num: f64 = parts[0].parse().ok()?;
    let denom: f64 = parts[1].parse().ok()?;
    Some(num / denom)
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
    fn test_parse_fraction() {
        assert_eq!(parse_fraction("30000/1001"), Some(29.97002997002997));
        assert_eq!(parse_fraction("30/1"), Some(30.0));
        assert_eq!(parse_fraction("invalid"), None);
    }

    #[test]
    fn test_parse_ffprobe_output() {
        let json = r#"{
            "format": {
                "duration": "60.5"
            },
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "30000/1001"
                },
                {
                    "codec_type": "audio"
                }
            ]
        }"#;

        let probe = parse_ffprobe_output(json).unwrap();
        assert_eq!(probe.duration_sec, 60.5);
        assert!(probe.has_video);
        assert!(probe.has_audio);
        assert_eq!(probe.resolution, (1920, 1080));
        assert!((probe.fps - 29.97).abs() < 0.01);
    }

    #[test]
    fn test_parse_ffprobe_minimal() {
        let json = r#"{
            "format": {},
            "streams": []
        }"#;

        let probe = parse_ffprobe_output(json).unwrap();
        assert_eq!(probe.duration_sec, 0.0);
        assert!(!probe.has_video);
        assert!(!probe.has_audio);
    }

    #[test]
    fn test_parse_ffprobe_invalid_json() {
        let result = parse_ffprobe_output("invalid json");
        assert!(result.is_err());
    }
}
