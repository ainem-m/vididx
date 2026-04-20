use async_trait::async_trait;
use serde_json::Value;
use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::adapter::AsrAdapter;
use vidx_core::{TranscriptSegment, TranscriptTimeline, VidxError};

/// Whisper.cpp ASR adapter.
pub struct WhisperCppAdapter {
    binary_path: String,
    model_path: String,
    threads: usize,
    language: String,
}

impl WhisperCppAdapter {
    /// Create a new WhisperCppAdapter.
    pub fn new(binary_path: &str, model_path: &str, threads: usize, language: &str) -> Self {
        Self {
            binary_path: binary_path.to_string(),
            model_path: expand_tilde(model_path),
            threads,
            language: language.to_string(),
        }
    }
}

#[async_trait]
impl AsrAdapter for WhisperCppAdapter {
    async fn transcribe(&self, wav_path: &Path) -> Result<TranscriptTimeline, VidxError> {
        // Check if binary exists
        if !is_tool_available(&self.binary_path).await {
            return Err(VidxError::ToolNotFound(format!(
                "whisper-cli not found at: {}",
                self.binary_path
            )));
        }

        // Run whisper-cli with JSON output
        let output_prefix = std::env::temp_dir().join(format!(
            "vidx_whisper_{}_{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0)
        ));
        let output_prefix_str = output_prefix.to_string_lossy().to_string();
        let output_json_path = output_prefix.with_extension("json");

        let output = Command::new(&self.binary_path)
            .arg("-m")
            .arg(&self.model_path)
            .arg("-t")
            .arg(self.threads.to_string())
            .arg("-l")
            .arg(&self.language)
            .arg("-oj")
            .arg("-of")
            .arg(&output_prefix_str)
            .arg("-f")
            .arg(wav_path)
            .output()
            .map_err(|e| VidxError::Asr(format!("whisper-cli execution failed: {}", e)))?;

        if !output.status.success() {
            return Err(VidxError::Asr(format!(
                "whisper-cli failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let json = std::fs::read_to_string(&output_json_path)
            .map_err(|e| VidxError::Asr(format!("Failed to read whisper JSON output: {}", e)))?;

        let parsed = parse_whisper_output(&json);
        let _ = std::fs::remove_file(&output_json_path);
        parsed
    }
}

/// Parse whisper-cli JSON output.
fn parse_whisper_output(json_str: &str) -> Result<TranscriptTimeline, VidxError> {
    let output: Value = serde_json::from_str(json_str)
        .map_err(|e| VidxError::Asr(format!("Invalid JSON: {}", e)))?;

    let mut segments = Vec::new();

    if let Some(segs) = output.get("result").and_then(|r| r.as_array()) {
        for seg in segs {
            let start_sec = seg.get("start").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let end_sec = seg.get("end").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let text = seg
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();

            // Skip empty segments
            if text.is_empty() {
                continue;
            }

            segments.push(TranscriptSegment {
                start_sec,
                end_sec,
                text,
                speaker: None,
                confidence: None,
            });
        }
    } else if let Some(segs) = output.get("transcription").and_then(|r| r.as_array()) {
        for seg in segs {
            let start_sec = seg
                .get("offsets")
                .and_then(|o| o.get("from"))
                .and_then(|v| v.as_u64())
                .map(|ms| ms as f64 / 1000.0)
                .unwrap_or(0.0);
            let end_sec = seg
                .get("offsets")
                .and_then(|o| o.get("to"))
                .and_then(|v| v.as_u64())
                .map(|ms| ms as f64 / 1000.0)
                .unwrap_or(0.0);
            let text = seg
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();

            if text.is_empty() {
                continue;
            }

            segments.push(TranscriptSegment {
                start_sec,
                end_sec,
                text,
                speaker: None,
                confidence: None,
            });
        }
    }

    Ok(TranscriptTimeline { segments })
}

async fn is_tool_available(tool_path: &str) -> bool {
    Command::new(tool_path)
        .arg("--help")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn expand_tilde(path: &str) -> String {
    if path == "~" {
        return dirs::home_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| path.to_string());
    }

    if let Some(stripped) = path.strip_prefix("~/") {
        return dirs::home_dir()
            .map(|p| p.join(stripped).to_string_lossy().to_string())
            .unwrap_or_else(|| path.to_string());
    }

    path.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_whisper_output() {
        let json = r#"{
            "result": [
                {
                    "start": 0.0,
                    "end": 2.5,
                    "text": "Hello world"
                },
                {
                    "start": 3.0,
                    "end": 5.5,
                    "text": "How are you?"
                }
            ]
        }"#;

        let timeline = parse_whisper_output(json).unwrap();
        assert_eq!(timeline.segments.len(), 2);
        assert_eq!(timeline.segments[0].text, "Hello world");
        assert_eq!(timeline.segments[0].start_sec, 0.0);
        assert_eq!(timeline.segments[0].end_sec, 2.5);
        assert_eq!(timeline.segments[1].text, "How are you?");
    }

    #[test]
    fn test_parse_whisper_empty_segments_filtered() {
        let json = r#"{
            "result": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Hello"
                },
                {
                    "start": 1.0,
                    "end": 2.0,
                    "text": "   "
                },
                {
                    "start": 2.0,
                    "end": 3.0,
                    "text": "World"
                }
            ]
        }"#;

        let timeline = parse_whisper_output(json).unwrap();
        assert_eq!(timeline.segments.len(), 2);
        assert_eq!(timeline.segments[0].text, "Hello");
        assert_eq!(timeline.segments[1].text, "World");
    }

    #[test]
    fn test_parse_whisper_invalid_json() {
        let result = parse_whisper_output("invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_whisper_no_result() {
        let json = r#"{ "other": "field" }"#;
        let timeline = parse_whisper_output(json).unwrap();
        assert!(timeline.segments.is_empty());
    }

    #[test]
    fn test_parse_whisper_transcription_format() {
        let json = r#"{
            "transcription": [
                {
                    "offsets": { "from": 0, "to": 2060 },
                    "text": "こんにちは"
                },
                {
                    "offsets": { "from": 2060, "to": 4000 },
                    "text": "世界"
                }
            ]
        }"#;

        let timeline = parse_whisper_output(json).unwrap();
        assert_eq!(timeline.segments.len(), 2);
        assert_eq!(timeline.segments[0].start_sec, 0.0);
        assert_eq!(timeline.segments[0].end_sec, 2.06);
        assert_eq!(timeline.segments[0].text, "こんにちは");
        assert_eq!(timeline.segments[1].text, "世界");
    }

    #[tokio::test]
    async fn test_whisper_cpp_adapter_nonexistent() {
        let adapter = WhisperCppAdapter::new("nonexistent_whisper", "/path/to/model", 8, "ja");
        let result = adapter.transcribe(Path::new("/tmp/audio.wav")).await;
        assert!(matches!(result, Err(VidxError::ToolNotFound(_))));
    }

    #[test]
    fn test_expand_tilde_keeps_plain_path() {
        assert_eq!(expand_tilde("/tmp/model.bin"), "/tmp/model.bin");
    }
}
