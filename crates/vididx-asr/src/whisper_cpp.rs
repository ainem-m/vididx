use async_trait::async_trait;
use serde_json::Value;
use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::adapter::AsrAdapter;
use vididx_core::{TranscriptSegment, TranscriptTimeline, VididxError};

/// Whisper.cpp ASR adapter.
pub struct WhisperCppAdapter {
    binary_path: String,
    model_path: String,
    threads: usize,
    language: String,
    initial_prompt: Option<String>,
    entropy_thold: f64,
    no_fallback: bool,
    suppress_nst: bool,
}

impl WhisperCppAdapter {
    /// Create a new WhisperCppAdapter.
    pub fn new(binary_path: &str, model_path: &str, threads: usize, language: &str) -> Self {
        Self {
            binary_path: binary_path.to_string(),
            model_path: expand_tilde(model_path),
            threads,
            language: language.to_string(),
            initial_prompt: None,
            entropy_thold: 2.40,
            no_fallback: false,
            suppress_nst: false,
        }
    }

    /// Create from config.
    pub fn from_config(cfg: &vididx_core::WhisperCppConfig, language: &str) -> Self {
        Self {
            binary_path: cfg.binary_path.clone(),
            model_path: expand_tilde(&cfg.model_path),
            threads: cfg.threads,
            language: language.to_string(),
            initial_prompt: cfg.initial_prompt.clone(),
            entropy_thold: cfg.entropy_thold,
            no_fallback: cfg.no_fallback,
            suppress_nst: cfg.suppress_nst,
        }
    }
}

#[async_trait]
impl AsrAdapter for WhisperCppAdapter {
    async fn transcribe(&self, wav_path: &Path) -> Result<TranscriptTimeline, VididxError> {
        // Check if binary exists
        if !is_tool_available(&self.binary_path).await {
            return Err(VididxError::ToolNotFound(format!(
                "whisper-cli not found at: {}",
                self.binary_path
            )));
        }

        // Run whisper-cli with JSON output
        let output_prefix = std::env::temp_dir().join(format!(
            "vididx_whisper_{}_{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0)
        ));
        let output_prefix_str = output_prefix.to_string_lossy().to_string();
        let output_json_path = output_prefix.with_extension("json");

        let mut cmd = Command::new(&self.binary_path);
        cmd.arg("-m")
            .arg(&self.model_path)
            .arg("-t")
            .arg(self.threads.to_string())
            .arg("-l")
            .arg(&self.language)
            .arg("--entropy-thold")
            .arg(self.entropy_thold.to_string());

        if let Some(prompt) = &self.initial_prompt {
            cmd.arg("--prompt").arg(prompt);
        }
        if self.no_fallback {
            cmd.arg("--no-fallback");
        }
        if self.suppress_nst {
            cmd.arg("--suppress-nst");
        }

        cmd.arg("-oj")
            .arg("-of")
            .arg(&output_prefix_str)
            .arg("-f")
            .arg(wav_path);

        let output = cmd
            .output()
            .map_err(|e| VididxError::Asr(format!("whisper-cli execution failed: {}", e)))?;

        if !output.status.success() {
            return Err(VididxError::Asr(format!(
                "whisper-cli failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let json = std::fs::read_to_string(&output_json_path)
            .map_err(|e| VididxError::Asr(format!("Failed to read whisper JSON output: {}", e)))?;

        let parsed = parse_whisper_output(&json);
        let _ = std::fs::remove_file(&output_json_path);
        parsed
    }
}

/// Parse whisper-cli JSON output.
fn parse_whisper_output(json_str: &str) -> Result<TranscriptTimeline, VididxError> {
    let output: Value = serde_json::from_str(json_str)
        .map_err(|e| VididxError::Asr(format!("Invalid JSON: {}", e)))?;

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

    Ok(TranscriptTimeline {
        segments: dedup_consecutive(segments),
    })
}

/// Remove consecutive segments with identical text (whisper hallucination loops).
/// When a run of duplicates is found, the first occurrence is kept with its end_sec
/// extended to cover the full run.
fn dedup_consecutive(segments: Vec<TranscriptSegment>) -> Vec<TranscriptSegment> {
    let mut out: Vec<TranscriptSegment> = Vec::with_capacity(segments.len());
    for seg in segments {
        match out.last_mut() {
            Some(prev) if prev.text == seg.text => {
                prev.end_sec = seg.end_sec.max(prev.end_sec);
            }
            _ => out.push(seg),
        }
    }
    out
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
        assert!(matches!(result, Err(VididxError::ToolNotFound(_))));
    }

    #[test]
    fn test_expand_tilde_keeps_plain_path() {
        assert_eq!(expand_tilde("/tmp/model.bin"), "/tmp/model.bin");
    }
}
