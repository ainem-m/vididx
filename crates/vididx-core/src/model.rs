use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Content type classification for chunks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ContentType {
    ScreenRecording,
    Slide,
    Conversation,
    Lecture,
    #[default]
    Unknown,
}

/// Kind of extracted frame.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FrameKind {
    Periodic,
    SceneChange,
    /// Frame extracted at the start of an ASR utterance (utterance mode).
    UtteranceStart,
}

/// Flags indicating presence of different modalities in a chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalityFlags {
    pub has_speech: bool,
    pub has_ocr: bool,
    pub has_visual: bool,
}

/// Reference to an extracted frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageRef {
    pub path: String,
    pub at_sec: f64,
    pub kind: FrameKind,
    pub analyzed: bool,
}

/// Source jump reference for navigation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceJumpRef {
    #[serde(rename = "type")]
    pub ref_type: String,
    pub start_sec: f64,
}

/// Processing metadata for a chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMeta {
    pub segmentation_rationale: Option<String>,
    pub asr_adapter: String,
    pub vlm_adapter: String,
    pub generated_at: DateTime<Utc>,
}

/// Main output chunk structure (JSONL record).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub schema_version: String,
    pub video_id: String,
    pub source_type: String,
    pub source_path: String,
    pub source_hash: String,
    pub chunk_id: String,
    pub parent_segment_id: String,
    pub start_sec: f64,
    pub end_sec: f64,
    pub start_tc: String,
    pub end_tc: String,
    pub duration_sec: f64,
    pub content_type: ContentType,
    pub speaker_info: Vec<String>,
    pub title: String,
    pub summary: String,
    pub transcript: String,
    pub ocr_text: Option<String>,
    pub visual_caption: Option<String>,
    pub keywords: Vec<String>,
    pub embedding_text: String,
    pub image_refs: Vec<ImageRef>,
    pub source_jump_ref: SourceJumpRef,
    pub modality_flags: ModalityFlags,
    pub processing_meta: ProcessingMeta,
}

/// Single segment from transcript with timing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptSegment {
    pub start_sec: f64,
    pub end_sec: f64,
    pub text: String,
    pub speaker: Option<String>,
    pub confidence: Option<f32>,
}

/// Complete transcript timeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptTimeline {
    pub segments: Vec<TranscriptSegment>,
}

/// Detected silence interval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SilenceInterval {
    pub start_sec: f64,
    pub end_sec: f64,
}

/// Detected scene change.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneChange {
    pub at_sec: f64,
    pub score: f64,
}

/// Media probe information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaProbe {
    pub duration_sec: f64,
    pub fps: f64,
    pub resolution: (u32, u32),
    pub has_video: bool,
    pub has_audio: bool,
}

/// Coarse segment (粗分割).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoarseSegment {
    pub index: usize,
    pub start_sec: f64,
    pub end_sec: f64,
    pub transcript_text: String,
}

/// Semantic chunk from meaning-based segmentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticChunk {
    pub start_sec: f64,
    pub end_sec: f64,
    pub transcript_text: String,
    pub rationale: String,
}

/// Normalized chunk after post-processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedChunk {
    pub chunk_id: String,
    pub parent_segment_id: String,
    pub start_sec: f64,
    pub end_sec: f64,
    pub transcript_text: String,
}

/// Extracted frame with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedFrame {
    pub path: PathBuf,
    pub at_sec: f64,
    pub kind: FrameKind,
    pub chunk_id: String,
    pub analyze: bool,
}

/// Analysis result for a single frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameAnalysis {
    pub frame_path: PathBuf,
    pub ocr_text: Option<String>,
    pub visual_caption: Option<String>,
}

/// Annotated chunk with title, summary, keywords.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotatedChunk {
    pub chunk_id: String,
    pub parent_segment_id: String,
    pub start_sec: f64,
    pub end_sec: f64,
    pub transcript_text: String,
    pub title: String,
    pub summary: String,
    pub keywords: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_roundtrip_serialize() {
        let chunk = Chunk {
            schema_version: "1.0".to_string(),
            video_id: "vid_001".to_string(),
            source_type: "local_mp4".to_string(),
            source_path: "/videos/sample.mp4".to_string(),
            source_hash: "sha256:abc123".to_string(),
            chunk_id: "vid_001_chunk_0001".to_string(),
            parent_segment_id: "vid_001_seg_0001".to_string(),
            start_sec: 0.0,
            end_sec: 60.0,
            start_tc: "00:00:00.000".to_string(),
            end_tc: "00:01:00.000".to_string(),
            duration_sec: 60.0,
            content_type: ContentType::ScreenRecording,
            speaker_info: vec!["speaker_a".to_string()],
            title: "Test chunk".to_string(),
            summary: "Test summary".to_string(),
            transcript: "Test transcript".to_string(),
            ocr_text: Some("Test OCR".to_string()),
            visual_caption: Some("Test caption".to_string()),
            keywords: vec!["test".to_string()],
            embedding_text: "embedding".to_string(),
            image_refs: vec![],
            source_jump_ref: SourceJumpRef {
                ref_type: "local".to_string(),
                start_sec: 0.0,
            },
            modality_flags: ModalityFlags {
                has_speech: true,
                has_ocr: true,
                has_visual: true,
            },
            processing_meta: ProcessingMeta {
                segmentation_rationale: Some("test".to_string()),
                asr_adapter: "whisper_cpp".to_string(),
                vlm_adapter: "claude".to_string(),
                generated_at: Utc::now(),
            },
        };

        let json = serde_json::to_string(&chunk).expect("serialize");
        let deserialized: Chunk = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(chunk.chunk_id, deserialized.chunk_id);
        assert_eq!(chunk.title, deserialized.title);
        assert_eq!(chunk.content_type, deserialized.content_type);
    }

    #[test]
    fn test_content_type_snake_case() {
        let ct = ContentType::ScreenRecording;
        let json = serde_json::to_string(&ct).unwrap();
        assert_eq!(json, "\"screen_recording\"");

        let ct2 = ContentType::Lecture;
        let json2 = serde_json::to_string(&ct2).unwrap();
        assert_eq!(json2, "\"lecture\"");
    }

    #[test]
    fn test_frame_kind_snake_case() {
        let fk = FrameKind::SceneChange;
        let json = serde_json::to_string(&fk).unwrap();
        assert_eq!(json, "\"scene_change\"");

        let fk2 = FrameKind::Periodic;
        let json2 = serde_json::to_string(&fk2).unwrap();
        assert_eq!(json2, "\"periodic\"");
    }

    #[test]
    fn test_content_type_default() {
        let ct = ContentType::default();
        assert_eq!(ct, ContentType::Unknown);
    }
}
