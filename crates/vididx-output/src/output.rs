use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::path::Path;
use vididx_core::{AnnotatedChunk, Chunk, VididxError};

#[derive(Debug, Serialize, Deserialize)]
pub struct OutputIndex {
    pub video_id: String,
    pub source_path: String,
    pub source_hash: String,
    pub config_hash: String,
    pub total_chunks: usize,
    pub total_duration_sec: f64,
    pub chunks_file: String,
    pub markdown_file: String,
    pub generated_at: DateTime<Utc>,
}

/// Write annotated chunks to a JSONL file (one JSON object per line).
pub async fn write_chunks_jsonl(
    chunks: &[AnnotatedChunk],
    output_path: &Path,
) -> Result<(), VididxError> {
    create_dir_all(output_path.parent().ok_or_else(|| {
        VididxError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Invalid output path",
        ))
    })?)
    .map_err(VididxError::Io)?;

    let file = File::create(output_path).map_err(VididxError::Io)?;
    let mut writer = BufWriter::new(file);

    for chunk in chunks {
        let json = serde_json::to_string(chunk).map_err(VididxError::Serde)?;
        writeln!(writer, "{}", json).map_err(VididxError::Io)?;
    }

    writer.flush().map_err(VididxError::Io)?;

    Ok(())
}

/// Write an index JSON file.
pub async fn write_index(index: &OutputIndex, output_path: &Path) -> Result<(), VididxError> {
    create_dir_all(output_path.parent().ok_or_else(|| {
        VididxError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Invalid output path",
        ))
    })?)
    .map_err(VididxError::Io)?;

    let json = serde_json::to_string_pretty(index).map_err(VididxError::Serde)?;

    std::fs::write(output_path, json).map_err(VididxError::Io)?;

    Ok(())
}

/// Write chunks to a markdown file, embedding image refs relative to the output directory.
pub async fn write_markdown(
    chunks: &[Chunk],
    video_id: &str,
    output_path: &Path,
) -> Result<(), VididxError> {
    create_dir_all(output_path.parent().ok_or_else(|| {
        VididxError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Invalid output path",
        ))
    })?)
    .map_err(VididxError::Io)?;

    let md_dir = output_path.parent().unwrap_or(Path::new("."));

    let file = File::create(output_path).map_err(VididxError::Io)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "# Video Analysis: {}\n", video_id).map_err(VididxError::Io)?;
    writeln!(writer, "## Contents\n").map_err(VididxError::Io)?;

    for (idx, chunk) in chunks.iter().enumerate() {
        let timestamp = format_timestamp(chunk.start_sec);
        writeln!(
            writer,
            "{}. [{}](#{}) - {}",
            idx + 1,
            chunk.title,
            idx,
            timestamp
        )
        .map_err(VididxError::Io)?;
    }

    writeln!(writer, "\n---\n").map_err(VididxError::Io)?;

    for (idx, chunk) in chunks.iter().enumerate() {
        let start_ts = format_timestamp(chunk.start_sec);
        let end_ts = format_timestamp(chunk.end_sec);

        writeln!(
            writer,
            "## {} - {} ({}–{})\n",
            idx + 1,
            chunk.title,
            start_ts,
            end_ts
        )
        .map_err(VididxError::Io)?;

        writeln!(writer, "**Summary:** {}\n", chunk.summary).map_err(VididxError::Io)?;
        writeln!(writer, "**Keywords:** {}\n", chunk.keywords.join(", "))
            .map_err(VididxError::Io)?;

        if let Some(ref ocr) = chunk.ocr_text {
            writeln!(writer, "**OCR:** {}\n", ocr).map_err(VididxError::Io)?;
        }

        if let Some(ref caption) = chunk.visual_caption {
            writeln!(writer, "**Visual Caption:** {}\n", caption).map_err(VididxError::Io)?;
        }

        for image_ref in &chunk.image_refs {
            let rel_path = relative_image_path(&image_ref.path, md_dir);
            let at = format_timestamp(image_ref.at_sec);
            writeln!(writer, "![Frame at {}]({})\n", at, rel_path).map_err(VididxError::Io)?;
        }

        if !chunk.transcript.is_empty() {
            writeln!(writer, "**Transcript:**\n```\n{}\n```\n", chunk.transcript)
                .map_err(VididxError::Io)?;
        }

        let flags = &chunk.modality_flags;
        writeln!(
            writer,
            "**Modality:** speech={}, ocr={}, visual={}\n",
            flags.has_speech, flags.has_ocr, flags.has_visual
        )
        .map_err(VididxError::Io)?;

        writeln!(writer, "\n---\n").map_err(VididxError::Io)?;
    }

    writer.flush().map_err(VididxError::Io)?;
    Ok(())
}

fn relative_image_path(image_path: &str, md_dir: &Path) -> String {
    let p = Path::new(image_path);
    p.strip_prefix(md_dir)
        .map(|rel| rel.to_string_lossy().into_owned())
        .unwrap_or_else(|_| image_path.to_owned())
}

fn format_timestamp(seconds: f64) -> String {
    let total_secs = seconds as u32;
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let secs = total_secs % 60;

    if hours > 0 {
        format!("{}:{:02}:{:02}", hours, minutes, secs)
    } else {
        format!("{}:{:02}", minutes, secs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_timestamp_seconds() {
        assert_eq!(format_timestamp(30.5), "0:30");
    }

    #[test]
    fn test_format_timestamp_minutes() {
        assert_eq!(format_timestamp(125.0), "2:05");
    }

    #[test]
    fn test_format_timestamp_hours() {
        assert_eq!(format_timestamp(3661.0), "1:01:01");
    }

    #[tokio::test]
    async fn test_write_chunks_jsonl() {
        let temp_dir = tempfile::tempdir().unwrap();
        let output_path = temp_dir.path().join("chunks.jsonl");

        let chunks = vec![AnnotatedChunk {
            chunk_id: "chunk_0".to_string(),
            parent_segment_id: "seg_0".to_string(),
            start_sec: 0.0,
            end_sec: 50.0,
            transcript_text: "Test content".to_string(),
            title: "Test Title".to_string(),
            summary: "Test summary".to_string(),
            keywords: vec!["test".to_string()],
        }];

        let result = write_chunks_jsonl(&chunks, &output_path).await;
        assert!(result.is_ok());
        assert!(output_path.exists());

        let content = std::fs::read_to_string(&output_path).unwrap();
        assert!(content.contains("chunk_0"));
        assert!(content.contains("Test Title"));
    }

    #[tokio::test]
    async fn test_write_markdown() {
        use chrono::Utc;
        use vididx_core::{
            Chunk, ContentType, FrameKind, ImageRef, ModalityFlags, ProcessingMeta, SourceJumpRef,
        };

        let temp_dir = tempfile::tempdir().unwrap();
        let output_path = temp_dir.path().join("output.md");
        let img_path = temp_dir.path().join("frame_000.jpg");
        std::fs::write(&img_path, b"fake").unwrap();

        let chunks = vec![Chunk {
            schema_version: "1.0".to_string(),
            video_id: "test_video".to_string(),
            source_type: "local_mp4".to_string(),
            source_path: "/tmp/test.mp4".to_string(),
            source_hash: "sha256:abc".to_string(),
            chunk_id: "test_video_chunk_0000".to_string(),
            parent_segment_id: String::new(),
            start_sec: 0.0,
            end_sec: 50.0,
            start_tc: "00:00:00.000".to_string(),
            end_tc: "00:00:50.000".to_string(),
            duration_sec: 50.0,
            content_type: ContentType::ScreenRecording,
            speaker_info: vec![],
            title: "Test Title".to_string(),
            summary: "Test summary".to_string(),
            transcript: "Test content".to_string(),
            ocr_text: Some("OCR text".to_string()),
            visual_caption: Some("Visual caption".to_string()),
            keywords: vec!["test".to_string()],
            embedding_text: "Test Title\nTest summary\nTest content".to_string(),
            image_refs: vec![ImageRef {
                path: img_path.to_string_lossy().to_string(),
                at_sec: 0.0,
                kind: FrameKind::Periodic,
                analyzed: true,
            }],
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
                segmentation_rationale: None,
                asr_adapter: "whisper_cpp".to_string(),
                vlm_adapter: "claude".to_string(),
                generated_at: Utc::now(),
            },
        }];

        let result = write_markdown(&chunks, "test_video", &output_path).await;
        assert!(result.is_ok());
        assert!(output_path.exists());

        let content = std::fs::read_to_string(&output_path).unwrap();
        assert!(content.contains("test_video"));
        assert!(content.contains("Test Title"));
        assert!(content.contains("Test summary"));
        assert!(content.contains("![Frame at"));
        assert!(content.contains("**OCR:** OCR text"));
        assert!(content.contains("**Visual Caption:** Visual caption"));
        assert!(content.contains("**Modality:** speech=true, ocr=true, visual=true"));
    }

    #[tokio::test]
    async fn test_write_index() {
        let temp_dir = tempfile::tempdir().unwrap();
        let output_path = temp_dir.path().join("index.json");

        let index = OutputIndex {
            video_id: "test_video".to_string(),
            source_path: "/tmp/test.mp4".to_string(),
            source_hash: "sha256:abc".to_string(),
            config_hash: "sha256:cfg".to_string(),
            total_chunks: 1,
            total_duration_sec: 50.0,
            chunks_file: "chunks.jsonl".to_string(),
            markdown_file: "output.md".to_string(),
            generated_at: Utc::now(),
        };

        let result = write_index(&index, &output_path).await;
        assert!(result.is_ok());
        assert!(output_path.exists());

        let content = std::fs::read_to_string(&output_path).unwrap();
        assert!(content.contains("test_video"));
        assert!(content.contains("sha256:abc"));
        assert!(content.contains("sha256:cfg"));
    }
}
