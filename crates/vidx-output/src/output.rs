use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::path::Path;
use serde::{Deserialize, Serialize};
use vidx_core::{AnnotatedChunk, VidxError};

#[derive(Debug, Serialize, Deserialize)]
pub struct OutputIndex {
    pub video_id: String,
    pub total_chunks: usize,
    pub total_duration_sec: f64,
    pub chunks_file: String,
    pub markdown_file: String,
}

/// Write annotated chunks to a JSONL file (one JSON object per line).
pub async fn write_chunks_jsonl(
    chunks: &[AnnotatedChunk],
    output_path: &Path,
) -> Result<(), VidxError> {
    create_dir_all(output_path.parent().ok_or_else(|| {
        VidxError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Invalid output path",
        ))
    })?)
    .map_err(|e| VidxError::Io(e))?;

    let file = File::create(output_path)
        .map_err(|e| VidxError::Io(e))?;
    let mut writer = BufWriter::new(file);

    for chunk in chunks {
        let json = serde_json::to_string(chunk)
            .map_err(|e| VidxError::Serde(e))?;
        writeln!(writer, "{}", json)
            .map_err(|e| VidxError::Io(e))?;
    }

    writer.flush()
        .map_err(|e| VidxError::Io(e))?;

    Ok(())
}

/// Write an index JSON file.
pub async fn write_index(
    index: &OutputIndex,
    output_path: &Path,
) -> Result<(), VidxError> {
    create_dir_all(output_path.parent().ok_or_else(|| {
        VidxError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Invalid output path",
        ))
    })?)
    .map_err(|e| VidxError::Io(e))?;

    let json = serde_json::to_string_pretty(index)
        .map_err(|e| VidxError::Serde(e))?;

    std::fs::write(output_path, json)
        .map_err(|e| VidxError::Io(e))?;

    Ok(())
}

/// Write chunks to a markdown file.
pub async fn write_markdown(
    chunks: &[AnnotatedChunk],
    video_id: &str,
    output_path: &Path,
) -> Result<(), VidxError> {
    create_dir_all(output_path.parent().ok_or_else(|| {
        VidxError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Invalid output path",
        ))
    })?)
    .map_err(|e| VidxError::Io(e))?;

    let file = File::create(output_path)
        .map_err(|e| VidxError::Io(e))?;
    let mut writer = BufWriter::new(file);

    // Write header
    writeln!(writer, "# Video Analysis: {}\n", video_id)
        .map_err(|e| VidxError::Io(e))?;

    // Write table of contents
    writeln!(writer, "## Contents\n")
        .map_err(|e| VidxError::Io(e))?;

    for (idx, chunk) in chunks.iter().enumerate() {
        let timestamp = format_timestamp(chunk.start_sec);
        writeln!(writer, "{}. [{}](#{}) - {}",
            idx + 1, chunk.title, idx, timestamp)
            .map_err(|e| VidxError::Io(e))?;
    }

    writeln!(writer, "\n---\n")
        .map_err(|e| VidxError::Io(e))?;

    // Write detailed sections
    for (idx, chunk) in chunks.iter().enumerate() {
        let start_ts = format_timestamp(chunk.start_sec);
        let end_ts = format_timestamp(chunk.end_sec);

        writeln!(writer, "## {} - {} ({}–{})\n",
            idx + 1, chunk.title, start_ts, end_ts)
            .map_err(|e| VidxError::Io(e))?;

        writeln!(writer, "**Summary:** {}\n", chunk.summary)
            .map_err(|e| VidxError::Io(e))?;

        writeln!(writer, "**Keywords:** {}\n", chunk.keywords.join(", "))
            .map_err(|e| VidxError::Io(e))?;

        if !chunk.transcript_text.is_empty() {
            writeln!(writer, "**Transcript:**\n```\n{}\n```\n", chunk.transcript_text)
                .map_err(|e| VidxError::Io(e))?;
        }

        writeln!(writer, "\n---\n")
            .map_err(|e| VidxError::Io(e))?;
    }

    writer.flush()
        .map_err(|e| VidxError::Io(e))?;

    Ok(())
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

        let chunks = vec![
            AnnotatedChunk {
                chunk_id: "chunk_0".to_string(),
                parent_segment_id: "seg_0".to_string(),
                start_sec: 0.0,
                end_sec: 50.0,
                transcript_text: "Test content".to_string(),
                title: "Test Title".to_string(),
                summary: "Test summary".to_string(),
                keywords: vec!["test".to_string()],
            },
        ];

        let result = write_chunks_jsonl(&chunks, &output_path).await;
        assert!(result.is_ok());
        assert!(output_path.exists());

        let content = std::fs::read_to_string(&output_path).unwrap();
        assert!(content.contains("chunk_0"));
        assert!(content.contains("Test Title"));
    }

    #[tokio::test]
    async fn test_write_markdown() {
        let temp_dir = tempfile::tempdir().unwrap();
        let output_path = temp_dir.path().join("output.md");

        let chunks = vec![
            AnnotatedChunk {
                chunk_id: "chunk_0".to_string(),
                parent_segment_id: "seg_0".to_string(),
                start_sec: 0.0,
                end_sec: 50.0,
                transcript_text: "Test content".to_string(),
                title: "Test Title".to_string(),
                summary: "Test summary".to_string(),
                keywords: vec!["test".to_string()],
            },
        ];

        let result = write_markdown(&chunks, "test_video", &output_path).await;
        assert!(result.is_ok());
        assert!(output_path.exists());

        let content = std::fs::read_to_string(&output_path).unwrap();
        assert!(content.contains("test_video"));
        assert!(content.contains("Test Title"));
        assert!(content.contains("Test summary"));
    }

    #[tokio::test]
    async fn test_write_index() {
        let temp_dir = tempfile::tempdir().unwrap();
        let output_path = temp_dir.path().join("index.json");

        let index = OutputIndex {
            video_id: "test_video".to_string(),
            total_chunks: 1,
            total_duration_sec: 50.0,
            chunks_file: "chunks.jsonl".to_string(),
            markdown_file: "output.md".to_string(),
        };

        let result = write_index(&index, &output_path).await;
        assert!(result.is_ok());
        assert!(output_path.exists());

        let content = std::fs::read_to_string(&output_path).unwrap();
        assert!(content.contains("test_video"));
    }
}
