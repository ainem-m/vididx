use serde::{Deserialize, Serialize};
use tera::Tera;
use vidx_core::{AnnotatedChunk, NormalizedChunk, VidxError};
use vidx_llm::AnthropicClient;

#[derive(Debug, Serialize, Deserialize)]
struct AnnotationResponse {
    title: String,
    summary: String,
    keywords: Vec<String>,
}

/// Generate annotations (title, summary, keywords) for a chunk using LLM.
pub async fn annotate_chunk(
    llm_client: &AnthropicClient,
    chunk: &NormalizedChunk,
) -> Result<AnnotatedChunk, VidxError> {
    let prompt = render_annotation_prompt(chunk)?;

    let system_prompt = "You are an expert at summarizing and annotating video content. \
        Analyze the provided transcript and generate a concise title, summary, and keywords.";

    match llm_client.call_with_json_mode(system_prompt, &prompt).await {
        Ok(response) => match parse_annotation_response(&response) {
            Ok(annotation) => Ok(AnnotatedChunk {
                chunk_id: chunk.chunk_id.clone(),
                parent_segment_id: chunk.parent_segment_id.clone(),
                start_sec: chunk.start_sec,
                end_sec: chunk.end_sec,
                transcript_text: chunk.transcript_text.clone(),
                title: annotation.title,
                summary: annotation.summary,
                keywords: annotation.keywords,
            }),
            Err(_) => Ok(create_fallback_annotation(chunk)),
        },
        Err(_) => Ok(create_fallback_annotation(chunk)),
    }
}

fn render_annotation_prompt(chunk: &NormalizedChunk) -> Result<String, VidxError> {
    let template_content = include_str!("../../../prompts/annotate_chunk.jinja");

    let mut tera = Tera::new("")
        .map_err(|e| VidxError::Segment(format!("Failed to create template engine: {}", e)))?;

    tera.add_raw_template("annotate", template_content)
        .map_err(|e| VidxError::Segment(format!("Failed to add template: {}", e)))?;

    let mut context = tera::Context::new();
    context.insert("transcript_text", &chunk.transcript_text);
    context.insert("duration_sec", &(chunk.end_sec - chunk.start_sec));

    tera.render("annotate", &context)
        .map_err(|e| VidxError::Segment(format!("Failed to render template: {}", e)))
}

fn parse_annotation_response(
    response: &serde_json::Value,
) -> Result<AnnotationResponse, VidxError> {
    serde_json::from_value(response.clone())
        .map_err(|e| VidxError::Segment(format!("Failed to parse annotation: {}", e)))
}

fn create_fallback_annotation(chunk: &NormalizedChunk) -> AnnotatedChunk {
    let first_words: Vec<&str> = chunk.transcript_text.split_whitespace().take(10).collect();
    let default_title = if !first_words.is_empty() {
        first_words.join(" ")
    } else {
        format!("Segment {}", chunk.chunk_id)
    };

    let default_title = if default_title.len() > 50 {
        format!("{}...", &default_title[..47])
    } else {
        default_title
    };

    AnnotatedChunk {
        chunk_id: chunk.chunk_id.clone(),
        parent_segment_id: chunk.parent_segment_id.clone(),
        start_sec: chunk.start_sec,
        end_sec: chunk.end_sec,
        transcript_text: chunk.transcript_text.clone(),
        title: default_title,
        summary: "Failed to generate summary from LLM".to_string(),
        keywords: vec!["content".to_string()],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_valid_annotation() {
        let response = serde_json::json!({
            "title": "Introduction to Rust",
            "summary": "An overview of Rust programming basics.",
            "keywords": ["rust", "programming", "basics"]
        });

        let result = parse_annotation_response(&response);
        assert!(result.is_ok());
        let annotation = result.unwrap();
        assert_eq!(annotation.title, "Introduction to Rust");
        assert_eq!(annotation.keywords.len(), 3);
    }

    #[test]
    fn test_parse_invalid_annotation() {
        let response = serde_json::json!({
            "wrong_field": "value"
        });

        let result = parse_annotation_response(&response);
        assert!(result.is_err());
    }

    #[test]
    fn test_fallback_annotation() {
        let chunk = NormalizedChunk {
            chunk_id: "test_chunk".to_string(),
            parent_segment_id: "segment_0".to_string(),
            start_sec: 0.0,
            end_sec: 50.0,
            transcript_text: "This is a test transcript with some content.".to_string(),
        };

        let result = create_fallback_annotation(&chunk);
        assert_eq!(result.chunk_id, "test_chunk");
        assert!(!result.title.is_empty());
        assert!(!result.summary.is_empty());
        assert_eq!(result.keywords.len(), 1);
    }

    #[test]
    fn test_fallback_long_title_truncation() {
        let chunk = NormalizedChunk {
            chunk_id: "test".to_string(),
            parent_segment_id: "seg".to_string(),
            start_sec: 0.0,
            end_sec: 50.0,
            transcript_text:
                "This is a very long transcript that should be truncated when used as a title \
                              because it exceeds the maximum title length of fifty characters."
                    .to_string(),
        };

        let result = create_fallback_annotation(&chunk);
        assert!(result.title.len() <= 50);
    }
}
