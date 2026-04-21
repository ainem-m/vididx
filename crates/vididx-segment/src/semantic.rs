use serde::{Deserialize, Serialize};
use tera::Tera;
use vididx_core::{CoarseSegment, SemanticChunk, VididxError};
use vididx_llm::AnthropicClient;

#[derive(Debug, Serialize, Deserialize)]
struct ChunkResponse {
    chunks: Vec<ChunkData>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChunkData {
    start_sec: f64,
    end_sec: f64,
    rationale: String,
}

/// Segment using LLM-based semantic boundary detection.
pub async fn semantic_chunk(
    llm_client: &AnthropicClient,
    coarse: &CoarseSegment,
    target_min: f64,
    target_max: f64,
) -> Result<Vec<SemanticChunk>, VididxError> {
    let prompt = render_semantic_prompt(coarse, target_min, target_max)?;

    let system_prompt = "You are an expert at identifying semantic boundaries in transcripts. \
        Analyze the provided transcript and return valid JSON chunks with semantic breakpoints.";

    // Try LLM-based semantic chunking with one retry
    let result = llm_client.call_with_json_mode(system_prompt, &prompt).await;

    let chunks = match result {
        Ok(response) => {
            match parse_and_validate_chunks(&response, coarse.end_sec) {
                Ok(chunks) => chunks,
                Err(_) => {
                    // Retry once on validation failure
                    let result = llm_client.call_with_json_mode(system_prompt, &prompt).await;
                    match result {
                        Ok(response) => {
                            match parse_and_validate_chunks(&response, coarse.end_sec) {
                                Ok(chunks) => chunks,
                                Err(_) => {
                                    // Fall back to equal division
                                    return Ok(equal_division(coarse, target_min, target_max));
                                }
                            }
                        }
                        Err(_) => {
                            // Fall back to equal division
                            return Ok(equal_division(coarse, target_min, target_max));
                        }
                    }
                }
            }
        }
        Err(_) => {
            // Fall back to equal division
            return Ok(equal_division(coarse, target_min, target_max));
        }
    };

    Ok(chunks)
}

fn render_semantic_prompt(
    coarse: &CoarseSegment,
    target_min: f64,
    target_max: f64,
) -> Result<String, VididxError> {
    let template_path = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());

    // Try to find the template from workspace root
    let template_file = format!("{}/../../prompts/semantic_chunking.jinja", template_path);

    let template_content = if std::path::Path::new(&template_file).exists() {
        std::fs::read_to_string(&template_file)
            .map_err(|e| VididxError::Segment(format!("Failed to read template: {}", e)))?
    } else {
        // Fallback to a direct template string if file not found
        include_str!("../../../prompts/semantic_chunking.jinja").to_string()
    };

    let mut tera = Tera::new("")
        .map_err(|e| VididxError::Segment(format!("Failed to create template engine: {}", e)))?;

    tera.add_raw_template("semantic_chunking", &template_content)
        .map_err(|e| VididxError::Segment(format!("Failed to add template: {}", e)))?;

    let mut context = tera::Context::new();
    context.insert("transcript_text", &coarse.transcript_text);
    context.insert("duration_sec", &coarse.end_sec);
    context.insert("target_min", &target_min);
    context.insert("target_max", &target_max);

    tera.render("semantic_chunking", &context)
        .map_err(|e| VididxError::Segment(format!("Failed to render template: {}", e)))
}

fn parse_and_validate_chunks(
    response: &serde_json::Value,
    duration_sec: f64,
) -> Result<Vec<SemanticChunk>, VididxError> {
    let chunk_response: ChunkResponse = serde_json::from_value(response.clone())
        .map_err(|e| VididxError::Segment(format!("Failed to parse chunks: {}", e)))?;

    let mut chunks = Vec::new();

    // Validate chunks
    if chunk_response.chunks.is_empty() {
        return Err(VididxError::Segment("No chunks in response".to_string()));
    }

    // Check first chunk starts at 0
    if (chunk_response.chunks[0].start_sec - 0.0).abs() > 0.01 {
        return Err(VididxError::Segment(
            "First chunk must start at 0".to_string(),
        ));
    }

    // Check last chunk ends at duration
    let last_chunk = &chunk_response.chunks[chunk_response.chunks.len() - 1];
    if (last_chunk.end_sec - duration_sec).abs() > 0.01 {
        return Err(VididxError::Segment(
            "Last chunk must end at duration".to_string(),
        ));
    }

    // Check continuity
    for i in 0..chunk_response.chunks.len() - 1 {
        let current = &chunk_response.chunks[i];
        let next = &chunk_response.chunks[i + 1];

        if current.end_sec > next.start_sec {
            return Err(VididxError::Segment("Chunks must not overlap".to_string()));
        }

        if (current.end_sec - next.start_sec).abs() > 0.01 {
            return Err(VididxError::Segment("Chunks must be continuous".to_string()));
        }
    }

    // Check all chunks have positive duration
    for chunk in &chunk_response.chunks {
        if chunk.end_sec <= chunk.start_sec {
            return Err(VididxError::Segment("Chunk end must be > start".to_string()));
        }
    }

    // Convert to SemanticChunk
    for chunk in chunk_response.chunks {
        let text = extract_transcript_segment(chunk.start_sec, chunk.end_sec);
        chunks.push(SemanticChunk {
            start_sec: chunk.start_sec,
            end_sec: chunk.end_sec,
            transcript_text: text,
            rationale: chunk.rationale,
        });
    }

    Ok(chunks)
}

fn extract_transcript_segment(start_sec: f64, end_sec: f64) -> String {
    format!("Segment from {:.1}s to {:.1}s", start_sec, end_sec)
}

fn equal_division(coarse: &CoarseSegment, target_min: f64, target_max: f64) -> Vec<SemanticChunk> {
    let duration = coarse.end_sec - coarse.start_sec;
    let num_chunks = (duration / ((target_min + target_max) / 2.0).max(1.0)).ceil() as usize;
    let num_chunks = num_chunks.max(1);

    let chunk_duration = duration / num_chunks as f64;

    let mut chunks = Vec::new();
    for i in 0..num_chunks {
        let start = coarse.start_sec + (i as f64) * chunk_duration;
        let end = if i == num_chunks - 1 {
            coarse.end_sec
        } else {
            coarse.start_sec + ((i + 1) as f64) * chunk_duration
        };

        chunks.push(SemanticChunk {
            start_sec: start,
            end_sec: end,
            transcript_text: coarse.transcript_text.clone(),
            rationale: "Fallback equal division".to_string(),
        });
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equal_division_single_chunk() {
        let coarse = CoarseSegment {
            index: 0,
            start_sec: 0.0,
            end_sec: 100.0,
            transcript_text: "test".to_string(),
        };

        let chunks = equal_division(&coarse, 50.0, 150.0);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start_sec, 0.0);
        assert_eq!(chunks[0].end_sec, 100.0);
    }

    #[test]
    fn test_equal_division_multiple_chunks() {
        let coarse = CoarseSegment {
            index: 0,
            start_sec: 0.0,
            end_sec: 300.0,
            transcript_text: "test".to_string(),
        };

        let chunks = equal_division(&coarse, 50.0, 150.0);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].start_sec, 0.0);
        assert_eq!(chunks[chunks.len() - 1].end_sec, 300.0);
    }

    #[test]
    fn test_parse_valid_chunks() {
        let response = serde_json::json!({
            "chunks": [
                {
                    "start_sec": 0.0,
                    "end_sec": 50.0,
                    "rationale": "Introduction"
                },
                {
                    "start_sec": 50.0,
                    "end_sec": 100.0,
                    "rationale": "Main content"
                }
            ]
        });

        let chunks = parse_and_validate_chunks(&response, 100.0);
        assert!(chunks.is_ok());
        let chunks = chunks.unwrap();
        assert_eq!(chunks.len(), 2);
    }

    #[test]
    fn test_parse_invalid_chunks_gap() {
        let response = serde_json::json!({
            "chunks": [
                {
                    "start_sec": 0.0,
                    "end_sec": 40.0,
                    "rationale": "Introduction"
                },
                {
                    "start_sec": 50.0,
                    "end_sec": 100.0,
                    "rationale": "Main content"
                }
            ]
        });

        let chunks = parse_and_validate_chunks(&response, 100.0);
        assert!(chunks.is_err());
    }
}
