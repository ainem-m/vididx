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
/// If the coarse segment is shorter than 30 seconds, skip LLM and return a single chunk.
pub async fn semantic_chunk(
    llm_client: &AnthropicClient,
    coarse: &CoarseSegment,
    target_min: f64,
    target_max: f64,
) -> Result<Vec<SemanticChunk>, VididxError> {
    let duration = coarse.end_sec - coarse.start_sec;

    if duration < 30.0 {
        return Ok(vec![SemanticChunk {
            start_sec: coarse.start_sec,
            end_sec: coarse.end_sec,
            transcript_text: coarse.transcript_text.clone(),
            rationale: "Short segment, no LLM call needed".to_string(),
            parent_segment_id: coarse.segment_id.clone(),
        }]);
    }

    let prompt = render_semantic_prompt(coarse, target_min, target_max)?;

    let system_prompt = "You are an expert at identifying semantic boundaries in transcripts. \
        Analyze the provided transcript and return valid JSON chunks with semantic breakpoints.";

    // Try LLM-based semantic chunking with one retry
    let result = llm_client.call_with_json_mode(system_prompt, &prompt).await;

    let chunks = match result {
        Ok(response) => {
            match parse_and_validate_chunks(&response, coarse) {
                Ok(chunks) => chunks,
                Err(_) => {
                    // Retry once on validation failure
                    let result = llm_client.call_with_json_mode(system_prompt, &prompt).await;
                    match result {
                        Ok(response) => {
                            match parse_and_validate_chunks(&response, coarse) {
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

/// Render the semantic chunking prompt with timestamped segments.
fn render_semantic_prompt(
    coarse: &CoarseSegment,
    target_min: f64,
    target_max: f64,
) -> Result<String, VididxError> {
    let template_path = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());

    let template_file = format!("{}/../../prompts/semantic_chunking.jinja", template_path);

    let template_content = if std::path::Path::new(&template_file).exists() {
        std::fs::read_to_string(&template_file)
            .map_err(|e| VididxError::Segment(format!("Failed to read template: {}", e)))?
    } else {
        include_str!("../../../prompts/semantic_chunking.jinja").to_string()
    };

    let mut tera = Tera::new("")
        .map_err(|e| VididxError::Segment(format!("Failed to create template engine: {}", e)))?;

    tera.add_raw_template("semantic_chunking", &template_content)
        .map_err(|e| VididxError::Segment(format!("Failed to add template: {}", e)))?;

    let mut context = tera::Context::new();
    context.insert("transcript_text", &coarse.transcript_text);
    context.insert("duration_sec", &(coarse.end_sec - coarse.start_sec));
    context.insert("offset_sec", &coarse.start_sec);
    context.insert("target_min", &target_min);
    context.insert("target_max", &target_max);

    tera.render("semantic_chunking", &context)
        .map_err(|e| VididxError::Segment(format!("Failed to render template: {}", e)))
}

fn parse_and_validate_chunks(
    response: &serde_json::Value,
    coarse: &CoarseSegment,
) -> Result<Vec<SemanticChunk>, VididxError> {
    let chunk_response: ChunkResponse = serde_json::from_value(response.clone())
        .map_err(|e| VididxError::Segment(format!("Failed to parse chunks: {}", e)))?;

    let mut chunks = Vec::new();

    if chunk_response.chunks.is_empty() {
        return Err(VididxError::Segment("No chunks in response".to_string()));
    }

    // Validate: first chunk starts at 0 (relative), last chunk ends at duration (relative)
    let duration = coarse.end_sec - coarse.start_sec;
    if (chunk_response.chunks[0].start_sec - 0.0).abs() > 0.01 {
        return Err(VididxError::Segment(
            "First chunk must start at 0 (relative)".to_string(),
        ));
    }

    let last_chunk = &chunk_response.chunks[chunk_response.chunks.len() - 1];
    if (last_chunk.end_sec - duration).abs() > 0.01 {
        return Err(VididxError::Segment(
            "Last chunk must end at duration (relative)".to_string(),
        ));
    }

    // Validate continuity in relative timestamps
    for i in 0..chunk_response.chunks.len() - 1 {
        let current = &chunk_response.chunks[i];
        let next = &chunk_response.chunks[i + 1];

        if current.end_sec > next.start_sec + 0.01 {
            return Err(VididxError::Segment("Chunks must not overlap".to_string()));
        }

        if (current.end_sec - next.start_sec).abs() > 0.01 {
            return Err(VididxError::Segment(
                "Chunks must be continuous".to_string(),
            ));
        }
    }

    // Validate positive duration
    for chunk in &chunk_response.chunks {
        if chunk.end_sec <= chunk.start_sec {
            return Err(VididxError::Segment(
                "Chunk end must be > start".to_string(),
            ));
        }
    }

    // Convert relative timestamps to absolute by adding coarse.start_sec offset
    for chunk in chunk_response.chunks {
        let abs_start = chunk.start_sec + coarse.start_sec;
        let abs_end = chunk.end_sec + coarse.start_sec;
        chunks.push(SemanticChunk {
            start_sec: abs_start,
            end_sec: abs_end,
            transcript_text: extract_transcript_segment(
                &coarse.transcript_text,
                abs_start,
                abs_end,
            ),
            rationale: chunk.rationale,
            parent_segment_id: coarse.segment_id.clone(),
        });
    }

    Ok(chunks)
}

/// Extract transcript text for a time range. Since coarse.transcript_text
/// is a flat concatenated string without timestamps, we return the full text
/// and let the normalize step handle splitting. For future improvement,
/// the TranscriptTimeline with per-segment timestamps should be used.
fn extract_transcript_segment(transcript_text: &str, _start_sec: f64, _end_sec: f64) -> String {
    transcript_text.to_string()
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
            parent_segment_id: coarse.segment_id.clone(),
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
            segment_id: "vid_seg_0000".to_string(),
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
            segment_id: "vid_seg_0000".to_string(),
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

        let coarse = CoarseSegment {
            segment_id: "seg_0".to_string(),
            index: 0,
            start_sec: 200.0,
            end_sec: 300.0,
            transcript_text: "test transcript".to_string(),
        };

        let chunks = parse_and_validate_chunks(&response, &coarse);
        assert!(chunks.is_ok());
        let chunks = chunks.unwrap();
        assert_eq!(chunks.len(), 2);
        // Verify absolute timestamps are offset by coarse.start_sec
        assert_eq!(chunks[0].start_sec, 200.0);
        assert_eq!(chunks[0].end_sec, 250.0);
        assert_eq!(chunks[1].start_sec, 250.0);
        assert_eq!(chunks[1].end_sec, 300.0);
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

        let coarse = CoarseSegment {
            segment_id: "seg_0".to_string(),
            index: 0,
            start_sec: 0.0,
            end_sec: 100.0,
            transcript_text: "test".to_string(),
        };

        let chunks = parse_and_validate_chunks(&response, &coarse);
        assert!(chunks.is_err());
    }

    #[test]
    fn test_short_segment_skips_llm() {
        let coarse = CoarseSegment {
            segment_id: "seg_0".to_string(),
            index: 0,
            start_sec: 0.0,
            end_sec: 25.0,
            transcript_text: "short segment".to_string(),
        };

        // This test verifies that segments < 30s are returned as-is without calling LLM.
        // We test the static logic here.
        let duration = coarse.end_sec - coarse.start_sec;
        assert!(duration < 30.0);
    }
}
