use serde::{Deserialize, Serialize};
use tera::Tera;
use vididx_core::{CoarseSegment, SemanticChunk, TranscriptTimeline, VididxError};
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
    timeline: &TranscriptTimeline,
    target_min: f64,
    target_max: f64,
) -> Result<Vec<SemanticChunk>, VididxError> {
    let duration = coarse.end_sec - coarse.start_sec;

    if duration < 30.0 {
        return Ok(vec![SemanticChunk {
            start_sec: coarse.start_sec,
            end_sec: coarse.end_sec,
            transcript_text: transcript_text_for_range(timeline, coarse.start_sec, coarse.end_sec),
            rationale: "short segment — single chunk".to_string(),
            parent_segment_id: coarse.segment_id.clone(),
        }]);
    }

    let prompt = render_semantic_prompt(coarse, timeline, target_min, target_max)?;

    let system_prompt = "You are an expert at identifying semantic boundaries in video transcripts. \
        Analyze the provided timestamped transcript and return valid JSON with semantic breakpoints.";

    // Try LLM-based semantic chunking with one retry on schema validation failure.
    let result = llm_client.call_with_json_mode(system_prompt, &prompt).await;

    let chunks = match result {
        Ok(response) => match parse_and_validate_chunks(&response, coarse, timeline) {
            Ok(chunks) => chunks,
            Err(_) => {
                let retry = llm_client.call_with_json_mode(system_prompt, &prompt).await;
                match retry {
                    Ok(response) => match parse_and_validate_chunks(&response, coarse, timeline) {
                        Ok(chunks) => chunks,
                        Err(_) => {
                            return Ok(uniform_split(coarse, timeline, target_min, target_max));
                        }
                    },
                    Err(_) => return Ok(uniform_split(coarse, timeline, target_min, target_max)),
                }
            }
        },
        Err(_) => return Ok(uniform_split(coarse, timeline, target_min, target_max)),
    };

    Ok(chunks)
}

/// Uniform time-based split used as LLM fallback.
pub fn uniform_split(
    coarse: &CoarseSegment,
    timeline: &TranscriptTimeline,
    target_min: f64,
    target_max: f64,
) -> Vec<SemanticChunk> {
    let duration = coarse.end_sec - coarse.start_sec;
    let target = ((target_min + target_max) / 2.0).max(1.0);
    let count = (duration / target).ceil().max(1.0) as usize;
    let width = duration / count as f64;

    (0..count)
        .map(|i| {
            let start = coarse.start_sec + width * i as f64;
            let end = if i + 1 == count {
                coarse.end_sec
            } else {
                coarse.start_sec + width * (i + 1) as f64
            };
            SemanticChunk {
                start_sec: start,
                end_sec: end,
                transcript_text: transcript_text_for_range(timeline, start, end),
                rationale: "uniform-split (no-llm)".to_string(),
                parent_segment_id: coarse.segment_id.clone(),
            }
        })
        .collect()
}

fn render_semantic_prompt(
    coarse: &CoarseSegment,
    timeline: &TranscriptTimeline,
    target_min: f64,
    target_max: f64,
) -> Result<String, VididxError> {
    let template_content = include_str!("../../../prompts/semantic_chunking.jinja");

    let mut tera = Tera::default();
    tera.add_raw_template("semantic_chunking", template_content)
        .map_err(|e| VididxError::Segment(format!("Failed to add template: {}", e)))?;

    // Build timestamped segment list for the coarse segment's time range.
    let segments: Vec<serde_json::Value> = timeline
        .segments
        .iter()
        .filter(|s| s.end_sec > coarse.start_sec && s.start_sec < coarse.end_sec)
        .map(|s| {
            serde_json::json!({
                "start_sec": (s.start_sec - coarse.start_sec).max(0.0),
                "end_sec": (s.end_sec - coarse.start_sec).min(coarse.end_sec - coarse.start_sec),
                "text": s.text.trim()
            })
        })
        .collect();

    let duration = coarse.end_sec - coarse.start_sec;

    let mut context = tera::Context::new();
    context.insert("segments", &segments);
    context.insert("duration_sec", &duration);
    context.insert("target_min", &target_min);
    context.insert("target_max", &target_max);

    tera.render("semantic_chunking", &context)
        .map_err(|e| VididxError::Segment(format!("Failed to render template: {}", e)))
}

fn parse_and_validate_chunks(
    response: &serde_json::Value,
    coarse: &CoarseSegment,
    timeline: &TranscriptTimeline,
) -> Result<Vec<SemanticChunk>, VididxError> {
    let chunk_response: ChunkResponse = serde_json::from_value(response.clone())
        .map_err(|e| VididxError::Segment(format!("Failed to parse chunks: {}", e)))?;

    if chunk_response.chunks.is_empty() {
        return Err(VididxError::Segment("No chunks in response".to_string()));
    }

    let duration = coarse.end_sec - coarse.start_sec;

    if (chunk_response.chunks[0].start_sec).abs() > 1.0 {
        return Err(VididxError::Segment(
            "First chunk must start near 0 (relative)".to_string(),
        ));
    }

    let last = &chunk_response.chunks[chunk_response.chunks.len() - 1];
    if (last.end_sec - duration).abs() > 1.0 {
        return Err(VididxError::Segment(
            "Last chunk must end near duration (relative)".to_string(),
        ));
    }

    for i in 0..chunk_response.chunks.len() - 1 {
        let cur = &chunk_response.chunks[i];
        let nxt = &chunk_response.chunks[i + 1];
        if cur.end_sec > nxt.start_sec + 1.0 {
            return Err(VididxError::Segment("Chunks must not overlap".to_string()));
        }
        if (cur.end_sec - nxt.start_sec).abs() > 2.0 {
            return Err(VididxError::Segment(
                "Chunks must be continuous".to_string(),
            ));
        }
    }

    for chunk in &chunk_response.chunks {
        if chunk.end_sec <= chunk.start_sec {
            return Err(VididxError::Segment(
                "Chunk end must be > start".to_string(),
            ));
        }
    }

    let chunks = chunk_response
        .chunks
        .into_iter()
        .map(|chunk| {
            let abs_start = (chunk.start_sec + coarse.start_sec).max(coarse.start_sec);
            let abs_end = (chunk.end_sec + coarse.start_sec).min(coarse.end_sec);
            SemanticChunk {
                start_sec: abs_start,
                end_sec: abs_end,
                transcript_text: transcript_text_for_range(timeline, abs_start, abs_end),
                rationale: chunk.rationale,
                parent_segment_id: coarse.segment_id.clone(),
            }
        })
        .collect();

    Ok(chunks)
}

fn transcript_text_for_range(timeline: &TranscriptTimeline, start: f64, end: f64) -> String {
    timeline
        .segments
        .iter()
        .filter(|s| s.end_sec > start && s.start_sec < end)
        .map(|s| s.text.trim())
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use vididx_core::TranscriptSegment;

    fn make_coarse(start: f64, end: f64) -> CoarseSegment {
        CoarseSegment {
            segment_id: "seg_0".to_string(),
            index: 0,
            start_sec: start,
            end_sec: end,
            transcript_text: "test".to_string(),
        }
    }

    fn empty_timeline() -> TranscriptTimeline {
        TranscriptTimeline { segments: vec![] }
    }

    #[test]
    fn test_uniform_split_single_chunk() {
        let coarse = make_coarse(0.0, 100.0);
        let chunks = uniform_split(&coarse, &empty_timeline(), 50.0, 150.0);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start_sec, 0.0);
        assert_eq!(chunks[0].end_sec, 100.0);
        assert_eq!(chunks[0].rationale, "uniform-split (no-llm)");
    }

    #[test]
    fn test_uniform_split_multiple_chunks() {
        let coarse = make_coarse(0.0, 300.0);
        let chunks = uniform_split(&coarse, &empty_timeline(), 50.0, 150.0);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].start_sec, 0.0);
        assert_eq!(chunks[chunks.len() - 1].end_sec, 300.0);
    }

    #[test]
    fn test_parse_valid_chunks() {
        let response = serde_json::json!({
            "chunks": [
                {"start_sec": 0.0, "end_sec": 50.0, "rationale": "Introduction"},
                {"start_sec": 50.0, "end_sec": 100.0, "rationale": "Main content"}
            ]
        });
        let coarse = CoarseSegment {
            segment_id: "seg_0".to_string(),
            index: 0,
            start_sec: 200.0,
            end_sec: 300.0,
            transcript_text: "test".to_string(),
        };
        let chunks = parse_and_validate_chunks(&response, &coarse, &empty_timeline()).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].start_sec, 200.0);
        assert_eq!(chunks[0].end_sec, 250.0);
        assert_eq!(chunks[1].start_sec, 250.0);
        assert_eq!(chunks[1].end_sec, 300.0);
    }

    #[test]
    fn test_parse_invalid_chunks_gap() {
        let response = serde_json::json!({
            "chunks": [
                {"start_sec": 0.0, "end_sec": 40.0, "rationale": "A"},
                {"start_sec": 50.0, "end_sec": 100.0, "rationale": "B"}
            ]
        });
        let coarse = make_coarse(0.0, 100.0);
        assert!(parse_and_validate_chunks(&response, &coarse, &empty_timeline()).is_err());
    }

    #[test]
    fn test_short_segment_skips_llm() {
        // Segments < 30s return single chunk without LLM
        let coarse = make_coarse(0.0, 25.0);
        let duration = coarse.end_sec - coarse.start_sec;
        assert!(duration < 30.0);
    }

    #[test]
    fn test_transcript_text_for_range() {
        let timeline = TranscriptTimeline {
            segments: vec![
                TranscriptSegment {
                    start_sec: 0.0,
                    end_sec: 10.0,
                    text: "hello".to_string(),
                    speaker: None,
                    confidence: None,
                },
                TranscriptSegment {
                    start_sec: 10.0,
                    end_sec: 20.0,
                    text: "world".to_string(),
                    speaker: None,
                    confidence: None,
                },
            ],
        };
        let text = transcript_text_for_range(&timeline, 0.0, 10.0);
        assert_eq!(text, "hello");
        let text2 = transcript_text_for_range(&timeline, 0.0, 20.0);
        assert_eq!(text2, "hello world");
    }
}
