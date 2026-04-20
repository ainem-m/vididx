use vidx_core::{NormalizedChunk, SemanticChunk};

const MIN_DURATION: f64 = 15.0;
const MAX_DURATION: f64 = 120.0;
const SPLIT_MIN: f64 = 30.0;
const SPLIT_MAX: f64 = 90.0;

/// Normalize chunks by merging short and splitting long.
pub fn normalize(
    chunks: Vec<SemanticChunk>,
    hard_min: f64,
    hard_max: f64,
    video_id: &str,
) -> Vec<NormalizedChunk> {
    if chunks.is_empty() {
        return Vec::new();
    }

    // Phase 1: Merge short chunks
    let merged = merge_short_chunks(chunks, hard_min);

    // Phase 2: Split long chunks
    let split = split_long_chunks(merged, hard_max);

    // Phase 3: Assign chunk IDs
    assign_chunk_ids(split, video_id)
}

fn merge_short_chunks(chunks: Vec<SemanticChunk>, hard_min: f64) -> Vec<SemanticChunk> {
    let mut result: Vec<SemanticChunk> = Vec::new();
    let threshold = hard_min.min(MIN_DURATION);

    let mut i = 0;
    while i < chunks.len() {
        let mut current = chunks[i].clone();

        // Check if current chunk is below threshold
        while current.end_sec - current.start_sec < threshold && i + 1 < chunks.len() {
            // Merge with next chunk
            let next = &chunks[i + 1];
            current.end_sec = next.end_sec;
            current.transcript_text = format!(
                "{} {}",
                current.transcript_text, next.transcript_text
            );
            current.rationale = format!("{} + {}", current.rationale, next.rationale);
            i += 1;
        }

        if current.end_sec - current.start_sec < threshold && !result.is_empty() {
            let previous = result.last_mut().expect("checked is_empty");
            previous.end_sec = current.end_sec;
            previous.transcript_text = format!(
                "{} {}",
                previous.transcript_text, current.transcript_text
            )
            .trim()
            .to_string();
            previous.rationale = format!("{} + {}", previous.rationale, current.rationale);
        } else {
            result.push(current);
        }
        i += 1;
    }

    result
}

fn split_long_chunks(chunks: Vec<SemanticChunk>, hard_max: f64) -> Vec<SemanticChunk> {
    let mut result = Vec::new();
    let threshold = hard_max.max(MAX_DURATION);

    for chunk in chunks {
        let duration = chunk.end_sec - chunk.start_sec;

        if duration <= threshold {
            result.push(chunk);
        } else {
            // Split this chunk
            let target_duration = ((SPLIT_MIN + SPLIT_MAX) / 2.0).min(threshold / 2.0);
            let num_splits = (duration / target_duration).ceil() as usize;

            let split_duration = duration / num_splits as f64;

            for j in 0..num_splits {
                let start = chunk.start_sec + (j as f64) * split_duration;
                let end = if j == num_splits - 1 {
                    chunk.end_sec
                } else {
                    chunk.start_sec + ((j + 1) as f64) * split_duration
                };

                result.push(SemanticChunk {
                    start_sec: start,
                    end_sec: end,
                    transcript_text: chunk.transcript_text.clone(),
                    rationale: format!("{} (part {})", chunk.rationale, j + 1),
                });
            }
        }
    }

    result
}

fn assign_chunk_ids(chunks: Vec<SemanticChunk>, video_id: &str) -> Vec<NormalizedChunk> {
    chunks
        .into_iter()
        .enumerate()
        .map(|(idx, chunk)| NormalizedChunk {
            chunk_id: format!("{}_{:04}", format!("{}_chunk", video_id), idx),
            parent_segment_id: String::new(),
            start_sec: chunk.start_sec,
            end_sec: chunk.end_sec,
            transcript_text: chunk.transcript_text,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_basic() {
        let chunks = vec![
            SemanticChunk {
                start_sec: 0.0,
                end_sec: 50.0,
                transcript_text: "part 1".to_string(),
                rationale: "intro".to_string(),
            },
            SemanticChunk {
                start_sec: 50.0,
                end_sec: 100.0,
                transcript_text: "part 2".to_string(),
                rationale: "main".to_string(),
            },
        ];

        let result = normalize(chunks, 15.0, 120.0, "test_video");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].chunk_id, "test_video_chunk_0000");
        assert_eq!(result[1].chunk_id, "test_video_chunk_0001");
    }

    #[test]
    fn test_normalize_merge_short() {
        let chunks = vec![
            SemanticChunk {
                start_sec: 0.0,
                end_sec: 10.0,
                transcript_text: "short".to_string(),
                rationale: "intro".to_string(),
            },
            SemanticChunk {
                start_sec: 10.0,
                end_sec: 60.0,
                transcript_text: "main".to_string(),
                rationale: "content".to_string(),
            },
        ];

        let result = normalize(chunks, 15.0, 120.0, "test_video");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].start_sec, 0.0);
        assert_eq!(result[0].end_sec, 60.0);
    }

    #[test]
    fn test_normalize_merge_short_trailing_chunk() {
        let chunks = vec![
            SemanticChunk {
                start_sec: 0.0,
                end_sec: 50.0,
                transcript_text: "main".to_string(),
                rationale: "content".to_string(),
            },
            SemanticChunk {
                start_sec: 50.0,
                end_sec: 55.0,
                transcript_text: "tail".to_string(),
                rationale: "tail".to_string(),
            },
        ];

        let result = normalize(chunks, 15.0, 120.0, "test_video");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].start_sec, 0.0);
        assert_eq!(result[0].end_sec, 55.0);
    }

    #[test]
    fn test_normalize_split_long() {
        let chunks = vec![SemanticChunk {
            start_sec: 0.0,
            end_sec: 200.0,
            transcript_text: "very long content".to_string(),
            rationale: "long".to_string(),
        }];

        let result = normalize(chunks, 15.0, 120.0, "test_video");
        assert!(result.len() > 1);
        assert_eq!(result[0].start_sec, 0.0);
        assert_eq!(result[result.len() - 1].end_sec, 200.0);
    }

    #[test]
    fn test_chunk_id_format() {
        let chunks = vec![SemanticChunk {
            start_sec: 0.0,
            end_sec: 50.0,
            transcript_text: "test".to_string(),
            rationale: "test".to_string(),
        }];

        let result = normalize(chunks, 15.0, 120.0, "my_video_123");
        assert_eq!(result[0].chunk_id, "my_video_123_chunk_0000");
    }

    #[test]
    fn test_normalize_empty() {
        let chunks = vec![];
        let result = normalize(chunks, 15.0, 120.0, "test");
        assert_eq!(result.len(), 0);
    }
}
