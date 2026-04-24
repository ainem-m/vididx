use vididx_core::{NormalizedChunk, SemanticChunk};

/// Normalize chunks by merging short and splitting long.
/// Merges shortest-first per SPEC §5 Stage 5.
pub fn normalize(
    chunks: Vec<SemanticChunk>,
    hard_min: f64,
    hard_max: f64,
    target_min: f64,
    target_max: f64,
    video_id: &str,
) -> Vec<NormalizedChunk> {
    if chunks.is_empty() {
        return Vec::new();
    }

    // Phase 1: Merge short chunks (shortest first per SPEC)
    let merged = merge_short_chunks(chunks, hard_min);

    // Phase 2: Split long chunks
    let split = split_long_chunks(merged, hard_max, target_min, target_max);

    // Phase 3: Assign chunk IDs
    assign_chunk_ids(split, video_id)
}

fn merge_short_chunks(chunks: Vec<SemanticChunk>, hard_min: f64) -> Vec<SemanticChunk> {
    if chunks.is_empty() {
        return Vec::new();
    }

    let mut result: Vec<SemanticChunk> = Vec::new();
    for chunk in chunks {
        result.push(chunk);
    }

    // Repeatedly find the shortest chunk below threshold and merge with shorter neighbor
    loop {
        let mut shortest_idx = None;
        let mut shortest_dur = f64::MAX;

        for (i, chunk) in result.iter().enumerate() {
            let dur = chunk.end_sec - chunk.start_sec;
            if dur < hard_min && dur < shortest_dur {
                shortest_idx = Some(i);
                shortest_dur = dur;
            }
        }

        let Some(idx) = shortest_idx else {
            break;
        };

        // Determine which neighbor is shorter
        let left_dur = if idx > 0 {
            result[idx - 1].end_sec - result[idx - 1].start_sec
        } else {
            f64::MAX
        };
        let right_dur = if idx + 1 < result.len() {
            result[idx + 1].end_sec - result[idx + 1].start_sec
        } else {
            f64::MAX
        };

        if left_dur == f64::MAX && right_dur == f64::MAX {
            // No neighbors to merge with; keep as-is
            break;
        }

        if left_dur <= right_dur {
            // Merge current into left neighbor
            result[idx - 1].end_sec = result[idx].end_sec;
            result[idx - 1].transcript_text = format!(
                "{} {}",
                result[idx - 1].transcript_text,
                result[idx].transcript_text
            )
            .trim()
            .to_string();
            result[idx - 1].rationale =
                format!("{} + {}", result[idx - 1].rationale, result[idx].rationale);
            result.remove(idx);
        } else {
            // Merge right neighbor into current
            result[idx].end_sec = result[idx + 1].end_sec;
            result[idx].transcript_text = format!(
                "{} {}",
                result[idx].transcript_text,
                result[idx + 1].transcript_text
            )
            .trim()
            .to_string();
            result[idx].rationale =
                format!("{} + {}", result[idx].rationale, result[idx + 1].rationale);
            result.remove(idx + 1);
        }
    }

    result
}

fn split_long_chunks(
    chunks: Vec<SemanticChunk>,
    hard_max: f64,
    target_min: f64,
    target_max: f64,
) -> Vec<SemanticChunk> {
    let mut result = Vec::new();

    for chunk in chunks {
        let duration = chunk.end_sec - chunk.start_sec;

        if duration <= hard_max {
            result.push(chunk);
        } else {
            let target_duration = ((target_min + target_max) / 2.0).min(hard_max / 2.0);
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
                    parent_segment_id: chunk.parent_segment_id.clone(),
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
            chunk_id: format!("{video_id}_chunk_{idx:04}"),
            parent_segment_id: chunk.parent_segment_id,
            start_sec: chunk.start_sec,
            end_sec: chunk.end_sec,
            transcript_text: chunk.transcript_text,
            rationale: chunk.rationale,
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
                parent_segment_id: String::new(),
            },
            SemanticChunk {
                start_sec: 50.0,
                end_sec: 100.0,
                transcript_text: "part 2".to_string(),
                rationale: "main".to_string(),
                parent_segment_id: String::new(),
            },
        ];

        let result = normalize(chunks, 15.0, 120.0, 30.0, 90.0, "test_video");
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
                parent_segment_id: String::new(),
            },
            SemanticChunk {
                start_sec: 10.0,
                end_sec: 60.0,
                transcript_text: "main".to_string(),
                rationale: "content".to_string(),
                parent_segment_id: String::new(),
            },
        ];

        let result = normalize(chunks, 15.0, 120.0, 30.0, 90.0, "test_video");
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
                parent_segment_id: String::new(),
            },
            SemanticChunk {
                start_sec: 50.0,
                end_sec: 55.0,
                transcript_text: "tail".to_string(),
                rationale: "tail".to_string(),
                parent_segment_id: String::new(),
            },
        ];

        let result = normalize(chunks, 15.0, 120.0, 30.0, 90.0, "test_video");
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
            parent_segment_id: String::new(),
        }];

        let result = normalize(chunks, 15.0, 120.0, 30.0, 90.0, "test_video");
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
            parent_segment_id: String::new(),
        }];

        let result = normalize(chunks, 15.0, 120.0, 30.0, 90.0, "my_video_123");
        assert_eq!(result[0].chunk_id, "my_video_123_chunk_0000");
    }

    #[test]
    fn test_normalize_empty() {
        let chunks = vec![];
        let result = normalize(chunks, 15.0, 120.0, 30.0, 90.0, "test");
        assert_eq!(result.len(), 0);
    }
}
