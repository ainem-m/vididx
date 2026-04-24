use vididx_core::{NormalizedChunk, TranscriptSegment, TranscriptTimeline, UtteranceSegmentConfig};

/// Convert ASR utterances directly into normalized chunks.
///
/// Adjacent utterances are merged when:
/// - the silence gap between them is ≤ `cfg.merge_gap_sec`, OR
/// - the accumulated pending chunk is shorter than `cfg.min_duration_sec`.
///
/// A trailing chunk shorter than `min_duration_sec` is merged backward into the
/// previous chunk rather than being emitted alone.
pub fn utterance_to_chunks(
    transcript: &TranscriptTimeline,
    cfg: &UtteranceSegmentConfig,
    video_id: &str,
) -> Vec<NormalizedChunk> {
    if transcript.segments.is_empty() {
        return Vec::new();
    }

    let merged = merge_utterances(
        &transcript.segments,
        cfg.merge_gap_sec,
        cfg.min_duration_sec,
    );

    merged
        .into_iter()
        .enumerate()
        .map(|(idx, (start_sec, end_sec, text))| NormalizedChunk {
            chunk_id: format!("{video_id}_chunk_{idx:04}"),
            parent_segment_id: String::new(),
            start_sec,
            end_sec,
            transcript_text: text,
            rationale: "utterance-boundary segmentation".to_string(),
        })
        .collect()
}

struct Pending {
    start_sec: f64,
    end_sec: f64,
    texts: Vec<String>,
}

impl Pending {
    fn duration(&self) -> f64 {
        self.end_sec - self.start_sec
    }

    fn into_tuple(self) -> (f64, f64, String) {
        let text = self
            .texts
            .iter()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        (self.start_sec, self.end_sec, text)
    }
}

fn merge_utterances(
    segments: &[TranscriptSegment],
    merge_gap_sec: f64,
    min_duration_sec: f64,
) -> Vec<(f64, f64, String)> {
    let mut result: Vec<Pending> = Vec::new();
    let mut pending = Pending {
        start_sec: segments[0].start_sec,
        end_sec: segments[0].end_sec,
        texts: vec![segments[0].text.clone()],
    };

    for seg in &segments[1..] {
        let gap = seg.start_sec - pending.end_sec;
        let too_short = pending.duration() < min_duration_sec;
        if gap <= merge_gap_sec || too_short {
            pending.end_sec = seg.end_sec;
            pending.texts.push(seg.text.clone());
        } else {
            result.push(pending);
            pending = Pending {
                start_sec: seg.start_sec,
                end_sec: seg.end_sec,
                texts: vec![seg.text.clone()],
            };
        }
    }

    // Trailing chunk: if too short, absorb into previous instead of emitting alone.
    if pending.duration() < min_duration_sec && !result.is_empty() {
        let prev = result.last_mut().expect("checked non-empty");
        prev.end_sec = pending.end_sec;
        prev.texts.extend(pending.texts);
    } else {
        result.push(pending);
    }

    result.into_iter().map(Pending::into_tuple).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start: f64, end: f64, text: &str) -> TranscriptSegment {
        TranscriptSegment {
            start_sec: start,
            end_sec: end,
            text: text.to_string(),
            speaker: None,
            confidence: None,
        }
    }

    fn cfg() -> UtteranceSegmentConfig {
        UtteranceSegmentConfig {
            merge_gap_sec: 0.5,
            min_duration_sec: 3.0,
        }
    }

    #[test]
    fn test_utterance_to_chunks_empty_transcript_returns_empty() {
        let timeline = TranscriptTimeline { segments: vec![] };
        assert!(utterance_to_chunks(&timeline, &cfg(), "vid").is_empty());
    }

    #[test]
    fn test_utterance_to_chunks_normal_segments_one_chunk_each() {
        // All segments are long enough and gaps are large → no merging.
        let timeline = TranscriptTimeline {
            segments: vec![
                seg(0.0, 6.0, "Hello world."),
                seg(7.0, 13.0, "This is a test."),
                seg(14.0, 20.0, "End of transcript."),
            ],
        };
        let chunks = utterance_to_chunks(&timeline, &cfg(), "vid");
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].start_sec, 0.0);
        assert_eq!(chunks[0].end_sec, 6.0);
        assert_eq!(chunks[2].end_sec, 20.0);
    }

    #[test]
    fn test_utterance_to_chunks_assigns_sequential_ids() {
        let timeline = TranscriptTimeline {
            segments: vec![seg(0.0, 5.0, "A."), seg(10.0, 15.0, "B.")],
        };
        let chunks = utterance_to_chunks(&timeline, &cfg(), "myvid");
        assert_eq!(chunks[0].chunk_id, "myvid_chunk_0000");
        assert_eq!(chunks[1].chunk_id, "myvid_chunk_0001");
    }

    #[test]
    fn test_utterance_to_chunks_small_gap_merges() {
        // Gap of 0.3s ≤ merge_gap_sec (0.5) → merge first two.
        let timeline = TranscriptTimeline {
            segments: vec![
                seg(0.0, 5.0, "First."),
                seg(5.3, 10.0, "Second."),
                seg(15.0, 21.0, "Third."),
            ],
        };
        let chunks = utterance_to_chunks(&timeline, &cfg(), "vid");
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].transcript_text, "First. Second.");
        assert_eq!(chunks[0].end_sec, 10.0);
    }

    #[test]
    fn test_utterance_to_chunks_tiny_segment_merges_with_next() {
        // "OK." is 1.5s < min_duration (3.0) with a large gap from previous → merges forward.
        let timeline = TranscriptTimeline {
            segments: vec![
                seg(0.0, 6.0, "Setup complete."),
                seg(10.0, 11.5, "OK."),
                seg(15.0, 21.0, "Now proceed."),
            ],
        };
        let chunks = utterance_to_chunks(&timeline, &cfg(), "vid");
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[1].transcript_text, "OK. Now proceed.");
        assert_eq!(chunks[1].start_sec, 10.0);
        assert_eq!(chunks[1].end_sec, 21.0);
    }

    #[test]
    fn test_utterance_to_chunks_tiny_trailing_merges_backward() {
        // "Hi." is 1.5s and the last segment → merges into previous.
        let timeline = TranscriptTimeline {
            segments: vec![seg(0.0, 8.0, "Main content."), seg(9.0, 10.5, "Hi.")],
        };
        let chunks = utterance_to_chunks(&timeline, &cfg(), "vid");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].end_sec, 10.5);
        assert!(chunks[0].transcript_text.contains("Hi."));
    }

    #[test]
    fn test_utterance_to_chunks_single_segment_emitted_even_if_short() {
        // Only one segment that is short: no previous to merge into, so emit it.
        let timeline = TranscriptTimeline {
            segments: vec![seg(0.0, 1.5, "Congratulations!")],
        };
        let chunks = utterance_to_chunks(&timeline, &cfg(), "vid");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].transcript_text, "Congratulations!");
    }

    #[test]
    fn test_utterance_to_chunks_concatenates_text_with_space() {
        let timeline = TranscriptTimeline {
            segments: vec![seg(0.0, 5.0, "  Hello  "), seg(5.2, 10.0, "  world.  ")],
        };
        let chunks = utterance_to_chunks(&timeline, &cfg(), "vid");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].transcript_text, "Hello world.");
    }
}
