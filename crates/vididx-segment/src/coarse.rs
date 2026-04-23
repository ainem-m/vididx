use vididx_core::{CoarseSegment, SceneChange, SilenceInterval, TranscriptTimeline, VididxError};

/// Coarse segmentation: divide video into ~300s segments, snapping to auxiliary signals.
pub fn coarse_segment(
    duration_sec: f64,
    transcript: &TranscriptTimeline,
    silence_intervals: &[SilenceInterval],
    scene_changes: &[SceneChange],
    max_duration: f64,
    snap_window: f64,
) -> Result<Vec<CoarseSegment>, VididxError> {
    if duration_sec <= 0.0 {
        return Err(VididxError::Segment(
            "Duration must be positive".to_string(),
        ));
    }

    let mut segments = Vec::new();
    let mut current_start = 0.0;
    let mut segment_index = 0;

    while current_start < duration_sec {
        let mut segment_end = (current_start + max_duration).min(duration_sec);

        // Try to snap to auxiliary signals within snap_window of the ideal boundary
        let snap_start = (segment_end - snap_window).max(current_start);
        let snap_end = (segment_end + snap_window).min(duration_sec);

        if let Some(new_end) = find_snap_point(
            segment_end,
            snap_start,
            snap_end,
            silence_intervals,
            scene_changes,
        ) {
            segment_end = new_end;
        }

        // Ensure segment_end is at least slightly past segment_start
        if segment_end <= current_start {
            segment_end = (current_start + 1.0).min(duration_sec);
        }

        // Collect transcript text for this segment
        let transcript_text = collect_transcript(transcript, current_start, segment_end);

        segments.push(CoarseSegment {
            segment_id: String::new(),
            index: segment_index,
            start_sec: current_start,
            end_sec: segment_end,
            transcript_text,
        });

        current_start = segment_end;
        segment_index += 1;
    }

    // Verify invariants
    if !segments.is_empty() {
        if (segments[0].start_sec - 0.0).abs() > 0.01 {
            return Err(VididxError::Segment(
                "First segment must start at 0".to_string(),
            ));
        }

        if (segments.last().unwrap().end_sec - duration_sec).abs() > 0.01 {
            return Err(VididxError::Segment(
                "Last segment must end at duration".to_string(),
            ));
        }

        for i in 0..segments.len() - 1 {
            if (segments[i].end_sec - segments[i + 1].start_sec).abs() > 0.01 {
                return Err(VididxError::Segment(
                    "Segments must be continuous".to_string(),
                ));
            }
        }

        for seg in &segments {
            if seg.end_sec <= seg.start_sec {
                return Err(VididxError::Segment(
                    "Segment end must be > start".to_string(),
                ));
            }
        }
    }

    Ok(segments)
}

/// Find a snap point near the target boundary.
fn find_snap_point(
    target: f64,
    window_start: f64,
    window_end: f64,
    silence_intervals: &[SilenceInterval],
    scene_changes: &[SceneChange],
) -> Option<f64> {
    let mut candidates = Vec::new();

    // Add silence boundaries
    for si in silence_intervals {
        if si.start_sec >= window_start && si.start_sec <= window_end {
            candidates.push(si.start_sec);
        }
        if si.end_sec >= window_start && si.end_sec <= window_end {
            candidates.push(si.end_sec);
        }
    }

    // Add scene changes
    for sc in scene_changes {
        if sc.at_sec >= window_start && sc.at_sec <= window_end {
            candidates.push(sc.at_sec);
        }
    }

    // Find closest to target, preferring later in case of tie
    candidates
        .iter()
        .min_by(|a, b| {
            let dist_a = (**a - target).abs();
            let dist_b = (**b - target).abs();
            match dist_a.partial_cmp(&dist_b).unwrap() {
                std::cmp::Ordering::Equal => b.partial_cmp(a).unwrap(),
                other => other,
            }
        })
        .copied()
}

/// Collect transcript segments within time range.
fn collect_transcript(timeline: &TranscriptTimeline, start_sec: f64, end_sec: f64) -> String {
    timeline
        .segments
        .iter()
        .filter(|seg| seg.start_sec < end_sec && seg.end_sec > start_sec)
        .map(|seg| seg.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coarse_segment_basic() {
        let transcript = TranscriptTimeline { segments: vec![] };

        let result = coarse_segment(
            100.0,
            &transcript,
            &[],
            &[],
            50.0, // max_duration
            10.0,
        );

        assert!(result.is_ok());
        let segs = result.unwrap();
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].start_sec, 0.0);
        assert_eq!(segs[0].end_sec, 50.0);
        assert_eq!(segs[1].start_sec, 50.0);
        assert_eq!(segs[1].end_sec, 100.0);
    }

    #[test]
    fn test_coarse_segment_with_snap() {
        let transcript = TranscriptTimeline { segments: vec![] };
        let silence = vec![SilenceInterval {
            start_sec: 45.0,
            end_sec: 55.0,
        }];

        let result = coarse_segment(
            100.0,
            &transcript,
            &silence,
            &[],
            50.0,
            10.0, // snap_window: 40-60 window around 50
        );

        assert!(result.is_ok());
        let segs = result.unwrap();
        // Should snap to 45 or 55 (silence boundaries within ±10 of the 50s mark)
        assert!(
            (segs[0].end_sec - 45.0).abs() < 0.01 || (segs[0].end_sec - 55.0).abs() < 0.01,
            "Expected snap to 45 or 55, got {}",
            segs[0].end_sec
        );
    }

    #[test]
    fn test_coarse_segment_continuity() {
        let transcript = TranscriptTimeline { segments: vec![] };

        let result = coarse_segment(300.0, &transcript, &[], &[], 100.0, 10.0);

        assert!(result.is_ok());
        let segs = result.unwrap();
        for i in 0..segs.len() - 1 {
            assert!((segs[i].end_sec - segs[i + 1].start_sec).abs() < 0.01);
        }
    }

    #[test]
    fn test_coarse_segment_zero_duration() {
        let transcript = TranscriptTimeline { segments: vec![] };

        let result = coarse_segment(0.0, &transcript, &[], &[], 100.0, 10.0);
        assert!(result.is_err());
    }
}
