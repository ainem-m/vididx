use std::path::Path;
use vidx_core::VidxError;

use crate::caption::VlmCaptionAdapter;
use crate::frame_analysis::ExtractedFrame;
use crate::ocr::OcrAdapter;

#[derive(Debug, Clone)]
pub struct EnrichedFrame {
    pub timestamp_sec: f64,
    pub image_path: String,
    pub dhash: u64,
    pub ocr_text: Option<String>,
    pub visual_caption: Option<String>,
}

/// Enrich extracted frames with OCR and VLM captions.
pub async fn enrich_frames(
    frames: Vec<ExtractedFrame>,
    ocr_adapter: &dyn OcrAdapter,
    _vlm_adapter: &dyn VlmCaptionAdapter,
    _max_vlm_per_chunk: usize,
) -> Result<Vec<EnrichedFrame>, VidxError> {
    let mut enriched = Vec::new();
    let languages = &["eng"];

    for frame in frames {
        let path = Path::new(&frame.image_path);

        // Apply OCR to all frames
        let ocr_text = ocr_adapter.extract_text(path, languages).await.ok();

        // For now, don't apply VLM here - that will be done selectively per chunk
        // This will be handled in a separate step that selects which frames to caption
        let visual_caption = None;

        enriched.push(EnrichedFrame {
            timestamp_sec: frame.timestamp_sec,
            image_path: frame.image_path,
            dhash: frame.dhash,
            ocr_text,
            visual_caption,
        });
    }

    Ok(enriched)
}

/// Select frames for VLM captioning and enrich them.
pub async fn enrich_with_vlm(
    frames: &mut [EnrichedFrame],
    vlm_adapter: &dyn VlmCaptionAdapter,
    chunk_boundaries: &[(f64, f64)],
    max_vlm_per_chunk: usize,
) -> Result<(), VidxError> {
    for (chunk_start, chunk_end) in chunk_boundaries {
        // Find frames in this chunk
        let frames_in_chunk: Vec<usize> = frames
            .iter()
            .enumerate()
            .filter(|(_, f)| f.timestamp_sec >= *chunk_start && f.timestamp_sec < *chunk_end)
            .map(|(idx, _)| idx)
            .collect();

        // Select up to max_vlm_per_chunk frames (prioritize evenly distributed)
        let selected = select_frames_for_vlm(&frames_in_chunk, max_vlm_per_chunk);

        // Apply VLM to selected frames
        for idx in selected {
            if idx < frames.len() {
                let path = Path::new(&frames[idx].image_path);
                if let Ok(caption) = vlm_adapter.caption(path).await {
                    frames[idx].visual_caption = Some(caption);
                }
            }
        }
    }

    Ok(())
}

fn select_frames_for_vlm(frame_indices: &[usize], max_count: usize) -> Vec<usize> {
    if frame_indices.is_empty() {
        return Vec::new();
    }

    if frame_indices.len() <= max_count {
        return frame_indices.to_vec();
    }

    // Evenly distribute selection across the frames
    let mut selected = Vec::new();
    let step = frame_indices.len() as f64 / max_count as f64;

    for i in 0..max_count {
        let idx = (i as f64 * step) as usize;
        if idx < frame_indices.len() {
            selected.push(frame_indices[idx]);
        }
    }

    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_frames_all_selected() {
        let frames = vec![0, 1, 2];
        let selected = select_frames_for_vlm(&frames, 3);
        assert_eq!(selected.len(), 3);
    }

    #[test]
    fn test_select_frames_some_selected() {
        let frames = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let selected = select_frames_for_vlm(&frames, 3);
        assert_eq!(selected.len(), 3);
        // Should be roughly evenly distributed
        assert!(selected[0] < selected[1]);
        assert!(selected[1] < selected[2]);
    }

    #[test]
    fn test_select_frames_empty() {
        let frames: Vec<usize> = vec![];
        let selected = select_frames_for_vlm(&frames, 3);
        assert_eq!(selected.len(), 0);
    }

    #[test]
    fn test_select_frames_single() {
        let frames = vec![5];
        let selected = select_frames_for_vlm(&frames, 3);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0], 5);
    }
}
