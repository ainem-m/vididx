//! Computer vision: OCR, VLM, perceptual hashing.

pub mod caption;
pub mod dhash;
pub mod enrich;
pub mod frame_analysis;
pub mod ocr;

pub use caption::{ClaudeVlmAdapter, VlmCaptionAdapter};
pub use dhash::{dhash_from_image, dhash_from_path, hamming_distance};
pub use enrich::{EnrichedFrame, enrich_frames, enrich_with_vlm};
pub use frame_analysis::{ExtractedFrame, analyze_frames};
pub use ocr::{OcrAdapter, TesseractAdapter};
