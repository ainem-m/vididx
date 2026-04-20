//! Computer vision: OCR, VLM, perceptual hashing.

pub mod caption;
pub mod dhash;
pub mod ocr;
pub mod frame_analysis;
pub mod enrich;

pub use caption::{ClaudeVlmAdapter, VlmCaptionAdapter};
pub use dhash::{dhash_from_image, dhash_from_path, hamming_distance};
pub use ocr::{OcrAdapter, TesseractAdapter};
pub use frame_analysis::{analyze_frames, ExtractedFrame};
pub use enrich::{enrich_frames, enrich_with_vlm, EnrichedFrame};
