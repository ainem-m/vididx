//! vididx-segment

mod annotate;
mod coarse;
mod normalize;
mod semantic;
mod utterance;

pub use annotate::annotate_chunk;
pub use coarse::coarse_segment;
pub use normalize::normalize;
pub use semantic::semantic_chunk;
pub use utterance::utterance_to_chunks;
