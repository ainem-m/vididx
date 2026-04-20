//! vidx-segment

mod coarse;
mod semantic;
mod normalize;
mod annotate;

pub use coarse::coarse_segment;
pub use semantic::semantic_chunk;
pub use normalize::normalize;
pub use annotate::annotate_chunk;
