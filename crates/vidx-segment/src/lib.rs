//! vidx-segment

mod annotate;
mod coarse;
mod normalize;
mod semantic;

pub use annotate::annotate_chunk;
pub use coarse::coarse_segment;
pub use normalize::normalize;
pub use semantic::semantic_chunk;
