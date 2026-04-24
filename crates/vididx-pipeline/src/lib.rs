//! Pipeline orchestration: Stage trait and manifest management.

pub mod manifest;
pub mod runner;
pub mod stage;

pub use manifest::Manifest;
pub use runner::{STAGES, run_pipeline, stage_name_to_index};
pub use stage::{JobContext, Stage, StageError};
