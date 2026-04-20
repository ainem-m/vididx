//! Pipeline orchestration: Stage trait and manifest management.

pub mod manifest;
pub mod stage;
pub mod runner;

pub use manifest::Manifest;
pub use stage::{JobContext, Stage, StageError};
pub use runner::run_pipeline;
