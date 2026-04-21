//! Pipeline orchestration: Stage trait and manifest management.

pub mod manifest;
pub mod runner;
pub mod stage;

pub use manifest::Manifest;
pub use runner::run_pipeline;
pub use stage::{JobContext, Stage, StageError};
