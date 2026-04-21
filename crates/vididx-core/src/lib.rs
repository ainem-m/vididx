//! Core types, errors, and utilities for vididx.

pub mod config;
pub mod error;
pub mod hash;
pub mod model;

pub use config::{Config, SegmentMode, UtteranceSegmentConfig, WhisperCppConfig};
pub use error::VididxError;
pub use model::*;
