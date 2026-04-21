//! Media processing: ffmpeg/ffprobe wrappers, frame extraction, silence detection.

pub mod audio;
pub mod frames;
pub mod probe;
pub mod scene;
pub mod silence;

pub use audio::extract_audio;
pub use frames::extract_frames;
pub use probe::probe;
pub use scene::detect_scene_changes;
pub use silence::detect_silence;
