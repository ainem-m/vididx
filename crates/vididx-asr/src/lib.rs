//! Speech-to-text adapters: whisper.cpp and other ASR backends.

pub mod adapter;
pub mod whisper_cpp;

pub use adapter::AsrAdapter;
pub use whisper_cpp::WhisperCppAdapter;
