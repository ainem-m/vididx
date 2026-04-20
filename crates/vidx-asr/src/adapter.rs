use async_trait::async_trait;
use std::path::Path;

use vidx_core::{TranscriptTimeline, VidxError};

/// Trait for speech-to-text adapters.
#[async_trait]
pub trait AsrAdapter: Send + Sync {
    /// Transcribe audio from a WAV file.
    async fn transcribe(&self, wav_path: &Path) -> Result<TranscriptTimeline, VidxError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyAsrAdapter;

    #[async_trait]
    impl AsrAdapter for DummyAsrAdapter {
        async fn transcribe(&self, _wav_path: &Path) -> Result<TranscriptTimeline, VidxError> {
            Ok(TranscriptTimeline { segments: vec![] })
        }
    }

    #[tokio::test]
    async fn test_dummy_adapter() {
        let adapter = DummyAsrAdapter;
        let result = adapter.transcribe(Path::new("/tmp/audio.wav")).await;
        assert!(result.is_ok());
        assert!(result.unwrap().segments.is_empty());
    }
}
