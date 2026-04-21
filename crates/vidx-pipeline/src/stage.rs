use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::manifest::Manifest;
use vidx_core::Config;

/// Job context shared across all pipeline stages.
#[derive(Debug, Clone)]
pub struct JobContext {
    pub video_id: String,
    pub source_type: String,
    pub source_ref: String,
    pub source_path: PathBuf,
    pub out_dir: PathBuf,
    pub config: Config,
    pub manifest: Arc<Mutex<Manifest>>,
}

/// Error type for stage execution.
#[derive(Debug, thiserror::Error)]
pub enum StageError {
    #[error("stage {stage}: {source}")]
    Run {
        stage: &'static str,
        #[source]
        source: anyhow::Error,
    },
    #[error("cache io: {0}")]
    CacheIo(#[from] std::io::Error),
    #[error("serde: {0}")]
    Serde(#[from] serde_json::Error),
}

/// Core trait for all pipeline stages.
#[async_trait]
pub trait Stage: Send + Sync {
    type Input: Serialize + Send + Sync;
    type Output: Serialize + for<'de> Deserialize<'de> + Send + Sync;

    const NAME: &'static str;

    /// Run the stage with given input.
    async fn run(&self, ctx: &JobContext, input: Self::Input) -> Result<Self::Output, StageError>;

    /// Compute cache key from input. Default uses SHA-256 of JSON.
    fn cache_key(&self, input: &Self::Input) -> String {
        let json_str = serde_json::to_string(input).unwrap_or_default();
        vidx_core::hash::sha256_str(&json_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Serialize, Deserialize)]
    struct DummyInput {
        value: i32,
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct DummyOutput {
        result: i32,
    }

    struct DummyStage;

    #[async_trait]
    impl Stage for DummyStage {
        type Input = DummyInput;
        type Output = DummyOutput;

        const NAME: &'static str = "dummy";

        async fn run(
            &self,
            _ctx: &JobContext,
            input: Self::Input,
        ) -> Result<Self::Output, StageError> {
            Ok(DummyOutput {
                result: input.value * 2,
            })
        }
    }

    #[test]
    fn test_cache_key_deterministic() {
        let stage = DummyStage;
        let input = DummyInput { value: 42 };

        let key1 = stage.cache_key(&input);
        let key2 = stage.cache_key(&input);

        assert_eq!(key1, key2);
        assert!(key1.starts_with("sha256:"));
    }

    #[test]
    fn test_cache_key_different_for_different_input() {
        let stage = DummyStage;
        let input1 = DummyInput { value: 42 };
        let input2 = DummyInput { value: 43 };

        let key1 = stage.cache_key(&input1);
        let key2 = stage.cache_key(&input2);

        assert_ne!(key1, key2);
    }

    #[tokio::test]
    async fn test_stage_run() {
        let stage = DummyStage;
        let temp_dir = tempfile::tempdir().unwrap();

        let manifest = Manifest::new("test_vid", "/path/to/video.mp4", "sha256:vid", "sha256:cfg");
        let ctx = JobContext {
            video_id: "test_vid".to_string(),
            source_type: "local_mp4".to_string(),
            source_ref: "/path/to/video.mp4".to_string(),
            source_path: PathBuf::from("/path/to/video.mp4"),
            out_dir: temp_dir.path().to_path_buf(),
            config: Config::default(),
            manifest: Arc::new(Mutex::new(manifest)),
        };

        let input = DummyInput { value: 21 };
        let output = stage.run(&ctx, input).await.unwrap();

        assert_eq!(output, DummyOutput { result: 42 });
    }
}
