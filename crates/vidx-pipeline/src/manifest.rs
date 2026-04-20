use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

use crate::stage::StageError;

/// Status of a pipeline stage execution.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StageStatus {
    Pending,
    Running,
    Done,
    Failed,
}

/// Record of a single stage's execution state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageRecord {
    pub status: StageStatus,
    pub input_hash: String,
    pub output_path: String,
    pub completed_at: Option<DateTime<Utc>>,
}

/// Manifest tracking the state of all stages for a single video.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub manifest_version: String,
    pub video_id: String,
    pub source_path: String,
    pub source_hash: String,
    pub config_hash: String,
    pub stages: BTreeMap<String, StageRecord>,
}

impl Manifest {
    /// Create a new manifest for a video processing job.
    pub fn new(video_id: &str, source_path: &str, source_hash: &str, config_hash: &str) -> Self {
        Self {
            manifest_version: "1.0".to_string(),
            video_id: video_id.to_string(),
            source_path: source_path.to_string(),
            source_hash: source_hash.to_string(),
            config_hash: config_hash.to_string(),
            stages: BTreeMap::new(),
        }
    }

    /// Load manifest from file.
    pub fn load(path: &Path) -> Result<Self, StageError> {
        let contents = std::fs::read_to_string(path)?;
        let manifest = serde_json::from_str(&contents)?;
        Ok(manifest)
    }

    /// Save manifest to file.
    pub fn save(&self, path: &Path) -> Result<(), StageError> {
        let contents = serde_json::to_string_pretty(self)?;
        std::fs::write(path, contents)?;
        Ok(())
    }

    /// Check if a stage has completed and its cache is valid.
    /// Returns true if: status == Done AND input_hash matches.
    pub fn is_cached(&self, stage_name: &str, input_hash: &str) -> bool {
        if let Some(record) = self.stages.get(stage_name) {
            record.status == StageStatus::Done && record.input_hash == input_hash
        } else {
            false
        }
    }

    /// Mark a stage as running.
    pub fn mark_running(&mut self, stage_name: &str, input_hash: &str) {
        self.stages.insert(
            stage_name.to_string(),
            StageRecord {
                status: StageStatus::Running,
                input_hash: input_hash.to_string(),
                output_path: String::new(),
                completed_at: None,
            },
        );
    }

    /// Mark a stage as done.
    pub fn mark_done(&mut self, stage_name: &str, output_path: &str) {
        if let Some(record) = self.stages.get_mut(stage_name) {
            record.status = StageStatus::Done;
            record.output_path = output_path.to_string();
            record.completed_at = Some(Utc::now());
        }
    }

    /// Mark a stage as failed.
    pub fn mark_failed(&mut self, stage_name: &str) {
        if let Some(record) = self.stages.get_mut(stage_name) {
            record.status = StageStatus::Failed;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_new() {
        let m = Manifest::new("vid_001", "/path/video.mp4", "sha256:vid", "sha256:cfg");
        assert_eq!(m.video_id, "vid_001");
        assert_eq!(m.manifest_version, "1.0");
        assert!(m.stages.is_empty());
    }

    #[test]
    fn test_manifest_save_and_load() {
        let temp_dir = tempfile::tempdir().unwrap();
        let manifest_path = temp_dir.path().join("manifest.json");

        let mut manifest = Manifest::new("vid_001", "/path/video.mp4", "sha256:vid", "sha256:cfg");
        manifest.mark_running("stage0", "hash123");
        manifest.mark_done("stage0", "output/stage0.json");

        manifest.save(&manifest_path).unwrap();

        let loaded = Manifest::load(&manifest_path).unwrap();
        assert_eq!(loaded.video_id, manifest.video_id);
        assert_eq!(loaded.stages.len(), 1);
        assert_eq!(loaded.stages["stage0"].status, StageStatus::Done);
    }

    #[test]
    fn test_is_cached_hit() {
        let mut manifest = Manifest::new("vid_001", "/path/video.mp4", "sha256:vid", "sha256:cfg");
        manifest.mark_running("stage0", "hash123");
        manifest.mark_done("stage0", "output/stage0.json");

        assert!(manifest.is_cached("stage0", "hash123"));
    }

    #[test]
    fn test_is_cached_miss_different_hash() {
        let mut manifest = Manifest::new("vid_001", "/path/video.mp4", "sha256:vid", "sha256:cfg");
        manifest.mark_running("stage0", "hash123");
        manifest.mark_done("stage0", "output/stage0.json");

        assert!(!manifest.is_cached("stage0", "hash456"));
    }

    #[test]
    fn test_is_cached_miss_not_done() {
        let mut manifest = Manifest::new("vid_001", "/path/video.mp4", "sha256:vid", "sha256:cfg");
        manifest.mark_running("stage0", "hash123");

        assert!(!manifest.is_cached("stage0", "hash123"));
    }

    #[test]
    fn test_is_cached_miss_not_exists() {
        let manifest = Manifest::new("vid_001", "/path/video.mp4", "sha256:vid", "sha256:cfg");

        assert!(!manifest.is_cached("stage0", "hash123"));
    }

    #[test]
    fn test_mark_done_updates_status_and_timestamp() {
        let mut manifest = Manifest::new("vid_001", "/path/video.mp4", "sha256:vid", "sha256:cfg");
        manifest.mark_running("stage0", "hash123");
        manifest.mark_done("stage0", "output/stage0.json");

        let record = &manifest.stages["stage0"];
        assert_eq!(record.status, StageStatus::Done);
        assert_eq!(record.output_path, "output/stage0.json");
        assert!(record.completed_at.is_some());
    }

    #[test]
    fn test_mark_failed() {
        let mut manifest = Manifest::new("vid_001", "/path/video.mp4", "sha256:vid", "sha256:cfg");
        manifest.mark_running("stage0", "hash123");
        manifest.mark_failed("stage0");

        let record = &manifest.stages["stage0"];
        assert_eq!(record.status, StageStatus::Failed);
    }
}
