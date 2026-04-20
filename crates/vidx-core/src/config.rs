use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::error::VidxError;
use crate::hash;

/// General configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GeneralConfig {
    pub out_dir: String,
    pub log_level: String,
    pub parallelism: usize,
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            out_dir: "./out".to_string(),
            log_level: "info".to_string(),
            parallelism: 4,
        }
    }
}

/// Media configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MediaConfig {
    pub ffmpeg_path: String,
    pub ffprobe_path: String,
    pub audio_sample_rate: u32,
}

impl Default for MediaConfig {
    fn default() -> Self {
        Self {
            ffmpeg_path: "ffmpeg".to_string(),
            ffprobe_path: "ffprobe".to_string(),
            audio_sample_rate: 16000,
        }
    }
}

/// Coarse segmentation config.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CoarseSegmentConfig {
    pub max_duration_sec: f64,
    pub snap_window_sec: f64,
}

impl Default for CoarseSegmentConfig {
    fn default() -> Self {
        Self {
            max_duration_sec: 300.0,
            snap_window_sec: 30.0,
        }
    }
}

/// Semantic segmentation config.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SemanticSegmentConfig {
    pub target_min_sec: f64,
    pub target_max_sec: f64,
    pub hard_min_sec: f64,
    pub hard_max_sec: f64,
}

impl Default for SemanticSegmentConfig {
    fn default() -> Self {
        Self {
            target_min_sec: 30.0,
            target_max_sec: 90.0,
            hard_min_sec: 15.0,
            hard_max_sec: 120.0,
        }
    }
}

/// Segment configuration container.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct SegmentConfig {
    pub coarse: CoarseSegmentConfig,
    pub semantic: SemanticSegmentConfig,
}

/// Frame extraction configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FramesConfig {
    pub periodic_interval_sec: f64,
    pub scene_change_threshold: f64,
    pub max_analyzed_per_chunk: usize,
    pub dhash_distance_threshold: usize,
}

impl Default for FramesConfig {
    fn default() -> Self {
        Self {
            periodic_interval_sec: 5.0,
            scene_change_threshold: 0.4,
            max_analyzed_per_chunk: 8,
            dhash_distance_threshold: 10,
        }
    }
}

/// Whisper.cpp specific config.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WhisperCppConfig {
    pub binary_path: String,
    pub model_path: String,
    pub threads: usize,
}

impl Default for WhisperCppConfig {
    fn default() -> Self {
        Self {
            binary_path: "whisper-cli".to_string(),
            model_path: "~/.cache/whisper.cpp/ggml-large-v3.bin".to_string(),
            threads: 8,
        }
    }
}

/// ASR configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AsrConfig {
    pub adapter: String,
    pub model: String,
    pub language: String,
    #[serde(default)]
    pub whisper_cpp: WhisperCppConfig,
}

impl Default for AsrConfig {
    fn default() -> Self {
        Self {
            adapter: "whisper_cpp".to_string(),
            model: "large-v3".to_string(),
            language: "auto".to_string(),
            whisper_cpp: WhisperCppConfig::default(),
        }
    }
}

/// OCR configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OcrConfig {
    pub adapter: String,
    pub languages: Vec<String>,
}

impl Default for OcrConfig {
    fn default() -> Self {
        Self {
            adapter: "tesseract".to_string(),
            languages: vec!["jpn".to_string(), "eng".to_string()],
        }
    }
}

/// Caption configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CaptionConfig {
    pub adapter: String,
    pub model: String,
    pub max_per_chunk: usize,
}

impl Default for CaptionConfig {
    fn default() -> Self {
        Self {
            adapter: "claude".to_string(),
            model: "claude-sonnet-4-6".to_string(),
            max_per_chunk: 3,
        }
    }
}

/// Vision configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct VisionConfig {
    pub ocr: OcrConfig,
    pub caption: CaptionConfig,
}

/// LLM configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    pub adapter: String,
    pub model: String,
    pub max_concurrency: usize,
    pub retry_max: usize,
    pub retry_backoff_ms: u64,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            adapter: "claude".to_string(),
            model: "claude-sonnet-4-6".to_string(),
            max_concurrency: 4,
            retry_max: 3,
            retry_backoff_ms: 1000,
        }
    }
}

/// Output configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputConfig {
    pub generate_markdown: bool,
    pub generate_jsonl: bool,
    pub embedding_text_fields: Vec<String>,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            generate_markdown: true,
            generate_jsonl: true,
            embedding_text_fields: vec![
                "title".to_string(),
                "summary".to_string(),
                "transcript".to_string(),
                "ocr_important".to_string(),
            ],
        }
    }
}

/// Secrets configuration (read from environment variables).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SecretsConfig {
    pub anthropic_api_key_env: String,
    pub openai_api_key_env: String,
    pub groq_api_key_env: String,
    #[serde(skip)]
    pub anthropic_api_key: Option<String>,
    #[serde(skip)]
    pub openai_api_key: Option<String>,
    #[serde(skip)]
    pub groq_api_key: Option<String>,
}

impl Default for SecretsConfig {
    fn default() -> Self {
        Self {
            anthropic_api_key_env: "ANTHROPIC_API_KEY".to_string(),
            openai_api_key_env: "OPENAI_API_KEY".to_string(),
            groq_api_key_env: "GROQ_API_KEY".to_string(),
            anthropic_api_key: None,
            openai_api_key: None,
            groq_api_key: None,
        }
    }
}

/// Main configuration struct.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub general: GeneralConfig,
    pub media: MediaConfig,
    pub segment: SegmentConfig,
    pub frames: FramesConfig,
    pub asr: AsrConfig,
    pub vision: VisionConfig,
    pub llm: LlmConfig,
    pub output: OutputConfig,
    #[serde(skip)]
    pub secrets: SecretsConfig,
}

impl Default for Config {
    fn default() -> Self {
        let mut secrets = SecretsConfig::default();
        secrets.load_from_env();

        Self {
            general: GeneralConfig::default(),
            media: MediaConfig::default(),
            segment: SegmentConfig::default(),
            frames: FramesConfig::default(),
            asr: AsrConfig::default(),
            vision: VisionConfig::default(),
            llm: LlmConfig::default(),
            output: OutputConfig::default(),
            secrets,
        }
    }
}

impl SecretsConfig {
    /// Load API keys from environment variables.
    fn load_from_env(&mut self) {
        self.anthropic_api_key = std::env::var(&self.anthropic_api_key_env).ok();
        self.openai_api_key = std::env::var(&self.openai_api_key_env).ok();
        self.groq_api_key = std::env::var(&self.groq_api_key_env).ok();
    }
}

impl Config {
    /// Load configuration from TOML file, with fallback to default.
    /// Tries: explicit path → ./vidx.toml → ~/.config/vidx/config.toml → default
    pub fn load(toml_path: Option<&Path>) -> Result<Self, VidxError> {
        let mut config = Self::default();

        let path_to_try = if let Some(p) = toml_path {
            Some(p.to_path_buf())
        } else {
            Self::find_default_config_path()
        };

        if let Some(path) = path_to_try.filter(|p| p.exists()) {
            let contents = std::fs::read_to_string(&path)?;
            let toml_value: toml::Table = toml::from_str(&contents)
                .map_err(|e| VidxError::Config(format!("Invalid TOML: {}", e)))?;

            Self::merge_toml(&mut config, toml_value)?;
        }

        config.secrets.load_from_env();
        Ok(config)
    }

    /// Find default config path: ./vidx.toml → ~/.config/vidx/config.toml
    fn find_default_config_path() -> Option<PathBuf> {
        let local = PathBuf::from("./vidx.toml");
        if local.exists() {
            return Some(local);
        }

        if let Some(home) = dirs::home_dir() {
            let home_config = home.join(".config/vidx/config.toml");
            if home_config.exists() {
                return Some(home_config);
            }
        }

        None
    }

    /// Merge TOML table into config (overwrites matching keys).
    /// Partial configs are allowed - only specified fields are overwritten.
    #[allow(clippy::collapsible_if)]
    fn merge_toml(config: &mut Config, toml: toml::Table) -> Result<(), VidxError> {
        if let Some(val) = toml.get("general") {
            if let Ok(partial) = val.clone().try_into::<GeneralConfig>() {
                config.general = partial;
            }
        }
        if let Some(val) = toml.get("media") {
            if let Ok(partial) = val.clone().try_into::<MediaConfig>() {
                config.media = partial;
            }
        }
        if let Some(val) = toml.get("segment") {
            if let Ok(partial) = val.clone().try_into::<SegmentConfig>() {
                config.segment = partial;
            }
        }
        if let Some(val) = toml.get("frames") {
            if let Ok(partial) = val.clone().try_into::<FramesConfig>() {
                config.frames = partial;
            }
        }
        if let Some(val) = toml.get("asr") {
            if let Ok(partial) = val.clone().try_into::<AsrConfig>() {
                config.asr = partial;
            }
        }
        if let Some(val) = toml.get("vision") {
            if let Ok(partial) = val.clone().try_into::<VisionConfig>() {
                config.vision = partial;
            }
        }
        if let Some(val) = toml.get("llm") {
            if let Ok(partial) = val.clone().try_into::<LlmConfig>() {
                config.llm = partial;
            }
        }
        if let Some(val) = toml.get("output") {
            if let Ok(partial) = val.clone().try_into::<OutputConfig>() {
                config.output = partial;
            }
        }
        Ok(())
    }

    /// Compute config hash for cache key validation.
    /// Excludes `secrets` to avoid API key changes invalidating cache.
    pub fn config_hash(&self) -> String {
        let hashable = serde_json::json!({
            "general": self.general,
            "media": self.media,
            "segment": self.segment,
            "frames": self.frames,
            "asr": self.asr,
            "vision": self.vision,
            "llm": self.llm,
            "output": self.output,
        });

        hash::sha256_str(&serde_json::to_string(&hashable).unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_default_config_values() {
        let cfg = Config::default();
        assert_eq!(cfg.segment.coarse.max_duration_sec, 300.0);
        assert_eq!(cfg.frames.periodic_interval_sec, 5.0);
        assert_eq!(cfg.llm.max_concurrency, 4);
    }

    #[test]
    fn test_config_load_with_partial_toml() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("vidx.toml");

        let toml_content = r#"
[general]
out_dir = "/custom/out"

[segment.coarse]
max_duration_sec = 600.0
"#;

        fs::write(&config_path, toml_content).unwrap();

        let cfg = Config::load(Some(&config_path)).unwrap();
        assert_eq!(cfg.general.out_dir, "/custom/out");
        assert_eq!(cfg.segment.coarse.max_duration_sec, 600.0);
        // Not overridden, should keep default
        assert_eq!(cfg.frames.periodic_interval_sec, 5.0);
    }

    #[test]
    fn test_config_load_invalid_toml() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("vidx.toml");

        fs::write(&config_path, "invalid [[[").unwrap();

        let result = Config::load(Some(&config_path));
        assert!(result.is_err());
        match result.unwrap_err() {
            VidxError::Config(_) => {}
            _ => panic!("Expected Config error"),
        }
    }

    #[test]
    fn test_config_hash_deterministic() {
        let cfg1 = Config::default();
        let hash1 = cfg1.config_hash();

        let cfg2 = Config::default();
        let hash2 = cfg2.config_hash();

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_config_hash_changes_with_field() {
        let mut cfg1 = Config::default();
        let hash1 = cfg1.config_hash();

        cfg1.segment.coarse.max_duration_sec = 600.0;
        let hash2 = cfg1.config_hash();

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_config_hash_excludes_secrets() {
        let mut cfg1 = Config::default();
        let hash1 = cfg1.config_hash();

        cfg1.secrets.anthropic_api_key = Some("new-key".to_string());
        let hash2 = cfg1.config_hash();

        assert_eq!(hash1, hash2, "secrets should not affect hash");
    }

    #[test]
    fn test_config_load_env_var() {
        unsafe {
            std::env::set_var("ANTHROPIC_API_KEY", "test-api-key");
        }
        let cfg = Config::default();
        assert_eq!(
            cfg.secrets.anthropic_api_key,
            Some("test-api-key".to_string())
        );
        unsafe {
            std::env::remove_var("ANTHROPIC_API_KEY");
        }
    }
}
