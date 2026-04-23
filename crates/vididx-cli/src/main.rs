use clap::{Parser, Subcommand};
use reqwest::blocking::Client;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use tokio::sync::Mutex;
use url::Url;
use vididx_core::{Config, VididxError};
use vididx_pipeline::{JobContext, Manifest, run_pipeline};

mod wix;

use wix::{
    WixDirectSegmentResolver, WixDownloadOptions, WixDownloader, WixSourceResolver,
    build_download_plan,
};

#[derive(Parser)]
#[command(
    name = "vididx",
    about = "Video indexing and RAG preprocessing",
    long_about = "A tool for processing videos into retrieval-ready JSON chunks for RAG systems"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Process a video through the pipeline
    Process {
        /// Path to a local video file, or a URL resolvable by yt-dlp
        video: String,

        /// Video ID (default: filename without extension)
        #[arg(long)]
        video_id: Option<String>,

        /// Start from this stage (0-9, default: 0)
        #[arg(long)]
        from: Option<usize>,

        /// End at this stage (0-9, default: 9)
        #[arg(long)]
        to: Option<usize>,

        /// Force re-processing even if cached
        #[arg(long)]
        force: bool,

        /// Output directory
        #[arg(long, short = 'o')]
        output: Option<PathBuf>,
    },

    /// Inspect output of a processing job
    Inspect {
        /// Output directory to inspect
        out_dir: PathBuf,
    },

    /// Validate a chunks JSONL file
    Validate {
        /// Path to chunks.jsonl file
        jsonl: PathBuf,
    },

    /// Estimate processing time and disk space for a video
    Estimate {
        /// Path to a local video file, or a URL resolvable by yt-dlp
        video: String,
    },

    /// Reconstruct a Wix-hosted video from a segment URL
    WixDownload {
        /// A Wix segment URL like .../seg-5-v1-a1.ts?token=...
        input: String,

        /// Output base directory
        #[arg(long, short = 'o', default_value = "testdata")]
        output: PathBuf,

        /// Override the inferred video ID for the output folder
        #[arg(long)]
        video_id: Option<String>,

        /// Maximum segment index to probe
        #[arg(long, default_value_t = 300)]
        max_segments: usize,

        /// Stop after this many consecutive missing segments
        #[arg(long, default_value_t = 3)]
        stop_after_misses: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum VideoInput {
    LocalPath(PathBuf),
    WebUrl(Url),
}

#[derive(Debug, Clone)]
struct PreparedInput {
    source_type: String,
    source_ref: String,
    local_video_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
struct EstimateInfo {
    label: String,
    size_mb: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ToolRequirement {
    name: &'static str,
    command: String,
}

const DIRECT_VIDEO_EXTENSIONS: &[&str] = &["mp4", "m4v", "mov", "webm", "mkv"];

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Process {
            video,
            video_id,
            from,
            to,
            force,
            output,
        } => {
            process_command(video, video_id, from, to, force, output).await?;
        }
        Commands::Inspect { out_dir } => {
            inspect_command(out_dir).await?;
        }
        Commands::Validate { jsonl } => {
            validate_command(jsonl).await?;
        }
        Commands::Estimate { video } => {
            estimate_command(video).await?;
        }
        Commands::WixDownload {
            input,
            output,
            video_id,
            max_segments,
            stop_after_misses,
        } => {
            wix_download_command(input, output, video_id, max_segments, stop_after_misses).await?;
        }
    }

    Ok(())
}

async fn process_command(
    video: String,
    video_id: Option<String>,
    from: Option<usize>,
    to: Option<usize>,
    force: bool,
    output: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let video_input = classify_video_input(&video)?;
    let config = Config::load(None)?;
    let (from_stage, to_stage) = normalize_stage_range(from, to)?;
    run_preflight_checks(&config, &video_input, from_stage, to_stage)?;

    let vid_id = video_id
        .unwrap_or_else(|| default_video_id(&video_input).unwrap_or_else(|| "video".to_string()));

    let out_dir = output.unwrap_or_else(|| PathBuf::from(format!("output/{}", vid_id)));
    fs::create_dir_all(&out_dir)?;
    let prepared = prepare_input_for_processing(&video_input, &config, &out_dir)?;

    let manifest_path = out_dir.join("manifest.json");
    let source_hash = vididx_core::hash::sha256_file(&prepared.local_video_path)?;
    let config_hash = format!(
        "sha256:{}",
        serde_json::to_string(&config).unwrap_or_default()
    );

    let manifest = if manifest_path.exists() && !force {
        Manifest::load(&manifest_path)
            .map_err(|e| VididxError::Config(format!("Failed to load manifest: {:?}", e)))?
    } else {
        Manifest::new(&vid_id, &prepared.source_ref, &source_hash, &config_hash)
    };

    // Create job context
    let ctx = JobContext {
        video_id: vid_id.clone(),
        source_type: prepared.source_type,
        source_ref: prepared.source_ref.clone(),
        source_path: prepared.local_video_path.clone(),
        out_dir,
        from_stage,
        to_stage,
        config,
        manifest: Arc::new(Mutex::new(manifest)),
    };

    eprintln!(
        "Processing video: {} (stages {}-{})",
        prepared.source_ref, from_stage, to_stage
    );
    run_pipeline(ctx).await?;

    eprintln!("✓ Processing complete");
    Ok(())
}

async fn inspect_command(out_dir: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = out_dir.join("manifest.json");
    check_file_exists(&manifest_path)?;

    let manifest = Manifest::load(&manifest_path)
        .map_err(|e| VididxError::Config(format!("Failed to load manifest: {:?}", e)))?;

    println!("Video ID: {}", manifest.video_id);
    println!("Source: {}", manifest.source_path);
    println!("Source Hash: {}", manifest.source_hash);
    println!("Config Hash: {}", manifest.config_hash);
    println!("\nStages:");

    for (stage_name, record) in &manifest.stages {
        println!(
            "  {} - {:?} ({})",
            stage_name, record.status, record.input_hash
        );
    }

    Ok(())
}

async fn validate_command(jsonl: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    check_file_exists(&jsonl)?;

    let content = fs::read_to_string(&jsonl)?;
    let mut valid_count = 0;
    let mut error_count = 0;

    for (line_no, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }

        match serde_json::from_str::<serde_json::Value>(line) {
            Ok(_) => {
                valid_count += 1;
            }
            Err(e) => {
                eprintln!("Line {}: {}", line_no + 1, e);
                error_count += 1;
            }
        }
    }

    println!("✓ Valid lines: {}", valid_count);
    if error_count > 0 {
        println!("✗ Invalid lines: {}", error_count);
        return Err("Validation failed".into());
    }

    Ok(())
}

async fn estimate_command(video: String) -> Result<(), Box<dyn std::error::Error>> {
    let video_input = classify_video_input(&video)?;
    let config = Config::load(None)?;
    let estimate = estimate_input(&video_input, &config)?;
    let size_mb = estimate.size_mb;

    // Rough estimates
    let est_chunks = (size_mb / 10.0).max(5.0) as usize;
    let est_output_mb = (size_mb * 0.5).ceil() as usize;
    let est_time_min = (size_mb / 50.0).ceil() as usize;

    println!("Video: {}", estimate.label);
    println!("File size: {:.1} MB", size_mb);
    println!("\nEstimates:");
    println!("  Chunks: ~{}", est_chunks);
    println!("  Output size: ~{} MB", est_output_mb);
    println!("  Processing time: ~{} min", est_time_min);

    Ok(())
}

async fn wix_download_command(
    input: String,
    output: PathBuf,
    video_id: Option<String>,
    max_segments: usize,
    stop_after_misses: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let input_for_log = input.clone();
    let output_for_log = output.clone();
    eprintln!("Resolving Wix segments from: {}", input_for_log);
    eprintln!("Writing output under: {}", output_for_log.display());

    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        fs::create_dir_all(&output).map_err(|err| err.to_string())?;

        let resolver = WixDirectSegmentResolver;
        let resolved = resolver
            .resolve(&input)
            .map_err(|err| format!("Failed to resolve Wix segment source: {}", err))?;
        let plan = build_download_plan(&resolved, &output, video_id.as_deref());
        let downloader = WixDownloader::new(http_client().map_err(|err| err.to_string())?);
        let options = WixDownloadOptions {
            max_segments,
            stop_after_misses,
        };

        downloader
            .download(&resolved, &plan, &options)
            .map_err(|err| format!("Failed to download Wix video: {}", err))
    })
    .await
    .map_err(|err| format!("Wix download task failed: {}", err))?
    .map_err(|err| -> Box<dyn std::error::Error> { err.into() })?;

    println!("Source: {}", result.source_ref);
    println!("Segments: {}", result.segment_count);
    println!("Segments dir: {}", result.segments_dir.display());
    println!("MP4: {}", result.output_mp4.display());

    Ok(())
}

fn classify_video_input(input: &str) -> Result<VideoInput, Box<dyn std::error::Error>> {
    if let Ok(url) = Url::parse(input)
        && matches!(url.scheme(), "http" | "https")
    {
        return Ok(VideoInput::WebUrl(url));
    }

    Ok(VideoInput::LocalPath(PathBuf::from(input)))
}

fn default_video_id(input: &VideoInput) -> Option<String> {
    match input {
        VideoInput::LocalPath(path) => path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .map(ToOwned::to_owned),
        VideoInput::WebUrl(url) => {
            let last_segment = url
                .path_segments()
                .and_then(|mut segments| segments.rfind(|segment| !segment.is_empty()))
                .map(sanitize_video_id);
            last_segment.or_else(|| url.host_str().map(sanitize_video_id))
        }
    }
}

fn sanitize_video_id(input: &str) -> String {
    let sanitized = input
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .to_string();

    if sanitized.is_empty() {
        "video".to_string()
    } else {
        sanitized
    }
}

fn prepare_input_for_processing(
    input: &VideoInput,
    config: &Config,
    out_dir: &Path,
) -> Result<PreparedInput, Box<dyn std::error::Error>> {
    match input {
        VideoInput::LocalPath(path) => {
            check_video_exists(path)?;
            Ok(PreparedInput {
                source_type: "local_mp4".to_string(),
                source_ref: path.to_string_lossy().to_string(),
                local_video_path: path.clone(),
            })
        }
        VideoInput::WebUrl(url) => {
            let downloader = configured_url_downloader(config)?;
            ensure_tool_available(downloader)?;
            let input_dir = out_dir.join("input");
            fs::create_dir_all(&input_dir)?;
            let local_video_path = download_video(downloader, url, &input_dir)
                .map_err(|err| format!("Failed to fetch video from URL {}: {}", url, err))?;
            Ok(PreparedInput {
                source_type: "web_url".to_string(),
                source_ref: url.as_str().to_string(),
                local_video_path,
            })
        }
    }
}

fn estimate_input(
    input: &VideoInput,
    config: &Config,
) -> Result<EstimateInfo, Box<dyn std::error::Error>> {
    match input {
        VideoInput::LocalPath(path) => {
            check_video_exists(path)?;
            let metadata = fs::metadata(path)?;
            Ok(EstimateInfo {
                label: path.display().to_string(),
                size_mb: bytes_to_mb(metadata.len()),
            })
        }
        VideoInput::WebUrl(url) => {
            let downloader = configured_url_downloader(config)?;
            ensure_tool_available(downloader)?;
            let size_mb = estimate_url_size_mb(downloader, url)?;
            Ok(EstimateInfo {
                label: url.as_str().to_string(),
                size_mb,
            })
        }
    }
}

fn configured_url_downloader(config: &Config) -> Result<&str, Box<dyn std::error::Error>> {
    match config.media.url_downloader.as_deref() {
        Some("yt-dlp") => Ok("yt-dlp"),
        Some(other) => Err(format!(
            "Unsupported media.url_downloader: {}. Only \"yt-dlp\" is supported for URL input.",
            other
        )
        .into()),
        None => Err(
            "URL input is disabled. Set media.url_downloader = \"yt-dlp\" in config to enable yt-dlp-backed URL downloads."
                .into(),
        ),
    }
}

fn bytes_to_mb(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn normalize_stage_range(
    from: Option<usize>,
    to: Option<usize>,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let from_stage = from.unwrap_or(0);
    let to_stage = to.unwrap_or(9);

    if from_stage > 9 {
        return Err(format!("Invalid --from stage {}. Expected 0-9.", from_stage).into());
    }
    if to_stage > 9 {
        return Err(format!("Invalid --to stage {}. Expected 0-9.", to_stage).into());
    }
    if from_stage > to_stage {
        return Err(format!(
            "Invalid stage range: --from {} is greater than --to {}.",
            from_stage, to_stage
        )
        .into());
    }

    Ok((from_stage, to_stage))
}

fn stage_range_includes(from_stage: usize, to_stage: usize, stage: usize) -> bool {
    (from_stage..=to_stage).contains(&stage)
}

fn tool_requirements(
    config: &Config,
    input: &VideoInput,
    from_stage: usize,
    to_stage: usize,
) -> Result<Vec<ToolRequirement>, String> {
    let mut requirements = Vec::new();

    if from_stage == 0 {
        requirements.push(ToolRequirement {
            name: "ffprobe",
            command: config.media.ffprobe_path.clone(),
        });
    }

    if stage_range_includes(from_stage, to_stage, 1)
        || stage_range_includes(from_stage, to_stage, 3)
        || stage_range_includes(from_stage, to_stage, 4)
        || stage_range_includes(from_stage, to_stage, 5)
    {
        requirements.push(ToolRequirement {
            name: "ffmpeg",
            command: config.media.ffmpeg_path.clone(),
        });
    }

    if stage_range_includes(from_stage, to_stage, 2) && config.asr.adapter == "whisper_cpp" {
        requirements.push(ToolRequirement {
            name: "whisper-cli",
            command: config.asr.whisper_cpp.binary_path.clone(),
        });
    }

    if to_stage == 9 && config.vision.ocr.adapter == "tesseract" {
        requirements.push(ToolRequirement {
            name: "tesseract",
            command: config.vision.ocr.adapter.clone(),
        });
    }

    if matches!(input, VideoInput::WebUrl(_)) {
        let downloader = configured_url_downloader(config).map_err(|err| err.to_string())?;
        requirements.push(ToolRequirement {
            name: "yt-dlp",
            command: downloader.to_string(),
        });
    }

    Ok(requirements)
}

fn run_preflight_checks(
    config: &Config,
    input: &VideoInput,
    from_stage: usize,
    to_stage: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let requirements =
        tool_requirements(config, input, from_stage, to_stage).map_err(|err| err.to_string())?;
    let mut failures = Vec::new();

    for requirement in requirements {
        if let Err(err) = ensure_tool_available(&requirement.command) {
            failures.push(format!("{} ({})", requirement.name, err));
        }
    }

    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!("Pre-flight check failed: {}", failures.join(", ")).into())
    }
}

fn ensure_tool_available(tool: &str) -> Result<(), Box<dyn std::error::Error>> {
    let output = Command::new(tool).arg("-version").output()?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!("Required tool not available: {}", tool).into())
    }
}

fn download_video(
    downloader: &str,
    url: &Url,
    out_dir: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    match download_video_with_ytdlp(downloader, url, out_dir) {
        Ok(path) => Ok(path),
        Err(ytdlp_error) => {
            if can_fallback_to_direct_http(url)? {
                download_video_direct_http(url, out_dir).map_err(|direct_error| {
                    format!(
                        "yt-dlp failed ({}) and direct-http fallback failed ({}). Only yt-dlp-resolvable URLs and direct video file links are supported.",
                        ytdlp_error, direct_error
                    )
                    .into()
                })
            } else {
                Err(format!(
                    "{}. Only yt-dlp-resolvable URLs and direct video file links are supported.",
                    ytdlp_error
                )
                .into())
            }
        }
    }
}

fn download_video_with_ytdlp(
    downloader: &str,
    url: &Url,
    out_dir: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let output_template = out_dir.join("source.%(ext)s");
    let output = Command::new(downloader)
        .arg("--no-playlist")
        .arg("--merge-output-format")
        .arg("mp4")
        .arg("-o")
        .arg(&output_template)
        .arg(url.as_str())
        .output()?;

    if !output.status.success() {
        return Err(format!(
            "yt-dlp download failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }

    find_downloaded_video(out_dir)
}

fn download_video_direct_http(
    url: &Url,
    out_dir: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let client = http_client()?;
    let response = client.get(url.as_str()).send()?;
    let response = response.error_for_status()?;

    let extension = infer_direct_video_extension(
        url,
        response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
    );
    let output_path = out_dir.join(format!("source.{}", extension));
    let mut file = fs::File::create(&output_path)?;
    let bytes = response.bytes()?;
    file.write_all(&bytes)?;

    Ok(output_path)
}

fn find_downloaded_video(out_dir: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut entries = fs::read_dir(out_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("source."))
        })
        .collect::<Vec<_>>();

    entries.sort();
    entries
        .into_iter()
        .next()
        .ok_or_else(|| "yt-dlp completed but no downloaded video was found".into())
}

fn fetch_ytdlp_metadata(
    downloader: &str,
    url: &Url,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let output = Command::new(downloader)
        .arg("--dump-single-json")
        .arg("--no-download")
        .arg(url.as_str())
        .output()?;

    if !output.status.success() {
        return Err(format!(
            "yt-dlp metadata fetch failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }

    serde_json::from_slice(&output.stdout).map_err(Into::into)
}

fn estimate_url_size_mb(downloader: &str, url: &Url) -> Result<f64, Box<dyn std::error::Error>> {
    match fetch_ytdlp_metadata(downloader, url) {
        Ok(metadata) => extract_estimated_size_mb(&metadata)
            .ok_or_else(|| "Could not estimate remote video size from yt-dlp metadata".into()),
        Err(ytdlp_error) => {
            if can_fallback_to_direct_http(url)? {
                estimate_direct_http_size_mb(url).map_err(|direct_error| {
                    format!(
                        "yt-dlp metadata fetch failed ({}) and direct-http fallback failed ({}). Only yt-dlp-resolvable URLs and direct video file links are supported.",
                        ytdlp_error, direct_error
                    )
                    .into()
                })
            } else {
                Err(format!(
                    "{}. Only yt-dlp-resolvable URLs and direct video file links are supported.",
                    ytdlp_error
                )
                .into())
            }
        }
    }
}

fn estimate_direct_http_size_mb(url: &Url) -> Result<f64, Box<dyn std::error::Error>> {
    let client = http_client()?;
    let response = client.head(url.as_str()).send()?;
    let response = response.error_for_status()?;
    let content_length = response
        .headers()
        .get(reqwest::header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .ok_or_else(|| "No Content-Length available for direct HTTP estimate".to_string())?;

    Ok(bytes_to_mb(content_length))
}

fn can_fallback_to_direct_http(url: &Url) -> Result<bool, Box<dyn std::error::Error>> {
    if has_direct_video_extension(url) {
        return Ok(true);
    }

    let client = http_client()?;
    let response = client.head(url.as_str()).send()?;
    if !response.status().is_success() {
        return Ok(false);
    }

    Ok(response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(is_video_content_type))
}

fn has_direct_video_extension(url: &Url) -> bool {
    url.path_segments()
        .and_then(|mut segments| segments.next_back())
        .and_then(|segment| {
            segment
                .rsplit_once('.')
                .map(|(_, ext)| ext.to_ascii_lowercase())
        })
        .is_some_and(|ext| {
            DIRECT_VIDEO_EXTENSIONS
                .iter()
                .any(|candidate| *candidate == ext)
        })
}

fn is_video_content_type(content_type: &str) -> bool {
    content_type
        .split(';')
        .next()
        .is_some_and(|value| value.trim().starts_with("video/"))
}

fn infer_direct_video_extension(url: &Url, content_type: Option<&str>) -> String {
    if let Some(extension) = url
        .path_segments()
        .and_then(|mut segments| segments.next_back())
        .and_then(|segment| {
            segment
                .rsplit_once('.')
                .map(|(_, ext)| ext.to_ascii_lowercase())
        })
        .filter(|ext| {
            DIRECT_VIDEO_EXTENSIONS
                .iter()
                .any(|candidate| *candidate == ext)
        })
    {
        return extension;
    }

    if let Some(mapped) = content_type.and_then(content_type_to_extension) {
        return mapped.to_string();
    }

    "mp4".to_string()
}

fn content_type_to_extension(content_type: &str) -> Option<&'static str> {
    let media_type = content_type.split(';').next()?.trim();
    match media_type {
        "video/mp4" => Some("mp4"),
        "video/x-m4v" => Some("m4v"),
        "video/quicktime" => Some("mov"),
        "video/webm" => Some("webm"),
        "video/x-matroska" => Some("mkv"),
        _ => None,
    }
}

fn http_client() -> Result<Client, Box<dyn std::error::Error>> {
    Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(Into::into)
}

fn extract_estimated_size_mb(metadata: &serde_json::Value) -> Option<f64> {
    metadata
        .get("filesize")
        .and_then(serde_json::Value::as_u64)
        .or_else(|| {
            metadata
                .get("filesize_approx")
                .and_then(serde_json::Value::as_u64)
        })
        .map(bytes_to_mb)
}

fn check_video_exists(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if !path.exists() {
        return Err(format!("Video file not found: {}", path.display()).into());
    }
    Ok(())
}

fn check_file_exists(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if !path.exists() {
        return Err(format!("File not found: {}", path.display()).into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parse_process() {
        let args = vec!["vididx", "process", "video.mp4"];
        let cli = Cli::try_parse_from(args).unwrap();

        match cli.command {
            Commands::Process { video, .. } => {
                assert_eq!(video, "video.mp4");
            }
            _ => panic!("Expected Process command"),
        }
    }

    #[test]
    fn test_cli_parse_process_with_options() {
        let args = vec![
            "vididx",
            "process",
            "video.mp4",
            "--video-id",
            "test_vid",
            "--force",
        ];
        let cli = Cli::try_parse_from(args).unwrap();

        match cli.command {
            Commands::Process {
                video,
                video_id,
                force,
                ..
            } => {
                assert_eq!(video, "video.mp4");
                assert_eq!(video_id, Some("test_vid".to_string()));
                assert!(force);
            }
            _ => panic!("Expected Process command"),
        }
    }

    #[test]
    fn test_tool_requirements_local_input_includes_core_local_tools() {
        let config = Config::default();
        let input = VideoInput::LocalPath(PathBuf::from("video.mp4"));

        let requirements = tool_requirements(&config, &input, 0, 9).unwrap();
        let commands = requirements
            .iter()
            .map(|item| item.command.as_str())
            .collect::<Vec<_>>();

        assert!(commands.contains(&"ffmpeg"));
        assert!(commands.contains(&"ffprobe"));
        assert!(commands.contains(&"whisper-cli"));
        assert!(commands.contains(&"tesseract"));
        assert!(!commands.contains(&"yt-dlp"));
    }

    #[test]
    fn test_tool_requirements_url_input_includes_url_downloader() {
        let mut config = Config::default();
        config.media.url_downloader = Some("yt-dlp".to_string());
        let input = VideoInput::WebUrl(Url::parse("https://example.com/video.mp4").unwrap());

        let requirements = tool_requirements(&config, &input, 0, 9).unwrap();
        let commands = requirements
            .iter()
            .map(|item| item.command.as_str())
            .collect::<Vec<_>>();

        assert!(commands.contains(&"yt-dlp"));
    }

    #[test]
    fn test_tool_requirements_url_input_errors_when_downloader_disabled() {
        let config = Config::default();
        let input = VideoInput::WebUrl(Url::parse("https://example.com/video.mp4").unwrap());

        let result = tool_requirements(&config, &input, 0, 9);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("URL input is disabled"));
    }

    #[test]
    fn test_tool_requirements_later_stage_skips_earlier_tools() {
        let config = Config::default();
        let input = VideoInput::LocalPath(PathBuf::from("video.mp4"));

        let requirements = tool_requirements(&config, &input, 8, 9).unwrap();
        let commands = requirements
            .iter()
            .map(|item| item.command.as_str())
            .collect::<Vec<_>>();

        assert!(!commands.contains(&"ffmpeg"));
        assert!(!commands.contains(&"ffprobe"));
        assert!(!commands.contains(&"whisper-cli"));
        assert!(commands.contains(&"tesseract"));
    }

    #[test]
    fn test_normalize_stage_range_defaults_to_full_pipeline() {
        let range = normalize_stage_range(None, None).unwrap();
        assert_eq!(range, (0, 9));
    }

    #[test]
    fn test_normalize_stage_range_rejects_inverted_range() {
        let result = normalize_stage_range(Some(7), Some(3));
        assert!(result.is_err());
    }

    #[test]
    fn test_cli_parse_inspect() {
        let args = vec!["vididx", "inspect", "output"];
        let cli = Cli::try_parse_from(args).unwrap();

        match cli.command {
            Commands::Inspect { out_dir } => {
                assert_eq!(out_dir, PathBuf::from("output"));
            }
            _ => panic!("Expected Inspect command"),
        }
    }

    #[test]
    fn test_cli_parse_validate() {
        let args = vec!["vididx", "validate", "chunks.jsonl"];
        let cli = Cli::try_parse_from(args).unwrap();

        match cli.command {
            Commands::Validate { jsonl } => {
                assert_eq!(jsonl, PathBuf::from("chunks.jsonl"));
            }
            _ => panic!("Expected Validate command"),
        }
    }

    #[test]
    fn test_cli_parse_estimate() {
        let args = vec!["vididx", "estimate", "video.mp4"];
        let cli = Cli::try_parse_from(args).unwrap();

        match cli.command {
            Commands::Estimate { video } => {
                assert_eq!(video, "video.mp4");
            }
            _ => panic!("Expected Estimate command"),
        }
    }

    #[test]
    fn test_cli_parse_wix_download() {
        let args = vec![
            "vididx",
            "wix-download",
            "https://repackager.wixmp.com/video.wixstatic.com/video/abc123/2160p/mp4/file.mp4/seg-5-v1-a1.ts?token=demo",
            "--output",
            "tmp",
            "--video-id",
            "sample",
            "--max-segments",
            "55",
            "--stop-after-misses",
            "4",
        ];

        let cli = Cli::try_parse_from(args).unwrap();

        match cli.command {
            Commands::WixDownload {
                input,
                output,
                video_id,
                max_segments,
                stop_after_misses,
            } => {
                assert_eq!(
                    input,
                    "https://repackager.wixmp.com/video.wixstatic.com/video/abc123/2160p/mp4/file.mp4/seg-5-v1-a1.ts?token=demo"
                );
                assert_eq!(output, PathBuf::from("tmp"));
                assert_eq!(video_id, Some("sample".to_string()));
                assert_eq!(max_segments, 55);
                assert_eq!(stop_after_misses, 4);
            }
            _ => panic!("Expected WixDownload command"),
        }
    }

    #[test]
    fn test_classify_video_input_http_url() {
        let input = classify_video_input("https://example.com/watch?v=123").unwrap();
        assert!(matches!(input, VideoInput::WebUrl(_)));
    }

    #[test]
    fn test_default_video_id_from_url_last_segment() {
        let input = classify_video_input("https://example.com/videos/demo-file").unwrap();
        assert_eq!(default_video_id(&input).as_deref(), Some("demo-file"));
    }

    #[test]
    fn test_extract_estimated_size_mb_prefers_filesize() {
        let metadata = serde_json::json!({
            "filesize": 52_428_800_u64,
            "filesize_approx": 104_857_600_u64
        });

        let size_mb = extract_estimated_size_mb(&metadata).unwrap();
        assert!((size_mb - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_has_direct_video_extension_mp4_url() {
        let url = Url::parse("https://cdn.example.com/video/sample.mp4").unwrap();
        assert!(has_direct_video_extension(&url));
    }

    #[test]
    fn test_is_video_content_type_accepts_video_media_type() {
        assert!(is_video_content_type("video/mp4"));
        assert!(is_video_content_type("video/mp4; charset=binary"));
        assert!(!is_video_content_type("text/html"));
    }

    #[test]
    fn test_infer_direct_video_extension_prefers_content_type() {
        let url = Url::parse("https://example.com/download").unwrap();
        assert_eq!(
            infer_direct_video_extension(&url, Some("video/webm")),
            "webm"
        );
    }

    #[test]
    fn test_configured_url_downloader_disabled_by_default() {
        let config = Config::default();
        let result = configured_url_downloader(&config);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "URL input is disabled. Set media.url_downloader = \"yt-dlp\" in config to enable yt-dlp-backed URL downloads."
        );
    }

    #[test]
    fn test_configured_url_downloader_accepts_ytdlp() {
        let mut config = Config::default();
        config.media.url_downloader = Some("yt-dlp".to_string());
        assert_eq!(configured_url_downloader(&config).unwrap(), "yt-dlp");
    }

    #[test]
    fn test_configured_url_downloader_rejects_unsupported_tool() {
        let mut config = Config::default();
        config.media.url_downloader = Some("curl".to_string());

        let result = configured_url_downloader(&config);

        assert_eq!(
            result.unwrap_err().to_string(),
            "Unsupported media.url_downloader: curl. Only \"yt-dlp\" is supported for URL input."
        );
    }
}
