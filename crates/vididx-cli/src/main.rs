use clap::{Parser, Subcommand};
use reqwest::blocking::Client;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use tokio::sync::Mutex;
use url::Url;
use vididx_core::{Chunk, Config, VididxError};
use vididx_pipeline::{JobContext, Manifest, run_pipeline, stage_name_to_index};

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
    /// Path to config file (default: ./vididx.toml → ~/.config/vididx/config.toml)
    #[arg(long, global = true)]
    config: Option<PathBuf>,

    /// Log level: trace|debug|info|warn|error
    #[arg(long, global = true, default_value = "info")]
    log_level: String,

    /// Output directory (overrides config general.out_dir)
    #[arg(long, global = true)]
    out_dir: Option<PathBuf>,

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

        /// Start from this stage (number 0-9 or name like stage6_semantic)
        #[arg(long)]
        from: Option<String>,

        /// End at this stage (number 0-9 or name like stage4_coarse)
        #[arg(long)]
        to: Option<String>,

        /// Force re-processing even if cached
        #[arg(long)]
        force: bool,

        /// Estimate cost only, no LLM calls (not yet implemented)
        #[arg(long)]
        dry_run: bool,
    },

    /// Inspect output of a processing job
    Inspect {
        /// Output directory to inspect
        out_dir: PathBuf,
    },

    /// Validate a chunks JSONL file against the Chunk schema
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
            dry_run,
        } => {
            process_command(
                video,
                video_id,
                from,
                to,
                force,
                dry_run,
                cli.config,
                cli.out_dir,
            )
            .await?;
        }
        Commands::Inspect { out_dir } => {
            inspect_command(out_dir).await?;
        }
        Commands::Validate { jsonl } => {
            validate_command(jsonl).await?;
        }
        Commands::Estimate { video } => {
            estimate_command(video, cli.config).await?;
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

#[allow(clippy::too_many_arguments)]
async fn process_command(
    video: String,
    video_id: Option<String>,
    from: Option<String>,
    to: Option<String>,
    force: bool,
    _dry_run: bool,
    config_path: Option<PathBuf>,
    out_dir_override: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let video_input = classify_video_input(&video)?;
    let mut config = Config::load(config_path.as_deref())?;

    // Apply --out-dir override
    if let Some(ref od) = out_dir_override {
        config.general.out_dir = od.to_string_lossy().into_owned();
    }

    let (from_stage, to_stage) = normalize_stage_range(from.as_deref(), to.as_deref())?;
    run_preflight_checks(&config, &video_input, from_stage, to_stage)?;

    let vid_id = video_id
        .unwrap_or_else(|| default_video_id(&video_input).unwrap_or_else(|| "video".to_string()));

    let out_dir = resolve_process_out_dir(&config, &vid_id);
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
        "Processing: {} (stages {}-{})",
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
    println!("Source:   {}", manifest.source_path);
    println!("Src hash: {}", manifest.source_hash);
    println!("Cfg hash: {}", manifest.config_hash);
    println!("\nStages:");

    for (stage_name, record) in &manifest.stages {
        println!(
            "  {:20} {:?}  hash={}",
            stage_name, record.status, record.input_hash
        );
    }

    Ok(())
}

/// Validate a JSONL file: every non-empty line must deserialize as a valid `Chunk`.
async fn validate_command(jsonl: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    check_file_exists(&jsonl)?;

    let content = fs::read_to_string(&jsonl)?;
    let mut valid = 0usize;
    let mut errors = 0usize;

    for (line_no, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Chunk>(line) {
            Ok(chunk) => {
                // Additional field-level checks
                let mut field_errors: Vec<String> = Vec::new();
                if chunk.schema_version != "1.0" {
                    field_errors.push(format!(
                        "schema_version must be '1.0', got '{}'",
                        chunk.schema_version
                    ));
                }
                if chunk.start_sec >= chunk.end_sec {
                    field_errors.push(format!(
                        "start_sec ({}) must be < end_sec ({})",
                        chunk.start_sec, chunk.end_sec
                    ));
                }
                if (chunk.duration_sec - (chunk.end_sec - chunk.start_sec)).abs() > 0.01 {
                    field_errors.push("duration_sec inconsistent with start/end".to_string());
                }
                if chunk.embedding_text.is_empty() {
                    field_errors.push("embedding_text must not be empty".to_string());
                }
                if field_errors.is_empty() {
                    valid += 1;
                } else {
                    for msg in &field_errors {
                        eprintln!("Line {}: {}", line_no + 1, msg);
                    }
                    errors += 1;
                }
            }
            Err(e) => {
                eprintln!("Line {}: schema error: {}", line_no + 1, e);
                errors += 1;
            }
        }
    }

    println!("✓ Valid chunks: {}", valid);
    if errors > 0 {
        println!("✗ Invalid chunks: {}", errors);
        return Err("Validation failed".into());
    }
    Ok(())
}

async fn estimate_command(
    video: String,
    config_path: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let video_input = classify_video_input(&video)?;
    let config = Config::load(config_path.as_deref())?;
    let estimate = estimate_input(&video_input, &config)?;
    let size_mb = estimate.size_mb;

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
    eprintln!("Resolving Wix segments from: {}", input);
    eprintln!("Writing output under: {}", output.display());

    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        fs::create_dir_all(&output).map_err(|e| e.to_string())?;
        let resolver = WixDirectSegmentResolver;
        let resolved = resolver
            .resolve(&input)
            .map_err(|e| format!("Failed to resolve: {}", e))?;
        let plan = build_download_plan(&resolved, &output, video_id.as_deref());
        let downloader = WixDownloader::new(http_client().map_err(|e| e.to_string())?);
        let options = WixDownloadOptions {
            max_segments,
            stop_after_misses,
        };
        downloader
            .download(&resolved, &plan, &options)
            .map_err(|e| format!("Download failed: {}", e))
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
    .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    println!("Source:        {}", result.source_ref);
    println!("Segments:      {}", result.segment_count);
    println!("Segments dir:  {}", result.segments_dir.display());
    println!("MP4:           {}", result.output_mp4.display());

    Ok(())
}

// ── input classification ──────────────────────────────────────────────────────

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
            .and_then(|s| s.to_str())
            .map(ToOwned::to_owned),
        VideoInput::WebUrl(url) => {
            let last = url
                .path_segments()
                .and_then(|mut s| s.rfind(|seg| !seg.is_empty()))
                .map(sanitize_video_id);
            last.or_else(|| url.host_str().map(sanitize_video_id))
        }
    }
}

fn sanitize_video_id(input: &str) -> String {
    let s = input
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
    if s.is_empty() { "video".to_string() } else { s }
}

fn resolve_process_out_dir(config: &Config, video_id: &str) -> PathBuf {
    PathBuf::from(&config.general.out_dir).join(video_id)
}

// ── stage range parsing ───────────────────────────────────────────────────────

fn normalize_stage_range(
    from: Option<&str>,
    to: Option<&str>,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let from_stage = match from {
        None => 0,
        Some(s) => stage_name_to_index(s).ok_or_else(|| {
            format!(
                "Unknown stage '{}'. Use 0-9 or a stage name like stage6_semantic.",
                s
            )
        })?,
    };
    let to_stage = match to {
        None => 9,
        Some(s) => stage_name_to_index(s).ok_or_else(|| {
            format!(
                "Unknown stage '{}'. Use 0-9 or a stage name like stage4_coarse.",
                s
            )
        })?,
    };
    if from_stage > to_stage {
        return Err(format!("Invalid range: --from {} > --to {}.", from_stage, to_stage).into());
    }
    Ok((from_stage, to_stage))
}

// ── preflight checks ──────────────────────────────────────────────────────────

fn stage_range_includes(from: usize, to: usize, stage: usize) -> bool {
    (from..=to).contains(&stage)
}

fn tool_requirements(
    config: &Config,
    input: &VideoInput,
    from_stage: usize,
    to_stage: usize,
) -> Result<Vec<ToolRequirement>, String> {
    let mut reqs = Vec::new();

    if from_stage == 0 {
        reqs.push(ToolRequirement {
            name: "ffprobe",
            command: config.media.ffprobe_path.clone(),
        });
    }

    // ffmpeg needed for audio(1), aux(3), frames(5)
    if stage_range_includes(from_stage, to_stage, 1)
        || stage_range_includes(from_stage, to_stage, 3)
        || stage_range_includes(from_stage, to_stage, 5)
    {
        reqs.push(ToolRequirement {
            name: "ffmpeg",
            command: config.media.ffmpeg_path.clone(),
        });
    }

    if stage_range_includes(from_stage, to_stage, 2) && config.asr.adapter == "whisper_cpp" {
        reqs.push(ToolRequirement {
            name: "whisper-cli",
            command: config.asr.whisper_cpp.binary_path.clone(),
        });
    }

    if to_stage >= 5 && config.vision.ocr.adapter == "tesseract" {
        reqs.push(ToolRequirement {
            name: "tesseract",
            command: config.vision.ocr.adapter.clone(),
        });
    }

    if matches!(input, VideoInput::WebUrl(_)) {
        let dl = configured_url_downloader(config).map_err(|e| e.to_string())?;
        reqs.push(ToolRequirement {
            name: "yt-dlp",
            command: dl.to_string(),
        });
    }

    Ok(reqs)
}

fn run_preflight_checks(
    config: &Config,
    input: &VideoInput,
    from_stage: usize,
    to_stage: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let reqs = tool_requirements(config, input, from_stage, to_stage).map_err(|e| e.to_string())?;
    let mut failures = Vec::new();
    for req in reqs {
        if let Err(e) = ensure_tool_available(&req.command) {
            failures.push(format!("{} ({})", req.name, e));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!("Pre-flight check failed: {}", failures.join(", ")).into())
    }
}

fn ensure_tool_available(tool: &str) -> Result<(), Box<dyn std::error::Error>> {
    let out = Command::new(tool).arg("-version").output()?;
    if out.status.success() {
        Ok(())
    } else {
        Err(format!("Required tool not available: {}", tool).into())
    }
}

// ── download helpers ──────────────────────────────────────────────────────────

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
                source_ref: path.to_string_lossy().into_owned(),
                local_video_path: path.clone(),
            })
        }
        VideoInput::WebUrl(url) => {
            let dl = configured_url_downloader(config)?;
            ensure_tool_available(dl)?;
            let input_dir = out_dir.join("input");
            fs::create_dir_all(&input_dir)?;
            let local = download_video(dl, url, &input_dir)
                .map_err(|e| format!("Failed to fetch from URL {}: {}", url, e))?;
            Ok(PreparedInput {
                source_type: "web_url".to_string(),
                source_ref: url.as_str().to_string(),
                local_video_path: local,
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
            Ok(EstimateInfo {
                label: path.display().to_string(),
                size_mb: bytes_to_mb(fs::metadata(path)?.len()),
            })
        }
        VideoInput::WebUrl(url) => {
            let dl = configured_url_downloader(config)?;
            ensure_tool_available(dl)?;
            Ok(EstimateInfo {
                label: url.as_str().to_string(),
                size_mb: estimate_url_size_mb(dl, url)?,
            })
        }
    }
}

fn configured_url_downloader(config: &Config) -> Result<&str, Box<dyn std::error::Error>> {
    match config.media.url_downloader.as_deref() {
        Some("yt-dlp") => Ok("yt-dlp"),
        Some(other) => Err(format!(
            "Unsupported media.url_downloader: {}. Only \"yt-dlp\" is supported.",
            other
        )
        .into()),
        None => Err(
            "URL input is disabled. Set media.url_downloader = \"yt-dlp\" in config to enable."
                .into(),
        ),
    }
}

fn bytes_to_mb(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn download_video(
    dl: &str,
    url: &Url,
    out_dir: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    match download_video_with_ytdlp(dl, url, out_dir) {
        Ok(path) => Ok(path),
        Err(ytdlp_err) => {
            if can_fallback_to_direct_http(url)? {
                download_video_direct_http(url, out_dir).map_err(|direct_err| {
                    format!(
                        "yt-dlp failed ({}) and direct-http fallback failed ({}).",
                        ytdlp_err, direct_err
                    )
                    .into()
                })
            } else {
                Err(format!(
                    "{}. Only yt-dlp-resolvable URLs and direct video links are supported.",
                    ytdlp_err
                )
                .into())
            }
        }
    }
}

fn download_video_with_ytdlp(
    dl: &str,
    url: &Url,
    out_dir: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let tpl = out_dir.join("source.%(ext)s");
    let out = Command::new(dl)
        .args(["--no-playlist", "--merge-output-format", "mp4", "-o"])
        .arg(&tpl)
        .arg(url.as_str())
        .output()?;
    if !out.status.success() {
        return Err(format!("yt-dlp failed: {}", String::from_utf8_lossy(&out.stderr)).into());
    }
    find_downloaded_video(out_dir)
}

fn download_video_direct_http(
    url: &Url,
    out_dir: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let client = http_client()?;
    let resp = client.get(url.as_str()).send()?.error_for_status()?;
    let ext = infer_direct_video_extension(
        url,
        resp.headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok()),
    );
    let path = out_dir.join(format!("source.{}", ext));
    let mut file = fs::File::create(&path)?;
    file.write_all(&resp.bytes()?)?;
    Ok(path)
}

fn find_downloaded_video(out_dir: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut entries: Vec<_> = fs::read_dir(out_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("source."))
        })
        .collect();
    entries.sort();
    entries
        .into_iter()
        .next()
        .ok_or_else(|| "yt-dlp completed but no output found".into())
}

fn fetch_ytdlp_metadata(
    dl: &str,
    url: &Url,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let out = Command::new(dl)
        .args(["--dump-single-json", "--no-download", url.as_str()])
        .output()?;
    if !out.status.success() {
        return Err(format!(
            "yt-dlp metadata failed: {}",
            String::from_utf8_lossy(&out.stderr)
        )
        .into());
    }
    serde_json::from_slice(&out.stdout).map_err(Into::into)
}

fn estimate_url_size_mb(dl: &str, url: &Url) -> Result<f64, Box<dyn std::error::Error>> {
    match fetch_ytdlp_metadata(dl, url) {
        Ok(meta) => extract_estimated_size_mb(&meta)
            .ok_or_else(|| "Could not estimate size from yt-dlp metadata".into()),
        Err(ytdlp_err) => {
            if can_fallback_to_direct_http(url)? {
                estimate_direct_http_size_mb(url).map_err(|direct_err| {
                    format!(
                        "yt-dlp failed ({}) and direct-http fallback failed ({}).",
                        ytdlp_err, direct_err
                    )
                    .into()
                })
            } else {
                Err(format!("{}.", ytdlp_err).into())
            }
        }
    }
}

fn estimate_direct_http_size_mb(url: &Url) -> Result<f64, Box<dyn std::error::Error>> {
    let resp = http_client()?
        .head(url.as_str())
        .send()?
        .error_for_status()?;
    let len = resp
        .headers()
        .get(reqwest::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .ok_or_else(|| "No Content-Length".to_string())?;
    Ok(bytes_to_mb(len))
}

fn can_fallback_to_direct_http(url: &Url) -> Result<bool, Box<dyn std::error::Error>> {
    if has_direct_video_extension(url) {
        return Ok(true);
    }
    let resp = http_client()?.head(url.as_str()).send()?;
    if !resp.status().is_success() {
        return Ok(false);
    }
    Ok(resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(is_video_content_type))
}

fn has_direct_video_extension(url: &Url) -> bool {
    url.path_segments()
        .and_then(|mut s| s.next_back())
        .and_then(|seg| {
            seg.rsplit_once('.')
                .map(|(_, ext)| ext.to_ascii_lowercase())
        })
        .is_some_and(|ext| DIRECT_VIDEO_EXTENSIONS.iter().any(|&c| c == ext))
}

fn is_video_content_type(ct: &str) -> bool {
    ct.split(';')
        .next()
        .is_some_and(|v| v.trim().starts_with("video/"))
}

fn infer_direct_video_extension(url: &Url, content_type: Option<&str>) -> String {
    if let Some(ext) = url
        .path_segments()
        .and_then(|mut s| s.next_back())
        .and_then(|seg| {
            seg.rsplit_once('.')
                .map(|(_, ext)| ext.to_ascii_lowercase())
        })
        .filter(|ext| DIRECT_VIDEO_EXTENSIONS.iter().any(|&c| c == ext))
    {
        return ext;
    }
    if let Some(mapped) = content_type.and_then(content_type_to_extension) {
        return mapped.to_string();
    }
    "mp4".to_string()
}

fn content_type_to_extension(ct: &str) -> Option<&'static str> {
    match ct.split(';').next()?.trim() {
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

fn extract_estimated_size_mb(meta: &serde_json::Value) -> Option<f64> {
    meta.get("filesize")
        .and_then(serde_json::Value::as_u64)
        .or_else(|| {
            meta.get("filesize_approx")
                .and_then(serde_json::Value::as_u64)
        })
        .map(bytes_to_mb)
}

fn check_video_exists(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if !path.exists() {
        Err(format!("Video file not found: {}", path.display()).into())
    } else {
        Ok(())
    }
}

fn check_file_exists(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if !path.exists() {
        Err(format!("File not found: {}", path.display()).into())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parse_process() {
        let cli = Cli::try_parse_from(["vididx", "process", "video.mp4"]).unwrap();
        match cli.command {
            Commands::Process { video, .. } => assert_eq!(video, "video.mp4"),
            _ => panic!("Expected Process"),
        }
    }

    #[test]
    fn test_cli_parse_process_with_stage_names() {
        let cli = Cli::try_parse_from([
            "vididx",
            "process",
            "video.mp4",
            "--from",
            "stage4_coarse",
            "--to",
            "stage8_annotate",
        ])
        .unwrap();
        match cli.command {
            Commands::Process { from, to, .. } => {
                assert_eq!(from.as_deref(), Some("stage4_coarse"));
                assert_eq!(to.as_deref(), Some("stage8_annotate"));
            }
            _ => panic!("Expected Process"),
        }
    }

    #[test]
    fn test_cli_global_options() {
        let cli = Cli::try_parse_from([
            "vididx",
            "--log-level",
            "debug",
            "--out-dir",
            "/tmp/test",
            "estimate",
            "video.mp4",
        ])
        .unwrap();
        assert_eq!(cli.log_level, "debug");
        assert_eq!(cli.out_dir, Some(PathBuf::from("/tmp/test")));
    }

    #[test]
    fn test_resolve_process_out_dir_uses_config_root_and_video_id() {
        let mut config = Config::default();
        config.general.out_dir = "/tmp/vididx-out".to_string();
        assert_eq!(
            resolve_process_out_dir(&config, "demo"),
            PathBuf::from("/tmp/vididx-out/demo")
        );
    }

    #[test]
    fn test_normalize_stage_range_by_name() {
        let (f, t) = normalize_stage_range(Some("stage4_coarse"), Some("stage8_annotate")).unwrap();
        assert_eq!(f, 4);
        assert_eq!(t, 8);
    }

    #[test]
    fn test_normalize_stage_range_by_number() {
        let (f, t) = normalize_stage_range(Some("3"), Some("7")).unwrap();
        assert_eq!(f, 3);
        assert_eq!(t, 7);
    }

    #[test]
    fn test_normalize_stage_range_unknown_name() {
        let result = normalize_stage_range(Some("stage99_unknown"), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_normalize_stage_range_inverted() {
        let result = normalize_stage_range(Some("7"), Some("3"));
        assert!(result.is_err());
    }

    #[test]
    fn test_normalize_stage_range_defaults() {
        let (f, t) = normalize_stage_range(None, None).unwrap();
        assert_eq!((f, t), (0, 9));
    }

    #[test]
    fn test_tool_requirements_local_input() {
        let config = Config::default();
        let input = VideoInput::LocalPath(PathBuf::from("video.mp4"));
        let reqs = tool_requirements(&config, &input, 0, 9).unwrap();
        let cmds: Vec<_> = reqs.iter().map(|r| r.command.as_str()).collect();
        assert!(cmds.contains(&"ffmpeg"));
        assert!(cmds.contains(&"ffprobe"));
        assert!(cmds.contains(&"whisper-cli"));
        assert!(cmds.contains(&"tesseract"));
        assert!(!cmds.contains(&"yt-dlp"));
    }

    #[test]
    fn test_tool_requirements_late_stage_skips_early_tools() {
        let config = Config::default();
        let input = VideoInput::LocalPath(PathBuf::from("video.mp4"));
        let reqs = tool_requirements(&config, &input, 8, 9).unwrap();
        let cmds: Vec<_> = reqs.iter().map(|r| r.command.as_str()).collect();
        assert!(!cmds.contains(&"ffmpeg"));
        assert!(!cmds.contains(&"ffprobe"));
        assert!(!cmds.contains(&"whisper-cli"));
        assert!(cmds.contains(&"tesseract"));
    }

    #[test]
    fn test_tool_requirements_url_needs_ytdlp() {
        let mut config = Config::default();
        config.media.url_downloader = Some("yt-dlp".to_string());
        let input = VideoInput::WebUrl(Url::parse("https://example.com/video.mp4").unwrap());
        let reqs = tool_requirements(&config, &input, 0, 9).unwrap();
        let cmds: Vec<_> = reqs.iter().map(|r| r.command.as_str()).collect();
        assert!(cmds.contains(&"yt-dlp"));
    }

    #[test]
    fn test_tool_requirements_url_fails_when_no_downloader() {
        let config = Config::default();
        let input = VideoInput::WebUrl(Url::parse("https://example.com/video.mp4").unwrap());
        assert!(tool_requirements(&config, &input, 0, 9).is_err());
    }

    #[test]
    fn test_classify_video_input_http_url() {
        let input = classify_video_input("https://example.com/watch?v=123").unwrap();
        assert!(matches!(input, VideoInput::WebUrl(_)));
    }

    #[test]
    fn test_classify_video_input_local_path() {
        let input = classify_video_input("video.mp4").unwrap();
        assert!(matches!(input, VideoInput::LocalPath(_)));
    }

    #[test]
    fn test_default_video_id_from_url() {
        let input = classify_video_input("https://example.com/videos/demo-file").unwrap();
        assert_eq!(default_video_id(&input).as_deref(), Some("demo-file"));
    }

    #[test]
    fn test_extract_estimated_size_mb() {
        let meta = serde_json::json!({"filesize": 52_428_800_u64});
        let mb = extract_estimated_size_mb(&meta).unwrap();
        assert!((mb - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_has_direct_video_extension() {
        let url = Url::parse("https://cdn.example.com/video.mp4").unwrap();
        assert!(has_direct_video_extension(&url));
    }

    #[test]
    fn test_configured_url_downloader_disabled_by_default() {
        let config = Config::default();
        assert!(configured_url_downloader(&config).is_err());
    }

    #[test]
    fn test_configured_url_downloader_ytdlp() {
        let mut config = Config::default();
        config.media.url_downloader = Some("yt-dlp".to_string());
        assert_eq!(configured_url_downloader(&config).unwrap(), "yt-dlp");
    }

    #[test]
    fn test_cli_parse_inspect() {
        let cli = Cli::try_parse_from(["vididx", "inspect", "output"]).unwrap();
        match cli.command {
            Commands::Inspect { out_dir } => assert_eq!(out_dir, PathBuf::from("output")),
            _ => panic!("Expected Inspect"),
        }
    }

    #[test]
    fn test_cli_parse_validate() {
        let cli = Cli::try_parse_from(["vididx", "validate", "chunks.jsonl"]).unwrap();
        match cli.command {
            Commands::Validate { jsonl } => assert_eq!(jsonl, PathBuf::from("chunks.jsonl")),
            _ => panic!("Expected Validate"),
        }
    }

    #[test]
    fn test_cli_parse_estimate() {
        let cli = Cli::try_parse_from(["vididx", "estimate", "video.mp4"]).unwrap();
        match cli.command {
            Commands::Estimate { video } => assert_eq!(video, "video.mp4"),
            _ => panic!("Expected Estimate"),
        }
    }
}
