use clap::{Parser, Subcommand};
use std::path::PathBuf;
use vidx_core::{VidxError, Config};
use vidx_pipeline::{JobContext, Manifest, run_pipeline};
use std::sync::Arc;
use tokio::sync::Mutex;
use std::fs;

#[derive(Parser)]
#[command(
    name = "vidx",
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
        /// Path to the video file
        video: PathBuf,

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
        /// Path to the video file
        video: PathBuf,
    },
}

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
    }

    Ok(())
}

async fn process_command(
    video: PathBuf,
    video_id: Option<String>,
    _from: Option<usize>,
    _to: Option<usize>,
    force: bool,
    output: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Pre-flight checks
    check_video_exists(&video)?;

    // Determine video ID
    let vid_id = video_id.unwrap_or_else(|| {
        video
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("video")
            .to_string()
    });

    // Determine output directory
    let out_dir = output.unwrap_or_else(|| PathBuf::from(format!("output/{}", vid_id)));
    fs::create_dir_all(&out_dir)?;

    // Load or create manifest
    let manifest_path = out_dir.join("manifest.json");
    let source_hash = vidx_core::hash::sha256_file(&video)?;
    let config = Config::load(None)?;
    let config_hash = format!("sha256:{}", serde_json::to_string(&config).unwrap_or_default());

    let manifest = if manifest_path.exists() && !force {
        Manifest::load(&manifest_path)
            .map_err(|e| VidxError::Config(format!("Failed to load manifest: {:?}", e)))?
    } else {
        Manifest::new(&vid_id, video.to_string_lossy().as_ref(), &source_hash, &config_hash)
    };

    // Create job context
    let ctx = JobContext {
        video_id: vid_id.clone(),
        source_path: video.clone(),
        out_dir,
        config,
        manifest: Arc::new(Mutex::new(manifest)),
    };

    // Run pipeline
    eprintln!("Processing video: {}", video.display());
    run_pipeline(ctx).await?;

    eprintln!("✓ Processing complete");
    Ok(())
}

async fn inspect_command(out_dir: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = out_dir.join("manifest.json");
    check_file_exists(&manifest_path)?;

    let manifest = Manifest::load(&manifest_path)
        .map_err(|e| VidxError::Config(format!("Failed to load manifest: {:?}", e)))?;

    println!("Video ID: {}", manifest.video_id);
    println!("Source: {}", manifest.source_path);
    println!("Source Hash: {}", manifest.source_hash);
    println!("Config Hash: {}", manifest.config_hash);
    println!("\nStages:");

    for (stage_name, record) in &manifest.stages {
        println!(
            "  {} - {:?} ({})",
            stage_name,
            record.status,
            record.input_hash
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

async fn estimate_command(video: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    check_video_exists(&video)?;

    let metadata = fs::metadata(&video)?;
    let size_bytes = metadata.len();
    let size_mb = size_bytes as f64 / (1024.0 * 1024.0);

    // Rough estimates
    let est_chunks = (size_mb / 10.0).max(5.0) as usize;
    let est_output_mb = (size_mb * 0.5).ceil() as usize;
    let est_time_min = (size_mb / 50.0).ceil() as usize;

    println!("Video: {}", video.display());
    println!("File size: {:.1} MB", size_mb);
    println!("\nEstimates:");
    println!("  Chunks: ~{}", est_chunks);
    println!("  Output size: ~{} MB", est_output_mb);
    println!("  Processing time: ~{} min", est_time_min);

    Ok(())
}

fn check_video_exists(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    if !path.exists() {
        return Err(format!("Video file not found: {}", path.display()).into());
    }
    Ok(())
}

fn check_file_exists(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
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
        let args = vec!["vidx", "process", "video.mp4"];
        let cli = Cli::try_parse_from(args).unwrap();

        match cli.command {
            Commands::Process { video, .. } => {
                assert_eq!(video, PathBuf::from("video.mp4"));
            }
            _ => panic!("Expected Process command"),
        }
    }

    #[test]
    fn test_cli_parse_process_with_options() {
        let args = vec![
            "vidx",
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
                assert_eq!(video, PathBuf::from("video.mp4"));
                assert_eq!(video_id, Some("test_vid".to_string()));
                assert!(force);
            }
            _ => panic!("Expected Process command"),
        }
    }

    #[test]
    fn test_cli_parse_inspect() {
        let args = vec!["vidx", "inspect", "output"];
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
        let args = vec!["vidx", "validate", "chunks.jsonl"];
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
        let args = vec!["vidx", "estimate", "video.mp4"];
        let cli = Cli::try_parse_from(args).unwrap();

        match cli.command {
            Commands::Estimate { video } => {
                assert_eq!(video, PathBuf::from("video.mp4"));
            }
            _ => panic!("Expected Estimate command"),
        }
    }
}
