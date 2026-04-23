use chrono::Utc;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::stage::JobContext;
use vididx_asr::{AsrAdapter, WhisperCppAdapter};
use vididx_core::{
    AnnotatedChunk, Chunk, ContentType, FrameKind, ImageRef, ModalityFlags, NormalizedChunk,
    ProcessingMeta, SceneChange, SegmentMode, SemanticChunk, SourceJumpRef, TranscriptTimeline,
    VididxError,
};
use vididx_llm::AnthropicClient;
use vididx_media::{detect_scene_changes, detect_silence, extract_audio, extract_frames, probe};
use vididx_output::{OutputIndex, write_index, write_markdown};
use vididx_segment::{annotate_chunk, coarse_segment, normalize, utterance_to_chunks};
use vididx_vision::{
    ClaudeVlmAdapter, OcrAdapter, TesseractAdapter, VlmCaptionAdapter, dhash_from_path,
    hamming_distance,
};

const STAGES: &[&str] = &[
    "stage0_probe",
    "stage1_audio",
    "stage2_asr",
    "stage3_aux",
    "stage4_coarse",
    "stage5_frames",
    "stage6_semantic",
    "stage7_normalize",
    "stage8_annotate",
    "stage9_output",
];

#[derive(Debug, Serialize, Deserialize)]
struct FrameArtifact {
    path: String,
    at_sec: f64,
    kind: FrameKind,
    analyzed: bool,
    ocr_text: Option<String>,
    visual_caption: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AudioStageArtifact {
    extracted: bool,
    wav_path: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct AuxData {
    silence: Vec<vididx_core::SilenceInterval>,
    scene: Vec<SceneChange>,
}

#[derive(Debug)]
struct FrameCandidate {
    path: String,
    at_sec: f64,
    kind: FrameKind,
    dhash: Option<u64>,
}

/// Run a pipeline for processing a video.
pub async fn run_pipeline(ctx: JobContext) -> Result<(), VididxError> {
    std::fs::create_dir_all(&ctx.out_dir)?;

    let api_key_present = std::env::var("ANTHROPIC_API_KEY")
        .map(|k| !k.trim().is_empty())
        .unwrap_or(false);
    if api_key_present {
        eprintln!(
            "API  ANTHROPIC_API_KEY detected — stage9 annotation and VLM caption will call \
             Anthropic API (pay-per-use)"
        );
    } else {
        eprintln!("API  no ANTHROPIC_API_KEY — all processing is local (no API charges)");
    }

    let input_hash = {
        let manifest = ctx.manifest.lock().await;
        format!("{}:{}", manifest.source_hash, manifest.config_hash)
    };

    let media_probe = run_or_load_stage(&ctx, 0, &input_hash, async {
        let result = probe(&ctx.config.media.ffprobe_path, &ctx.source_path).await?;
        Ok((ctx.out_dir.join("probe.json"), result))
    })
    .await?;
    if should_stop_after(&ctx, 0) {
        return Ok(());
    }

    let audio_stage = run_or_load_stage(&ctx, 1, &input_hash, async {
        let audio_path = ctx.out_dir.join("audio").join("audio_16k.wav");
        if media_probe.has_audio {
            extract_audio(&ctx.config.media.ffmpeg_path, &ctx.source_path, &audio_path).await?;
        }

        Ok((
            ctx.out_dir.join("audio.json"),
            AudioStageArtifact {
                extracted: media_probe.has_audio,
                wav_path: audio_path.to_string_lossy().to_string(),
            },
        ))
    })
    .await?;
    if should_stop_after(&ctx, 1) {
        return Ok(());
    }
    let audio_path = PathBuf::from(&audio_stage.wav_path);

    let transcript = run_or_load_stage(&ctx, 2, &input_hash, async {
        let timeline = transcribe_or_empty(&ctx, &audio_path, media_probe.has_audio).await;
        Ok((ctx.out_dir.join("transcript.json"), timeline))
    })
    .await?;
    if should_stop_after(&ctx, 2) {
        return Ok(());
    }

    let aux = run_or_load_stage(&ctx, 3, &input_hash, async {
        let silence_intervals = if media_probe.has_audio {
            detect_silence(&ctx.config.media.ffmpeg_path, &ctx.source_path, -35.0, 0.5).await?
        } else {
            Vec::new()
        };
        let scene_changes = if media_probe.has_video {
            detect_scene_changes(
                &ctx.config.media.ffmpeg_path,
                &ctx.source_path,
                ctx.config.frames.scene_change_threshold as f32,
            )
            .await?
        } else {
            Vec::new()
        };
        Ok((
            ctx.out_dir.join("aux.json"),
            AuxData {
                silence: silence_intervals,
                scene: scene_changes,
            },
        ))
    })
    .await?;
    let silence = aux.silence;
    let scenes = aux.scene;
    if should_stop_after(&ctx, 3) {
        return Ok(());
    }

    let mode = ctx.config.segment.mode;

    let frame_artifacts = run_or_load_stage(&ctx, 5, &input_hash, async {
        let frames = if media_probe.has_video {
            let safe_end = if media_probe.duration_sec > 0.5 {
                media_probe.duration_sec - 0.5
            } else {
                media_probe.duration_sec
            };
            let stamps = if mode == SegmentMode::Utterance {
                transcript
                    .segments
                    .iter()
                    .map(|s| s.start_sec.clamp(0.0, safe_end))
                    .collect::<Vec<_>>()
            } else {
                let coarse_count = (media_probe.duration_sec
                    / ctx.config.segment.coarse.max_duration_sec)
                    .ceil()
                    .max(1.0) as usize;
                select_frame_timestamps(
                    media_probe.duration_sec,
                    &scenes,
                    ctx.config.frames.periodic_interval_sec,
                    ctx.config.frames.max_analyzed_per_chunk * coarse_count,
                )
            };
            let out_dir = ctx.out_dir.join("images").join(&ctx.video_id);
            let paths = extract_frames(
                &ctx.config.media.ffmpeg_path,
                &ctx.source_path,
                &stamps,
                &out_dir,
                85,
            )
            .await?;
            let candidates = paths
                .into_iter()
                .zip(stamps.into_iter())
                .map(|(path, at_sec)| FrameCandidate {
                    path: path.to_string_lossy().to_string(),
                    at_sec,
                    kind: if mode == SegmentMode::Utterance {
                        FrameKind::UtteranceStart
                    } else {
                        nearest_frame_kind(at_sec, &scenes)
                    },
                    dhash: dhash_from_path(&path).ok(),
                })
                .collect::<Vec<_>>();

            if mode == SegmentMode::Utterance {
                candidates
                    .into_iter()
                    .map(|c| FrameArtifact {
                        path: c.path,
                        at_sec: c.at_sec,
                        kind: c.kind,
                        analyzed: true,
                        ocr_text: None,
                        visual_caption: None,
                    })
                    .collect()
            } else {
                dedupe_frame_candidates(candidates, ctx.config.frames.dhash_distance_threshold)
            }
        } else {
            Vec::new()
        };
        Ok((ctx.out_dir.join("frames.json"), frames))
    })
    .await?;
    if should_stop_after(&ctx, 5) {
        return Ok(());
    }

    let mut coarse = run_or_load_stage(&ctx, 4, &input_hash, async {
        let coarse = coarse_segment(
            media_probe.duration_sec,
            &transcript,
            &silence,
            &scenes,
            ctx.config.segment.coarse.max_duration_sec,
            ctx.config.segment.coarse.snap_window_sec,
        )?;
        Ok((ctx.out_dir.join("coarse.json"), coarse))
    })
    .await?;
    for seg in &mut coarse {
        seg.segment_id = format!("{}_seg_{:04}", ctx.video_id, seg.index);
    }
    if should_stop_after(&ctx, 4) {
        return Ok(());
    }

    let semantic = run_or_load_stage(&ctx, 6, &input_hash, async {
        let semantic = match mode {
            SegmentMode::Utterance => transcript
                .segments
                .iter()
                .map(|s| SemanticChunk {
                    start_sec: s.start_sec,
                    end_sec: s.end_sec,
                    transcript_text: s.text.clone(),
                    rationale: "utterance-boundary".to_string(),
                    parent_segment_id: String::new(),
                })
                .collect::<Vec<_>>(),
            SegmentMode::Chapter => coarse
                .iter()
                .map(|seg| SemanticChunk {
                    start_sec: seg.start_sec,
                    end_sec: seg.end_sec,
                    transcript_text: seg.transcript_text.clone(),
                    rationale: "chapter (coarse only)".to_string(),
                    parent_segment_id: seg.segment_id.clone(),
                })
                .collect::<Vec<_>>(),
            SegmentMode::Semantic => coarse
                .iter()
                .flat_map(|segment| {
                    heuristic_semantic_chunks(
                        segment.start_sec,
                        segment.end_sec,
                        &transcript,
                        ctx.config.segment.semantic.target_min_sec,
                        ctx.config.segment.semantic.target_max_sec,
                        &segment.segment_id,
                    )
                })
                .collect::<Vec<_>>(),
        };
        Ok((ctx.out_dir.join("semantic.json"), semantic))
    })
    .await?;
    if should_stop_after(&ctx, 6) {
        return Ok(());
    }

    let normalized = run_or_load_stage(&ctx, 7, &input_hash, async {
        let normalized = match mode {
            SegmentMode::Utterance => {
                utterance_to_chunks(&transcript, &ctx.config.segment.utterance, &ctx.video_id)
            }
            SegmentMode::Chapter => normalize(
                semantic,
                ctx.config.segment.semantic.hard_min_sec,
                ctx.config.segment.semantic.hard_max_sec,
                &ctx.video_id,
            ),
            SegmentMode::Semantic => normalize(
                semantic,
                ctx.config.segment.semantic.hard_min_sec,
                ctx.config.segment.semantic.hard_max_sec,
                &ctx.video_id,
            ),
        };
        Ok((ctx.out_dir.join("normalized.json"), normalized))
    })
    .await?;
    if should_stop_after(&ctx, 7) {
        return Ok(());
    }

    let frame_artifacts = enrich_frame_artifacts(&ctx, frame_artifacts, &normalized).await;

    let annotated = run_or_load_stage(&ctx, 8, &input_hash, async {
        let llm_client = anthropic_client_from_env(&ctx);
        if llm_client.is_some() {
            eprintln!(
                "  [API] annotate: {} Anthropic API calls (model: {})",
                normalized.len(),
                ctx.config.llm.model
            );
        } else {
            eprintln!(
                "  [local] annotate: {} chunks (fallback, no API key)",
                normalized.len()
            );
        }
        let mut annotated = Vec::with_capacity(normalized.len());
        for chunk in &normalized {
            let item = match llm_client.as_ref() {
                Some(client) => annotate_chunk(client, chunk)
                    .await
                    .unwrap_or_else(|_| fallback_annotate(chunk)),
                None => fallback_annotate(chunk),
            };
            annotated.push(item);
        }
        Ok((ctx.out_dir.join("annotated.json"), annotated))
    })
    .await?;
    if should_stop_after(&ctx, 8) {
        return Ok(());
    }

    let (source_path, source_hash) = {
        let manifest = ctx.manifest.lock().await;
        (manifest.source_path.clone(), manifest.source_hash.clone())
    };

    let content_type = media_probe.guess_content_type();
    let mut chunks = build_chunks(
        &ctx,
        &source_path,
        &source_hash,
        &annotated,
        &frame_artifacts,
        media_probe.has_audio,
        content_type,
    );

    // Relocate frames to chunk-specific directories per SPEC.
    let frame_artifacts = relocate_frames_to_chunk_dirs(&ctx, &mut chunks, frame_artifacts)?;
    write_json_pretty(&ctx.out_dir.join("frames.json"), &frame_artifacts)?;

    write_chunks_jsonl(
        &chunks,
        &ctx.out_dir.join(format!("{}.chunks.jsonl", ctx.video_id)),
    )?;

    write_markdown(
        &chunks,
        &ctx.video_id,
        &ctx.out_dir.join(format!("{}.md", ctx.video_id)),
    )
    .await?;

    let index = OutputIndex {
        video_id: ctx.video_id.clone(),
        source_path: source_path.clone(),
        source_hash: source_hash.clone(),
        config_hash: {
            let manifest = ctx.manifest.lock().await;
            manifest.config_hash.clone()
        },
        total_chunks: chunks.len(),
        total_duration_sec: media_probe.duration_sec,
        chunks_file: format!("{}.chunks.jsonl", ctx.video_id),
        markdown_file: format!("{}.md", ctx.video_id),
        generated_at: Utc::now(),
    };
    write_index(
        &index,
        &ctx.out_dir.join(format!("{}.index.json", ctx.video_id)),
    )
    .await?;

    {
        let mut manifest = ctx.manifest.lock().await;
        let output_path = ctx.out_dir.join(format!("{}.chunks.jsonl", ctx.video_id));
        manifest.mark_done("stage9_output", &output_path.to_string_lossy());
        let manifest_path = ctx.out_dir.join("manifest.json");
        manifest
            .save(&manifest_path)
            .map_err(stage_error_to_vididx)?;
    }

    Ok(())
}

async fn run_stage<T, F>(
    ctx: &JobContext,
    stage_name: &'static str,
    input_hash: &str,
    f: F,
) -> Result<T, VididxError>
where
    T: Serialize,
    F: std::future::Future<Output = Result<(PathBuf, T), VididxError>>,
{
    {
        let mut manifest = ctx.manifest.lock().await;
        manifest.mark_running(stage_name, input_hash);
        manifest
            .save(&ctx.out_dir.join("manifest.json"))
            .map_err(stage_error_to_vididx)?;
    }

    match f.await {
        Ok((path, value)) => {
            write_json_pretty(&path, &value)?;
            let mut manifest = ctx.manifest.lock().await;
            manifest.mark_done(stage_name, &path.to_string_lossy());
            manifest
                .save(&ctx.out_dir.join("manifest.json"))
                .map_err(stage_error_to_vididx)?;
            eprintln!("✓ {}", stage_name);
            Ok(value)
        }
        Err(err) => {
            let mut manifest = ctx.manifest.lock().await;
            manifest.mark_failed(stage_name);
            manifest
                .save(&ctx.out_dir.join("manifest.json"))
                .map_err(stage_error_to_vididx)?;
            eprintln!("✗ {} - {}", stage_name, err);
            Err(err)
        }
    }
}

async fn run_or_load_stage<T, F>(
    ctx: &JobContext,
    stage_index: usize,
    input_hash: &str,
    f: F,
) -> Result<T, VididxError>
where
    T: Serialize + DeserializeOwned,
    F: std::future::Future<Output = Result<(PathBuf, T), VididxError>>,
{
    let stage_name = STAGES[stage_index];
    if let Some(path) = cached_stage_output_path(ctx, stage_name, input_hash).await {
        eprintln!("↺ {} (cached)", stage_name);
        return read_json(&path);
    }

    if should_load_from_previous_run(ctx, stage_index) {
        let path = stage_artifact_path(ctx, stage_index);
        eprintln!("↺ {} (loaded)", STAGES[stage_index]);
        return read_json(&path);
    }

    run_stage(ctx, stage_name, input_hash, f).await
}

async fn transcribe_or_empty(
    ctx: &JobContext,
    audio_path: &Path,
    has_audio: bool,
) -> TranscriptTimeline {
    if !has_audio || !audio_path.exists() {
        return TranscriptTimeline { segments: vec![] };
    }

    let adapter =
        WhisperCppAdapter::from_config(&ctx.config.asr.whisper_cpp, &ctx.config.asr.language);

    match adapter.transcribe(audio_path).await {
        Ok(timeline) => timeline,
        Err(err) => {
            eprintln!("! ASR fallback to empty transcript: {}", err);
            TranscriptTimeline { segments: vec![] }
        }
    }
}

fn should_load_from_previous_run(ctx: &JobContext, stage_index: usize) -> bool {
    stage_index < ctx.from_stage
}

async fn cached_stage_output_path(
    ctx: &JobContext,
    stage_name: &str,
    input_hash: &str,
) -> Option<PathBuf> {
    let manifest = ctx.manifest.lock().await;
    manifest
        .cached_output_path(stage_name, input_hash)
        .map(PathBuf::from)
}

fn should_stop_after(ctx: &JobContext, stage_index: usize) -> bool {
    stage_index == ctx.to_stage && stage_index < STAGES.len() - 1
}

fn stage_artifact_path(ctx: &JobContext, stage_index: usize) -> PathBuf {
    match stage_index {
        0 => ctx.out_dir.join("probe.json"),
        1 => ctx.out_dir.join("audio.json"),
        2 => ctx.out_dir.join("transcript.json"),
        3 => ctx.out_dir.join("aux.json"),
        4 => ctx.out_dir.join("coarse.json"),
        5 => ctx.out_dir.join("frames.json"),
        6 => ctx.out_dir.join("semantic.json"),
        7 => ctx.out_dir.join("normalized.json"),
        8 => ctx.out_dir.join("annotated.json"),
        _ => unreachable!("invalid stage index"),
    }
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, VididxError> {
    let bytes = std::fs::read(path)?;
    serde_json::from_slice(&bytes).map_err(Into::into)
}

fn select_frame_timestamps(
    duration_sec: f64,
    scenes: &[SceneChange],
    periodic_interval_sec: f64,
    max_frames: usize,
) -> Vec<f64> {
    if duration_sec <= 0.0 || max_frames == 0 {
        return Vec::new();
    }

    let safe_end = if duration_sec > 0.5 {
        duration_sec - 0.5
    } else {
        duration_sec
    };
    let mut stamps = Vec::new();

    if periodic_interval_sec > 0.0 {
        let mut t = periodic_interval_sec;
        while t < safe_end {
            stamps.push(t);
            t += periodic_interval_sec;
        }
    }

    for scene in scenes {
        if scene.at_sec > 0.0 && scene.at_sec < safe_end {
            stamps.push(scene.at_sec);
        }
    }

    for stamp in &mut stamps {
        *stamp = stamp.clamp(0.0, safe_end);
    }

    stamps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    stamps.dedup_by(|a, b| (*a - *b).abs() < 0.25);

    if stamps.len() > max_frames {
        let last = stamps.len() - 1;
        let mut reduced = Vec::with_capacity(max_frames);
        for idx in 0..max_frames {
            let pos = idx as f64 * last as f64 / (max_frames.saturating_sub(1).max(1)) as f64;
            reduced.push(stamps[pos.round() as usize]);
        }
        reduced.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        reduced.dedup_by(|a, b| (*a - *b).abs() < 0.25);
        return reduced;
    }

    stamps
}

fn nearest_frame_kind(at_sec: f64, scenes: &[SceneChange]) -> FrameKind {
    if scenes
        .iter()
        .any(|scene| (scene.at_sec - at_sec).abs() < 0.25)
    {
        FrameKind::SceneChange
    } else {
        FrameKind::Periodic
    }
}

fn heuristic_semantic_chunks(
    start_sec: f64,
    end_sec: f64,
    transcript: &TranscriptTimeline,
    target_min_sec: f64,
    target_max_sec: f64,
    parent_segment_id: &str,
) -> Vec<SemanticChunk> {
    let duration = (end_sec - start_sec).max(0.0);
    if duration <= 0.0 {
        return Vec::new();
    }

    let target = ((target_min_sec + target_max_sec) / 2.0).max(1.0);
    let count = (duration / target).ceil().max(1.0) as usize;
    let width = duration / count as f64;

    (0..count)
        .map(|idx| {
            let chunk_start = start_sec + width * idx as f64;
            let chunk_end = if idx + 1 == count {
                end_sec
            } else {
                start_sec + width * (idx + 1) as f64
            };

            SemanticChunk {
                start_sec: chunk_start,
                end_sec: chunk_end,
                transcript_text: transcript_text_for_range(transcript, chunk_start, chunk_end),
                parent_segment_id: parent_segment_id.to_string(),
                rationale: if transcript.segments.is_empty() {
                    "time-based split without transcript".to_string()
                } else {
                    "time-based split within coarse segment".to_string()
                },
            }
        })
        .collect()
}

fn transcript_text_for_range(
    transcript: &TranscriptTimeline,
    start_sec: f64,
    end_sec: f64,
) -> String {
    transcript
        .segments
        .iter()
        .filter(|segment| segment.end_sec > start_sec && segment.start_sec < end_sec)
        .map(|segment| segment.text.trim())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn fallback_annotate(chunk: &NormalizedChunk) -> AnnotatedChunk {
    let transcript = chunk.transcript_text.trim();
    let title = if transcript.is_empty() {
        format!(
            "Segment {} ({})",
            chunk.chunk_id,
            format_time_range(chunk.start_sec, chunk.end_sec)
        )
    } else {
        let words = transcript
            .split_whitespace()
            .take(8)
            .collect::<Vec<_>>()
            .join(" ");
        let mut chars: Vec<char> = words.chars().collect();
        if chars.len() > 40 {
            chars.truncate(37);
            chars.extend(['.', '.', '.']);
        }
        chars.into_iter().collect::<String>()
    };

    let summary = if transcript.is_empty() {
        "No transcript available. Chunk generated from timing and visual boundaries.".to_string()
    } else {
        summarize_text(transcript, 180)
    };

    let keywords = extract_keywords(transcript, &title);

    AnnotatedChunk {
        chunk_id: chunk.chunk_id.clone(),
        parent_segment_id: chunk.parent_segment_id.clone(),
        start_sec: chunk.start_sec,
        end_sec: chunk.end_sec,
        transcript_text: chunk.transcript_text.clone(),
        title,
        summary,
        keywords,
    }
}

fn summarize_text(text: &str, limit: usize) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let char_count = compact.chars().count();
    if char_count <= limit {
        compact
    } else {
        let prefix = compact
            .chars()
            .take(limit.saturating_sub(3))
            .collect::<String>();
        format!("{prefix}...")
    }
}

fn extract_keywords(transcript: &str, title: &str) -> Vec<String> {
    let mut keywords = title
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .chain(transcript.split(|c: char| !c.is_alphanumeric() && c != '_'))
        .filter_map(|word| {
            let normalized = word.trim().to_lowercase();
            if normalized.len() >= 4 {
                Some(normalized)
            } else {
                None
            }
        })
        .filter(|word| {
            !matches!(
                word.as_str(),
                "this" | "that" | "with" | "from" | "chunk" | "segment"
            )
        })
        .collect::<Vec<_>>();

    keywords.sort();
    keywords.dedup();
    keywords.truncate(8);

    if keywords.is_empty() {
        vec!["video".to_string()]
    } else {
        keywords
    }
}

fn build_chunks(
    ctx: &JobContext,
    source_path: &str,
    source_hash: &str,
    annotated: &[AnnotatedChunk],
    frame_artifacts: &[FrameArtifact],
    has_audio: bool,
    content_type: ContentType,
) -> Vec<Chunk> {
    let last_chunk_id = annotated.last().map(|c| c.chunk_id.as_str());
    annotated
        .iter()
        .map(|chunk| {
            let is_last = last_chunk_id == Some(&chunk.chunk_id);
            let image_refs = frame_artifacts
                .iter()
                .filter(|frame| {
                    frame.at_sec >= chunk.start_sec
                        && if is_last {
                            frame.at_sec <= chunk.end_sec
                        } else {
                            frame.at_sec < chunk.end_sec
                        }
                })
                .map(|frame| ImageRef {
                    path: frame.path.clone(),
                    at_sec: frame.at_sec,
                    kind: frame.kind.clone(),
                    analyzed: frame.analyzed,
                })
                .collect::<Vec<_>>();

            let (ocr_text, visual_caption) =
                collect_chunk_visual_data(chunk.start_sec, chunk.end_sec, frame_artifacts);
            let has_ocr = ocr_text.is_some();
            let has_visual = !image_refs.is_empty();
            let embedding_text = build_embedding_text(
                &chunk.title,
                &chunk.summary,
                &chunk.transcript_text,
                ocr_text.as_deref(),
                visual_caption.as_deref(),
                &ctx.config.output.embedding_text_fields,
            );

            Chunk {
                schema_version: "1.0".to_string(),
                video_id: ctx.video_id.clone(),
                source_type: ctx.source_type.clone(),
                source_path: source_path.to_string(),
                source_hash: source_hash.to_string(),
                chunk_id: chunk.chunk_id.clone(),
                parent_segment_id: chunk.parent_segment_id.clone(),
                start_sec: chunk.start_sec,
                end_sec: chunk.end_sec,
                start_tc: format_timecode(chunk.start_sec),
                end_tc: format_timecode(chunk.end_sec),
                duration_sec: (chunk.end_sec - chunk.start_sec).max(0.0),
                content_type: content_type.clone(),
                speaker_info: vec![],
                title: chunk.title.clone(),
                summary: chunk.summary.clone(),
                transcript: chunk.transcript_text.clone(),
                ocr_text,
                visual_caption,
                keywords: chunk.keywords.clone(),
                embedding_text,
                image_refs,
                source_jump_ref: SourceJumpRef {
                    ref_type: "local".to_string(),
                    start_sec: chunk.start_sec,
                },
                modality_flags: ModalityFlags {
                    has_speech: has_audio && !chunk.transcript_text.trim().is_empty(),
                    has_ocr,
                    has_visual,
                },
                processing_meta: ProcessingMeta {
                    segmentation_rationale: Some(match ctx.config.segment.mode {
                        SegmentMode::Utterance => "utterance-boundary segmentation".to_string(),
                        SegmentMode::Semantic => {
                            "coarse+time-based semantic segmentation".to_string()
                        }
                        SegmentMode::Chapter => "chapter (coarse-only) segmentation".to_string(),
                    }),
                    asr_adapter: ctx.config.asr.adapter.clone(),
                    vlm_adapter: ctx.config.vision.caption.adapter.clone(),
                    generated_at: Utc::now(),
                },
            }
        })
        .collect()
}

fn build_embedding_text(
    title: &str,
    summary: &str,
    transcript: &str,
    ocr_text: Option<&str>,
    visual_caption: Option<&str>,
    fields: &[String],
) -> String {
    // Use filtered OCR for embedding to keep noise out of retrieval vectors.
    let ocr_for_embedding = ocr_text.map(filter_ocr_for_embedding);

    let mut parts: Vec<&str> = Vec::new();
    for field in fields {
        match field.as_str() {
            "title" => parts.push(title.trim()),
            "summary" => parts.push(summary.trim()),
            "transcript" => parts.push(transcript.trim()),
            "ocr_important" | "ocr" => {
                if let Some(ocr) = ocr_for_embedding.as_deref().filter(|s| !s.is_empty()) {
                    parts.push(ocr.trim());
                }
            }
            "visual_caption" => {
                if let Some(caption) = visual_caption {
                    parts.push(caption.trim());
                }
            }
            _ => {}
        }
    }

    parts
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Extract and deduplicate OCR/caption data for a chunk.
/// Returns (raw_ocr_text, visual_caption) where raw_ocr preserves the original
/// filtered at line-level for easier downstream inspection.
fn collect_chunk_visual_data(
    start_sec: f64,
    end_sec: f64,
    frame_artifacts: &[FrameArtifact],
) -> (Option<String>, Option<String>) {
    let mut ocr_lines: Vec<String> = Vec::new();
    let mut seen_ocr_lines: HashSet<String> = HashSet::new();
    let mut caption_parts = Vec::new();
    let mut seen_caption = HashSet::new();

    for frame in frame_artifacts
        .iter()
        .filter(|frame| frame.at_sec >= start_sec && frame.at_sec <= end_sec)
    {
        if let Some(text) = frame.ocr_text.as_deref() {
            for line in text.lines() {
                let normalized = normalize_ocr_line(line);
                if !normalized.is_empty()
                    && !is_ocr_noise_line(&normalized)
                    && seen_ocr_lines.insert(normalized.clone())
                {
                    ocr_lines.push(line.trim().to_string());
                }
            }
        }

        if let Some(text) = frame.visual_caption.as_deref() {
            let normalized = text.trim();
            if !normalized.is_empty() && seen_caption.insert(normalized.to_string()) {
                caption_parts.push(normalized.to_string());
            }
        }
    }

    let ocr_text = (!ocr_lines.is_empty()).then(|| ocr_lines.join("\n"));
    let visual_caption = (!caption_parts.is_empty()).then(|| caption_parts.join(" "));
    (ocr_text, visual_caption)
}

/// Normalize an OCR line for deduplication and noise detection.
fn normalize_ocr_line(line: &str) -> String {
    line.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Heuristic noise detection for a single OCR line.
fn is_ocr_noise_line(line: &str) -> bool {
    // Too short
    if line.chars().count() < 3 {
        return true;
    }

    let alpha_numeric = line.chars().filter(|c| c.is_alphanumeric()).count();
    let total = line.chars().count().max(1);

    // Mostly symbols
    if alpha_numeric * 10 < total * 3 {
        return true;
    }

    // Too many short tokens (menu bars, toolbar dumps)
    let words: Vec<&str> = line.split_whitespace().collect();
    if !words.is_empty() {
        let short_words = words.iter().filter(|w| w.chars().count() <= 2).count();
        if short_words * 2 > words.len() {
            return true;
        }
    }

    // Common OS / browser UI noise (English)
    let lower = line.to_lowercase();
    let ui_noise = [
        "recycle bin",
        "trash",
        "desktop",
        "downloads",
        "documents",
        "quick access",
        "frequent folders",
        "recent files",
        "one drive",
        "this pc",
        "3d objects",
        "network",
        "chrome",
        "safari",
        "firefox",
        "file explorer",
        "bookmarks",
        "home",
        "share",
        "view",
        "new tab",
        "new window",
    ];
    if ui_noise.contains(&lower.as_str()) {
        return true;
    }

    // Stand-alone single words that are almost always UI chrome
    if lower.split_whitespace().count() == 1 {
        let single_word_noise = [
            "file",
            "edit",
            "view",
            "help",
            "tools",
            "window",
            "new",
            "open",
            "save",
            "cut",
            "copy",
            "paste",
            "undo",
            "redo",
            "print",
            "find",
            "replace",
            "close",
            "exit",
            "back",
            "forward",
            "refresh",
            "stop",
            "search",
            "home",
            "zoom",
            "fullscreen",
            "settings",
            "preferences",
            "options",
            "cancel",
            "ok",
            "done",
            "apply",
            "next",
            "previous",
            "first",
            "last",
            "add",
            "remove",
            "delete",
            "rename",
            "sort",
        ];
        if single_word_noise.contains(&lower.as_str()) {
            return true;
        }
    }

    false
}

/// Further filter OCR text for inclusion in embedding_text.
/// Keeps only the most informative lines and limits total length.
fn filter_ocr_for_embedding(raw_ocr: &str) -> String {
    let mut seen: HashSet<String> = HashSet::new();
    let mut kept: Vec<String> = Vec::new();

    for line in raw_ocr.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Skip short lines in embedding context
        if trimmed.chars().count() < 5 {
            continue;
        }

        // Re-apply noise filter for embedding (catches long toolbar dumps)
        if is_ocr_noise_line(trimmed) {
            continue;
        }

        let norm = normalize_ocr_line(trimmed);
        if seen.insert(norm) {
            kept.push(trimmed.to_string());
        }
    }

    // Limit total lines to avoid swamping the embedding vector
    const MAX_OCR_LINES_FOR_EMBEDDING: usize = 8;
    if kept.len() > MAX_OCR_LINES_FOR_EMBEDDING {
        kept.truncate(MAX_OCR_LINES_FOR_EMBEDDING);
    }

    kept.join("\n")
}

fn relocate_frames_to_chunk_dirs(
    ctx: &JobContext,
    chunks: &mut [Chunk],
    mut frame_artifacts: Vec<FrameArtifact>,
) -> Result<Vec<FrameArtifact>, VididxError> {
    let images_dir = ctx.out_dir.join("images");
    if !images_dir.exists() {
        return Ok(frame_artifacts);
    }

    // Build a map: frame filename -> target chunk directory.
    // With half-open intervals in build_chunks, each frame belongs to exactly one chunk.
    let mut frame_targets: std::collections::HashMap<String, PathBuf> =
        std::collections::HashMap::new();
    for chunk in chunks.iter() {
        let chunk_img_dir = images_dir.join(&chunk.chunk_id);
        std::fs::create_dir_all(&chunk_img_dir).map_err(VididxError::Io)?;
        for image_ref in &chunk.image_refs {
            let filename = Path::new(&image_ref.path)
                .file_name()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or_else(|| "frame.jpg".to_string());
            frame_targets
                .entry(filename)
                .or_insert(chunk_img_dir.clone());
        }
    }

    // Move each unique frame to its target directory and compute new relative paths.
    let mut moved_paths: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for chunk in chunks.iter_mut() {
        for image_ref in chunk.image_refs.iter_mut() {
            let old_path = PathBuf::from(&image_ref.path);
            let filename = old_path
                .file_name()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or_else(|| "frame.jpg".to_string());

            if let Some(target_dir) = frame_targets.get(&filename) {
                let new_abs = target_dir.join(&filename);

                // Move file only once.
                if old_path.exists() && old_path.parent() != Some(target_dir.as_path()) {
                    std::fs::rename(&old_path, &new_abs).map_err(VididxError::Io)?;
                }

                // Always store as a relative path from out_dir.
                let rel = new_abs
                    .strip_prefix(&ctx.out_dir)
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_else(|_| new_abs.to_string_lossy().into_owned());
                moved_paths.insert(filename, rel.clone());
                image_ref.path = rel;
            }
        }
    }

    // Update frame_artifacts with the new paths as well.
    for frame in &mut frame_artifacts {
        let filename = Path::new(&frame.path)
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_else(|| "frame.jpg".to_string());
        if let Some(new_path) = moved_paths.get(&filename) {
            frame.path = new_path.clone();
        } else {
            // Fallback: relativize even if the file was not in any chunk target.
            let p = PathBuf::from(&frame.path);
            frame.path = p
                .strip_prefix(&ctx.out_dir)
                .map(|r| r.to_string_lossy().into_owned())
                .unwrap_or_else(|_| frame.path.clone());
        }
    }

    // Remove legacy flat directory and any leftover dedup-excluded frames.
    let legacy_dir = images_dir.join(&ctx.video_id);
    if legacy_dir.exists() {
        let _ = std::fs::remove_dir_all(&legacy_dir);
    }

    Ok(frame_artifacts)
}

fn dedupe_frame_candidates(
    candidates: Vec<FrameCandidate>,
    distance_threshold: usize,
) -> Vec<FrameArtifact> {
    let mut selected_hashes = Vec::new();
    let mut selected = Vec::new();

    for candidate in candidates {
        let is_duplicate = candidate.dhash.is_some_and(|hash| {
            selected_hashes
                .iter()
                .any(|seen| hamming_distance(hash, *seen) <= distance_threshold)
        });

        if is_duplicate {
            continue;
        }

        if let Some(hash) = candidate.dhash {
            selected_hashes.push(hash);
        }

        selected.push(FrameArtifact {
            path: candidate.path,
            at_sec: candidate.at_sec,
            kind: candidate.kind,
            analyzed: true,
            ocr_text: None,
            visual_caption: None,
        });
    }

    selected
}

async fn enrich_frame_artifacts(
    ctx: &JobContext,
    mut frames: Vec<FrameArtifact>,
    chunks: &[NormalizedChunk],
) -> Vec<FrameArtifact> {
    if frames.is_empty() {
        return frames;
    }

    let ocr_adapter = TesseractAdapter::new(ocr_binary_path(ctx));
    let ocr_languages = ctx
        .config
        .vision
        .ocr
        .languages
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();

    for frame in &mut frames {
        frame.ocr_text = ocr_adapter
            .extract_text(Path::new(&frame.path), &ocr_languages)
            .await
            .ok()
            .and_then(|text| {
                let trimmed = text.trim();
                if trimmed.is_empty() || !is_meaningful_ocr_text(trimmed) {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            });
    }

    if let Some(vlm_adapter) = claude_vlm_from_env(ctx) {
        let selected =
            select_caption_frame_indices(&frames, chunks, ctx.config.vision.caption.max_per_chunk);
        eprintln!(
            "  [API] VLM caption: {} Anthropic API calls (model: {})",
            selected.len(),
            ctx.config.vision.caption.model
        );
        for index in selected {
            if let Some(frame) = frames.get_mut(index) {
                frame.visual_caption = vlm_adapter
                    .caption(Path::new(&frame.path))
                    .await
                    .ok()
                    .and_then(|text| {
                        let trimmed = text.trim();
                        (!trimmed.is_empty()).then(|| trimmed.to_string())
                    });
            }
        }
    }

    frames
}

fn ocr_binary_path(ctx: &JobContext) -> &str {
    &ctx.config.vision.ocr.adapter
}

fn select_caption_frame_indices(
    frames: &[FrameArtifact],
    chunks: &[NormalizedChunk],
    max_per_chunk: usize,
) -> Vec<usize> {
    if max_per_chunk == 0 {
        return Vec::new();
    }

    let mut selected = Vec::new();

    for chunk in chunks {
        let mut indices = frames
            .iter()
            .enumerate()
            .filter(|(_, frame)| frame.at_sec >= chunk.start_sec && frame.at_sec <= chunk.end_sec)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();

        indices.sort_by(|left, right| {
            let left_frame = &frames[*left];
            let right_frame = &frames[*right];
            match (
                left_frame.kind == FrameKind::SceneChange,
                right_frame.kind == FrameKind::SceneChange,
            ) {
                (true, false) => Ordering::Less,
                (false, true) => Ordering::Greater,
                _ => left_frame
                    .at_sec
                    .partial_cmp(&right_frame.at_sec)
                    .unwrap_or(Ordering::Equal),
            }
        });

        selected.extend(indices.into_iter().take(max_per_chunk));
    }

    selected.sort_unstable();
    selected.dedup();
    selected
}

/// Filter out OCR noise: too short, mostly symbols, or common UI artifacts.
fn is_meaningful_ocr_text(text: &str) -> bool {
    // Reject very short strings (likely icons or single letters)
    if text.chars().count() < 3 {
        return false;
    }

    // Reject if less than 40% alphabetic characters
    let total = text.chars().count().max(1);
    let alpha = text.chars().filter(|c| c.is_alphabetic()).count();
    if alpha * 10 < total * 4 {
        return false;
    }

    true
}

fn anthropic_client_from_env(ctx: &JobContext) -> Option<AnthropicClient> {
    std::env::var("ANTHROPIC_API_KEY").ok().and_then(|api_key| {
        let trimmed = api_key.trim();
        (!trimmed.is_empty()).then(|| {
            AnthropicClient::new(
                trimmed,
                &ctx.config.llm.model,
                ctx.config.llm.max_concurrency,
            )
        })
    })
}

fn claude_vlm_from_env(ctx: &JobContext) -> Option<ClaudeVlmAdapter> {
    std::env::var("ANTHROPIC_API_KEY").ok().and_then(|api_key| {
        let trimmed = api_key.trim();
        (!trimmed.is_empty())
            .then(|| ClaudeVlmAdapter::new(trimmed, &ctx.config.vision.caption.model))
    })
}

fn format_time_range(start_sec: f64, end_sec: f64) -> String {
    format!(
        "{}-{}",
        format_timecode(start_sec),
        format_timecode(end_sec)
    )
}

fn format_timecode(seconds: f64) -> String {
    let total_ms = (seconds.max(0.0) * 1000.0).round() as u64;
    let hours = total_ms / 3_600_000;
    let minutes = (total_ms % 3_600_000) / 60_000;
    let secs = (total_ms % 60_000) / 1000;
    let millis = total_ms % 1000;

    format!("{hours:02}:{minutes:02}:{secs:02}.{millis:03}")
}

fn write_json_pretty<T: Serialize>(path: &Path, value: &T) -> Result<(), VididxError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(value)?;
    std::fs::write(path, json)?;
    Ok(())
}

fn write_chunks_jsonl(chunks: &[Chunk], path: &Path) -> Result<(), VididxError> {
    use std::io::Write;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    for chunk in chunks {
        serde_json::to_writer(&mut writer, chunk)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

fn stage_error_to_vididx(err: crate::stage::StageError) -> VididxError {
    VididxError::Config(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Manifest;
    use vididx_core::Config;
    use vididx_core::TranscriptSegment;

    #[test]
    fn test_select_frame_timestamps_bounds() {
        let scenes = vec![
            SceneChange {
                at_sec: 10.0,
                score: 0.0,
            },
            SceneChange {
                at_sec: 50.0,
                score: 0.0,
            },
        ];
        let stamps = select_frame_timestamps(60.0, &scenes, 20.0, 5);
        assert!(!stamps.is_empty());
        assert!(stamps.iter().all(|sec| *sec >= 0.0 && *sec <= 60.0));
        assert!(stamps.len() <= 5);
    }

    #[test]
    fn test_heuristic_semantic_chunks_cover_interval() {
        let timeline = TranscriptTimeline {
            segments: vec![TranscriptSegment {
                start_sec: 0.0,
                end_sec: 180.0,
                text: "text".to_string(),
                speaker: None,
                confidence: None,
            }],
        };
        let chunks = heuristic_semantic_chunks(0.0, 180.0, &timeline, 30.0, 90.0, "seg_0");
        assert!(!chunks.is_empty());
        assert_eq!(chunks.first().unwrap().start_sec, 0.0);
        assert_eq!(chunks.last().unwrap().end_sec, 180.0);
    }

    #[test]
    fn test_build_embedding_text_skips_empty_parts() {
        let text = build_embedding_text(
            "title",
            "",
            "body",
            None,
            None,
            &[
                "title".to_string(),
                "summary".to_string(),
                "transcript".to_string(),
            ],
        );
        assert_eq!(text, "title\n\nbody");
    }

    #[test]
    fn test_heuristic_semantic_chunks_split_transcript_by_timeline() {
        let timeline = TranscriptTimeline {
            segments: vec![
                TranscriptSegment {
                    start_sec: 0.0,
                    end_sec: 10.0,
                    text: "intro".to_string(),
                    speaker: None,
                    confidence: None,
                },
                TranscriptSegment {
                    start_sec: 10.0,
                    end_sec: 20.0,
                    text: "details".to_string(),
                    speaker: None,
                    confidence: None,
                },
            ],
        };

        let chunks = heuristic_semantic_chunks(0.0, 20.0, &timeline, 10.0, 10.0, "seg_0");

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].transcript_text, "intro");
        assert_eq!(chunks[1].transcript_text, "details");
    }

    #[test]
    fn test_build_chunks_includes_visual_analysis() {
        let mut config = Config::default();
        config
            .output
            .embedding_text_fields
            .push("visual_caption".to_string());
        let manifest = Manifest::new("test_video", "/tmp/test.mp4", "sha256:src", "sha256:cfg");
        let ctx = JobContext {
            video_id: "test_video".to_string(),
            source_type: "local_mp4".to_string(),
            source_ref: "/tmp/test.mp4".to_string(),
            source_path: PathBuf::from("/tmp/test.mp4"),
            out_dir: PathBuf::from("/tmp/out"),
            from_stage: 0,
            to_stage: 9,
            config,
            manifest: std::sync::Arc::new(tokio::sync::Mutex::new(manifest)),
        };
        let annotated = vec![AnnotatedChunk {
            chunk_id: "test_video_chunk_0000".to_string(),
            parent_segment_id: String::new(),
            start_sec: 0.0,
            end_sec: 30.0,
            transcript_text: "transcript".to_string(),
            title: "title".to_string(),
            summary: "summary".to_string(),
            keywords: vec!["keyword".to_string()],
        }];
        let frames = vec![
            FrameArtifact {
                path: "frame0.jpg".to_string(),
                at_sec: 5.0,
                kind: FrameKind::SceneChange,
                analyzed: true,
                ocr_text: Some("Save Project".to_string()),
                visual_caption: Some("Settings dialog".to_string()),
            },
            FrameArtifact {
                path: "frame1.jpg".to_string(),
                at_sec: 12.0,
                kind: FrameKind::Periodic,
                analyzed: false,
                ocr_text: None,
                visual_caption: None,
            },
        ];

        let chunks = build_chunks(
            &ctx,
            "/tmp/test.mp4",
            "sha256:src",
            &annotated,
            &frames,
            true,
            ContentType::ScreenRecording,
        );

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].ocr_text.as_deref(), Some("Save Project"));
        assert_eq!(chunks[0].visual_caption.as_deref(), Some("Settings dialog"));
        assert!(chunks[0].modality_flags.has_ocr);
        assert!(chunks[0].image_refs[0].analyzed);
        assert!(chunks[0].embedding_text.contains("Save Project"));
        assert!(chunks[0].embedding_text.contains("Settings dialog"));
    }

    #[test]
    fn test_ocr_binary_path_uses_configured_adapter() {
        let mut config = Config::default();
        config.vision.ocr.adapter = "custom-tesseract".to_string();
        let manifest = Manifest::new("test_video", "/tmp/test.mp4", "sha256:src", "sha256:cfg");
        let ctx = JobContext {
            video_id: "test_video".to_string(),
            source_type: "local_mp4".to_string(),
            source_ref: "/tmp/test.mp4".to_string(),
            source_path: PathBuf::from("/tmp/test.mp4"),
            out_dir: PathBuf::from("/tmp/out"),
            from_stage: 0,
            to_stage: 9,
            config,
            manifest: std::sync::Arc::new(tokio::sync::Mutex::new(manifest)),
        };

        assert_eq!(ocr_binary_path(&ctx), "custom-tesseract");
    }

    #[test]
    fn test_should_load_from_previous_run_respects_from_stage() {
        let ctx = JobContext {
            video_id: "test_video".to_string(),
            source_type: "local_mp4".to_string(),
            source_ref: "/tmp/test.mp4".to_string(),
            source_path: PathBuf::from("/tmp/test.mp4"),
            out_dir: PathBuf::from("/tmp/out"),
            from_stage: 4,
            to_stage: 9,
            config: Config::default(),
            manifest: std::sync::Arc::new(tokio::sync::Mutex::new(Manifest::new(
                "test_video",
                "/tmp/test.mp4",
                "sha256:src",
                "sha256:cfg",
            ))),
        };

        assert!(should_load_from_previous_run(&ctx, 0));
        assert!(should_load_from_previous_run(&ctx, 3));
        assert!(!should_load_from_previous_run(&ctx, 4));
    }

    #[test]
    fn test_stage_artifact_path_maps_stage_numbers() {
        let ctx = JobContext {
            video_id: "test_video".to_string(),
            source_type: "local_mp4".to_string(),
            source_ref: "/tmp/test.mp4".to_string(),
            source_path: PathBuf::from("/tmp/test.mp4"),
            out_dir: PathBuf::from("/tmp/out"),
            from_stage: 0,
            to_stage: 9,
            config: Config::default(),
            manifest: std::sync::Arc::new(tokio::sync::Mutex::new(Manifest::new(
                "test_video",
                "/tmp/test.mp4",
                "sha256:src",
                "sha256:cfg",
            ))),
        };

        assert_eq!(
            stage_artifact_path(&ctx, 0),
            PathBuf::from("/tmp/out/probe.json")
        );
        assert_eq!(
            stage_artifact_path(&ctx, 8),
            PathBuf::from("/tmp/out/annotated.json")
        );
    }

    #[tokio::test]
    async fn test_cached_stage_output_path_reads_matching_manifest_entry() {
        let mut manifest = Manifest::new("test_video", "/tmp/test.mp4", "sha256:src", "sha256:cfg");
        manifest.mark_running("stage3_aux", "sha256:input");
        manifest.mark_done("stage3_aux", "/tmp/out/aux.json");
        let ctx = JobContext {
            video_id: "test_video".to_string(),
            source_type: "local_mp4".to_string(),
            source_ref: "/tmp/test.mp4".to_string(),
            source_path: PathBuf::from("/tmp/test.mp4"),
            out_dir: PathBuf::from("/tmp/out"),
            from_stage: 0,
            to_stage: 9,
            config: Config::default(),
            manifest: std::sync::Arc::new(tokio::sync::Mutex::new(manifest)),
        };

        let path = cached_stage_output_path(&ctx, "stage3_aux", "sha256:input").await;

        assert_eq!(path, Some(PathBuf::from("/tmp/out/aux.json")));
    }
}
