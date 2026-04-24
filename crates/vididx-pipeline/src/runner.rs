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
use vididx_segment::{
    annotate_chunk, coarse_segment, normalize, semantic_chunk, uniform_split, utterance_to_chunks,
};
use vididx_vision::{
    ClaudeVlmAdapter, OcrAdapter, TesseractAdapter, VlmCaptionAdapter, dhash_from_path,
    hamming_distance,
};

/// Stage identifiers in pipeline order (matches SPEC §7.1).
pub const STAGES: &[&str] = &[
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

/// Convert a stage name or numeric string to an index into STAGES.
pub fn stage_name_to_index(s: &str) -> Option<usize> {
    // Try numeric first
    if let Ok(n) = s.parse::<usize>() {
        if n < STAGES.len() {
            return Some(n);
        }
        return None;
    }
    STAGES.iter().position(|&name| name == s)
}

#[derive(Debug, Serialize, Deserialize)]
struct AudioArtifact {
    wav_path: Option<String>,
    sample_rate: u32,
    channels: u32,
}

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

/// Run the full video processing pipeline.
pub async fn run_pipeline(ctx: JobContext) -> Result<(), VididxError> {
    std::fs::create_dir_all(&ctx.out_dir)?;

    let api_key_present = std::env::var("ANTHROPIC_API_KEY")
        .map(|k| !k.trim().is_empty())
        .unwrap_or(false);
    if api_key_present {
        eprintln!(
            "INFO ANTHROPIC_API_KEY detected — LLM semantic chunking, annotation and VLM caption enabled"
        );
    } else {
        eprintln!("INFO no ANTHROPIC_API_KEY — all processing is local (no API charges)");
    }

    let input_hash = {
        let manifest = ctx.manifest.lock().await;
        format!("{}:{}", manifest.source_hash, manifest.config_hash)
    };

    // Stage 0: media probe
    let media_probe = run_or_load_stage(&ctx, 0, &input_hash, async {
        let result = probe(&ctx.config.media.ffprobe_path, &ctx.source_path).await?;
        Ok((ctx.out_dir.join("probe.json"), result))
    })
    .await?;
    if should_stop_after(&ctx, 0) {
        return Ok(());
    }

    // Stage 1: audio extraction
    let audio_artifact = run_or_load_stage(&ctx, 1, &input_hash, async {
        let audio_dir = ctx.out_dir.join("audio");
        let wav_path = audio_dir.join("audio_16k.wav");
        let artifact = if media_probe.has_audio {
            extract_audio(&ctx.config.media.ffmpeg_path, &ctx.source_path, &wav_path).await?;
            AudioArtifact {
                wav_path: Some(wav_path.to_string_lossy().into_owned()),
                sample_rate: ctx.config.media.audio_sample_rate,
                channels: 1,
            }
        } else {
            AudioArtifact {
                wav_path: None,
                sample_rate: ctx.config.media.audio_sample_rate,
                channels: 0,
            }
        };
        Ok((ctx.out_dir.join("audio.json"), artifact))
    })
    .await?;
    if should_stop_after(&ctx, 1) {
        return Ok(());
    }

    // Stage 2: ASR
    let transcript = run_or_load_stage(&ctx, 2, &input_hash, async {
        let timeline = transcribe_or_empty(
            &ctx,
            audio_artifact.wav_path.as_deref(),
            media_probe.has_audio,
        )
        .await;
        Ok((ctx.out_dir.join("transcript.json"), timeline))
    })
    .await?;
    if should_stop_after(&ctx, 2) {
        return Ok(());
    }

    // Stage 3: auxiliary signals (silence + scene change)
    let aux = run_or_load_stage(&ctx, 3, &input_hash, async {
        let silence = if media_probe.has_audio {
            detect_silence(&ctx.config.media.ffmpeg_path, &ctx.source_path, -35.0, 0.5).await?
        } else {
            Vec::new()
        };
        let scenes = if media_probe.has_video {
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
                silence,
                scene: scenes,
            },
        ))
    })
    .await?;
    let silence = aux.silence;
    let scenes = aux.scene;
    if should_stop_after(&ctx, 3) {
        return Ok(());
    }

    // Stage 4: coarse segmentation
    let mut coarse = run_or_load_stage(&ctx, 4, &input_hash, async {
        let segs = coarse_segment(
            media_probe.duration_sec,
            &transcript,
            &silence,
            &scenes,
            ctx.config.segment.coarse.max_duration_sec,
            ctx.config.segment.coarse.snap_window_sec,
        )?;
        Ok((ctx.out_dir.join("coarse.json"), segs))
    })
    .await?;
    // Assign segment IDs (idempotent, based on index)
    for seg in &mut coarse {
        seg.segment_id = format!("{}_seg_{:04}", ctx.video_id, seg.index);
    }
    if should_stop_after(&ctx, 4) {
        return Ok(());
    }

    let mode = ctx.config.segment.mode;

    // Stage 5: frame extraction + OCR + VLM (before semantic chunking per SPEC)
    let frame_artifacts = run_or_load_stage(&ctx, 5, &input_hash, async {
        let frames = if media_probe.has_video {
            let safe_end = (media_probe.duration_sec - 0.5).max(0.0);

            let timestamps = if mode == SegmentMode::Utterance {
                // Utterance mode: one frame per utterance start
                transcript
                    .segments
                    .iter()
                    .map(|s| s.start_sec.clamp(0.0, safe_end))
                    .collect::<Vec<_>>()
            } else {
                // Periodic + scene-change extraction
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

            let img_dir = ctx.out_dir.join("images").join(&ctx.video_id);
            let paths = extract_frames(
                &ctx.config.media.ffmpeg_path,
                &ctx.source_path,
                &timestamps,
                &img_dir,
                85,
            )
            .await?;

            let candidates = paths
                .into_iter()
                .zip(timestamps.into_iter())
                .map(|(path, at_sec)| FrameCandidate {
                    path: path.to_string_lossy().into_owned(),
                    at_sec,
                    kind: if mode == SegmentMode::Utterance {
                        FrameKind::UtteranceStart
                    } else {
                        nearest_frame_kind(at_sec, &scenes)
                    },
                    dhash: dhash_from_path(&path).ok(),
                })
                .collect::<Vec<_>>();

            let selected = if mode == SegmentMode::Utterance {
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
            };

            // OCR all selected frames
            let mut enriched = enrich_with_ocr(&ctx, selected).await;

            // VLM caption: select subset per chunk budget
            // At this point chunks are not yet assigned; use coarse segments as budget proxy.
            let budget_chunks = coarse_to_budget_chunks(&coarse);
            if let Some(vlm) = claude_vlm_from_env(&ctx) {
                let indices = select_caption_frame_indices(
                    &enriched,
                    &budget_chunks,
                    ctx.config.vision.caption.max_per_chunk,
                );
                eprintln!(
                    "  [API] VLM caption: {} calls (model: {})",
                    indices.len(),
                    ctx.config.vision.caption.model
                );
                for idx in indices {
                    if let Some(frame) = enriched.get_mut(idx) {
                        frame.visual_caption = vlm
                            .caption(Path::new(&frame.path))
                            .await
                            .ok()
                            .and_then(|t| {
                                let s = t.trim().to_string();
                                (!s.is_empty()).then_some(s)
                            });
                    }
                }
            }

            enriched
        } else {
            Vec::new()
        };

        Ok((ctx.out_dir.join("frames.json"), frames))
    })
    .await?;
    if should_stop_after(&ctx, 5) {
        return Ok(());
    }

    // Stage 6: semantic chunking (LLM or fallback)
    let semantic = run_or_load_stage(&ctx, 6, &input_hash, async {
        let chunks = match mode {
            SegmentMode::Utterance => {
                // Utterance mode: ASR boundaries → SemanticChunk (normalize handles merge/split)
                transcript
                    .segments
                    .iter()
                    .map(|s| SemanticChunk {
                        start_sec: s.start_sec,
                        end_sec: s.end_sec,
                        transcript_text: s.text.trim().to_string(),
                        rationale: "utterance-boundary segmentation".to_string(),
                        parent_segment_id: String::new(),
                    })
                    .collect::<Vec<_>>()
            }
            SegmentMode::Chapter => {
                // Chapter mode: coarse segments become chunks directly
                coarse
                    .iter()
                    .map(|seg| SemanticChunk {
                        start_sec: seg.start_sec,
                        end_sec: seg.end_sec,
                        transcript_text: seg.transcript_text.clone(),
                        rationale: "chapter (coarse-only) segmentation".to_string(),
                        parent_segment_id: seg.segment_id.clone(),
                    })
                    .collect()
            }
            SegmentMode::CoarseSemantic => {
                let llm = anthropic_client_from_env(&ctx);
                let mut all_chunks = Vec::new();
                for seg in &coarse {
                    let chunks = match llm.as_ref() {
                        Some(client) => {
                            match semantic_chunk(
                                client,
                                seg,
                                &transcript,
                                ctx.config.segment.semantic.target_min_sec,
                                ctx.config.segment.semantic.target_max_sec,
                            )
                            .await
                            {
                                Ok(c) => c,
                                Err(err) => {
                                    eprintln!(
                                        "! semantic_chunk fallback for {}: {}",
                                        seg.segment_id, err
                                    );
                                    uniform_split(
                                        seg,
                                        &transcript,
                                        ctx.config.segment.semantic.target_min_sec,
                                        ctx.config.segment.semantic.target_max_sec,
                                    )
                                }
                            }
                        }
                        None => {
                            eprintln!(
                                "  [local] semantic: no API key, uniform-split for {}",
                                seg.segment_id
                            );
                            uniform_split(
                                seg,
                                &transcript,
                                ctx.config.segment.semantic.target_min_sec,
                                ctx.config.segment.semantic.target_max_sec,
                            )
                        }
                    };
                    all_chunks.extend(chunks);
                }
                all_chunks
            }
        };
        Ok((ctx.out_dir.join("semantic.json"), chunks))
    })
    .await?;
    if should_stop_after(&ctx, 6) {
        return Ok(());
    }

    // Stage 7: normalize (merge short / split long / assign chunk IDs)
    let normalized = run_or_load_stage(&ctx, 7, &input_hash, async {
        let norm = match mode {
            SegmentMode::Utterance => {
                utterance_to_chunks(&transcript, &ctx.config.segment.utterance, &ctx.video_id)
            }
            SegmentMode::Chapter | SegmentMode::CoarseSemantic => normalize(
                semantic,
                if mode == SegmentMode::Utterance {
                    ctx.config.segment.utterance.min_duration_sec
                } else {
                    ctx.config.segment.semantic.hard_min_sec
                },
                ctx.config.segment.semantic.hard_max_sec,
                ctx.config.segment.semantic.target_min_sec,
                ctx.config.segment.semantic.target_max_sec,
                &ctx.video_id,
            ),
        };
        Ok((ctx.out_dir.join("normalized.json"), norm))
    })
    .await?;
    if should_stop_after(&ctx, 7) {
        return Ok(());
    }

    // Stage 8: annotation (LLM title/summary/keywords or fallback)
    let annotated = run_or_load_stage(&ctx, 8, &input_hash, async {
        let llm = anthropic_client_from_env(&ctx);
        if llm.is_some() {
            eprintln!(
                "  [API] annotate: {} calls (model: {})",
                normalized.len(),
                ctx.config.llm.model
            );
        } else {
            eprintln!(
                "  [local] annotate: {} chunks (no-llm fallback)",
                normalized.len()
            );
        }
        let mut annotated = Vec::with_capacity(normalized.len());
        for chunk in &normalized {
            let item = match llm.as_ref() {
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

    // Stage 9: output (JSONL + Markdown + index.json)
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

    // Relocate frames to per-chunk directories per SPEC.
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
        manifest
            .save(&ctx.out_dir.join("manifest.json"))
            .map_err(stage_error_to_vididx)?;
    }

    Ok(())
}

// ── helpers ──────────────────────────────────────────────────────────────────

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
            eprintln!("✗ {} — {}", stage_name, err);
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

    // Cache hit: same input hash in manifest
    if let Some(path) = cached_stage_output_path(ctx, stage_name, input_hash).await {
        eprintln!("↺ {} (cached)", stage_name);
        return read_json(&path);
    }

    // --from N: load artifact from previous run without re-running
    if should_load_from_previous_run(ctx, stage_index) {
        let path = stage_artifact_path(ctx, stage_index);
        eprintln!("↺ {} (loaded from previous run)", stage_name);
        return read_json(&path);
    }

    run_stage(ctx, stage_name, input_hash, f).await
}

async fn transcribe_or_empty(
    ctx: &JobContext,
    wav_path: Option<&str>,
    has_audio: bool,
) -> TranscriptTimeline {
    let Some(path_str) = wav_path else {
        return TranscriptTimeline { segments: vec![] };
    };
    let path = Path::new(path_str);
    if !has_audio || !path.exists() {
        return TranscriptTimeline { segments: vec![] };
    }

    let adapter =
        WhisperCppAdapter::from_config(&ctx.config.asr.whisper_cpp, &ctx.config.asr.language);

    match adapter.transcribe(path).await {
        Ok(tl) => tl,
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
        _ => unreachable!("invalid stage index: {}", stage_index),
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

    let safe_end = (duration_sec - 0.5).max(0.0);
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

async fn enrich_with_ocr(ctx: &JobContext, mut frames: Vec<FrameArtifact>) -> Vec<FrameArtifact> {
    let adapter = TesseractAdapter::new(ocr_binary_path(ctx));
    let langs = ctx
        .config
        .vision
        .ocr
        .languages
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();

    for frame in &mut frames {
        frame.ocr_text = adapter
            .extract_text(Path::new(&frame.path), &langs)
            .await
            .ok()
            .and_then(|text| {
                let t = text.trim().to_string();
                (is_meaningful_ocr_text(&t)).then_some(t)
            });
    }

    frames
}

/// Proxy budget-chunk list for VLM caption selection before normalize assigns real chunk IDs.
/// Uses coarse segments as proxies.
fn coarse_to_budget_chunks(
    coarse: &[vididx_core::CoarseSegment],
) -> Vec<vididx_core::NormalizedChunk> {
    coarse
        .iter()
        .enumerate()
        .map(|(i, seg)| vididx_core::NormalizedChunk {
            chunk_id: format!("_budget_{}", i),
            parent_segment_id: seg.segment_id.clone(),
            start_sec: seg.start_sec,
            end_sec: seg.end_sec,
            transcript_text: seg.transcript_text.clone(),
            rationale: String::new(),
        })
        .collect()
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
        let chars: Vec<char> = transcript.chars().collect();
        if chars.len() <= 40 {
            chars.into_iter().collect()
        } else {
            let truncated: String = chars[..40].iter().collect();
            format!("{}...", truncated)
        }
    };

    let summary = if transcript.is_empty() {
        "No transcript available.".to_string()
    } else {
        let chars: Vec<char> = transcript.chars().collect();
        if chars.len() <= 150 {
            chars.into_iter().collect()
        } else {
            format!("{}...", chars[..150].iter().collect::<String>())
        }
    };

    AnnotatedChunk {
        chunk_id: chunk.chunk_id.clone(),
        parent_segment_id: chunk.parent_segment_id.clone(),
        start_sec: chunk.start_sec,
        end_sec: chunk.end_sec,
        transcript_text: chunk.transcript_text.clone(),
        title,
        summary,
        keywords: vec![],
        rationale: chunk.rationale.clone(),
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
    let last_id = annotated.last().map(|c| c.chunk_id.as_str());
    annotated
        .iter()
        .map(|chunk| {
            let is_last = last_id == Some(chunk.chunk_id.as_str());
            let image_refs = frame_artifacts
                .iter()
                .filter(|f| {
                    f.at_sec >= chunk.start_sec
                        && if is_last {
                            f.at_sec <= chunk.end_sec
                        } else {
                            f.at_sec < chunk.end_sec
                        }
                })
                .map(|f| ImageRef {
                    path: f.path.clone(),
                    at_sec: f.at_sec,
                    kind: f.kind.clone(),
                    analyzed: f.analyzed,
                })
                .collect::<Vec<_>>();

            let (ocr_text, visual_caption) =
                collect_chunk_visual_data(chunk.start_sec, chunk.end_sec, frame_artifacts);
            let has_ocr = ocr_text.is_some();
            // SPEC §3.2: has_visual = VLM caption の有無
            let has_visual = visual_caption.is_some();

            let embedding_text = build_embedding_text(
                &chunk.title,
                &chunk.summary,
                &chunk.transcript_text,
                ocr_text.as_deref(),
                visual_caption.as_deref(),
                &ctx.config.output.embedding_text_fields,
            );

            // Determine segmentation rationale per SPEC §3.2
            let seg_rationale = if chunk.rationale.contains("uniform-split") {
                "uniform-split (no-llm)".to_string()
            } else if chunk.rationale.is_empty()
                || chunk.rationale.contains("no-llm")
                || chunk.rationale.contains("fallback")
            {
                "no-llm fallback".to_string()
            } else {
                chunk.rationale.clone()
            };

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
                    segmentation_rationale: Some(seg_rationale),
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
                if let Some(cap) = visual_caption {
                    parts.push(cap.trim());
                }
            }
            _ => {}
        }
    }

    parts
        .into_iter()
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn collect_chunk_visual_data(
    start_sec: f64,
    end_sec: f64,
    frames: &[FrameArtifact],
) -> (Option<String>, Option<String>) {
    let mut ocr_lines: Vec<String> = Vec::new();
    let mut seen_ocr: HashSet<String> = HashSet::new();
    let mut captions: Vec<String> = Vec::new();
    let mut seen_cap: HashSet<String> = HashSet::new();

    for frame in frames
        .iter()
        .filter(|f| f.at_sec >= start_sec && f.at_sec <= end_sec)
    {
        if let Some(text) = frame.ocr_text.as_deref() {
            for line in text.lines() {
                let norm = normalize_ocr_line(line);
                if !norm.is_empty() && !is_ocr_noise_line(&norm) && seen_ocr.insert(norm) {
                    ocr_lines.push(line.trim().to_string());
                }
            }
        }
        if let Some(cap) = frame.visual_caption.as_deref() {
            let t = cap.trim().to_string();
            if !t.is_empty() && seen_cap.insert(t.clone()) {
                captions.push(t);
            }
        }
    }

    let ocr_text = (!ocr_lines.is_empty()).then(|| ocr_lines.join("\n"));
    let visual_caption = (!captions.is_empty()).then(|| captions.join(" "));
    (ocr_text, visual_caption)
}

fn normalize_ocr_line(line: &str) -> String {
    line.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn is_ocr_noise_line(line: &str) -> bool {
    if line.chars().count() < 3 {
        return true;
    }
    let alpha = line.chars().filter(|c| c.is_alphanumeric()).count();
    let total = line.chars().count().max(1);
    if alpha * 10 < total * 3 {
        return true;
    }
    let words: Vec<&str> = line.split_whitespace().collect();
    if !words.is_empty() {
        let short = words.iter().filter(|w| w.chars().count() <= 2).count();
        if short * 2 > words.len() {
            return true;
        }
    }
    let lower = line.to_lowercase();
    let ui_noise = [
        "recycle bin",
        "trash",
        "desktop",
        "downloads",
        "documents",
        "quick access",
        "recent files",
        "one drive",
        "this pc",
        "chrome",
        "safari",
        "firefox",
        "file explorer",
        "new tab",
        "new window",
    ];
    if ui_noise.contains(&lower.as_str()) {
        return true;
    }
    if lower.split_whitespace().count() == 1 {
        let single = [
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
        if single.contains(&lower.as_str()) {
            return true;
        }
    }
    false
}

fn filter_ocr_for_embedding(raw: &str) -> String {
    let mut seen: HashSet<String> = HashSet::new();
    let mut kept: Vec<String> = Vec::new();
    for line in raw.lines() {
        let t = line.trim();
        if t.is_empty() || t.chars().count() < 5 || is_ocr_noise_line(t) {
            continue;
        }
        let norm = normalize_ocr_line(t);
        if seen.insert(norm) {
            kept.push(t.to_string());
        }
    }
    const MAX: usize = 8;
    if kept.len() > MAX {
        kept.truncate(MAX);
    }
    kept.join("\n")
}

fn relocate_frames_to_chunk_dirs(
    ctx: &JobContext,
    chunks: &mut [Chunk],
    mut frames: Vec<FrameArtifact>,
) -> Result<Vec<FrameArtifact>, VididxError> {
    let images_dir = ctx.out_dir.join("images");
    if !images_dir.exists() {
        return Ok(frames);
    }

    let mut frame_targets: std::collections::HashMap<String, PathBuf> = Default::default();
    for chunk in chunks.iter() {
        let chunk_dir = images_dir.join(&chunk.chunk_id);
        std::fs::create_dir_all(&chunk_dir).map_err(VididxError::Io)?;
        for ir in &chunk.image_refs {
            let name = Path::new(&ir.path)
                .file_name()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or_else(|| "frame.jpg".into());
            frame_targets.entry(name).or_insert(chunk_dir.clone());
        }
    }

    let mut moved: std::collections::HashMap<String, String> = Default::default();
    for chunk in chunks.iter_mut() {
        for ir in chunk.image_refs.iter_mut() {
            let old = PathBuf::from(&ir.path);
            let name = old
                .file_name()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or_else(|| "frame.jpg".into());
            if let Some(target) = frame_targets.get(&name) {
                let new_abs = target.join(&name);
                if old.exists() && old.parent() != Some(target.as_path()) {
                    std::fs::rename(&old, &new_abs).map_err(VididxError::Io)?;
                }
                let rel = new_abs
                    .strip_prefix(&ctx.out_dir)
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_else(|_| new_abs.to_string_lossy().into_owned());
                moved.insert(name, rel.clone());
                ir.path = rel;
            }
        }
    }

    for frame in &mut frames {
        let name = Path::new(&frame.path)
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_else(|| "frame.jpg".into());
        if let Some(new) = moved.get(&name) {
            frame.path = new.clone();
        } else {
            let p = PathBuf::from(&frame.path);
            frame.path = p
                .strip_prefix(&ctx.out_dir)
                .map(|r| r.to_string_lossy().into_owned())
                .unwrap_or_else(|_| frame.path.clone());
        }
    }

    let legacy = images_dir.join(&ctx.video_id);
    if legacy.exists() {
        let _ = std::fs::remove_dir_all(&legacy);
    }

    Ok(frames)
}

fn dedupe_frame_candidates(
    candidates: Vec<FrameCandidate>,
    threshold: usize,
) -> Vec<FrameArtifact> {
    let mut seen_hashes: Vec<u64> = Vec::new();
    let mut selected = Vec::new();

    for c in candidates {
        let is_dup = c.dhash.is_some_and(|h| {
            seen_hashes
                .iter()
                .any(|&s| hamming_distance(h, s) <= threshold)
        });
        if is_dup {
            continue;
        }
        if let Some(h) = c.dhash {
            seen_hashes.push(h);
        }
        selected.push(FrameArtifact {
            path: c.path,
            at_sec: c.at_sec,
            kind: c.kind,
            analyzed: true,
            ocr_text: None,
            visual_caption: None,
        });
    }
    selected
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
        let mut indices: Vec<usize> = frames
            .iter()
            .enumerate()
            .filter(|(_, f)| f.at_sec >= chunk.start_sec && f.at_sec <= chunk.end_sec)
            .map(|(i, _)| i)
            .collect();
        indices.sort_by(|&a, &b| {
            let fa = &frames[a];
            let fb = &frames[b];
            match (
                fa.kind == FrameKind::SceneChange,
                fb.kind == FrameKind::SceneChange,
            ) {
                (true, false) => Ordering::Less,
                (false, true) => Ordering::Greater,
                _ => fa.at_sec.partial_cmp(&fb.at_sec).unwrap_or(Ordering::Equal),
            }
        });
        selected.extend(indices.into_iter().take(max_per_chunk));
    }
    selected.sort_unstable();
    selected.dedup();
    selected
}

fn is_meaningful_ocr_text(text: &str) -> bool {
    if text.chars().count() < 3 {
        return false;
    }
    let total = text.chars().count().max(1);
    let alpha = text.chars().filter(|c| c.is_alphabetic()).count();
    alpha * 10 >= total * 4
}

fn ocr_binary_path(ctx: &JobContext) -> &str {
    &ctx.config.vision.ocr.adapter
}

fn anthropic_client_from_env(ctx: &JobContext) -> Option<AnthropicClient> {
    std::env::var("ANTHROPIC_API_KEY").ok().and_then(|k| {
        let t = k.trim().to_string();
        (!t.is_empty()).then(|| {
            AnthropicClient::new(&t, &ctx.config.llm.model, ctx.config.llm.max_concurrency)
        })
    })
}

fn claude_vlm_from_env(ctx: &JobContext) -> Option<ClaudeVlmAdapter> {
    std::env::var("ANTHROPIC_API_KEY").ok().and_then(|k| {
        let t = k.trim().to_string();
        (!t.is_empty()).then(|| ClaudeVlmAdapter::new(&t, &ctx.config.vision.caption.model))
    })
}

fn format_time_range(start: f64, end: f64) -> String {
    format!("{}-{}", format_timecode(start), format_timecode(end))
}

fn format_timecode(seconds: f64) -> String {
    let total_ms = (seconds.max(0.0) * 1000.0).round() as u64;
    let h = total_ms / 3_600_000;
    let m = (total_ms % 3_600_000) / 60_000;
    let s = (total_ms % 60_000) / 1000;
    let ms = total_ms % 1000;
    format!("{h:02}:{m:02}:{s:02}.{ms:03}")
}

fn write_json_pretty<T: Serialize>(path: &Path, value: &T) -> Result<(), VididxError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(value)?)?;
    Ok(())
}

fn write_chunks_jsonl(chunks: &[Chunk], path: &Path) -> Result<(), VididxError> {
    use std::io::Write;
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p)?;
    }
    let file = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::new(file);
    for chunk in chunks {
        serde_json::to_writer(&mut w, chunk)?;
        w.write_all(b"\n")?;
    }
    w.flush()?;
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

    fn make_ctx(from: usize, to: usize) -> JobContext {
        JobContext {
            video_id: "test_video".to_string(),
            source_type: "local_mp4".to_string(),
            source_ref: "/tmp/test.mp4".to_string(),
            source_path: PathBuf::from("/tmp/test.mp4"),
            out_dir: PathBuf::from("/tmp/out"),
            from_stage: from,
            to_stage: to,
            config: Config::default(),
            manifest: std::sync::Arc::new(tokio::sync::Mutex::new(Manifest::new(
                "test_video",
                "/tmp/test.mp4",
                "sha256:src",
                "sha256:cfg",
            ))),
        }
    }

    #[test]
    fn test_stage_name_to_index_numeric() {
        assert_eq!(stage_name_to_index("0"), Some(0));
        assert_eq!(stage_name_to_index("9"), Some(9));
        assert_eq!(stage_name_to_index("10"), None);
    }

    #[test]
    fn test_stage_name_to_index_named() {
        assert_eq!(stage_name_to_index("stage0_probe"), Some(0));
        assert_eq!(stage_name_to_index("stage5_frames"), Some(5));
        assert_eq!(stage_name_to_index("stage6_semantic"), Some(6));
        assert_eq!(stage_name_to_index("stage9_output"), Some(9));
        assert_eq!(stage_name_to_index("stage_unknown"), None);
    }

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
        assert!(stamps.iter().all(|&s| (0.0..=60.0).contains(&s)));
        assert!(stamps.len() <= 5);
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
    fn test_should_load_from_previous_run_respects_from_stage() {
        let ctx = make_ctx(4, 9);
        assert!(should_load_from_previous_run(&ctx, 0));
        assert!(should_load_from_previous_run(&ctx, 3));
        assert!(!should_load_from_previous_run(&ctx, 4));
    }

    #[test]
    fn test_stage_artifact_path_maps_stage_numbers() {
        let ctx = make_ctx(0, 9);
        assert_eq!(
            stage_artifact_path(&ctx, 0),
            PathBuf::from("/tmp/out/probe.json")
        );
        assert_eq!(
            stage_artifact_path(&ctx, 1),
            PathBuf::from("/tmp/out/audio.json")
        );
        assert_eq!(
            stage_artifact_path(&ctx, 5),
            PathBuf::from("/tmp/out/frames.json")
        );
        assert_eq!(
            stage_artifact_path(&ctx, 8),
            PathBuf::from("/tmp/out/annotated.json")
        );
    }

    #[test]
    fn test_has_visual_reflects_vlm_caption_not_frame_presence() {
        let ctx = make_ctx(0, 9);
        let annotated = vec![AnnotatedChunk {
            chunk_id: "test_video_chunk_0000".to_string(),
            parent_segment_id: String::new(),
            start_sec: 0.0,
            end_sec: 30.0,
            transcript_text: "transcript".to_string(),
            title: "title".to_string(),
            summary: "summary".to_string(),
            keywords: vec![],
            rationale: "test".to_string(),
        }];
        // Frame has no visual_caption
        let frames_no_caption = vec![FrameArtifact {
            path: "frame0.jpg".to_string(),
            at_sec: 5.0,
            kind: FrameKind::Periodic,
            analyzed: false,
            ocr_text: None,
            visual_caption: None,
        }];
        let chunks_no_vlm = build_chunks(
            &ctx,
            "/tmp/test.mp4",
            "sha256:src",
            &annotated,
            &frames_no_caption,
            true,
            ContentType::ScreenRecording,
        );
        assert!(
            !chunks_no_vlm[0].modality_flags.has_visual,
            "has_visual must be false when no VLM caption"
        );

        // Frame has visual_caption
        let frames_with_caption = vec![FrameArtifact {
            path: "frame0.jpg".to_string(),
            at_sec: 5.0,
            kind: FrameKind::SceneChange,
            analyzed: true,
            ocr_text: None,
            visual_caption: Some("A settings dialog".to_string()),
        }];
        let chunks_with_vlm = build_chunks(
            &ctx,
            "/tmp/test.mp4",
            "sha256:src",
            &annotated,
            &frames_with_caption,
            true,
            ContentType::ScreenRecording,
        );
        assert!(
            chunks_with_vlm[0].modality_flags.has_visual,
            "has_visual must be true when VLM caption present"
        );
    }

    #[tokio::test]
    async fn test_cached_stage_output_path_returns_path_on_hit() {
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
