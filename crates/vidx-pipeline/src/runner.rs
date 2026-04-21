use chrono::Utc;
use serde::Serialize;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::stage::JobContext;
use vidx_asr::{AsrAdapter, WhisperCppAdapter};
use vidx_core::{
    AnnotatedChunk, Chunk, ContentType, FrameKind, ImageRef, ModalityFlags, NormalizedChunk,
    ProcessingMeta, SceneChange, SemanticChunk, SourceJumpRef, TranscriptTimeline, VidxError,
};
use vidx_llm::AnthropicClient;
use vidx_media::{detect_scene_changes, detect_silence, extract_audio, extract_frames, probe};
use vidx_output::{OutputIndex, write_index, write_markdown};
use vidx_segment::{annotate_chunk, coarse_segment, normalize};
use vidx_vision::{
    ClaudeVlmAdapter, OcrAdapter, TesseractAdapter, VlmCaptionAdapter, dhash_from_path,
    hamming_distance,
};

const STAGES: &[&str] = &[
    "stage0_probe",
    "stage1_audio",
    "stage2_asr",
    "stage3_silence",
    "stage4_scene",
    "stage5_frames",
    "stage6_coarse",
    "stage7_semantic",
    "stage8_normalize",
    "stage9_annotate",
];

#[derive(Debug, Serialize)]
struct FrameArtifact {
    path: String,
    at_sec: f64,
    kind: FrameKind,
    analyzed: bool,
    ocr_text: Option<String>,
    visual_caption: Option<String>,
}

#[derive(Debug)]
struct FrameCandidate {
    path: String,
    at_sec: f64,
    kind: FrameKind,
    dhash: Option<u64>,
}

/// Run a pipeline for processing a video.
pub async fn run_pipeline(ctx: JobContext) -> Result<(), VidxError> {
    std::fs::create_dir_all(&ctx.out_dir)?;

    let input_hash = {
        let manifest = ctx.manifest.lock().await;
        format!("{}:{}", manifest.source_hash, manifest.config_hash)
    };

    let media_probe = run_stage(&ctx, STAGES[0], &input_hash, async {
        let result = probe(&ctx.config.media.ffprobe_path, &ctx.source_path).await?;
        Ok((ctx.out_dir.join("stage0_probe.json"), result))
    })
    .await?;

    let audio_path = ctx.out_dir.join("audio").join("audio.wav");
    let stage1_payload = run_stage(&ctx, STAGES[1], &input_hash, async {
        if media_probe.has_audio {
            extract_audio(&ctx.config.media.ffmpeg_path, &ctx.source_path, &audio_path).await?;
        }

        #[derive(Serialize)]
        struct AudioStage {
            extracted: bool,
            wav_path: String,
        }

        Ok((
            ctx.out_dir.join("stage1_audio.json"),
            AudioStage {
                extracted: media_probe.has_audio,
                wav_path: audio_path.to_string_lossy().to_string(),
            },
        ))
    })
    .await?;
    let _ = stage1_payload;

    let transcript = run_stage(&ctx, STAGES[2], &input_hash, async {
        let timeline = transcribe_or_empty(&ctx, &audio_path, media_probe.has_audio).await;
        Ok((ctx.out_dir.join("stage2_asr.json"), timeline))
    })
    .await?;

    let silence = run_stage(&ctx, STAGES[3], &input_hash, async {
        let intervals = if media_probe.has_audio {
            detect_silence(&ctx.config.media.ffmpeg_path, &ctx.source_path, -35.0, 0.5).await?
        } else {
            Vec::new()
        };
        Ok((ctx.out_dir.join("stage3_silence.json"), intervals))
    })
    .await?;

    let scenes = run_stage(&ctx, STAGES[4], &input_hash, async {
        let changes = if media_probe.has_video {
            detect_scene_changes(
                &ctx.config.media.ffmpeg_path,
                &ctx.source_path,
                ctx.config.frames.scene_change_threshold as f32,
            )
            .await?
        } else {
            Vec::new()
        };
        Ok((ctx.out_dir.join("stage4_scene.json"), changes))
    })
    .await?;

    let frame_artifacts = run_stage(&ctx, STAGES[5], &input_hash, async {
        let frames = if media_probe.has_video {
            let stamps = select_frame_timestamps(
                media_probe.duration_sec,
                &scenes,
                ctx.config.frames.periodic_interval_sec,
                ctx.config.frames.max_analyzed_per_chunk.max(8),
            );
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
                    kind: nearest_frame_kind(at_sec, &scenes),
                    dhash: dhash_from_path(&path).ok(),
                })
                .collect::<Vec<_>>();

            dedupe_frame_candidates(candidates, ctx.config.frames.dhash_distance_threshold)
        } else {
            Vec::new()
        };
        Ok((ctx.out_dir.join("stage5_frames.json"), frames))
    })
    .await?;

    let coarse = run_stage(&ctx, STAGES[6], &input_hash, async {
        let coarse = coarse_segment(
            media_probe.duration_sec,
            &transcript,
            &silence,
            &scenes,
            ctx.config.segment.coarse.max_duration_sec,
            ctx.config.segment.coarse.snap_window_sec,
        )?;
        Ok((ctx.out_dir.join("stage6_coarse.json"), coarse))
    })
    .await?;

    let semantic = run_stage(&ctx, STAGES[7], &input_hash, async {
        let semantic = coarse
            .iter()
            .flat_map(|segment| {
                heuristic_semantic_chunks(
                    segment.start_sec,
                    segment.end_sec,
                    &transcript,
                    ctx.config.segment.semantic.target_min_sec,
                    ctx.config.segment.semantic.target_max_sec,
                )
            })
            .collect::<Vec<_>>();
        Ok((ctx.out_dir.join("stage7_semantic.json"), semantic))
    })
    .await?;

    let normalized = run_stage(&ctx, STAGES[8], &input_hash, async {
        let normalized = normalize(
            semantic,
            ctx.config.segment.semantic.hard_min_sec,
            ctx.config.segment.semantic.hard_max_sec,
            &ctx.video_id,
        );
        Ok((ctx.out_dir.join("stage8_normalize.json"), normalized))
    })
    .await?;

    let frame_artifacts = enrich_frame_artifacts(&ctx, frame_artifacts, &normalized).await;

    let annotated = run_stage(&ctx, STAGES[9], &input_hash, async {
        let llm_client = anthropic_client_from_env(&ctx);
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
        Ok((ctx.out_dir.join("stage9_annotate.json"), annotated))
    })
    .await?;

    let (source_path, source_hash) = {
        let manifest = ctx.manifest.lock().await;
        (manifest.source_path.clone(), manifest.source_hash.clone())
    };

    let chunks = build_chunks(
        &ctx,
        &source_path,
        &source_hash,
        &annotated,
        &frame_artifacts,
        media_probe.has_audio,
    );
    write_chunks_jsonl(
        &chunks,
        &ctx.out_dir.join(format!("{}.chunks.jsonl", ctx.video_id)),
    )?;

    write_markdown(
        &annotated,
        &ctx.video_id,
        &ctx.out_dir.join(format!("{}.md", ctx.video_id)),
    )
    .await?;

    write_index(
        &OutputIndex {
            video_id: ctx.video_id.clone(),
            total_chunks: chunks.len(),
            total_duration_sec: media_probe.duration_sec,
            chunks_file: format!("{}.chunks.jsonl", ctx.video_id),
            markdown_file: format!("{}.md", ctx.video_id),
        },
        &ctx.out_dir.join(format!("{}.index.json", ctx.video_id)),
    )
    .await?;

    let manifest_path = ctx.out_dir.join("manifest.json");
    let manifest = ctx.manifest.lock().await;
    manifest.save(&manifest_path).map_err(stage_error_to_vidx)?;

    Ok(())
}

async fn run_stage<T, F>(
    ctx: &JobContext,
    stage_name: &'static str,
    input_hash: &str,
    f: F,
) -> Result<T, VidxError>
where
    T: Serialize,
    F: std::future::Future<Output = Result<(PathBuf, T), VidxError>>,
{
    {
        let mut manifest = ctx.manifest.lock().await;
        manifest.mark_running(stage_name, input_hash);
        manifest
            .save(&ctx.out_dir.join("manifest.json"))
            .map_err(stage_error_to_vidx)?;
    }

    match f.await {
        Ok((path, value)) => {
            write_json_pretty(&path, &value)?;
            let mut manifest = ctx.manifest.lock().await;
            manifest.mark_done(stage_name, &path.to_string_lossy());
            manifest
                .save(&ctx.out_dir.join("manifest.json"))
                .map_err(stage_error_to_vidx)?;
            eprintln!("✓ {}", stage_name);
            Ok(value)
        }
        Err(err) => {
            let mut manifest = ctx.manifest.lock().await;
            manifest.mark_failed(stage_name);
            manifest
                .save(&ctx.out_dir.join("manifest.json"))
                .map_err(stage_error_to_vidx)?;
            eprintln!("✗ {} - {}", stage_name, err);
            Err(err)
        }
    }
}

async fn transcribe_or_empty(
    ctx: &JobContext,
    audio_path: &Path,
    has_audio: bool,
) -> TranscriptTimeline {
    if !has_audio || !audio_path.exists() {
        return TranscriptTimeline { segments: vec![] };
    }

    let adapter = WhisperCppAdapter::new(
        &ctx.config.asr.whisper_cpp.binary_path,
        &ctx.config.asr.whisper_cpp.model_path,
        ctx.config.asr.whisper_cpp.threads,
        &ctx.config.asr.language,
    );

    match adapter.transcribe(audio_path).await {
        Ok(timeline) => timeline,
        Err(err) => {
            eprintln!("! ASR fallback to empty transcript: {}", err);
            TranscriptTimeline { segments: vec![] }
        }
    }
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
    let mut stamps = vec![0.0, (duration_sec / 2.0).min(safe_end)];

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
        let mut words = transcript
            .split_whitespace()
            .take(8)
            .collect::<Vec<_>>()
            .join(" ");
        if words.len() > 60 {
            words.truncate(60);
        }
        words
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
) -> Vec<Chunk> {
    annotated
        .iter()
        .map(|chunk| {
            let image_refs = frame_artifacts
                .iter()
                .filter(|frame| frame.at_sec >= chunk.start_sec && frame.at_sec <= chunk.end_sec)
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
                content_type: ContentType::ScreenRecording,
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
                    segmentation_rationale: Some(
                        "coarse+time-based semantic segmentation".to_string(),
                    ),
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
) -> String {
    [
        Some(title.trim()),
        Some(summary.trim()),
        Some(transcript.trim()),
        ocr_text.map(str::trim),
        visual_caption.map(str::trim),
    ]
    .into_iter()
    .flatten()
    .filter(|part| !part.is_empty())
    .collect::<Vec<_>>()
    .join("\n\n")
}

fn collect_chunk_visual_data(
    start_sec: f64,
    end_sec: f64,
    frame_artifacts: &[FrameArtifact],
) -> (Option<String>, Option<String>) {
    let mut ocr_parts = Vec::new();
    let mut caption_parts = Vec::new();
    let mut seen_ocr = HashSet::new();
    let mut seen_caption = HashSet::new();

    for frame in frame_artifacts
        .iter()
        .filter(|frame| frame.at_sec >= start_sec && frame.at_sec <= end_sec)
    {
        if let Some(text) = frame.ocr_text.as_deref() {
            let normalized = text.trim();
            if !normalized.is_empty() && seen_ocr.insert(normalized.to_string()) {
                ocr_parts.push(normalized.to_string());
            }
        }

        if let Some(text) = frame.visual_caption.as_deref() {
            let normalized = text.trim();
            if !normalized.is_empty() && seen_caption.insert(normalized.to_string()) {
                caption_parts.push(normalized.to_string());
            }
        }
    }

    let ocr_text = (!ocr_parts.is_empty()).then(|| ocr_parts.join("\n"));
    let visual_caption = (!caption_parts.is_empty()).then(|| caption_parts.join(" "));
    (ocr_text, visual_caption)
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
                (!trimmed.is_empty()).then(|| trimmed.to_string())
            });
    }

    if let Some(vlm_adapter) = claude_vlm_from_env(ctx) {
        let selected =
            select_caption_frame_indices(&frames, chunks, ctx.config.vision.caption.max_per_chunk);
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

fn write_json_pretty<T: Serialize>(path: &Path, value: &T) -> Result<(), VidxError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(value)?;
    std::fs::write(path, json)?;
    Ok(())
}

fn write_chunks_jsonl(chunks: &[Chunk], path: &Path) -> Result<(), VidxError> {
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

fn stage_error_to_vidx(err: crate::stage::StageError) -> VidxError {
    VidxError::Config(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Manifest;
    use vidx_core::Config;
    use vidx_core::TranscriptSegment;

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
        let chunks = heuristic_semantic_chunks(0.0, 180.0, &timeline, 30.0, 90.0);
        assert!(!chunks.is_empty());
        assert_eq!(chunks.first().unwrap().start_sec, 0.0);
        assert_eq!(chunks.last().unwrap().end_sec, 180.0);
    }

    #[test]
    fn test_build_embedding_text_skips_empty_parts() {
        let text = build_embedding_text("title", "", "body", None, None);
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

        let chunks = heuristic_semantic_chunks(0.0, 20.0, &timeline, 10.0, 10.0);

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].transcript_text, "intro");
        assert_eq!(chunks[1].transcript_text, "details");
    }

    #[test]
    fn test_build_chunks_includes_visual_analysis() {
        let config = Config::default();
        let manifest = Manifest::new("test_video", "/tmp/test.mp4", "sha256:src", "sha256:cfg");
        let ctx = JobContext {
            video_id: "test_video".to_string(),
            source_type: "local_mp4".to_string(),
            source_ref: "/tmp/test.mp4".to_string(),
            source_path: PathBuf::from("/tmp/test.mp4"),
            out_dir: PathBuf::from("/tmp/out"),
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
            config,
            manifest: std::sync::Arc::new(tokio::sync::Mutex::new(manifest)),
        };

        assert_eq!(ocr_binary_path(&ctx), "custom-tesseract");
    }
}
