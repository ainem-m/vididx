# Video RAG Preprocessor — 要件定義書 v3 (実装投入版)

- **対象実装者**: AIエージェント(Claude Code)
- **言語 / ランタイム**: Rust (stable, edition 2021 以上) / tokio async
- **成果物形態**: ネイティブCLIバイナリ + ライブラリクレート群 (cargo workspace)
- **ベースバージョン**: v2(階層的意味分割案)を確定仕様化したもの

---

## 1. 目的 / ゴール

ローカル動画(mp4)を入力として、下流のベクトルDB/RAG に直接投入可能な
**retrieval-ready な JSON チャンク群** と **人間可読な Markdown** を生成する CLI ツールを構築する。

**非ゴール(スコープ外)**

- ベクトルDBへの投入そのもの(JSONファイルの生成まで)
- 埋め込み(embedding)ベクトルの生成 ※下流で行う
- 動画ファイルのダウンロード(ローカル mp4 のみ対象)
- GUI 提供(CLI のみ)
- 文字起こしモデルや VLM そのものの学習/fine-tuning

---

## 2. 用語集

| 用語 | 定義 |
|---|---|
| **Segment (粗分割単位)** | 動画を話題単位より大きく一次分割した区間。上限5分(既定)。 |
| **Chunk (意味分割単位)** | 最終的な retrieval 単位。目標30〜90秒、上限120秒。 |
| **Frame** | 動画から抽出した静止画。定期抽出+画面変化検出で収集。 |
| **補助信号 (auxiliary signal)** | 意味境界の推定に使う副次情報。無音・画面変化・OCR変化・話者交代。 |
| **Manifest** | 1動画処理ジョブ全体の状態/キャッシュ記録(JSON)。 |
| **Artifact** | 各Stageが出力する中間成果物。manifestから参照される。 |
| **Adapter** | 外部ツール/API呼び出しを抽象化した trait 実装。 |

---

## 3. 未決着4論点の確定仕様

v2で保留だった4点をデフォルト値として確定。すべて `config.toml` で上書き可能にする。

| 論点 | 確定値(既定) | 根拠 |
|---|---|---|
| **A. 粗分割の上限** | **5分(300秒)** | 意味境界判定の安定性と LLM 入力長のバランス。10分は境界がぼやけ、3分は外部API呼び出し回数が増える。 |
| **B. 画像抽出ルール** | **定期(5秒毎) + 画面変化時追加** | 抽出コストは安く、網羅性とピーク検出を両立できる。 |
| **C. 埋め込み対象テキスト** | **title + summary + transcript + 重要OCR** を連結して1文字列で出力(embedding は下流実施) | 下流での再構築を避けるため、`embedding_text` フィールドを事前計算して保持する。 |
| **D. 1チャンク目標長** | **30〜90秒(上限120秒、下限15秒)** | 下限以下は結合、上限以上は再分割。 |

---

## 4. 全体アーキテクチャ

```
┌──────────────────────────────────────────────────────────────┐
│                    vidx-cli (バイナリ)                        │
└───────┬──────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│              vidx-pipeline (オーケストレータ)                 │
│  Stage0 → Stage1 → Stage2 → Stage3 → Stage4 → Stage5 → ...   │
└───────┬──────────────────────────────────────────────────────┘
        │
        ├── vidx-media   (ffmpeg呼び出し / メディア解析)
        ├── vidx-asr     (文字起こし: whisper.cpp / API)
        ├── vidx-vision  (OCR / Visual Caption)
        ├── vidx-segment (粗分割 / 意味分割)
        ├── vidx-llm    (summary / title / keywords)
        ├── vidx-output  (JSON / Markdown 生成)
        └── vidx-core    (ドメイン型 / エラー / 設定)
```

**設計原則**

1. **外部依存は必ず trait + Adapter** で抽象化(モック差し替え可能)
2. **各 Stage は idempotent**(入力ハッシュ一致ならキャッシュヒット)
3. **Stage間の受け渡しは型付き struct のみ**(dict/HashMap 禁止)
4. **非同期I/Oは tokio、CPU重い処理は spawn_blocking または rayon**
5. **エラーは thiserror で定義、呼び出し側は anyhow で集約**

---

## 5. 処理パイプライン(Stage 定義)

各Stageは以下を満たす trait を実装する(`vidx-pipeline::Stage`)。

```rust
#[async_trait]
pub trait Stage {
    type Input;
    type Output;
    const NAME: &'static str;

    async fn run(&self, ctx: &JobContext, input: Self::Input) -> Result<Self::Output, StageError>;
    fn cache_key(&self, input: &Self::Input) -> String; // 入力ハッシュ
}
```

### Stage 0: メディア解析 (`vidx-media`)

- **入力**: `mp4ファイルパス`
- **出力**: `MediaProbe { duration_sec, video_stream, audio_stream, fps, resolution, ... }`
- **実装**: `ffprobe` をサブプロセス実行し、JSON出力をパース
- **失敗モード**: ファイル欠損 / コーデック非対応 / 音声トラックなし
- **受け入れ基準**:
  - 1時間以内の mp4 で duration が ±0.1秒 誤差内で取得できる
  - 音声トラックなしの動画でも(警告ログを出して)処理継続可能

### Stage 1: 文字起こし (`vidx-asr`)

- **入力**: `mp4ファイルパス`, `MediaProbe`
- **出力**: `TranscriptTimeline { segments: Vec<TranscriptSegment> }`
  - `TranscriptSegment { start_sec, end_sec, text, speaker?, confidence? }`
- **実装**: `AsrAdapter` trait を差し替えられるようにする
  - 既定: `WhisperCppAdapter`(whisper.cpp をサブプロセス実行、出力JSONをパース)
  - 代替: `OpenAiWhisperAdapter`(OpenAI API), `GroqWhisperAdapter`(Groq API)
- **コスト制御**: 音声のみ抽出→16kHz mono wav に変換してから ASR に投入
- **失敗モード**: 無音動画 → 空の timeline を返す(エラーにしない)
- **受け入れ基準**:
  - 30分の日本語動画で、最低1分あたり30セグメント以上の粒度が得られる
  - タイムスタンプは単調増加である
  - 各セグメントの `text` は空文字列ではない(空ならセグメントごと除外)

### Stage 2: 補助信号抽出 (`vidx-media` + `vidx-vision`)

並列実行可能な3つのサブStage:

#### 2a. 無音区間検出

- **入力**: `mp4ファイルパス`
- **出力**: `Vec<SilenceInterval { start_sec, end_sec }>`
- **実装**: `ffmpeg -af silencedetect=noise=-30dB:d=0.6` の stderr をパース
- **既定閾値**: -30dB / 最低0.6秒

#### 2b. 画面変化(scene change)検出

- **入力**: `mp4ファイルパス`
- **出力**: `Vec<SceneChange { at_sec, score }>`
- **実装**: `ffmpeg -vf select='gt(scene,0.4)',showinfo`
- **既定閾値**: scene スコア 0.4

#### 2c. (遅延) 話者交代検出

- **v3では必須実装ではなく**、`TranscriptSegment.speaker` が埋まっていれば交代点を抽出するだけのポストプロセスで良い
- ASR が diarization 未対応の場合、このサブStageは skip される

### Stage 3: 粗分割 / Coarse Segmentation (`vidx-segment`)

- **入力**: `MediaProbe`, `TranscriptTimeline`, `Vec<SilenceInterval>`, `Vec<SceneChange>`
- **出力**: `Vec<CoarseSegment { index, start_sec, end_sec, transcript_text }>`
- **アルゴリズム(ハイブリッド粗分割)**:
  1. 動画全体を仮想的に 300秒(既定)の枠に区切る
  2. 各枠の境界付近(±30秒)に無音区間 or 画面変化があれば、そちらにスナップ
  3. 枠のどこにも補助信号がなければ、そのまま300秒で切る
  4. 各枠に属する transcript セグメントを連結して `transcript_text` にする
- **不変条件**:
  - すべての CoarseSegment は `end_sec > start_sec`
  - 連続する CoarseSegment は `prev.end_sec == next.start_sec`(境界は連続)
  - 動画全体をカバーする(最初の start_sec == 0、最後の end_sec == duration)

### Stage 4: 意味分割 / Semantic Chunking (`vidx-segment` + `vidx-llm`)

- **入力**: 1つの `CoarseSegment`(並列処理可能)
- **出力**: `Vec<SemanticChunk { start_sec, end_sec, transcript_text, rationale }>`
- **アルゴリズム**:
  1. LLM に CoarseSegment の transcript(タイムスタンプ付き)を渡す
  2. 以下の指示で境界推定: 「話題転換 / 手順切替 / 質問→回答 / スライド節 / 操作フェーズ変更 を境界とせよ」
  3. LLM は `[{start_sec, end_sec, rationale}, ...]` を JSON で返す
  4. 返却JSONを検証(スキーマ / 時刻整合性 / 抜け漏れ)
- **LLM プロンプトは `prompts/semantic_chunking.jinja` として外部化**
- **コスト制御**:
  - CoarseSegment の transcript が非常に短い(<30秒相当)場合は LLM を呼ばずそのまま1チャンクとする
  - 並列度は `config.llm.max_concurrency`(既定4)で制限
- **失敗時フォールバック**:
  - LLM がスキーマ違反JSONを返したら1回だけ再試行
  - 再試行も失敗したら、CoarseSegment を目標長で均等分割して警告ログを出す

### Stage 5: 後処理(結合・再分割) (`vidx-segment`)

- **入力**: `Vec<SemanticChunk>`(全 CoarseSegment 分を連結したもの)
- **出力**: `Vec<NormalizedChunk>`
- **ルール**:
  1. 15秒未満のチャンクは隣接チャンクと結合(短い方から処理)
  2. 120秒超のチャンクは 30〜90秒目標で時刻等分で再分割
  3. 結合/分割時は transcript_text も整合するように再構築
  4. 最終的にすべてのチャンクに連番 `chunk_id` を付与: `{video_id}_chunk_{NNNN}`

### Stage 6: 画像抽出・選別 (`vidx-media`)

- **入力**: `MediaProbe`, `Vec<SceneChange>`, `Vec<NormalizedChunk>`
- **出力**: `Vec<ExtractedFrame { path, at_sec, kind: Periodic|SceneChange, chunk_id }>`
- **抽出ルール(既定)**:
  - **定期抽出**: 5秒ごとに1フレーム(`Periodic`)
  - **画面変化抽出**: SceneChange の各 `at_sec` で1フレーム(`SceneChange`)
  - どちらの条件でも同一チャンク内で同じ秒数に近い(±1秒)フレームは1枚に重複排除
- **出力形式**: jpeg(quality=85)、`images/{video_id}/chunk_{NNNN}/frame_{MM}.jpg`
- **実装**: `ffmpeg -ss {t} -i {video} -frames:v 1 -q:v 3 out.jpg` をチャンクごとにバッチ実行
- **選別(OCR/caption対象の決定)**:
  - `kind=SceneChange` のものは常に OCR/caption 対象
  - `kind=Periodic` のものは、直前の OCR/caption 対象から **dHash 距離が閾値以上**のものだけ対象にする(類似フレームの重複解析を避ける)
  - 既定の dHash ハミング距離閾値: 10
  - 解析対象フレームは1チャンクあたり最大8枚(超過分は等間隔で間引き)

### Stage 7: OCR / Visual Caption (`vidx-vision`)

- **入力**: Stage6 で「解析対象」とマークされたフレーム群
- **出力**: `FrameAnalysis { frame_path, ocr_text?, visual_caption? }`
- **アダプタ**:
  - OCR: 既定 `TesseractAdapter`(tesseract を `tesseract image.jpg - -l jpn+eng` でサブプロセス実行)
  - Visual Caption: 既定 `ClaudeVlmAdapter`(Anthropic Messages API に画像を base64 で投入)
  - 代替: `GeminiVlmAdapter`, `OpenAiVlmAdapter`
- **コスト制御**:
  - OCR は全対象フレームに実施(ローカルなので安い)
  - VLM caption は**チャンクあたり最大3枚**に制限(kind=SceneChange を優先)
- **失敗モード**:
  - OCRが空文字列 → そのまま空で記録、エラーにしない
  - VLM 呼び出し失敗 → リトライ3回(指数バックオフ)、全失敗なら `visual_caption=null` で続行

### Stage 8: チャンク注釈生成 (`vidx-llm`)

- **入力**: `NormalizedChunk` + そのチャンクに紐づく `FrameAnalysis[]`
- **出力**: `AnnotatedChunk`(title, summary, keywords を付与)
- **LLM 1回の呼び出しで title / summary / keywords をまとめて生成**(プロンプト外部化)
  - title: 40文字以内の日本語
  - summary: 2〜4文、150文字程度
  - keywords: 3〜7個、名詞句
- **並列度**: `config.llm.max_concurrency`(既定4)
- **失敗時**:
  - スキーマ違反 → 1回リトライ → 失敗なら transcript の先頭40文字を title、先頭150文字を summary に fallback

### Stage 9: 出力 (`vidx-output`)

- **入力**: `Vec<AnnotatedChunk>` + `FrameAnalysis[]`
- **出力**:
  1. `{video_id}.chunks.jsonl` — 1行1チャンクのJSONL(retrieval-ready)
  2. `{video_id}.index.json` — manifest + メタデータ
  3. `{video_id}.md` — 人間可読Markdown(時系列、画像サムネ参照つき)
- **JSONL の各レコード**は第6章のスキーマ通り

---

## 6. データモデル

### 6.1 出力 JSONL スキーマ(1チャンク1レコード)

```json
{
  "schema_version": "1.0",
  "video_id": "vid_001",
  "source_type": "local_mp4",
  "source_path": "/videos/sample.mp4",
  "source_hash": "sha256:...",
  "chunk_id": "vid_001_chunk_0007",
  "parent_segment_id": "vid_001_seg_0002",
  "start_sec": 612.4,
  "end_sec": 684.8,
  "start_tc": "00:10:12.400",
  "end_tc": "00:11:24.800",
  "duration_sec": 72.4,
  "content_type": "screen_recording",
  "speaker_info": ["speaker_a"],
  "title": "検索条件を変更して結果を比較する場面",
  "summary": "検索条件の変更により表示結果がどう変わるかを説明している。",
  "transcript": "ここで検索条件を変えると…",
  "ocr_text": "Filter Settings / Results / Export",
  "visual_caption": "アプリの設定画面が表示され、左側にフィルタ項目、右側に結果一覧がある。",
  "keywords": ["検索条件", "フィルタ", "結果比較"],
  "embedding_text": "検索条件を変更して結果を比較する場面\n\n検索条件の変更により表示結果が…\n\nここで検索条件を変えると…\n\n[OCR] Filter Settings / Results / Export",
  "image_refs": [
    {"path": "images/vid_001/chunk_0007/frame_01.jpg", "at_sec": 615.0, "kind": "scene_change", "analyzed": true},
    {"path": "images/vid_001/chunk_0007/frame_02.jpg", "at_sec": 620.0, "kind": "periodic", "analyzed": false}
  ],
  "source_jump_ref": {"type": "local", "start_sec": 612.4},
  "modality_flags": {"has_speech": true, "has_ocr": true, "has_visual": true},
  "processing_meta": {
    "segmentation_rationale": "操作フェーズがフィルタ設定から結果確認に切り替わった",
    "asr_adapter": "whisper_cpp_large_v3",
    "vlm_adapter": "claude_sonnet_4",
    "generated_at": "2026-04-20T12:00:00Z"
  }
}
```

### 6.2 Rust 型定義(抜粋、`vidx-core::model`)

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContentType {
    ScreenRecording,
    Slide,
    Conversation,
    Lecture,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub schema_version: String,       // "1.0"
    pub video_id: String,
    pub source_type: String,          // "local_mp4"
    pub source_path: String,
    pub source_hash: String,          // "sha256:..."
    pub chunk_id: String,
    pub parent_segment_id: String,
    pub start_sec: f64,
    pub end_sec: f64,
    pub start_tc: String,             // "HH:MM:SS.mmm"
    pub end_tc: String,
    pub duration_sec: f64,
    pub content_type: ContentType,
    pub speaker_info: Vec<String>,
    pub title: String,
    pub summary: String,
    pub transcript: String,
    pub ocr_text: Option<String>,
    pub visual_caption: Option<String>,
    pub keywords: Vec<String>,
    pub embedding_text: String,
    pub image_refs: Vec<ImageRef>,
    pub source_jump_ref: SourceJumpRef,
    pub modality_flags: ModalityFlags,
    pub processing_meta: ProcessingMeta,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageRef {
    pub path: String,
    pub at_sec: f64,
    pub kind: FrameKind,
    pub analyzed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrameKind {
    Periodic,
    SceneChange,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalityFlags {
    pub has_speech: bool,
    pub has_ocr: bool,
    pub has_visual: bool,
}

// 他の型(SourceJumpRef, ProcessingMeta, CoarseSegment, SemanticChunk,
// NormalizedChunk, AnnotatedChunk, FrameAnalysis, MediaProbe など)は省略
```

### 6.3 Manifest(中間成果物管理、`{out_dir}/{video_id}/manifest.json`)

```json
{
  "manifest_version": "1.0",
  "video_id": "vid_001",
  "source_path": "/videos/sample.mp4",
  "source_hash": "sha256:...",
  "config_hash": "sha256:...",
  "stages": {
    "stage0_probe":      {"status": "done", "input_hash": "...", "output_path": ".../probe.json",      "completed_at": "..."},
    "stage1_asr":        {"status": "done", "input_hash": "...", "output_path": ".../transcript.json", "completed_at": "..."},
    "stage2_aux":        {"status": "done", "input_hash": "...", "output_path": ".../aux.json",        "completed_at": "..."},
    "stage3_coarse":     {"status": "done", "input_hash": "...", "output_path": ".../coarse.json",     "completed_at": "..."},
    "stage4_semantic":   {"status": "done", "input_hash": "...", "output_path": ".../semantic.json",   "completed_at": "..."},
    "stage5_normalize":  {"status": "done", "input_hash": "...", "output_path": ".../normalized.json", "completed_at": "..."},
    "stage6_frames":     {"status": "done", "input_hash": "...", "output_path": ".../frames.json",     "completed_at": "..."},
    "stage7_vision":     {"status": "done", "input_hash": "...", "output_path": ".../vision.json",     "completed_at": "..."},
    "stage8_annotate":   {"status": "done", "input_hash": "...", "output_path": ".../annotated.json",  "completed_at": "..."},
    "stage9_output":     {"status": "done", "input_hash": "...", "output_path": ".../chunks.jsonl",    "completed_at": "..."}
  }
}
```

**キャッシュキー算出**:
`input_hash = sha256(canonicalize_json(stage_input) ‖ config_hash ‖ source_hash ‖ stage_name)`

Stage が `done` かつ `input_hash` が一致すれば、そのStageはスキップ。

---

## 7. CLI 仕様

```
vidx [GLOBAL_OPTS] <COMMAND> [ARGS...]

GLOBAL OPTIONS:
  --config <PATH>        設定ファイルパス(既定: ./vidx.toml → ~/.config/vidx/config.toml)
  --log-level <LEVEL>    trace|debug|info|warn|error (既定: info)
  --out-dir <PATH>       出力先ディレクトリ(既定: ./out)

COMMANDS:
  process <VIDEO>        mp4を処理してチャンクJSONL+Markdownを生成
    --video-id <ID>      明示的なvideo_id(既定: ファイル名から自動生成)
    --from <STAGE>       指定Stageから再開(既定: 未完了の最初のStage)
    --to <STAGE>         指定Stageで停止(既定: stage9)
    --force              キャッシュ無視で全再実行
    --dry-run            コスト見積りのみ(LLM呼び出しなし)

  inspect <OUT_DIR>      出力済みのmanifest/中間成果物を表示
  validate <JSONL>       生成JSONLのスキーマ検証
  estimate <VIDEO>       処理コスト見積り(秒数・API呼び出し回数・想定料金)

EXAMPLES:
  vidx process ./sample.mp4
  vidx process ./sample.mp4 --from stage4_semantic --force
  vidx process ./sample.mp4 --to stage3_coarse        # 粗分割までで停止
  vidx estimate ./sample.mp4
```

### 7.1 Stage名の正規化

CLI引数で使う Stage 識別子は manifest の key と同じ:
`stage0_probe, stage1_asr, stage2_aux, stage3_coarse, stage4_semantic, stage5_normalize, stage6_frames, stage7_vision, stage8_annotate, stage9_output`

---

## 8. 設定ファイル(`vidx.toml`)

```toml
[general]
out_dir = "./out"
log_level = "info"
parallelism = 4

[media]
ffmpeg_path = "ffmpeg"
ffprobe_path = "ffprobe"
audio_sample_rate = 16000

[segment.coarse]
max_duration_sec = 300        # 論点A
snap_window_sec = 30          # 補助信号へのスナップ許容範囲

[segment.semantic]
target_min_sec = 30           # 論点D
target_max_sec = 90           # 論点D
hard_min_sec = 15
hard_max_sec = 120

[frames]
periodic_interval_sec = 5     # 論点B
scene_change_threshold = 0.4
max_analyzed_per_chunk = 8
dhash_distance_threshold = 10

[asr]
adapter = "whisper_cpp"       # whisper_cpp | openai | groq
model = "large-v3"
language = "ja"

[asr.whisper_cpp]
binary_path = "whisper-cli"
model_path = "~/.cache/whisper.cpp/ggml-large-v3.bin"
threads = 8

[vision.ocr]
adapter = "tesseract"
languages = ["jpn", "eng"]

[vision.caption]
adapter = "claude"
model = "claude-sonnet-4-6"
max_per_chunk = 3

[llm]
adapter = "claude"
model = "claude-sonnet-4-6"
max_concurrency = 4
retry_max = 3
retry_backoff_ms = 1000

[output]
generate_markdown = true
generate_jsonl = true
embedding_text_fields = ["title", "summary", "transcript", "ocr_important"]  # 論点C

[secrets]
# 環境変数から読む前提。設定ファイルに直書きしない。
anthropic_api_key_env = "ANTHROPIC_API_KEY"
openai_api_key_env = "OPENAI_API_KEY"
```

**環境変数 > CLI flag > 設定ファイル > デフォルト** の優先順位で上書き。

---

## 9. ディレクトリ構造

### 9.1 リポジトリ(cargo workspace)

```
vidx/
├── Cargo.toml                     # workspace 定義
├── vidx.toml.example              # 設定ファイルサンプル
├── README.md
├── crates/
│   ├── vidx-core/                 # ドメイン型、エラー、設定ロード
│   │   ├── Cargo.toml
│   │   └── src/{lib.rs, model.rs, config.rs, error.rs, hash.rs}
│   ├── vidx-media/                # ffmpeg/ffprobe 呼び出し
│   │   └── src/{lib.rs, probe.rs, audio.rs, frames.rs, silence.rs, scene.rs}
│   ├── vidx-asr/                  # 文字起こし
│   │   └── src/{lib.rs, adapter.rs, whisper_cpp.rs, openai.rs}
│   ├── vidx-vision/               # OCR / VLM
│   │   └── src/{lib.rs, ocr.rs, caption.rs, tesseract.rs, claude_vlm.rs, dhash.rs}
│   ├── vidx-segment/              # 粗分割 / 意味分割 / 正規化
│   │   └── src/{lib.rs, coarse.rs, semantic.rs, normalize.rs}
│   ├── vidx-llm/                  # summary/title/keywords
│   │   └── src/{lib.rs, client.rs, prompts.rs, annotate.rs}
│   ├── vidx-output/               # JSONL / Markdown
│   │   └── src/{lib.rs, jsonl.rs, markdown.rs}
│   ├── vidx-pipeline/             # Stage trait / オーケストレーション / manifest
│   │   └── src/{lib.rs, stage.rs, manifest.rs, runner.rs}
│   └── vidx-cli/                  # CLI バイナリ
│       └── src/main.rs
├── prompts/
│   ├── semantic_chunking.jinja
│   └── annotate_chunk.jinja
└── tests/
    ├── fixtures/                  # 小さなサンプル動画、期待出力JSON
    └── integration/
```

### 9.2 出力(`--out-dir`配下)

```
out/
└── {video_id}/
    ├── manifest.json
    ├── probe.json
    ├── transcript.json
    ├── aux.json
    ├── coarse.json
    ├── semantic.json
    ├── normalized.json
    ├── frames.json
    ├── vision.json
    ├── annotated.json
    ├── {video_id}.chunks.jsonl      # ★ 最終成果物
    ├── {video_id}.index.json        # ★ 最終成果物
    ├── {video_id}.md                # ★ 最終成果物
    ├── audio/
    │   └── audio_16k.wav            # ASR 用一時ファイル(保持可)
    └── images/
        └── chunk_{NNNN}/
            ├── frame_01.jpg
            └── frame_02.jpg
```

---

## 10. 外部依存

### 10.1 ランタイム依存(システム側)

| ツール | 用途 | インストール確認コマンド |
|---|---|---|
| ffmpeg / ffprobe | 音声抽出、フレーム抽出、無音検出、scene change検出 | `ffmpeg -version`, `ffprobe -version` |
| whisper.cpp (既定) | 文字起こし | `whisper-cli --help` |
| tesseract (既定) | OCR | `tesseract --version` |

CLI 起動時に `pre-flight check` として存在確認を行い、不足していればエラーメッセージで案内。

### 10.2 Rust crates(主要)

```toml
[workspace.dependencies]
tokio        = { version = "1", features = ["full"] }
async-trait  = "0.1"
serde        = { version = "1", features = ["derive"] }
serde_json   = "1"
toml         = "0.8"
clap         = { version = "4", features = ["derive"] }
anyhow       = "1"
thiserror    = "1"
tracing      = "0.1"
tracing-subscriber = "0.3"
reqwest      = { version = "0.12", features = ["json", "rustls-tls"] }
sha2         = "0.10"
hex          = "0.4"
indicatif    = "0.17"
image        = "0.25"             # dHash 計算用
tera         = "1"                # プロンプトテンプレート(Jinja相当)
tokio-retry  = "0.3"
chrono       = { version = "0.4", features = ["serde"] }
```

### 10.3 API 依存(任意、アダプタ経由)

| API | 用途 | 必要な環境変数 |
|---|---|---|
| Anthropic Messages API | semantic chunking, annotate, VLM caption | `ANTHROPIC_API_KEY` |
| OpenAI API (代替) | ASR / LLM / VLM | `OPENAI_API_KEY` |
| Groq API (代替) | 高速 Whisper | `GROQ_API_KEY` |

---

## 11. 中間成果物と再実行性

- **すべての Stage は中間成果物を JSON として永続化**
- **`--from <stage>` で再開**可能、未指定時は manifest を見て未完了の最初の Stage から開始
- **`config_hash` が変わったら全Stage再実行**(設定変更 = キャッシュ無効化)
- **個別Stageだけ再実行** する `--only <stage>` オプションも提供(前後のStageは既存成果物を読み直す)

---

## 12. ログと観測性

- `tracing` による構造化ログ(JSON 出力オプション `--log-format=json`)
- 各 Stage 開始/終了で span を張り、duration とコスト(API呼び出し回数)を記録
- `--log-level=debug` で LLM プロンプト/レスポンスもダンプ(秘匿情報を含むため既定はoff)
- 進捗バーは `indicatif`(TTY時のみ)

---

## 13. エラーハンドリング

```rust
#[derive(Debug, thiserror::Error)]
pub enum VidxError {
    #[error("media error: {0}")]           Media(String),
    #[error("asr error: {0}")]              Asr(String),
    #[error("llm error: {0}")]              Llm(String),
    #[error("vision error: {0}")]           Vision(String),
    #[error("segment error: {0}")]          Segment(String),
    #[error("config error: {0}")]           Config(String),
    #[error("io error: {0}")]               Io(#[from] std::io::Error),
    #[error("serde error: {0}")]            Serde(#[from] serde_json::Error),
    #[error("external tool not found: {0}")]ToolNotFound(String),
}
```

- **リトライ対象**: 外部API(5xx, タイムアウト, rate limit)→ 指数バックオフ最大3回
- **リトライ対象外**: 設定エラー / スキーマ違反 / ファイル欠損
- **部分失敗の扱い**: 1チャンクの VLM caption が全失敗しても、そのチャンクは `visual_caption=null` で出力を継続

---

## 14. 非機能要件

| 項目 | 要件 |
|---|---|
| 処理対象長 | 単一動画の想定上限は **2時間**(これ以上は未保証) |
| 精度 vs コスト | **コスト優先**。ただし意味検索に耐える粒度は維持 |
| 並列性 | LLM/VLM 呼び出しは `max_concurrency` で制御、ffmpeg は逐次 |
| 冪等性 | 同一入力 + 同一 config → 同一出力(ただし LLM 応答の揺らぎはあり) |
| 再実行性 | 任意の Stage から再開可能 |
| プラットフォーム | Linux / macOS(x86_64, aarch64)。Windows は best-effort |
| メモリ | 2時間動画の処理で常駐メモリ 2GB 以下 |
| セキュリティ | API キーは必ず環境変数、ログには出さない |

---

## 15. 受け入れ基準 (Acceptance Criteria)

全体として、以下がすべて満たされること。

1. **E2E 成功**: サンプル10分動画(fixtures に同梱)を `vidx process` で処理し、chunks.jsonl / index.json / Markdown の3ファイルが生成される
2. **スキーマ検証**: 生成された全 JSONL 行が `vidx validate` で pass
3. **チャンク粒度**: 全チャンクが `hard_min_sec(15) <= duration <= hard_max_sec(120)` の範囲内
4. **時刻整合性**: 全チャンクで `start_sec < end_sec`、連続する2チャンクで重複なし(隣接チャンクはgapゼロを強制しない)
5. **キャッシュ**: 2回目の `vidx process` は全Stageがキャッシュヒットして LLM API を呼び出さない(`--dry-run` で確認可能)
6. **部分再実行**: `--from stage4_semantic` で Stage4 以降のみが実行される
7. **fail-soft**: 意図的に API キーを無効化した状態でも、caption/summary が null/fallback で埋まるだけで全体は完走する
8. **Markdown 可読性**: 生成 Markdown には時刻アンカー / 画像参照 / summary がすべて含まれる

---

## 16. テスト戦略

| 種類 | 対象 | 備考 |
|---|---|---|
| ユニット | `vidx-segment`(coarse/semantic/normalize), `vidx-core::hash`, `vidx-vision::dhash` | 外部依存なしで全網羅 |
| アダプタ契約 | 各 Adapter trait に対してモック実装 | trait レベルで挙動を固定 |
| ゴールデン | サンプル動画 → 期待 JSONL との差分比較(LLM出力部分は除外しスキーマ/時刻のみ検証) | `tests/fixtures/` |
| 統合 | `vidx process` の E2E(mock LLM adapter 使用) | CI で実行可能に |
| 負荷 | 2時間動画での完走確認 | nightly 手動 |

---

## 17. Claude Code 向け実装タスク分割

各タスクは **1セッション(=1 PR)で完結するサイズ** に切ってあり、依存関係を矢印で示す。
タスクID は `T-NN`。

### フェーズ1: 基盤(依存なし)

- **T-01**: `vidx-core` の model.rs(全構造体定義) + error.rs(VidxError) + hash.rs(sha256 ヘルパ) を実装。単体テストで各構造体の round-trip serialize/deserialize を確認。
- **T-02**: `vidx-core::config` で `vidx.toml` を読み込む。欠損キーは既定値、環境変数 > CLI flag > toml > default の優先順位。`config_hash()` も実装。
- **T-03**: `vidx-pipeline::stage::Stage` trait + `vidx-pipeline::manifest` を実装。ダミーStageで保存/読み込み/キャッシュヒット判定をテスト。

### フェーズ2: メディア層(T-01 依存)

- **T-04**: `vidx-media::probe`(ffprobe 呼び出しラッパ)。存在チェック含む。
- **T-05**: `vidx-media::audio`(mp4 → 16kHz mono wav 抽出)。
- **T-06**: `vidx-media::silence`(ffmpeg silencedetect のstderr パース)。
- **T-07**: `vidx-media::scene`(ffmpeg scene change 検出)。
- **T-08**: `vidx-media::frames`(指定秒数のフレーム抽出、バッチ実行)。

### フェーズ3: ASR / Vision 層(T-01, T-04, T-05 依存)

- **T-09**: `vidx-asr::adapter::AsrAdapter` trait + `whisper_cpp.rs` 実装。
- **T-10**: `vidx-vision::dhash` + `vidx-vision::ocr::tesseract.rs`。
- **T-11**: `vidx-vision::caption::claude_vlm.rs`(Anthropic API、base64 画像投入、リトライ)。

### フェーズ4: 分割ロジック(T-01, T-09 依存)

- **T-12**: `vidx-segment::coarse`(ハイブリッド粗分割)。補助信号無しでも動く純粋関数として実装、プロパティテストで不変条件を確認。
- **T-13**: `vidx-llm::client`(Anthropic Messages API クライアント、JSON mode、リトライ、同時実行制限)。
- **T-14**: `vidx-segment::semantic`(T-13 を使う、プロンプトは `prompts/semantic_chunking.jinja`)。スキーマ違反時のフォールバックも実装。
- **T-15**: `vidx-segment::normalize`(短すぎ結合 / 長すぎ再分割 / chunk_id 付与)。純粋関数 + 網羅テスト。

### フェーズ5: 注釈 + フレーム選別(T-10, T-11, T-13, T-15 依存)

- **T-16**: `vidx-media::frames` 呼び出し + dHash 選別で `ExtractedFrame` 群を決定する Stage6 実装。
- **T-17**: Stage7 実装(OCR 全対象、VLM caption は最大3枚/chunk)。
- **T-18**: Stage8 実装(title/summary/keywords を一括生成、fallback 含む)。

### フェーズ6: 出力 + オーケストレーション(全前段依存)

- **T-19**: `vidx-output::jsonl`(embedding_text の構築含む) + `vidx-output::markdown`(時刻アンカー付き)。
- **T-20**: `vidx-pipeline::runner`(Stage0〜9 を連結、キャッシュ判定、進捗表示)。
- **T-21**: `vidx-cli`(clap で全サブコマンド実装、pre-flight check 含む)。

### フェーズ7: 品質保証

- **T-22**: E2E テスト(`tests/integration/process_happy_path.rs`) — Mock LLM adapter を注入してサンプル動画を完走させる。
- **T-23**: ゴールデンテスト — 固定のサンプル動画 + Mock で JSONL 差分比較(LLM生成部は「存在確認」のみ)。
- **T-24**: `vidx validate` サブコマンドで JSON Schema バリデーション実装。

### 依存グラフ(簡略)

```
T-01 ──┬─→ T-02 ──→ T-03 ──→ T-20 ──→ T-21 ──→ T-22, T-23, T-24
       ├─→ T-04 ──→ T-05 ──→ T-09 ──→ T-14 ──→ T-15 ──→ T-18 ──→ T-19
       ├─→ T-06, T-07 ─────→ T-12 ─────↑        ↑
       └─→ T-08 ─────────→ T-16 ──→ T-17 ──────┘
                            ↑           ↑
                           T-10, T-11 ──┘
                                T-13 ──↑
```

### タスク実行時の Claude Code への指示テンプレ

Claude Code に投げるときの定型(タスクごとにコピペ想定):

```
@tasks/T-XX.md を読み、以下の制約で実装してください。

制約:
- crate: <該当crate>
- 既存コードを壊さない(他crateのpublic APIを変更しない)
- 追加 dependency が必要なら Cargo.toml に明記
- cargo fmt / cargo clippy --all-targets -- -D warnings を通す
- 単体テストを `#[cfg(test)]` で同ファイルに追加
- ドキュメントコメント(///)を pub 項目すべてに付ける

完了条件:
- cargo test -p <crate> が pass
- 受け入れ基準(spec §15 の該当項目)を満たす
```

---

## 18. 付録 A: サンプル LLM プロンプト(semantic chunking)

`prompts/semantic_chunking.jinja`:

```
あなたは動画字幕を意味単位に分割するアシスタントです。

【入力】以下はタイムスタンプ付き文字起こしです。
---
{% for seg in segments %}
[{{ seg.start_sec }}-{{ seg.end_sec }}] {{ seg.text }}
{% endfor %}
---

【タスク】
この区間を「話題転換」「手順切替」「質問→回答」「スライド節」「操作フェーズ変更」の観点で
意味単位に分割してください。1チャンクの目安は {{ target_min }}〜{{ target_max }}秒です。

【出力】厳密に以下のJSON配列のみを出力。前置きや説明は書かない。
[
  {"start_sec": <float>, "end_sec": <float>, "rationale": "<短い日本語>"}
]

制約:
- start_sec は昇順、重複なし
- 最初のチャンクの start_sec は {{ segments[0].start_sec }}
- 最後のチャンクの end_sec は {{ segments[-1].end_sec }}
- 隣接チャンクは end_sec == 次の start_sec
```

---

## 19. 付録 B: この仕様書で残る判断ポイント(将来バージョン)

現v3では**すべて既定値で確定**しているが、運用して以下が課題になる可能性がある:

1. **動画種別自動判定**: `content_type` の自動推定(現状は設定 or 固定 "unknown")
2. **話者ダイアライゼーション**: whisper.cpp 単体では弱い。pyannote 併用など
3. **長尺動画の分散処理**: 2時間超を想定した場合のチャンクジョブ分割
4. **インクリメンタル取り込み**: 同じ動画の別エンコードを処理したときの差分検出
5. **品質スコア**: 各チャンクに retrieval 向きの品質スコアを付与

これらは v4 以降の検討事項として Out of Scope。

---

**変更履歴**

- v1: 初期案(秒数ベース分割前提)
- v2: 階層的意味分割に方針転換、4論点を保留
- **v3 (本版)**: 4論点を既定値で確定、Rust workspace 構造・Stage trait・受け入れ基準・タスク分割を追加(実装投入版)

