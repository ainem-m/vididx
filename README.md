# vididx

`vididx` は、ローカルの mp4 動画を中心に、補助的に `yt-dlp` で解決可能な動画 URL を、RAG 向けの JSONL チャンク群と人間可読な Markdown に変換する Rust 製 CLI です。

ASR・シーン変化検出・粗分割・意味分割・正規化・出力生成を cargo workspace として分離し、将来的な外部 API / ローカルツール差し替えを前提に設計しています。

## 何ができるか

- ローカル mp4 を入力にして retrieval-ready な JSONL を生成
- 補助機能として `yt-dlp` で解決可能な動画 URL も処理可能
- チャンクごとの Markdown サマリを生成
- `ffprobe` によるメディア解析
- `ffmpeg` による音声抽出、無音検出、シーン変化検出、フレーム抽出
- `whisper.cpp` アダプタ経由の文字起こし
- 粗分割、意味分割、長さ正規化
- manifest ベースのステージ記録

## 現状のステータス

このリポジトリは公開可能な形に整理中ですが、現時点ではまだフル機能版ではありません。

- 実装済み: `vididx-core`, `vididx-media`, `vididx-asr`, `vididx-output`, `vididx-segment` の主要機能
- 実装済み: `vididx-cli` の基本サブコマンド (`process`, `inspect`, `validate`, `estimate`)
- 実装済み: pipeline runner の基本経路、OCR 統合、vision enrich の fail-soft
- 部分実装: semantic chunking の LLM 統合、注釈生成、`yt-dlp` 経由の URL 入力
- 未完了: VLM の本統合、CLI の pre-flight check、`--from/--to` の段階実行

現状の `run_pipeline` は一連の出力を生成できますが、一部ステージはヒューリスティックまたは fallback 実装です。

## インストール

Rust stable が必要です。edition は `2024` です。

```bash
cargo build --workspace
```

CLI バイナリは以下で実行できます。

```bash
cargo run -p vididx-cli -- --help
```

## 依存ツール

最低限、以下の外部ツールが必要です。

- `ffmpeg`
- `ffprobe`
- `whisper-cli` (`whisper.cpp`)

OCR / VLM 統合や `yt-dlp` 経由の URL 入力では以下も利用します。

- `tesseract`
- `yt-dlp` (`media.url_downloader = "yt-dlp"` を設定した場合のみ)
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `GROQ_API_KEY`

## クイックスタート

```bash
cargo run -p vididx-cli -- process ./sample.mp4
```

`yt-dlp` で解決可能な URL を使う場合:

```bash
cargo run -p vididx-cli -- process "https://example.com/path/to/video"
```

出力先を指定する場合:

```bash
cargo run -p vididx-cli -- process ./sample.mp4 --video-id demo --output ./output/demo
```

生成結果の確認:

```bash
cargo run -p vididx-cli -- inspect ./output/demo
cargo run -p vididx-cli -- validate ./output/demo/demo.chunks.jsonl
```

見積もり:

```bash
cargo run -p vididx-cli -- estimate ./sample.mp4
```

`yt-dlp` で解決可能な URL を見積もる場合:

```bash
cargo run -p vididx-cli -- estimate "https://example.com/path/to/video"
```

## サブコマンド

```text
vididx process <VIDEO> [--video-id ID] [--from STAGE] [--to STAGE] [--force] [--output DIR]
vididx inspect <OUT_DIR>
vididx validate <JSONL>
vididx estimate <VIDEO>
```

注意:

- `--from` / `--to` は CLI 引数としては存在しますが、現状の実装では pipeline に未接続です
- 設定ファイルの明示指定オプションはまだありません。`Config::load(None)` により `./vididx.toml` または `~/.config/vididx/config.toml` を自動読込します
- URL 入力は既定では無効です。`media.url_downloader = "yt-dlp"` を設定した場合のみ有効になります
- URL 入力は `yt-dlp` で解決可能な URL と単純な直動画リンクのみを対象とします

## 出力物

`process` 実行後、通常は出力ディレクトリ配下に以下が生成されます。

- `manifest.json`
- `stage0_probe.json` から `stage9_annotate.json`
- `images/<video_id>/...`
- `<video_id>.chunks.jsonl`
- `<video_id>.index.json`
- `<video_id>.md`

JSONL の各行はチャンク単位のレコードで、少なくとも以下のような情報を持ちます。

- `chunk_id`
- `start_sec`, `end_sec`
- `title`, `summary`, `keywords`
- `transcript`
- `embedding_text`
- `image_refs`
- `processing_meta`

## 設定

設定は TOML で上書きできます。現在は以下の探索順です。

1. `./vididx.toml`
2. `~/.config/vididx/config.toml`
3. どちらも無ければデフォルト設定

最小例:

```toml
[media]
ffmpeg_path = "ffmpeg"
ffprobe_path = "ffprobe"
url_downloader = "yt-dlp"

[asr]
adapter = "whisper_cpp"
language = "ja"

[asr.whisper_cpp]
binary_path = "whisper-cli"
model_path = "/path/to/ggml-large-v3.bin"
threads = 8

[segment.coarse]
max_duration_sec = 300.0
snap_window_sec = 30.0

[frames]
periodic_interval_sec = 5.0
scene_change_threshold = 0.4
```

URL 入力について:

- `media.url_downloader = "yt-dlp"` を設定すると `process <URL>` と `estimate <URL>` が有効になります
- 取得時はまず `yt-dlp` を試します
- `yt-dlp` が失敗した場合、URL が直動画リンクっぽいか `HEAD` が `video/*` を返すときだけ `direct-http` に fallback します
- 一般の HTML ページ解析やサイト個別対応は行いません
- つまり、対応範囲は `yt-dlp` で解決できる URL と単純な直動画リンクに限られます

設定例ファイルは [vididx.toml.example](/Users/ainem/vididx/vididx.toml.example) に置いてあります。

API キーは設定ファイルではなく環境変数から読みます。

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GROQ_API_KEY=...
```

## 開発

このプロジェクトは t-wada 式 TDD を前提に進めています。詳細は [AGENTS.md](/Users/ainem/vididx/AGENTS.md) と [SPEC.md](/Users/ainem/vididx/SPEC.md) を参照してください。

よく使うコマンド:

```bash
cargo test --workspace
cargo fmt
cargo clippy --all-targets -- -D warnings
```

2026-04-20 時点で `cargo test --workspace` は全件 pass しています。

## Workspace

```text
crates/
  vididx-core      ドメイン型、設定、エラー、ハッシュ
  vididx-media     ffmpeg / ffprobe ラッパ
  vididx-asr       ASR adapter
  vididx-vision    OCR / caption / frame enrich
  vididx-segment   粗分割、意味分割、正規化、fallback 注釈
  vididx-llm       LLM client
  vididx-output    JSONL / Markdown 出力
  vididx-pipeline  stage / manifest / runner
  vididx-cli       バイナリエントリポイント
```

## ライセンス

MIT
