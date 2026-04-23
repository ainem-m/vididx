# vididx 改善計画

## 優先度: 高

### 1. whisper.cpp `model_path` の `~` 展開 ✅
**問題**: `~/.cache/whisper.cpp/...` の `~` がシェル展開されずファイル不存在エラー
**修正**: `shellexpand` クレートを追加し、`model_path` と `binary_path` を展開

### 2. `fallback_annotate` の文字数カウントがバイト単位 ✅
**問題**: `words.len()` は `String` のバイト数。日本語UTF-8で1文字=3バイト
**修正**: `.chars().count()` と `.chars().take(40).collect()` を使う

### 3. `select_frame_timestamps` の `max_frames` が chunk 数を無視 ✅
**問題**: 動画全体で8枚しかフレームを取らない。長尺動画でフレーム不足
**修正**: `max_frames = max_analyzed_per_chunk * (duration / coarse_duration)` の最小保証

## 優先度: 中

### 4. `merge_toml` のパース失敗を黙殺 ✅
**問題**: 設定ミスがあっても無視される
**修正**: `try_into` の `Err` を `eprintln!` で報告

### 5. `FrameKind::UtteranceStart` が SPEC にない ✅
**問題**: JSONL出力に `"utterance_start"` が入る
**修正**: `#[serde(rename = "periodic")]` でマップ

### 6. `build_embedding_text` が `embedding_text_fields` 設定を無視 ✅
**問題**: `config.output.embedding_text_fields` をカスタマイズしても効かない
**修正**: 設定に基づいて動的に構築

### 7. `extract_single_frame` の `-ss` 配置 ✅
**問題**: `-ss` を `-i` 前に置くとキーフレームシークで近似値
**修正**: `-ss` を `-i` の後に配置（精度優先）

## 優先度: 低

### 8. `content_type` が常に `ScreenRecording` ✅
**問題**: 固定値
**修正**: 解像度・縦横比から推定（16:9→Lecture, 4:3→Slide 等）

### 9. `config_hash()` の `unwrap_or_default()` ✅
**問題**: serialize失敗時に空文字列のハッシュになる
**修正**: `expect("config serialize")` に変更

---

## 実装順序

1. 依存関係の追加（shellexpand）
2. whisper.cpp model_path 展開
3. fallback_annotate 文字数修正
4. select_frame_timestamps max_frames 修正
5. merge_toml 警告出力
6. FrameKind::UtteranceStart の JSON マッピング
7. embedding_text_fields 対応
8. extract_single_frame -ss 配置修正
9. content_type 自動推定
10. config_hash unwrap 修正
11. テスト実行・修正
12. cargo fmt / clippy
