# CLAUDE.md — vididx 開発ガイド

## プロジェクト概要

ローカルmp4動画をRAG投入可能なJSONLチャンク群に変換するCLIツール。
仕様の全詳細は `SPEC.md` を参照。

- **言語**: Rust (stable, edition 2024)
- **非同期**: tokio
- **成果物**: `vididx` バイナリ + cargo workspace

---

## 開発原則: t-wada式TDD

**Red → Green → Refactor** サイクルを厳守する。

1. **Red**: 失敗するテストを先に書く。テストが失敗することを確認してから実装に入る
2. **Green**: テストを通す最小限の実装を書く。きれいさより動くことを優先
3. **Refactor**: テストが通ったままコードを整理する。重複排除・命名改善・抽象化

### テストの書き方

- テストは実装ファイルと同じファイルの `#[cfg(test)]` ブロックに書く
- テスト名は `test_<対象>_<条件>_<期待結果>` の形式(例: `test_coarse_segment_no_aux_signals_splits_at_300s`)
- 外部依存(ffmpeg, LLM API)はすべて trait + Adapter で抽象化し、テストではモックを使う
- 純粋関数はプロパティテストも書く(`proptest` クレートを活用)
- 「動作する仕様書」としてテストを書く。コメントで「何をテストしているか」ではなく、テスト名と構造で表現する

### タスクの進め方

各タスク(`tasks/T-XX.md`)は以下の順で進める:

```
1. tasks/T-XX.md を読む
2. 受け入れ基準を確認する
3. テストを書く(Red)
4. cargo test で失敗を確認する
5. 実装する(Green)
6. cargo test で全pass を確認する
7. cargo fmt && cargo clippy --all-targets -- -D warnings を通す
8. Refactorが必要なら行う(テストが通ったまま)
9. 完了
```

---

## Cargo Workspace 構成

```
vididx/
├── Cargo.toml            # workspace 定義
├── CLAUDE.md             # 本ファイル
├── SPEC.md               # 要件定義書(読み取り専用)
├── vididx.toml.example
├── crates/
│   ├── vididx-core/        # ドメイン型・エラー・設定・ハッシュ
│   ├── vididx-media/       # ffmpeg/ffprobe ラッパ
│   ├── vididx-asr/         # 文字起こし adapter
│   ├── vididx-vision/      # OCR / VLM caption
│   ├── vididx-segment/     # 粗分割・意味分割・正規化
│   ├── vididx-llm/         # LLM クライアント・注釈生成
│   ├── vididx-output/      # JSONL / Markdown 出力
│   ├── vididx-pipeline/    # Stage trait・manifest・runner
│   └── vididx-cli/         # バイナリエントリポイント
├── prompts/
│   ├── semantic_chunking.jinja
│   └── annotate_chunk.jinja
└── tasks/
    └── T-XX.md           # タスク定義(1タスク1ファイル)
```

---

## コーディング規則

### 全般

- `cargo fmt`(rustfmt デフォルト設定)を常に適用
- `cargo clippy --all-targets -- -D warnings` を常に通す
- コメントは「なぜ(Why)」だけ書く。「何を(What)」はコード自体で表現する
- `pub` な型・関数には `///` ドキュメントコメントを付ける

### エラーハンドリング

- クレート固有エラーは `thiserror` で定義
- 呼び出し側の集約は `anyhow` を使う
- `unwrap()` / `expect()` はテストコード以外禁止(`expect()` を使う場合でも理由を明記)

### 非同期・並列

- 非同期I/Oは `tokio`
- CPU重い処理は `tokio::task::spawn_blocking` または `rayon`
- LLM/VLM 呼び出しの並列度は `config.llm.max_concurrency` で制御

### 設計原則

- **外部依存は必ず trait + Adapter**で抽象化(ffmpeg, whisper.cpp, API等)
- **Stage間の受け渡しは型付き struct のみ**(HashMap / `serde_json::Value` の素通しは禁止)
- **各 Stage は idempotent**(同一入力ハッシュならキャッシュヒット)
- 依存方向: `vididx-cli` → `vididx-pipeline` → 各クレート → `vididx-core`

---

## 主要な外部ツール依存

| ツール | 確認コマンド |
|--------|-------------|
| ffmpeg / ffprobe | `ffmpeg -version` / `ffprobe -version` |
| whisper-cli (whisper.cpp) | `whisper-cli --help` |
| tesseract | `tesseract --version` |

CLI起動時に pre-flight check で存在確認する。

---

## 環境変数

APIキーは必ず環境変数から読む。設定ファイルや `git` に含めない。

```
ANTHROPIC_API_KEY
OPENAI_API_KEY
GROQ_API_KEY
```

---

## よく使うコマンド

```bash
# ビルド
cargo build --workspace

# テスト(全クレート)
cargo test --workspace

# 特定クレートのテスト
cargo test -p vididx-core

# lint
cargo clippy --all-targets -- -D warnings

# フォーマット
cargo fmt

# フォーマット確認のみ(CI向け)
cargo fmt -- --check
```

---

## タスク一覧(進捗)

各タスクの詳細は `tasks/T-XX.md` を参照。

この表はコード実態ベースで保守的に更新する。`完了` は少なくともクレート実装とテストが揃っているもの、`部分実装` はコードはあるが runner/CLI への接続や受け入れ基準が未達のもの、`未着手` は主経路に未接続のもの。

| ID | タイトル | フェーズ | 状態 |
|----|---------|---------|------|
| T-01 | vididx-core: model / error / hash | 1: 基盤 | ✅ 完了 |
| T-02 | vididx-core: config / config_hash | 1: 基盤 | ✅ 完了 |
| T-03 | vididx-pipeline: Stage trait / manifest | 1: 基盤 | ✅ 完了 |
| T-04 | vididx-media: ffprobe ラッパ | 2: メディア層 | ✅ 完了 |
| T-05 | vididx-media: 音声抽出 (16kHz mono wav) | 2: メディア層 | ✅ 完了 |
| T-06 | vididx-media: silencedetect | 2: メディア層 | ✅ 完了 |
| T-07 | vididx-media: scene change 検出 | 2: メディア層 | ✅ 完了 |
| T-08 | vididx-media: フレーム抽出 | 2: メディア層 | ✅ 完了 |
| T-09 | vididx-asr: adapter + whisper_cpp | 3: ASR/Vision層 | ✅ 完了 |
| T-10 | vididx-vision: dHash + tesseract | 3: ASR/Vision層 | ✅ 完了 |
| T-11 | vididx-vision: Claude VLM caption | 3: ASR/Vision層 | ⚠️ タスク定義なし / コードのみ |
| T-12 | vididx-segment: 粗分割(coarse) | 4: 分割層 | ✅ 完了 |
| T-13 | vididx-llm: LLM クライアント | 4: 分割層 | ✅ 完了 |
| T-14 | vididx-segment: 意味分割(semantic) | 4: 分割層 | ⚠️ 部分実装 |
| T-15 | vididx-segment: 正規化(normalize) | 4: 分割層 | ✅ 完了 |
| T-16 | Stage6: フレーム抽出・選別 | 5: パイプライン統合 | ⚠️ 部分実装 |
| T-17 | Stage7: OCR / VLM | 5: パイプライン統合 | ❌ 未着手 |
| T-18 | Stage8: 注釈生成 | 5: パイプライン統合 | ⚠️ 部分実装 |
| T-19 | vididx-output: JSONL / Markdown | 6: 出力層 | ✅ 完了 |
| T-20 | vididx-pipeline: runner | 6: 出力層 | ⚠️ 部分実装 |
| T-21 | vididx-cli: バイナリ | 6: 出力層 | ⚠️ 部分実装 |
