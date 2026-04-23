# vididx 改善計画 — レビュー完了版

## 優先度: Critical（本番動作を阻害）

### C1. パイプラインStage番号・名称がSPECと不一致
**ファイル**: `runner.rs:24-35`
**問題**: SPECは `stage0_probe → stage1_asr → stage2_aux → stage3_coarse → stage4_semantic → stage5_normalize → stage6_frames → stage7_vision → stage8_annotate → stage9_output`。実装は `stage0_probe → stage1_audio → stage2_asr → stage3_aux → stage4_coarse → stage5_frames → stage6_semantic → stage7_normalize → stage8_annotate → stage9_output` とズレている。audioはSPECでは独立Stageではない。
**修正**: SPEC準拠のStage名・番号に修正。`stage1_audio`を削除し、音声抽出をStage1(ASR)の内部ステップに組み込む。

### C2. 意味分割(Stage4)がLLMを使わずヒューリスティックのみ
**ファイル**: `semantic.rs`, `runner.rs:609-648`
**問題**: SPECの核心機能であるLLMベース意味分割が未実装。時間均等分割のみ。
**修正**: `semantic_chunk()`で`AnthropicClient`を呼び出し、`prompts/semantic_chunking.jinja`テンプレートを使用してLLM分割を実装。スキーマ違反時の1回リトライ + フォールバックも実装。

### C3. semantic_chunkのタイムスタンプ補正がない
**ファイル**: `semantic.rs:98-130`
**問題**: LLMが返す0ベース相対タイムスタンプをcoarse segmentの`start_sec`でオフセットしていない。
**修正**: `semantic_chunk()`の返り値の各セグメントに`coarse.start_sec`を加算する。

### C4. extract_transcript_segmentがスタブ
**ファイル**: `semantic.rs:176-178`
**問題**: `format!("Segment from {:.1}s to {:.1}s", ...)` というダミー文字列を返す。
**修正**: coarse segmentのtranscript_textをセグメント時刻範囲でフィルタリングして抽出するロジックを実装。

### C5. 粗分割のスナップ窓が広すぎる
**ファイル**: `coarse.rs:94-131`
**問題**: `snap_start = current_start` でセグメント全体がスナップ範囲になる。SPECは「境界付近±30秒」。
**修正**: `snap_start = segment_end - snap_window` に変更し、境界付近のみを候補範囲にする。

### C6. VLM captionアダプタがスタブ
**ファイル**: `caption.rs:48-51`
**問題**: `format!("Visual caption for {}", filename)` を返すだけでAPI呼び出しなし。
**修正**: Anthropic Messages APIを呼び出す実装。base64画像アップロード、リトライ(3回・指数バックオフ)、フォールバック(`visual_caption: None`)を実装。

### C7. FrameKind::UtteranceStartがPeriodicと同じserde名
**ファイル**: `model.rs:28-29`
**問題**: `#[serde(rename = "periodic")]` によりデシリアライズ時にUtteranceStartが不可到達。データ消失。
**修正**: `UtteranceStart`に独自のserde名 `"utterance_start"` を付与。`#![allow(unreachable_patterns)]`を削除。

### C8. sha256_fileがファイル全体をメモリ読込
**ファイル**: `hash.rs:20-21`
**問題**: 数GB動画でOOM。SPEC §14「常駐メモリ2GB以下」に違反。
**修正**: `std::io::BufReader` + `Sha256::update()` のストリーミングハッシュに変更。

### C9. annotationフォールバックがSPECと不一致
**ファイル**: `annotate.rs:65-88`
**問題**: SPECは「transcriptの先頭40文字→title、先頭150文字→summary」だが、実装は「最初の10単語→title、固定メッセージ→summary」。
**修正**: `transcript.chars().take(40)`, `transcript.chars().take(150)` に変更。

### C10. 日本語文字列のバイトスライスでパニック可能性
**ファイル**: `annotate.rs:73`
**問題**: `&default_title[..47]` はバイト単位スライスでマルチバイト文字境界でパニック。
**修正**: `.chars().take(40).collect::<String>()` に変更。

### C11. VididxErrorがnon_exhaustiveでない
**ファイル**: `error.rs:3-22`
**問題**: 下流クレートのmatchが破壊的変更を受ける。
**修正**: `#[non_exhaustive]` をenumに追加。

---

## 優先度: High（SPEC違反・設計上の重大問題）

### H1. vididx-mediaにtrait抽象化なし
**ファイル**: `lib.rs`, `probe.rs`, `audio.rs`, `silence.rs`, `scene.rs`, `frames.rs`
**問題**: 全関数がasync fnでffmpegを直接呼び出し、モック差し替え不可能。SPEC §4設計原則1違反。
**修正**: `MediaAdapter` traitを定義し、`FfmpegAdapter` とmock実装を提供。各関数をtraitメソッドに移行。

### H2. 同期ブロッキング呼び出しがasyncコンテキストで多点
**ファイル**: `probe.rs`, `audio.rs`, `silence.rs`, `scene.rs`, `whisper_cpp.rs`, `ocr.rs`
**問題**: `std::process::Command::output()` がtokioスレッドをブロック。
**修正**: `tokio::task::spawn_blocking` でラップする。

### H3. LLMクライアントが全エラーを無条件リトライ
**ファイル**: `client.rs:43-60`
**問題**: 401/400等も3回リトライ。SPEC §13は「5xx/タイムアウト/rate limitのみ」。
**修正**: ステータスコードを見て429/5xx/timeoutのみリトライ。

### H4. プロンプトテンプレートにタイムスタンプ付きセグメントが未送信
**ファイル**: `semantic.rs:96-100`
**問題**: `transcript_text`（フラット文字列）のみ渡す。LLMが時刻境界を推定できない。
**修正**: `TranscriptSegment`の配列をテンプレートに渡す。

### H5. 短いセグメントのLLM呼び出しスキップ未実装
**ファイル**: `semantic.rs:19-68`
**問題**: SPEC §5 Stage4の「30秒未満なら1チャンク」ロジックがない。
**修正**: coarse duration < 30秒の場合、LLM呼び出しをスキップしてそのまま1チャンク返す。

### H6. annotate_chunkがFrameAnalysisを受け取らない
**ファイル**: `annotate.rs:14-17`
**問題**: SPEC §8は「NormalizedChunk + FrameAnalysis[]」が入力。
**修正**: シグネチャに`frame_analyses: &[FrameAnalysis]`を追加し、プロンプトにOCR/caption情報を含める。

### H7. segment_idが常に空文字列
**ファイル**: `coarse.rs:48`
**問題**: SPECの`parent_segment_id`フォーマット(`vid_001_seg_0002`)が生成されない。
**修正**: `coarse_segment()`に`video_id`パラメータを追加し、`format!("{video_id}_seg_{index:04}")`でsegment_idを生成。

### H8. Stage traitが本番パイプラインで未使用
**ファイル**: `stage.rs:40-55`, `runner.rs`
**問題**: SPEC設計の核心であるStage traitがダミーテストのみ。
**修正**: runnerの各ステージをStage trait実装にリファクタリング。（大規模変更、Phase2で対応）

### H9. Stage7(OCR/VLM)がキャッシュ対応ステージとして扱われていない
**ファイル**: `runner.rs:1200-1257`
**問題**: manifestに個別ステージとして記録されず、再開不可。
**修正**: `enrich_frame_artifacts()`を`run_or_load_stage`で包む。

### H10. parallelism/max_concurrencyが使用されていない
**ファイル**: `config.rs:12`, `runner.rs:338-346`
**問題**: LLM呼び出しが逐次forループ。
**修正**: `tokio::sync::Semaphore`でmax_concurrency制御を実装。

### H11. 正規化マージが「短い方から」処理していない
**ファイル**: `normalize.rs:27-60`
**問題**: 左→右貪欲マージ。SPEC §5 Stage5は「短い方から処理」。
**修正**: durationでソートして短い方からマージするアルゴリズムに変更。

### H12. ffmpegに-nostdinフラグがない
**ファイル**: `audio.rs:27`, `silence.rs:21`, `scene.rs:20`, `frames.rs:73`
**問題**: stdin接続時にffmpegがハングする可能性。
**修正**: 全ffmpegコマンドに`-nostdin`を追加。

### H13. フレーム抽出が1フレーム1ffmpegプロセス
**ファイル**: `frames.rs:30-40`
**問題**: SPECは「チャンクごとにバッチ実行」。2時間動画で〜1440回のプロセス生成。
**修正**: ffmpegのFPSフィルタまたは`-vf select`で1回の呼び出しで複数フレーム抽出。

### H14. Scene changeのscoreが常に0.0
**ファイル**: `scene.rs:52,57`
**問題**: SPECは `SceneChange { at_sec, score }` を定義。
**修正**: ffmpegの`showinfo`出力からscore値をパース、または閾値をデフォルト値として設定。

### H15. 設定languageデフォルトが"auto"
**ファイル**: `config.rs:198`
**問題**: SPEC §8は `language = "ja"`。
**修正**: デフォルトを`"ja"`に変更。

### H16. reqwest TLS設定がworkspaceと不一致
**ファイル**: `vididx-llm/Cargo.toml:13`
**問題**: `features = ["json"]`のみ。workspaceは`["json", "rustls-tls"]`。
**修正**: workspace依存に合わせる。

### H17. LLMクライアントがmarkdown code fenceを除去しない
**ファイル**: `client.rs:110-119`
**問題**: LLMが`\`\`\`json ... \`\`\``で返した場合にパース失敗。
**修正**: 応答からcode fenceをstripする前処理を追加。

---

## 優先度: Medium（品質・保守性・SPEC準拠）

### M1. unwrap()/expect()がテスト外で使用
**ファイル**: `config.rs:483`, `normalize.rs:47`, `coarse.rs:126-127`
**修正**: AGENTS.md違反。適切なエラー伝播に変更。

### M2. 設定パースエラーをeprintlnで出力し無視
**ファイル**: `config.rs:458-463`
**修正**: `tracing::warn!` を使用し、重大なパースエラーはエラーで返す。

### M3. ドメイン型に不変条件検証メソッドがない
**ファイル**: `model.rs`
**修正**: `CoarseSegment::validate()`, `NormalizedChunk::validate()` 等を追加。

### M4. 設定値のバリデーションがない
**ファイル**: `config.rs`
**修正**: `Config::validate()` を追加（hard_min < hard_max 等）。

### M5. output::write_chunks_jsonlがAnnotatedChunkを引数に取る
**ファイル**: `output.rs:22-45`
**修正**: SPEC §6.1の完全なChunkスキーマを直列化するよう修正。

### M6. Markdownタイムスタンプがミリ秒を含まない
**ファイル**: `output.rs:153-170`
**修正**: `HH:MM:SS.mmm` 形式を使用。

### M7. OutputIndexがSPECのindex.jsonスキーマと不一致
**ファイル**: `output.rs:8-19`
**修正**: `manifest_version`と`stages`フィールドを追加。

### M8. Manifestキャッシュヒット時にファイル存在確認がない
**ファイル**: `manifest.rs:76-83`
**修正**: キャッシュヒット時にパスのファイル存在を検証。

### M9. --force実行時に古い成果物が削除されない
**ファイル**: `main.rs:192-197`
**修正**: 出力ディレクトリ内の古いファイルをクリーンアップ。

### M10. Stage9(出力)がrun_or_load_stageに包まれていない
**ファイル**: `runner.rs:354-416`
**修正**: 出力生成もキャッシュ可能ステージとして扱う。

### M11. OCR言語設定がハードコード["eng"]
**ファイル**: `enrich.rs:25`
**修正**: 設定の言語リストを使用。

### M12. VLM選別がScene Change優先でない
**ファイル**: `enrich.rs:65-79`
**修正**: `kind=SceneChange`を優先して選別するロジックを追加。

### M13. is_tool_availableが5ファイルにコピペ
**ファイル**: `probe.rs`, `audio.rs`, `silence.rs`, `scene.rs`, `frames.rs`
**修正**: 共通ユーティリティに抽出。

### M14. フレームの±1秒重複排除が未実装
**ファイル**: `frames.rs`
**修正**: 同一チャンク内で±1秒以内のフレームをマージするロジックを追加。

### M15. Regexが毎呼び出しで再コンパイル
**ファイル**: `silence.rs:43-47`, `scene.rs:42-45`
**修正**: `std::sync::LazyLock` でキャッシュ。

### M16. AnnotatedChunkにrationale/ocr_text/visual_captionが欠落
**ファイル**: `model.rs:205-214`
**修正**: フィールドを追加し、下流に伝播させる。

### M17. equal_divisionが全サブチャンクにcoarse全体のtranscriptをコピー
**ファイル**: `semantic.rs:198-199`
**修正**: 時間範囲でフィルタリングしたtranscriptを各チャンクに設定。

### M18. SPLIT_MIN/SPLIT_MAXがハードコード
**ファイル**: `normalize.rs:3-4`
**修正**: 設定のtarget_min_sec/target_max_secを使用。

### M19. extract_audioが同期FS操作をasync fn内で使用
**ファイル**: `audio.rs:21-24`
**修正**: `tokio::fs::create_dir_all` に変更。

### M20. 注釈プロンプトがSPECと不一致
**ファイル**: `prompts/annotate_chunk.jinja:8`
**修正**: title=40文字以内、keywords=3〜7個に修正。

### M21. CLI不足オプション
**ファイル**: `main.rs`
**修正**: `--dry-run`, `--config`, `--only`, `--log-level`, `--log-format` を追加。

### M22. validateサブコマンドがJSON構文チェックのみ
**ファイル**: `main.rs:245-275`
**修正**: SPEC §6.1のスキーマ検証を実装。

### M23. url_downloader/WixDownloadがSPEC外機能
**ファイル**: `config.rs:33`, `main.rs:31-97`, `wix.rs`
**修正**: 将来の削除または明確なドキュメント化を検討。

### M24. VlmCaptionAdapterにkindフィールドがなく優先選別が不可能
**ファイル**: `enrich.rs`
**修正**: `EnrichedFrame`に`kind`フィールドを追加。

### M25. extract_single_frameのquality計算がSPECと不一致
**ファイル**: `frames.rs:67-68`
**修正**: SPECの`quality=85` → `-q:v 3`に合わせるか、計算式を修正。

---

## 優先度: Low（改善推奨・コード品質）

### L1. Resolutionタプル(u32, u32)を名前付き構造体に
### L2. guess_content_typeのテスト追加
### L3. ModalityFlagsにDefault deriveを追加
### L4. duration_secのデフォルト0.0をエラーに変更
### L5. fpsのデフォルト0.0をエラーに変更
### L6. テストのenv::set_varをunsafeに（edition 2024）
### L7. toml/sha2/hex/dirs/chrono/tempfileをworkspace depsに統合
### L8. inspectサブコマンドの情報拡充
### L9. estimateサブコマンドの実装拡充
### L10. source_type "web_url"のSPEC外対応
### L11. parse_timecodeの可変長フォーマット対応
### L12. 末尾silence_startのログ出力
### L13. vididx-mediaにtracingログを追加
### L14. parallelism設定の使用
### L15. SegmentMode::Utterance/Chapterのドキュメント化
### L16. is_meaningful_ocr_text / is_ocr_noise_lineの統合

---

## 実装フェーズ計画

### Phase 1: Critical修正（C1〜C11）
1. C11 — `VididxError`に`#[non_exhaustive]`追加
2. C7 — `FrameKind::UtteranceStart`のserde名修正
3. C8 — `sha256_file`のストリーミング化
4. C5 — 粗分割スナップ窓の修正
5. C10 — 日本語バイトスライス→chars対応
6. C9 — annotationフォールバック修正
7. C1 — Stage番号・名称のSPEC準拠修正
8. C3 — semantic_chunkタイムスタンプ補正追加
9. C4 — extract_transcript_segment実装
10. C2 — LLMベース意味分割実装
11. C6 — ClaudeVlmAdapter実装

### Phase 2: High修正（H1〜H17）
12. H12 — ffmpegに-nostdin追加
13. H15 — languageデフォルトを"ja"に変更
14. H16 — reqwest TLS設定修正
15. H17 — LLM応答のcode fence除去
16. H3 — LLMリトライ条件分岐
17. H14 — Scene change scoreの修正
18. H7 — segment_id生成修正
19. H5 — 短いセグメントのLLMスキップ
20. H6 — annotate_chunkにFrameAnalysis追加
21. H4 — プロンプトテンプレートにセグメント配列渡し
22. H11 — 正規化マージアルゴリズム修正
23. H13 — フレーム抽出バッチ化
24. H2 — 同期ブロッキング呼び出しのspawn_blocking化
25. H9 — Stage7をキャッシュ対応ステージ化
26. H10 — LLM並列度制御実装
27. H1 — MediaAdapter trait導入（大規模変更）
28. H8 — Stage traitの本番利用（大規模変更）

### Phase 3: Medium修正（M1〜M25）
29. M13 — is_tool_available共通化
30. M15 — RegexのLazyLock化
31. M1 — unwrap/expect除去
32. M2 — 設定パースエラーのtracing化
33. M4 — Config::validate()追加
34. M3 — ドメイン型のvalidate()追加
35. M16 — AnnotatedChunkに欠落フィールド追加
36. M5 — output::write_chunks_jsonlのsig修正
37. M20 — 注釈プロンプトSPEC準拠
38. M6 — Markdownタイムスタンプ修正
39. M7 — OutputIndexスキーマ修正
40. M8 — Manifestファイル存在確認
41. M9 — --force時の古い成果物クリーンアップ
41. M10 — Stage9のキャッシュ対応化
43. M11 — OCR言語設定の動的化
44. M12 — VLM選別のScene Change優先
45. M24 — EnrichedFrameにkind追加
46. M14 — フレーム±1秒重複排除
47. M25 — フレームquality計算修正
48. M17 — equal_divisionのtranscript分割
49. M18 — SPLIT_MIN/SPLIT_MAXの設定駆動化
50. M19 — extract_audioのasync化
51. M21 — CLI不足オプション追加
52. M22 — validateサブコマンドのスキーマ検証化
53. M23 — SPEC外機能の整理
54. M29 — Resolution型の改善
55. M30 — guess_content_typeテスト追加
56. M31 — ModalityFlags Default derive
57. M32 — テストunsafe set_var対応

### 各タスク完了後の検証
- `cargo test --workspace`
- `cargo clippy --all-targets -- -D warnings`
- `cargo fmt`