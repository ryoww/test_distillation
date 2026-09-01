# DSPy_Shizuoka_V3 引き渡しマニフェスト（別部門・再現用）

> 現在のUbuntu/H200向け運用手順、Qwen3.6-27B + Qwen3.8-27B の2モデル構成、
> preflight、修正後の再評価手順は `OPERATIONS.md` を正とします。本書の旧Windows
> パスと旧単一endpointは、引き渡し時点の履歴情報です。

別部門が同じ実験（参照あり／参照フリーの学習・評価）を再現するために
**圧縮して送付すべきファイル一覧**です。仕様は `specification/DSPy_Shizuoka_V3_Specification.pdf` を参照。

---

## 1. ソースコード（必須）

| パス | 役割 |
|---|---|
| `train_gepa_v3.py` | 学習・評価エントリポイント（`--no-reference` で参照フリー） |
| `src/__init__.py` | パッケージ初期化 |
| `src/data_loader.py` | データ読込・層化分割（seed=42, 40/20）・Example 化 |
| `src/requirement_builder.py` | LLM 入力要件テキスト生成 |
| `src/signatures.py` | 3 つの Signature（初期指示文） |
| `src/modules.py` | `AlgorithmGenerator` パイプライン |
| `src/metrics_v3.py` | 評価メトリック本体（`compute_v3_score` 等） |
| `src/gepa_feedback_v3.py` | GEPA 用フィードバック生成（参照フリー制御含む） |
| `src/best_known.py` | 既知最良／セルフベースライン管理 |
| `src/lm_config.py` | Qwen3.6 generation / Qwen3.8 reflection のローカルendpoint設定 |
| `src/utils/__init__.py` | utils 初期化 |
| `src/utils/safe_exec.py` | AST制限付き別プロセス実行（multiprocessing spawn） |
| `src/utils/feasibility.py` | 制約充足検査 |
| `src/utils/scorer.py` | 目的値計算補助 |

### 補助スクリプト（任意・あると便利）
- `eval_gepa_v3.py`, `quick_eval_v3.py`, `train_eval_v3.py`

---

## 2. データ（必須）

| 内容 | 現在の場所 | 送付形態 |
|---|---|---|
| 問題 100 問 `prob_001.json` 〜 `prob_100.json` | `C:\Users\...\OptimizationDataCreater\data\` | `data/problems/` 等にまとめて同梱 |

> **注意**: 現在の作業環境には `data/problems/` として100問が揃っていますが、
> root `.gitignore` の `data/` 規則によりGit管理外です。ZIP等で必ず同梱し、
> `data_manifest.json` のSHA-256をpreflightで照合してください。

---

## 3. 学習済み成果物（任意：再学習せず評価だけ再現する場合）

| ファイル | 内容 |
|---|---|
| `compiled_program_v3_gepa_phaseE.json` | 参照ありモードの学習済みプログラム（進化後指示文 ~17KB を含む） |
| `compiled_program_v3_gepa_phaseF_noref.json` | 参照フリーモードの学習済みプログラム |
| `evaluation_results_v3_gepa_phaseE.json` | 参照ありモードの評価結果 |
| `evaluation_results_v3_gepa_phaseF_noref.json` | 参照フリーモードの評価結果 |
| `data/best_known.jsonl` | 参照ありモードの既知最良レジストリ |
| `data/best_known_noref.jsonl` | 参照フリーモードの既知最良レジストリ |

`train_gepa_v3.py --eval-only --program-path <上記 json>` で再評価可能。
成果物を省略して再学習する場合、事前検査は
`scripts/preflight.py --offline --skip-programs` を使用します。

---

## 4. ドキュメント（必須）

| ファイル | 内容 |
|---|---|
| `specification/DSPy_Shizuoka_V3_Specification.pdf` (+`.tex`) | **再現用仕様書**（アルゴリズム詳細・メトリック・手順・精度比較） |
| `Report/DSPy_Shizuoka_V3_Report1.pdf` (+`.tex`) | 開発経緯レポート（参照フリー検証含む） |
| `HANDOVER.md` | 本ファイル |

---

## 5. 送付から除外してよいもの（容量削減）

- `outputs/*/gepa_logs/`（GEPA 実行ログ、数百 MB）
- `phase_*_train.log`（学習の生ログ、合計 ~7 GB）
- `src/**/__pycache__/`
- `data/*.jsonl.pre_*`（バックアップ）
- 旧成果物: `compiled_program_v3_gepa.json`, `evaluation_results_v3*.json`(phase 無し),
  `compiled_prompts*.txt`, 調査用スクリプト
  (`analyze_results.py`, `check_ref.py`, `investigate_*.py`, `re_eval_*.py`)

---

## 6. 実行環境（受領側で用意）

- Python **3.11.15**（`.python-version`、`uv.lock` で固定）
- DSPy **3.3.0b1** / GEPA **0.1.1**
- ライブラリ: `ortools, scipy, pulp, networkx, numpy`
- generation: Qwen3.6-27B (`127.0.0.1:7501/v1`)
- reflection: Qwen3.8-27B (`127.0.0.1:7502/v1`)
- 環境変数 `PYTHONIOENCODING=utf-8`

環境構築は `uv sync --locked`、事前検査は
`uv run python scripts/preflight.py --offline` を実行してください。実運用コマンドは
`OPERATIONS.md` を参照します。
