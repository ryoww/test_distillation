# モデル配信の設定まとめ（vLLM と llama.cpp）

下沢 亮太郎（2026-10-07 作成）

評価・GEPA・学習済みモデルの配信で使っている vLLM と llama.cpp の設定を、用途別にまとめた。手順の正本は `docs/serving_runtimes.md`（ルート）と `ProgAndSpec/DSPy_Shizuoka_V3_handover/OPERATIONS.md` にあり、ここは「どの場面で何をどう設定しているか」の一覧である。

## 1. 共通の原則

- モデルは Python プロセスに直接ロードせず、**OpenAI 互換 API** として起動し、評価スクリプトや DSPy は HTTP で叩く。
- サーバー用の venv と、学習・クライアント用の venv は分ける（vLLM は CUDA ドライバと Torch wheel の相性が強い）。
- モデル・キャッシュ・ランタイムはリポジトリ相対が既定。外部ディスクは `DISTILL_*` 環境変数で指す。
- API キーは `local` 固定、待ち受けは `127.0.0.1` のみ。秘密情報はコマンドやログに書かない。
- GPU は H200 × 2（各 141 GB）を Slurm で共有。1 ジョブ 1 GPU を基本にし、ポートはジョブ ID から決めて衝突を避ける。

## 2. どのモデルを、どのランタイムで配信しているか

| モデル | ランタイム | 環境 | 理由 |
|---|---|---|---|
| Qwen3.6-27B、Qwen3.8-27B（GEPA の生成・反省、評価） | vLLM 0.20.0 | `/home/yy-lab/test_DSPy/.runtime/vllm/vllm-cu13`（Python 3.13、CUDA 13） | 引き継ぎ時からの構成。`model_manifest.json` に revision を固定 |
| Gemma 4 12B（素・LoRA 焼き込み・全層学習） | vLLM 0.28.0 | `.runtime/vllm-0.28` | Gemma 4 は encoder なしの統合アーキテクチャで、0.20 に登録がない |
| Agents-A1-4B（素・LoRA 焼き込み） | vLLM 0.20.0 | Qwen と同じ環境 | Qwen3.5 系なので 0.20 で動く |
| GLM-5.3-Flash（320B MoE、Q8 GGUF 347 GB） | llama.cpp `llama-server` | `.runtime/llama.cpp`（自前ビルド、CUDA 13） | vLLM 0.28 が未対応。VRAM に収まらないエキスパートを RAM に回せる |

学習した adapter は vLLM の LoRA 配信を使わず、`scripts/merge_adapter.py` でベースに焼き込んでから通常モデルとして配信する。vLLM の LoRA 対応は対象モジュールに制約があり（Agents-A1-4B の線形 attention など）、焼き込めば配信経路が 1 つで済む。

## 3. vLLM の設定（用途別）

すべて `scripts/serve_vllm.py` 経由で起動する。ラッパーが知らない引数はそのまま `vllm serve` に渡る。

| 用途 | スクリプト | モデル / GPU | 文脈長 | 出力枠 | その他 |
|---|---|---|---|---|---|
| 学習済みモデルの評価 | `slurm_eval_solver.sbatch` | 1 モデル / GPU 1 枚 | `--max-model-len 53248`（大規模問題の入力が最長約 15k token） | 8,192 | `--gpu-memory-utilization 0.85`、`--max-num-seqs 16`、同時 8 リクエスト、`--reasoning-parser gemma4` または `qwen3` |
| Qwen の評価（思考あり） | 同上 | 1 モデル / GPU 1 枚 | 131,072 | 65,536 | 思考が 32k を超えて本文が空になる問題があったため（9 月）、既定を 65k に |
| GEPA（Qwen 生成 + Qwen 反省） | `slurm_gepa_retrain.sbatch` | 生成 GPU 0、反省 GPU 1 | 131,072 | 65,536 | `--speculative-config {"method":"mtp","num_speculative_tokens":1}`（MTP 投機）、`VLLM_USE_DEEP_GEMM=0` ほか DeepGEMM を切る |
| GEPA（学習済み student + Qwen3.8 反省） | `slurm_gepa_student.sbatch` | 既定は GPU 2 枚。`SHARE_GPU=1` で 1 枚に同居 | student 53,248、反省 131,072 | student 8,192 | 同居時は student `--gpu-memory-utilization 0.33`、反省 `0.55`。12B（約 24 GB）と 27B（約 52 GB）が 1 枚に収まる。student が載り切ってから反省側を起動する（vLLM は起動時に空きメモリを確かめるため） |

ポートは評価が `7700 + ジョブID % 100`、GEPA の student が `7600 + ジョブID % 50`、反省がその +50。手動起動の 7501 と衝突しない。

### 思考（thinking）の制御とテンプレートの整合

- 思考の有無は chat template の `enable_thinking` を `extra_body.chat_template_kwargs` で渡して切り替える。
- **Qwen3 系・Agents-A1-4B の SFT モデルは `enable_thinking: false` で配信する。** 学習データは assistant の先頭に空の思考ブロックを描画しているので、既定（思考あり）で推論すると学習で見ていない状態から思考を始め、枠を使い切る（9 月、140 問中 56 問が枠切れ）。
- **Gemma 4 の SFT モデルは学習・推論とも `enable_thinking: true`。** Gemma のテンプレートは true のとき system に `<|think|>` を立てて assistant 直前に何も挟まず、学習で見た token 列と推論の入力が一致する。素の Gemma を思考なしで測るときだけ false。
- `--reasoning-parser`（`gemma4` / `qwen3`）で思考を `reasoning_content` に分離し、コード抽出を汚さない。

### vLLM 0.28（Gemma 4）だけに要る準備

- FlashInfer が起動時に sampling カーネルを nvcc で JIT する。PATH の nvcc はヘッダを持たないラッパーなので、`CUDA_HOME=.runtime/cuda-13.0` を明示する。
- conda 版の toolkit に `curand*.h` がないため、torch が持ち込む pip の CUDA 13 ヘッダから curand だけをリンクしたディレクトリを `NVCC_APPEND_FLAGS=-I...` で足す。`lib64` → `lib` のシンボリックリンクも要る。
- 0.20 用の `--gdn-prefill-backend triton` は、nvcc なしでも起動できるようにする既定。

## 4. llama.cpp の設定（GLM-5.3-Flash）

`scripts/serve_llamacpp.py` 経由。評価は `slurm_eval_glm.sbatch`。

| 項目 | 値 | 理由 |
|---|---|---|
| モデル | `model/gguf/GLM-5.3-Flash-Q8_0-fp8src.gguf`（347 GB） | 公式 FP8 から自前変換。unsloth 系の GGUF は `glm5next` という旧アーキテクチャ名で読めない |
| GPU | 2 枚、`--fit`（既定 on） | 載り切らないエキスパートを自動で RAM に回す |
| `--fit-target 8192` | 各 GPU に 8 GB の余白を残す | 既定の 1 GB では、ctx 49152 の初回 decode で 4.9 GB の計算バッファが確保できず落ちた（job 784） |
| `--n-cpu-moe` | 使わない | 明示すると `--fit` の GPU 間配分も止まり、1 枚に 161 GB を割り当てて読み込みで落ちた（job 785） |
| `--ctx-size` | 評価 49,152（手動起動の既定は 32,768） | 入力約 15k + 思考と回答。KV を増やすほど GPU に置けるエキスパートが減って遅くなる |
| `--parallel` | 1 | 生成は RAM 上のエキスパート計算で律速され、4 本同時でも合計 15.6 → 18.2 tok/s とほぼ伸びない |
| `--load-mode none` | RAM に回す層を mmap せず読み込む | mmap だと前処理が 335 → 83 tok/s に落ちる |
| 思考量 | `reasoning_effort` を `extra_body` で low / high / max | 既定は思考あり。`max_tokens` が小さいと思考の途中で切れて本文が空になる |
| 読み込み時間 | 初回約 4 分半、ページキャッシュ有で 2〜3 分 | `/health` が `{"status":"ok"}` を返してから使う |

実測は生成約 14 token/秒。大規模問題は 1 問 30〜40 分かかり、3 問で正解 0 だったため教師には使わない（本編）。量子化を Q4 程度まで落とせば全部 GPU に載って数倍速くなる見込みはあるが、未実施。

## 5. 落とし穴と対処

| 症状 | 原因 | 対処 |
|---|---|---|
| vLLM が OOM で落ちる | 文脈長か同時数が大きい | `--max-model-len` を下げる、`--gpu-memory-utilization` を下げる。評価では入力 + 出力枠 + 4,096 が文脈長に収まるかを起動前に検査している |
| 400 が全リクエストで返る | `max_tokens` + 入力が `max_model_len` を超える | `MAX_MODEL_LEN` を上げる（評価ジョブは起動前に止まる） |
| SFT モデルが枠切ればかり | 学習時と推論時の思考テンプレートが不一致 | 上記のとおり Qwen 系は false、Gemma 4 は true に揃える |
| GLM の起動が落ちる | 計算バッファの余白不足、または `--n-cpu-moe` 明示 | `--fit-target 8192`、`--n-cpu-moe` は使わない |
| GLM が 15 tok/s より大きく遅い | 他プロセスが GPU を使い、RAM に回る層が増えた | GPU 2 枚とも空けてから起動。`-lv 4` で何層が RAM に回ったかを確認 |
| 7501 で `couldn't bind` | 手動起動の vLLM か前回の llama-server が残っている | `ss -ltn` で確認して止める。Slurm ジョブはジョブ ID からポートを決めるので衝突しない |
| 同じコードの正解判定が実行ごとに変わる | 時間制限付きソルバーが CPU の混雑で解の質を落とす | 判定を伴う処理（データ作成、参照解付け）は GPU ジョブと同時に流さない（`OPERATIONS.md` 13 章、`RESCORE_REPORT.md` 32.4 節） |
| 評価と学習が GPU 2 枚を取り合う | 2 枚要求のジョブは両方空くまで待つ | GEPA は `SHARE_GPU=1` で 1 枚に同居させる |

## 6. 参照

- `docs/serving_runtimes.md`: 環境構築、手動起動、疎通確認、障害切り分けの正本
- `ProgAndSpec/DSPy_Shizuoka_V3_handover/scripts/slurm_eval_solver.sbatch`、`slurm_gepa_retrain.sbatch`、`slurm_gepa_student.sbatch`、`slurm_eval_glm.sbatch`: 用途別の起動設定
- `ProgAndSpec/DSPy_Shizuoka_V3_handover/model_manifest.json`: Qwen の revision と配信設定
- `scripts/serve_vllm.py`、`scripts/serve_llamacpp.py`: ラッパーの既定値
