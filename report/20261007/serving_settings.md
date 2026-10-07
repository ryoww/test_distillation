# モデル配信の設定（vLLM / llama.cpp）

下沢 亮太郎（2026-10-07）。正本は `docs/serving_runtimes.md` と `OPERATIONS.md`。

**原則**: OpenAI 互換 API で起動し HTTP で叩く。サーバー用と学習用の venv は分ける。待ち受けは 127.0.0.1、API キーは `local`。GPU は H200 × 2 を Slurm で共有し、ポートはジョブ ID から決める。adapter は焼き込んでから通常モデルとして配信する。

## モデルとランタイム

| モデル | ランタイム | 理由 |
|---|---|---|
| Qwen3.6 / 3.8 27B、Agents-A1-4B | vLLM 0.20（`test_DSPy/.runtime/vllm/vllm-cu13`） | 引き継ぎ時の構成。revision は `model_manifest.json` |
| Gemma 4 12B | vLLM 0.28（`.runtime/vllm-0.28`） | 0.20 に登録がない。起動時に FlashInfer が nvcc で JIT するため `CUDA_HOME` と curand ヘッダの指定が要る |
| GLM-5.3-Flash（320B、Q8 GGUF 347 GB） | llama.cpp（`.runtime/llama.cpp`） | vLLM 未対応。VRAM に載らないエキスパートを RAM に回す |

## vLLM の設定

| 用途 | 文脈長 | 出力枠 | その他 |
|---|---|---|---|
| 学習済みモデルの評価（`slurm_eval_solver.sbatch`） | 53,248 | 8,192 | GPU 1 枚、メモリ 0.85、同時 16、`--reasoning-parser gemma4` / `qwen3` |
| Qwen の思考あり評価 | 131,072 | 65,536 | 32k では思考が切れて本文が空になるため |
| GEPA（Qwen 生成 + Qwen 反省、`slurm_gepa_retrain.sbatch`） | 131,072 | 65,536 | GPU 1 枚ずつ、MTP 投機 1 token、DeepGEMM 無効 |
| GEPA（学習済み student + Qwen3.8、`slurm_gepa_student.sbatch`） | student 53,248 | 8,192 | `SHARE_GPU=1` で 1 枚に同居（メモリ 0.33 + 0.55）。student が載ってから反省側を起動 |

**思考テンプレートの整合**: `chat_template_kwargs.enable_thinking` で切り替える。Qwen 系と 4B の SFT モデルは **false**（学習データが空の思考ブロックを描画しているため。true だと枠切れ）。Gemma 4 の SFT モデルは学習・推論とも **true**（token 列が一致する）。

## llama.cpp の設定（GLM）

| 項目 | 値 | 理由 |
|---|---|---|
| `--fit`（既定）+ `--fit-target 8192` | 各 GPU に 8 GB の余白 | 余白 1 GB では ctx 49k の計算バッファで落ちた |
| `--n-cpu-moe` | 使わない | 明示すると GPU 間の配分も止まり、読み込みで落ちた |
| `--ctx-size` | 49,152（手動は 32,768） | KV を増やすほど GPU に置ける層が減る |
| `--parallel` | 1 | RAM 律速で並列しても合計がほぼ伸びない |
| `--load-mode none` | mmap しない | mmap だと前処理が 4 分の 1 に落ちる |
| 思考量 | `reasoning_effort` low / high / max | 既定は思考あり |

実測 14 tok/s、1 問 30〜40 分。教師には使わない。

## 落とし穴

| 症状 | 対処 |
|---|---|
| vLLM が OOM | 文脈長かメモリ率を下げる |
| 400 が全件 | `max_tokens` + 入力が文脈長を超えている。評価ジョブは起動前に止まる |
| SFT モデルが枠切ればかり | 思考テンプレートを学習時と揃える |
| GLM の起動が落ちる / 遅い | `--fit-target 8192`、`--n-cpu-moe` を使わない、GPU 2 枚を空けてから起動 |
| 7501 で bind 失敗 | 手動起動のサーバーが残っている。`ss -ltn` で確認 |
| 同じコードの正解判定が変わる | CPU の混雑で時間制限付きソルバーの解が落ちる。判定は GPU ジョブと重ねない |
