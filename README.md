# Small-model sequence distillation

`r0b0tlab/qwen3.8-max-glm5.2-distillation-51389` の教師応答を使い、
`InternScience/Agents-A1-4B` を assistant-only LoRA/QLoRA で SFT するための
小規模な実験環境です。教師モデルを同時に起動するオンライン蒸留ではなく、
保存済みの教師応答を学習する sequence-level distillation です。

## 重要な制約

- データセットは混合ライセンスで、公開カード上の用途は
  **controlled, noncommercial research only** です。商用利用や再配布の前に
  `LICENSE` と行ごとの provenance を確認してください。
- validation/test は benchmark 由来の development holdout であり、汚染のない
  能力評価には使えません。最終評価には独立した非重複データが必要です。
- 初回レシピは `sft_final` の text-only 行（tool 定義のない行）を使います。
  tool-use 蒸留はテンプレートとツール実行環境を別途検証してから追加してください。

## ディレクトリ

すべてリポジトリ相対が既定値です。別ストレージを使う場合だけ環境変数で
上書きします。

| 用途 | 既定値 | 上書き |
|---|---|---|
| モデル/Hugging Face cache | `./model/` | `DISTILL_MODEL_DIR` |
| データセット cache | `./data/` | `DISTILL_DATA_DIR` |
| 学習結果 | `./outputs/` | `DISTILL_OUTPUT_DIR` |
| 一般 cache | `./.cache/` | `DISTILL_CACHE_DIR` |
| venv | `./.runtime/` | `DISTILL_RUNTIME_DIR` |

## 1. 学習環境

CUDA 13 + H200 用の独立した venv を作ります。CUDA拡張をビルドする場合は、
先にプロジェクトローカルの `nvcc` を導入します。どちらも `sudo` は不要です。

```bash
uv run scripts/bootstrap_cuda_env.py
uv run scripts/bootstrap_train_env.py
uv run scripts/install_fast_kernels.py
```

導入先は `./.runtime/cuda-13.0/` で、現在のサーバーには CUDA
`13.0.48` を固定しています。ビルド時は次のように指定します。
`install_fast_kernels.py` は Agents-A1 用の `causal-conv1d` と
`flash-linear-attention` をソースビルドします。初回は十数分かかりますが、
以後の学習では再コンパイルされません。

```bash
export CUDA_HOME="$PWD/.runtime/cuda-13.0"
export PATH="$CUDA_HOME/bin:$PATH"
```

### 動作確認済みの高速カーネル構成

2026-07-29 に、H200 NVL上で以下の構成を確認済みです。

| コンポーネント | バージョン |
|---|---|
| CUDA compiler | 13.0.48 |
| PyTorch | 2.13.0+cu130 |
| causal-conv1d | 1.6.2.post1 |
| flash-linear-attention | 0.5.2 |

`causal-conv1d` はソース配布からビルドします。NVIDIAのCondaパッケージでは
CUDAヘッダーとライブラリが `targets/x86_64-linux/` 以下に置かれるため、
通常の `CUDA_HOME` だけでは `cuda_runtime_api.h` を見つけられない場合が
あります。`install_fast_kernels.py` は `CPATH`、`LIBRARY_PATH`、
`LD_LIBRARY_PATH` を自動設定してこの差を吸収します。

1サンプル、2,048 tokens上限、LoRA、gradient accumulation 1のsmoke testでは、
PyTorch fallbackの約6秒に対し、高速カーネル導入後の `train_runtime` は
4.607秒でした。約23%の短縮ですが、サンプル数が1件だけの参考値です。
実際の改善率は系列長、padding、batch size、gradient accumulation、
データ読み込み時間によって変わるため、pilot全体のthroughputでも比較してください。

## 2. データと tokenizer の検証

最初に少数行を再レンダリングし、assistant 部分だけに label が付くことを確認します。
このコマンドはモデル本体を読み込みません。

```bash
.runtime/train/bin/python scripts/validate_dataset.py --limit 32
```

## 3. 1 step smoke training

GPU 0 だけを使います。初回は BF16 LoRA が最も単純で、H200 141GB には十分収まります。

```bash
CUDA_VISIBLE_DEVICES=0 .runtime/train/bin/python scripts/train_distillation.py \
  --max-train-samples 32 \
  --max-eval-samples 8 \
  --max-length 2048 \
  --max-steps 1 \
  --run-name smoke-agents-a1-4b
```

成功すると adapter は
`outputs/smoke-agents-a1-4b/adapter/`、実験条件は
`outputs/smoke-agents-a1-4b/run_config.json` に保存されます。

## 4. Pilot 学習

Agents-A1 の Qwen3.5 linear-attention 学習は、上記の高速カーネルを利用します。
カーネルを入れない構成では Transformers の PyTorch fallback に切り替わります。
いきなり46,250件を回さず、まず5,000件・2,048 tokensで品質と所要時間を
測ってください。長すぎる例は黙って切らずに除外し、件数を
`dataset_stats.json` に記録します。

```bash
CUDA_VISIBLE_DEVICES=0 .runtime/train/bin/python scripts/train_distillation.py \
  --max-train-samples 5000 \
  --max-eval-samples 256 \
  --max-length 2048 \
  --num-train-epochs 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 2e-5 \
  --run-name agents-a1-4b-sft-final-pilot
```

VRAMをさらに節約する場合は `--load-in-4bit` を加えます。H200 では BF16 LoRA の
ほうが単純かつ高速なので既定値にはしていません。

pilot の結果を確認後、`--max-train-samples` を外し、必要なら
`--max-length 4096` に上げると全件学習になります。

フルファインチューニングの時間見積もりは、LoRAとは別に
`--full-finetune --max-steps 3` で測定できます。H200・最大長2048・勾配蓄積16の
10 optimizer steps（160例）の実測では計算部分が約7.1〜8秒/step、前処理・保存・
評価込みで183秒でした。全学習可能39,788例（2,487 steps）では、保存処理を含めた
概算5時間半〜6時間です。ベンチマークは
`outputs/agents-a1-4b-fullft-benchmark-10steps/` に保存されています。

## 5. vLLM で adapter を配信

`~/test_DSPy` と同じく、vLLM は学習環境と分離した venv で動かします。
vLLM/llama.cpp/Ollama の運用手順（バージョン選定、起動、疎通確認、障害切り分け）は
[`docs/serving_runtimes.md`](docs/serving_runtimes.md) にまとめています。
蒸留前後のモデル応答差の実例と、学習に使ったデータセットの構造・train/val/test
の代表例は [`docs/distillation_examples.md`](docs/distillation_examples.md)
にあります。

```bash
uv run scripts/bootstrap_vllm_env.py

CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_vllm.py \
  --adapter-path outputs/agents-a1-4b-sft-final-pilot/adapter \
  --served-model-name agents-a1-4b-distilled \
  --max-model-len 4096
```

このサーバー向けの起動スクリプトは、`nvcc` を要求しない Triton GDN prefill と、
利用不能な DeepGEMM warmup の無効化を既定値にしています。初回だけ
`torch.compile` と CUDA graph capture に数分かかります。

疎通確認:

```bash
curl http://127.0.0.1:7501/v1/chat/completions \
  -H 'Authorization: Bearer local' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "agents-a1-4b-distilled",
    "messages": [{"role": "user", "content": "Explain why ocean currents affect climate."}],
    "temperature": 0,
    "max_tokens": 256
  }'
```

adapter を指定しない場合はベースモデルを配信するため、同じ評価セットで
base と distilled adapter を比較できます。
