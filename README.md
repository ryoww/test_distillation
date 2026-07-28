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

CUDA 13 + H200 用の独立した venv を作ります。

```bash
uv run scripts/bootstrap_train_env.py
```

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

このサーバーには `nvcc` がなく、Agents-A1 の Qwen3.5 linear-attention 学習は
Transformers の PyTorch fallback を使います。いきなり46,250件を回さず、
まず5,000件・2,048 tokensで品質と所要時間を測ってください。長すぎる例は
黙って切らずに除外し、件数を `dataset_stats.json` に記録します。

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

## 5. vLLM で adapter を配信

`~/test_DSPy` と同じく、vLLM は学習環境と分離した venv で動かします。

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
