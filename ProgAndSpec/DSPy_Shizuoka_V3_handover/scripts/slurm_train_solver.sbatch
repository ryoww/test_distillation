#!/bin/bash
#SBATCH --job-name=train-solver
#SBATCH --partition=p1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#
# 検証済み (問題文, コード) 対で Agents-A1-4B を LoRA 学習し、この問題集専用の solver にする。
# データは scripts/build_sft_dataset.py が data/sft/ に書いたもの。学習はルートの
# .runtime/train 環境と scripts/train_distillation.py を使う。
#
#   cd ProgAndSpec/DSPy_Shizuoka_V3_handover
#   sbatch scripts/slurm_train_solver.sbatch
#
# 上書きできる環境変数: RUN_NAME, DATASET_DIR, EPOCHS, LR, LORA_R, MAX_LENGTH, EXTRA_ARGS
set -euo pipefail

HANDOVER_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
ROOT_DIR="$(cd "$HANDOVER_DIR/../.." && pwd)"
RUN_NAME="${RUN_NAME:-solver-agents-a1-4b-lora-$(date +%Y%m%d)}"
DATASET_DIR="${DATASET_DIR:-$HANDOVER_DIR/data/sft}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
LORA_R="${LORA_R:-32}"
MAX_LENGTH="${MAX_LENGTH:-6144}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
PYTHON="$ROOT_DIR/.runtime/train/bin/python"

[[ -x "$PYTHON" ]] || { echo "training env missing: $PYTHON (run uv run scripts/bootstrap_train_env.py at root)" >&2; exit 2; }
[[ -f "$DATASET_DIR/train.jsonl" && -f "$DATASET_DIR/validation.jsonl" ]] || { echo "dataset missing in $DATASET_DIR" >&2; exit 2; }

echo "[$(date +%T)] job $SLURM_JOB_ID on $(hostname); GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "[$(date +%T)] run=$RUN_NAME data=$DATASET_DIR epochs=$EPOCHS lr=$LR lora_r=$LORA_R max_length=$MAX_LENGTH"
echo "[$(date +%T)] train rows: $(wc -l < "$DATASET_DIR/train.jsonl"), validation rows: $(wc -l < "$DATASET_DIR/validation.jsonl")"

cd "$ROOT_DIR"
export CUDA_HOME="$ROOT_DIR/.runtime/cuda-13.0"
export PATH="$CUDA_HOME/bin:$PATH"
# Why not sacrifice alpha: r と alpha を 1:2 に保ち、小さなデータでも学習率を上げすぎない。
"$PYTHON" scripts/train_distillation.py \
    --dataset-dir "$DATASET_DIR" \
    --run-name "$RUN_NAME" \
    --max-length "$MAX_LENGTH" \
    --max-eval-samples 512 \
    --lora-r "$LORA_R" \
    --lora-alpha $((LORA_R * 2)) \
    --learning-rate "$LR" \
    --warmup-steps 30 \
    --num-train-epochs "$EPOCHS" \
    --gradient-accumulation-steps 8 \
    --save-steps 200 \
    $EXTRA_ARGS
echo "[$(date +%T)] done: $ROOT_DIR/outputs/$RUN_NAME/adapter"
