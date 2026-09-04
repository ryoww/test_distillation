#!/bin/bash
#SBATCH --job-name=eval-generated-qwen38
#SBATCH --partition=p1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#
# Qwen3.8 を 1 プロンプトぶんだけ評価する GPU 1 枚のジョブ。Qwen3.8 は Qwen3.6 の
# 3 倍以上遅いので、改善前と改善後を別ジョブにして、空いた GPU から順に走らせる。
# PROMPT=before / after を --export で渡し、2 本投入する。
#
#   cd ProgAndSpec/DSPy_Shizuoka_V3_handover
#   export QWEN_HF_HOME=... QWEN_VLLM_ENV=... RUN_NAME=generated140-qwen38-YYYYMMDD
#   PROMPT=before sbatch --export=ALL scripts/slurm_eval_generated_qwen38.sbatch
#   PROMPT=after  sbatch --export=ALL scripts/slurm_eval_generated_qwen38.sbatch
#
# 結果は outputs/prompt_model_comparisons/${RUN_NAME}-${PROMPT}/ に shard 単位で出る。
set -euo pipefail

: "${QWEN_HF_HOME:?set QWEN_HF_HOME to the Hugging Face cache holding the Qwen models}"
: "${QWEN_VLLM_ENV:?set QWEN_VLLM_ENV to the vLLM virtualenv directory}"
: "${PROMPT:?set PROMPT to before or after}"

HANDOVER_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
ROOT_DIR="$(cd "$HANDOVER_DIR/../.." && pwd)"
RUN_NAME="${RUN_NAME:-generated140-qwen38-$(date +%Y%m%d)}"
DATA_DIR="${DATA_DIR:-data/problems_generated}"
SHARDS="${SHARDS:-4}"
# 出力枠は思考と本文の合計。65536 にすると 32768 で切れていた思考が入るが、1 問 20 分超に
# なるので LM の timeout も合わせて延ばす。
MAX_TOKENS="${MAX_TOKENS:-32768}"
LM_TIMEOUT="${LM_TIMEOUT:-1800}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-1800}"
# Why not 固定ポート: before と after のジョブが同じノードで同時に走るので、ジョブ ID で分ける。
PORT=$((7600 + SLURM_JOB_ID % 100))
LOG_DIR="$HANDOVER_DIR/outputs/slurm/$RUN_NAME-$PROMPT"
mkdir -p "$LOG_DIR"

cd "$HANDOVER_DIR"
manifest_field() {
    uv run python -c "import json,sys; m=json.load(open('model_manifest.json')); print(m[sys.argv[1]][sys.argv[2]])" "$1" "$2"
}
MODEL="$(manifest_field reflection model_id)"
REV="$(manifest_field reflection revision)"

echo "[$(date +%T)] job $SLURM_JOB_ID on $(hostname); GPU: ${CUDA_VISIBLE_DEVICES:-unset}; port $PORT"
echo "[$(date +%T)] run=$RUN_NAME-$PROMPT data=$DATA_DIR shards=$SHARDS model=$MODEL max_tokens=$MAX_TOKENS lm_timeout=$LM_TIMEOUT"

(
    cd "$ROOT_DIR"
    HF_HOME="$QWEN_HF_HOME" \
    VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 VLLM_DEEP_GEMM_WARMUP=skip \
    uv run scripts/serve_vllm.py \
        --vllm-env-dir "$QWEN_VLLM_ENV" \
        --model-path "$MODEL" \
        --model-revision "$REV" \
        --served-model-name "$MODEL" \
        --port "$PORT" \
        --max-model-len 131072 \
        --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
        --gpu-memory-utilization 0.85 \
        --max-num-seqs 16 \
        --no-enable-log-requests --disable-log-stats \
        --disable-uvicorn-access-log --generation-config vllm \
        --reasoning-parser qwen3
) > "$LOG_DIR/vllm_$PORT.log" 2>&1 &
SERVER_PID=$!
trap 'echo "[$(date +%T)] stopping server"; kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT

waited=0
until curl -sf -H "Authorization: Bearer local" "http://127.0.0.1:$PORT/v1/models" > /dev/null; do
    if (( waited >= READY_TIMEOUT_SEC )); then
        echo "[$(date +%T)] server on port $PORT did not come up within ${READY_TIMEOUT_SEC}s" >&2
        exit 1
    fi
    sleep 15
    waited=$((waited + 15))
done
echo "[$(date +%T)] server ready on port $PORT after ${waited}s"

echo "[$(date +%T)] starting $PROMPT comparison"
uv run python scripts/compare_prompt_models.py \
    --data-dir "$DATA_DIR" --run-name "$RUN_NAME-$PROMPT" \
    --only-model qwen3_8_27b --only-prompt "$PROMPT" \
    --qwen38-api-base "http://127.0.0.1:$PORT/v1" \
    --max-tokens "$MAX_TOKENS" --lm-timeout "$LM_TIMEOUT" \
    --parallel --shards "$SHARDS" 2>&1 | tee "$LOG_DIR/compare.log"
echo "[$(date +%T)] done: outputs/prompt_model_comparisons/$RUN_NAME-$PROMPT/"
