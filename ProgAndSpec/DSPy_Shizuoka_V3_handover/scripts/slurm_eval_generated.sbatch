#!/bin/bash
#SBATCH --job-name=eval-generated
#SBATCH --partition=p1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#
# 生成ホールドアウトで改善前後プロンプト × Qwen3.6 / Qwen3.8 を評価する。
# handover ディレクトリから sbatch する。GPU0 に Qwen3.6、GPU1 に Qwen3.8 を立て、
# 両サーバーが応答したら compare_prompt_models.py が 4 条件 × 2 shard を同時に流す。
#
#   cd ProgAndSpec/DSPy_Shizuoka_V3_handover
#   export QWEN_HF_HOME=... QWEN_VLLM_ENV=...
#   sbatch scripts/slurm_eval_generated.sbatch
#
# 上書きできる環境変数: RUN_NAME, DATA_DIR, SHARDS, READY_TIMEOUT_SEC
set -euo pipefail

: "${QWEN_HF_HOME:?set QWEN_HF_HOME to the Hugging Face cache holding both Qwen models}"
: "${QWEN_VLLM_ENV:?set QWEN_VLLM_ENV to the vLLM virtualenv directory}"

HANDOVER_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
ROOT_DIR="$(cd "$HANDOVER_DIR/../.." && pwd)"
RUN_NAME="${RUN_NAME:-generated140-$(date +%Y%m%d-%H%M)}"
DATA_DIR="${DATA_DIR:-data/problems_generated}"
SHARDS="${SHARDS:-2}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-1800}"
LOG_DIR="$HANDOVER_DIR/outputs/slurm/$RUN_NAME"
mkdir -p "$LOG_DIR"

cd "$HANDOVER_DIR"
manifest_field() {
    uv run python -c "import json,sys; m=json.load(open('model_manifest.json')); print(m[sys.argv[1]][sys.argv[2]])" "$1" "$2"
}
GEN_MODEL="$(manifest_field generation model_id)"
GEN_REV="$(manifest_field generation revision)"
REF_MODEL="$(manifest_field reflection model_id)"
REF_REV="$(manifest_field reflection revision)"
GEN_PORT=7501
REF_PORT=7502

echo "[$(date +%T)] job $SLURM_JOB_ID on $(hostname); GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "[$(date +%T)] run=$RUN_NAME data=$DATA_DIR shards=$SHARDS"

start_server() {
    # $1: GPU index inside the allocation, $2: model, $3: revision, $4: port
    (
        cd "$ROOT_DIR"
        HF_HOME="$QWEN_HF_HOME" CUDA_VISIBLE_DEVICES="$1" \
        VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 VLLM_DEEP_GEMM_WARMUP=skip \
        uv run scripts/serve_vllm.py \
            --vllm-env-dir "$QWEN_VLLM_ENV" \
            --model-path "$2" \
            --model-revision "$3" \
            --served-model-name "$2" \
            --port "$4" \
            --max-model-len 131072 \
            --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
            --gpu-memory-utilization 0.85 \
            --max-num-seqs 16 \
            --no-enable-log-requests --disable-log-stats \
            --disable-uvicorn-access-log --generation-config vllm \
            --reasoning-parser qwen3
    ) > "$LOG_DIR/vllm_$4.log" 2>&1 &
    echo $!
}

wait_ready() {
    # $1: port, $2: label
    local waited=0
    until curl -sf -H "Authorization: Bearer local" "http://127.0.0.1:$1/v1/models" > /dev/null; do
        if (( waited >= READY_TIMEOUT_SEC )); then
            echo "[$(date +%T)] $2 on port $1 did not come up within ${READY_TIMEOUT_SEC}s" >&2
            return 1
        fi
        sleep 15
        waited=$((waited + 15))
    done
    echo "[$(date +%T)] $2 ready on port $1 after ${waited}s"
}

GEN_PID="$(start_server 0 "$GEN_MODEL" "$GEN_REV" "$GEN_PORT")"
REF_PID="$(start_server 1 "$REF_MODEL" "$REF_REV" "$REF_PORT")"
# Why not サーバーを別ジョブにする: 2 GPU を 1 ジョブで確保すれば、片方だけ空くのを待って
# 半端な条件が走ることがなく、終了時にまとめて止められる。
trap 'echo "[$(date +%T)] stopping servers"; kill "$GEN_PID" "$REF_PID" 2>/dev/null || true; wait "$GEN_PID" "$REF_PID" 2>/dev/null || true' EXIT

wait_ready "$GEN_PORT" "$GEN_MODEL"
wait_ready "$REF_PORT" "$REF_MODEL"

uv run python scripts/preflight.py > "$LOG_DIR/preflight.json"
echo "[$(date +%T)] preflight ok: $(uv run python -c "import json;print(json.load(open('$LOG_DIR/preflight.json'))['ok'])")"

echo "[$(date +%T)] starting comparison"
uv run python scripts/compare_prompt_models.py \
    --data-dir "$DATA_DIR" \
    --run-name "$RUN_NAME" \
    --parallel --shards "$SHARDS" \
    2>&1 | tee "$LOG_DIR/compare.log"
echo "[$(date +%T)] done: outputs/prompt_model_comparisons/$RUN_NAME/factorial_comparison.json"
