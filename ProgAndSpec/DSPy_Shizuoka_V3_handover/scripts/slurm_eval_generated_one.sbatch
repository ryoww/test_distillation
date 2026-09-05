#!/bin/bash
#SBATCH --job-name=eval-generated-one
#SBATCH --partition=p1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#
# 生成ホールドアウトで、1 モデル × 1 プロンプト条件を GPU 1 枚で評価する。
# 条件ごとにジョブを分けると、空いた GPU から順に走り、遅い Qwen3.8 が速い Qwen3.6 を
# 待たせない。
#
#   cd ProgAndSpec/DSPy_Shizuoka_V3_handover
#   export QWEN_HF_HOME=... QWEN_VLLM_ENV=... RUN_NAME=generated140-YYYYMMDD
#   MODEL_LABEL=qwen3_8_27b PROMPT=before sbatch --export=ALL scripts/slurm_eval_generated_one.sbatch
#   MODEL_LABEL=qwen3_6_27b PROMPT=compact PROGRAM=prompts/compiled_program_v3_compact.json \
#     sbatch --export=ALL scripts/slurm_eval_generated_one.sbatch
#
# PROMPT が before / after 以外のときは PROGRAM（保存済み DSPy プログラム）が必要。
# 結果は outputs/prompt_model_comparisons/${RUN_NAME}-${MODEL_LABEL}-${PROMPT}/ に出る。
set -euo pipefail

: "${QWEN_HF_HOME:?set QWEN_HF_HOME to the Hugging Face cache holding the Qwen models}"
: "${QWEN_VLLM_ENV:?set QWEN_VLLM_ENV to the vLLM virtualenv directory}"
: "${PROMPT:?set PROMPT to before, after, or an extra program label}"
MODEL_LABEL="${MODEL_LABEL:-qwen3_8_27b}"
PROGRAM="${PROGRAM:-}"
if [[ "$PROMPT" != "before" && "$PROMPT" != "after" && -z "$PROGRAM" ]]; then
    echo "PROMPT=$PROMPT needs PROGRAM=<saved program json>" >&2
    exit 2
fi

HANDOVER_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
ROOT_DIR="$(cd "$HANDOVER_DIR/../.." && pwd)"
RUN_NAME="${RUN_NAME:-generated140-$(date +%Y%m%d)}"
DATA_DIR="${DATA_DIR:-data/problems_generated}"
SHARDS="${SHARDS:-4}"
# 出力枠は思考と本文の合計。Qwen3.8 は 32768 では思考が切れるので 65536 を既定にする。
MAX_TOKENS="${MAX_TOKENS:-65536}"
LM_TIMEOUT="${LM_TIMEOUT:-5400}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-1800}"
# Why not 固定ポート: 同じノードで複数ジョブが同時に走るので、ジョブ ID で分ける。
PORT=$((7600 + SLURM_JOB_ID % 100))
LABEL="$RUN_NAME-$MODEL_LABEL-$PROMPT"
LOG_DIR="$HANDOVER_DIR/outputs/slurm/$LABEL"
mkdir -p "$LOG_DIR"

cd "$HANDOVER_DIR"
manifest_field() {
    uv run python -c "import json,sys; m=json.load(open('model_manifest.json')); print(m[sys.argv[1]][sys.argv[2]])" "$1" "$2"
}
case "$MODEL_LABEL" in
    qwen3_6_27b) ROLE=generation; API_FLAG=--qwen36-api-base ;;
    qwen3_8_27b) ROLE=reflection; API_FLAG=--qwen38-api-base ;;
    *) echo "MODEL_LABEL must be qwen3_6_27b or qwen3_8_27b" >&2; exit 2 ;;
esac
MODEL="$(manifest_field "$ROLE" model_id)"
REV="$(manifest_field "$ROLE" revision)"

echo "[$(date +%T)] job $SLURM_JOB_ID on $(hostname); GPU: ${CUDA_VISIBLE_DEVICES:-unset}; port $PORT"
echo "[$(date +%T)] run=$LABEL data=$DATA_DIR shards=$SHARDS model=$MODEL max_tokens=$MAX_TOKENS lm_timeout=$LM_TIMEOUT program=${PROGRAM:-builtin}"

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

EXTRA=()
if [[ -n "$PROGRAM" ]]; then
    EXTRA=(--extra-program "$PROMPT=$PROGRAM")
fi
echo "[$(date +%T)] starting $MODEL_LABEL / $PROMPT"
uv run python scripts/compare_prompt_models.py \
    --data-dir "$DATA_DIR" --run-name "$LABEL" \
    --only-model "$MODEL_LABEL" --only-prompt "$PROMPT" "${EXTRA[@]}" \
    "$API_FLAG" "http://127.0.0.1:$PORT/v1" \
    --max-tokens "$MAX_TOKENS" --lm-timeout "$LM_TIMEOUT" \
    --parallel --shards "$SHARDS" 2>&1 | tee "$LOG_DIR/compare.log"
echo "[$(date +%T)] done: outputs/prompt_model_comparisons/$LABEL/"
