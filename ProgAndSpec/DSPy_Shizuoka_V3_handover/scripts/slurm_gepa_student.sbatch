#!/bin/bash
#SBATCH --job-name=gepa-student
#SBATCH --partition=p1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#
# 特化学習したモデル（student、GPU 0 の vLLM 0.28）の指示文を、Qwen3.8（GPU 1）の反省で GEPA 進化させる。
# プロンプト最適化と fine-tuning を交互に回すときのプロンプト側（scripts/gepa_student.py）。
#
#   cd ProgAndSpec/DSPy_Shizuoka_V3_handover
#   export QWEN_HF_HOME=/home/yy-lab/test_DSPy/model/hf_home QWEN_VLLM_ENV=/home/yy-lab/test_DSPy/.runtime/vllm/vllm-cu13
#   STUDENT_PATH=/var/tmp/yy-lab-ft/<merged> RUN_DIR=outputs/bt-r1/gepa sbatch --export=ALL scripts/slurm_gepa_student.sbatch
#
# 上書きできる環境変数: GEPA_ARGS（gepa_student.py に足す引数）、RESUME=1（同じ RUN_DIR で続きから）
set -euo pipefail

: "${STUDENT_PATH:?set STUDENT_PATH to the merged student model directory}"
: "${RUN_DIR:?set RUN_DIR, e.g. outputs/bt-r1/gepa}"
: "${QWEN_HF_HOME:?set QWEN_HF_HOME to the Hugging Face cache holding Qwen3.8}"
: "${QWEN_VLLM_ENV:?set QWEN_VLLM_ENV to the vLLM 0.20 virtualenv for Qwen}"
HANDOVER_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
ROOT_DIR="$(cd "$HANDOVER_DIR/../.." && pwd)"
STUDENT_VLLM_ENV="${STUDENT_VLLM_ENV:-$ROOT_DIR/.runtime/vllm-0.28}"
GEPA_ARGS="${GEPA_ARGS:-}"
RESUME="${RESUME:-0}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-1800}"
STU_PORT=$((7600 + ${SLURM_JOB_ID:-0} % 50))
REF_PORT=$((STU_PORT + 50))
LOG_DIR="$HANDOVER_DIR/$RUN_DIR/logs"
mkdir -p "$LOG_DIR"
cd "$HANDOVER_DIR"
REF_MODEL="$(uv run python -c "import json;print(json.load(open('model_manifest.json'))['reflection']['model_id'])")"
REF_REV="$(uv run python -c "import json;print(json.load(open('model_manifest.json'))['reflection']['revision'])")"
echo "[$(date +%T)] job ${SLURM_JOB_ID:-local}; student=$STUDENT_PATH reflection=$REF_MODEL run_dir=$RUN_DIR resume=$RESUME"

# vLLM 0.28 の FlashInfer は起動時に nvcc で JIT する（slurm_eval_solver.sbatch と同じ準備）。
export CUDA_HOME="$ROOT_DIR/.runtime/cuda-13.0"
export PATH="$CUDA_HOME/bin:$PATH"
CURAND_INC="$CUDA_HOME/curand-include"
if [[ -f "$CURAND_INC/curand_kernel.h" ]]; then export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-} -I$CURAND_INC"; fi
[[ -e "$CUDA_HOME/lib64" ]] || ln -s lib "$CUDA_HOME/lib64"

(
    cd "$ROOT_DIR"
    CUDA_VISIBLE_DEVICES=0 exec setsid uv run scripts/serve_vllm.py --vllm-env-dir "$STUDENT_VLLM_ENV" \
        --model-path "$STUDENT_PATH" --served-model-name student --port "$STU_PORT" \
        --max-model-len 53248 --gpu-memory-utilization 0.85 --max-num-seqs 16 \
        --disable-log-stats --disable-uvicorn-access-log --reasoning-parser gemma4
) > "$LOG_DIR/vllm_student.log" 2>&1 &
STU_PID=$!
(
    cd "$ROOT_DIR"
    HF_HOME="$QWEN_HF_HOME" CUDA_VISIBLE_DEVICES=1 \
    VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 VLLM_DEEP_GEMM_WARMUP=skip \
    exec setsid uv run scripts/serve_vllm.py --vllm-env-dir "$QWEN_VLLM_ENV" \
        --model-path "$REF_MODEL" --model-revision "$REF_REV" --served-model-name "$REF_MODEL" \
        --port "$REF_PORT" --max-model-len 131072 \
        --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
        --gpu-memory-utilization 0.85 --max-num-seqs 16 --disable-log-stats \
        --disable-uvicorn-access-log --generation-config vllm --reasoning-parser qwen3
) > "$LOG_DIR/vllm_reflection.log" 2>&1 &
REF_PID=$!
stop_servers() {
    echo "[$(date +%T)] stopping servers"
    kill -TERM -- "-$STU_PID" "-$REF_PID" 2>/dev/null || true
    wait "$STU_PID" "$REF_PID" 2>/dev/null || true
}
trap stop_servers EXIT

for port in "$STU_PORT" "$REF_PORT"; do
    waited=0
    until curl -sf -H "Authorization: Bearer local" "http://127.0.0.1:$port/v1/models" > /dev/null; do
        if (( waited >= READY_TIMEOUT_SEC )); then echo "server on $port not ready" >&2; exit 1; fi
        sleep 15; waited=$((waited + 15))
    done
    echo "[$(date +%T)] server on $port ready after ${waited}s"
done

RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then RESUME_ARGS=(--resume); fi
# shellcheck disable=SC2086 # GEPA_ARGS is a space-separated option list
uv run python scripts/gepa_student.py --run-dir "$RUN_DIR" \
    --student-model student --student-api-base "http://127.0.0.1:$STU_PORT/v1" \
    --reflection-model "$REF_MODEL" --reflection-api-base "http://127.0.0.1:$REF_PORT/v1" \
    "${RESUME_ARGS[@]}" $GEPA_ARGS 2>&1 | tee -a "$LOG_DIR/gepa.log"
echo "[$(date +%T)] done: $RUN_DIR/instruction.md"
