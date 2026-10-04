#!/bin/bash
#SBATCH --job-name=eval-glm
#SBATCH --partition=p1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#
# GLM-5.3-Flash（GGUF Q8、320B / 活性 18B）を llama.cpp で GPU 2 枚 + RAM に載せ、
# scripts/evaluate_solver_model.py で問題集を単発で解かせる。配信の手順と性能はリポジトリ
# ルートの docs/serving_runtimes.md「GLM-5.3-Flash の起動と停止」にある。
#
#   cd ProgAndSpec/DSPy_Shizuoka_V3_handover
#   DATA_DIR=data/problems_hard_gen/subsets/glm_pilot RUN_NAME=glm-pilot-20261004 \
#     EFFORTS="max high" sbatch --export=ALL scripts/slurm_eval_glm.sbatch
#
# EFFORTS に並べた reasoning_effort ごとに、同じ問題集を順に解かせる（サーバーの読み込みは 1 回）。
# 結果は outputs/prompt_model_comparisons/${RUN_NAME}-<effort>/glm53_flash_<effort>__shard01of01/ に出る。
set -euo pipefail

HANDOVER_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
ROOT_DIR="$(cd "$HANDOVER_DIR/../.." && pwd)"
: "${DATA_DIR:?set DATA_DIR to the problem directory to solve}"
RUN_NAME="${RUN_NAME:-glm-$(date +%Y%m%d)}"
EFFORTS="${EFFORTS:-max}"
# 入力（最長で約 15k tokens）と思考・回答の合計が収まる長さ。KV を増やすと GPU に置ける
# エキスパートが減って生成が遅くなるので、必要な分だけにする。
CTX_SIZE="${CTX_SIZE:-49152}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
LM_TIMEOUT="${LM_TIMEOUT:-5400}"
EXEC_TIMEOUT="${EXEC_TIMEOUT:-900}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-1800}"
# Why not 7501 固定: 手作業で立てた vLLM / llama.cpp と衝突しないよう、ジョブ ID で分ける。
PORT=$((7600 + ${SLURM_JOB_ID:-0} % 100))
LOG_DIR="$HANDOVER_DIR/outputs/slurm/$RUN_NAME"
mkdir -p "$LOG_DIR"

echo "[$(date +%T)] job ${SLURM_JOB_ID:-local} on $(hostname); GPUs: ${CUDA_VISIBLE_DEVICES:-unset}; port $PORT"
echo "[$(date +%T)] data=$DATA_DIR efforts=[$EFFORTS] ctx=$CTX_SIZE max_tokens=$MAX_TOKENS exec_timeout=$EXEC_TIMEOUT"

(
    cd "$ROOT_DIR"
    exec uv run scripts/serve_llamacpp.py --port "$PORT" --ctx-size "$CTX_SIZE"
) > "$LOG_DIR/llama_server_$PORT.log" 2>&1 &
SERVER_PID=$!
trap 'echo "[$(date +%T)] stopping server"; kill -- -"$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true; pkill -P "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT

waited=0
until curl -sf "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[$(date +%T)] server exited during load" >&2
        tail -20 "$LOG_DIR/llama_server_$PORT.log" >&2
        exit 1
    fi
    if (( waited >= READY_TIMEOUT_SEC )); then
        echo "[$(date +%T)] server did not come up within ${READY_TIMEOUT_SEC}s" >&2
        exit 1
    fi
    sleep 15
    waited=$((waited + 15))
done
echo "[$(date +%T)] server ready after ${waited}s"

cd "$HANDOVER_DIR"
status=0
for effort in $EFFORTS; do
    label="glm53_flash_${effort}"
    echo "[$(date +%T)] solving with reasoning_effort=$effort"
    uv run python scripts/evaluate_solver_model.py \
        --data-dir "$DATA_DIR" \
        --api-base "http://127.0.0.1:$PORT/v1" \
        --model glm-5.3-flash \
        --label "$label" \
        --run-name "$RUN_NAME-$effort" \
        --max-tokens "$MAX_TOKENS" \
        --timeout "$LM_TIMEOUT" \
        --exec-timeout "$EXEC_TIMEOUT" \
        --concurrency 1 \
        --extra-body "{\"reasoning_effort\": \"$effort\"}" 2>&1 | tee "$LOG_DIR/eval-$effort.log" || status=$?
done
echo "[$(date +%T)] done (status=$status)"
exit $status
