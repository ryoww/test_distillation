#!/bin/bash
#SBATCH --job-name=eval-solver
#SBATCH --partition=p1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#
# Agents-A1-4B（素のまま、または LoRA を焼き込んだマージ済みモデル）を vLLM で配信し、SFT と
# 同じ入力形式で問題集を解かせて採点する。MODEL_PATH を省くと base モデルの対照条件になる。
# adapter は先にルートの scripts/merge_adapter.py で焼き込む（vLLM の LoRA 対応に依存しない）。
#
#   cd ProgAndSpec/DSPy_Shizuoka_V3_handover
#   export QWEN_VLLM_ENV=/home/yy-lab/test_DSPy/.runtime/vllm/vllm-cu13
#   MODEL_PATH=/path/to/outputs/<run>/merged LABEL=sft__agents_a1_4b sbatch --export=ALL scripts/slurm_eval_solver.sbatch
#   LABEL=base__agents_a1_4b sbatch --export=ALL scripts/slurm_eval_solver.sbatch
#
# DATA_DIRS はスペース区切りの問題ディレクトリ。既定は SFT のテスト集合と生成 140 問。
# 汎化の測定: DATA_DIRS=data/problems EVAL_ARGS=--exclude-templated で出荷 100 問のうち雛形外の 72 問。
set -euo pipefail

: "${LABEL:?set LABEL, e.g. sft__agents_a1_4b}"
: "${QWEN_VLLM_ENV:?set QWEN_VLLM_ENV to the vLLM virtualenv directory}"
MODEL_PATH="${MODEL_PATH:-}"
HANDOVER_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
ROOT_DIR="$(cd "$HANDOVER_DIR/../.." && pwd)"
RUN_NAME="${RUN_NAME:-solver-eval-$(date +%Y%m%d)}"
DATA_DIRS="${DATA_DIRS:-data/sft/problems_test data/problems_generated}"
MAX_TOKENS="${MAX_TOKENS:-6144}"
# 入力（約 3k tokens）と MAX_TOKENS の合計が収まる長さにする。
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
# Why not 思考あり: SFT データの assistant 冒頭は空の思考ブロックで描画されているので、推論も
# 同じ描画（enable_thinking=false）にする。素のモデルを思考ありで測るときだけ上書きする。
EXTRA_BODY="${EXTRA_BODY:-{\"chat_template_kwargs\": {\"enable_thinking\": false}}}"
CONCURRENCY="${CONCURRENCY:-8}"
# 例: EVAL_ARGS=--exclude-templated で雛形化済みの問題を除き、汎化だけを測る。
EVAL_ARGS="${EVAL_ARGS:-}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-1500}"
# 入力はおよそ 3k tokens。出力枠と合わせて context に収まらなければ全リクエストが 400 になる。
if (( MAX_TOKENS + 4096 > MAX_MODEL_LEN )); then
    echo "MAX_TOKENS=$MAX_TOKENS + prompt does not fit MAX_MODEL_LEN=$MAX_MODEL_LEN; raise MAX_MODEL_LEN" >&2
    exit 2
fi
PORT=$((7700 + SLURM_JOB_ID % 100))
SERVED="agents-a1-4b-solver"
LOG_DIR="$HANDOVER_DIR/outputs/slurm/$RUN_NAME-$LABEL"
mkdir -p "$LOG_DIR"

echo "[$(date +%T)] job $SLURM_JOB_ID on $(hostname); GPU: ${CUDA_VISIBLE_DEVICES:-unset}; port $PORT"
echo "[$(date +%T)] label=$LABEL model=${MODEL_PATH:-base} data=[$DATA_DIRS] max_tokens=$MAX_TOKENS extra_body=$EXTRA_BODY"

SERVE_ARGS=(--vllm-env-dir "$QWEN_VLLM_ENV" --served-model-name "$SERVED" --port "$PORT" --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization 0.85)
if [[ -n "$MODEL_PATH" ]]; then
    # Why not --adapter-path: マージ済みなら通常モデルと同じ経路で配信でき、LoRA 対応の差に左右されない。
    SERVE_ARGS+=(--model-path "$MODEL_PATH" --model-revision "")
fi
(
    cd "$ROOT_DIR"
    # Why --reasoning-parser: 思考を開くモデルの <think> を content から分離し、コード抽出を汚さない。
    uv run scripts/serve_vllm.py "${SERVE_ARGS[@]}" --max-num-seqs 16 --disable-log-stats --disable-uvicorn-access-log --reasoning-parser qwen3
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

cd "$HANDOVER_DIR"
status=0
for data_dir in $DATA_DIRS; do
    tag="$(basename "$data_dir")"
    echo "[$(date +%T)] evaluating $tag"
    uv run python scripts/evaluate_solver_model.py \
        --data-dir "$data_dir" \
        --api-base "http://127.0.0.1:$PORT/v1" \
        --model "$SERVED" \
        --label "$LABEL" \
        --run-name "$RUN_NAME-$tag" \
        --max-tokens "$MAX_TOKENS" --extra-body "$EXTRA_BODY" \
        --concurrency "$CONCURRENCY" $EVAL_ARGS 2>&1 | tee "$LOG_DIR/eval-$tag.log" || status=$?
done
echo "[$(date +%T)] done (status=$status)"
exit "$status"
