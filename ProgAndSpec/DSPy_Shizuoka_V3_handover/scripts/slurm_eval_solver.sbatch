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
# MODEL_PATH に Hugging Face の repo id を渡せば他のモデル（Gemma 4、Ministral 3 など）も同じ
# 手順で測れる。その場合は REASONING_PARSER と EXTRA_BODY をそのモデルの思考制御に合わせる。
#
#   cd ProgAndSpec/DSPy_Shizuoka_V3_handover
#   export QWEN_VLLM_ENV=/home/yy-lab/test_DSPy/.runtime/vllm/vllm-cu13
#   MODEL_PATH=/path/to/outputs/<run>/merged LABEL=sft__agents_a1_4b sbatch --export=ALL scripts/slurm_eval_solver.sbatch
#   LABEL=base__agents_a1_4b sbatch --export=ALL scripts/slurm_eval_solver.sbatch
#
# DATA_DIRS はスペース区切りの問題ディレクトリ。既定は SFT のテスト集合と生成 140 問。
# 汎化の測定: EXCLUDE_TEMPLATED_DIRS=data/problems で、そのディレクトリだけ雛形化済みの 28 問を除く
# （出荷 100 問 → 雛形外 72 問）。生成集合は全問が雛形由来なので、そこには付けない。
set -euo pipefail

: "${LABEL:?set LABEL, e.g. sft__agents_a1_4b}"
: "${QWEN_VLLM_ENV:?set QWEN_VLLM_ENV to the vLLM virtualenv directory}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_REVISION="${MODEL_REVISION:-}"
VLLM_ENV="${VLLM_ENV:-$QWEN_VLLM_ENV}"
# Why: ルートの serve_vllm.py は HF_HOME 未設定ならルートの model/hf_home を見る。Qwen と同じ
# キャッシュに置いたモデルを使うときは MODEL_HF_HOME でそこを指す。
if [[ -n "${MODEL_HF_HOME:-}" ]]; then export HF_HOME="$MODEL_HF_HOME"; fi
SERVED="${SERVED:-agents-a1-4b-solver}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
EXCLUDE_TEMPLATED_DIRS="${EXCLUDE_TEMPLATED_DIRS:-}"
SYSTEM_PREFIX_FILE="${SYSTEM_PREFIX_FILE:-}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-600}"
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
LOG_DIR="$HANDOVER_DIR/outputs/slurm/$RUN_NAME-$LABEL"
mkdir -p "$LOG_DIR"

echo "[$(date +%T)] job $SLURM_JOB_ID on $(hostname); GPU: ${CUDA_VISIBLE_DEVICES:-unset}; port $PORT"
echo "[$(date +%T)] label=$LABEL model=${MODEL_PATH:-base} data=[$DATA_DIRS] max_tokens=$MAX_TOKENS extra_body=$EXTRA_BODY"

SERVE_ARGS=(--vllm-env-dir "$VLLM_ENV" --served-model-name "$SERVED" --port "$PORT" --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization 0.85)
if [[ -n "$MODEL_PATH" ]]; then
    # Why not --adapter-path: マージ済みなら通常モデルと同じ経路で配信でき、LoRA 対応の差に左右されない。
    SERVE_ARGS+=(--model-path "$MODEL_PATH" --model-revision "$MODEL_REVISION")
fi
(
    cd "$ROOT_DIR"
    # Why --reasoning-parser: 思考を開くモデルの <think> を content から分離し、コード抽出を汚さない。
    uv run scripts/serve_vllm.py "${SERVE_ARGS[@]}" --max-num-seqs 16 --disable-log-stats --disable-uvicorn-access-log --reasoning-parser "$REASONING_PARSER" $VLLM_EXTRA_ARGS
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
    dir_args=()
    for excluded in $EXCLUDE_TEMPLATED_DIRS; do
        [[ "$excluded" == "$data_dir" ]] && dir_args+=(--exclude-templated)
    done
    if [[ -n "$SYSTEM_PREFIX_FILE" ]]; then dir_args+=(--system-prefix-file "$SYSTEM_PREFIX_FILE"); fi
    echo "[$(date +%T)] evaluating $tag ${dir_args[*]:-}"
    uv run python scripts/evaluate_solver_model.py \
        --data-dir "$data_dir" \
        --api-base "http://127.0.0.1:$PORT/v1" \
        --model "$SERVED" \
        --label "$LABEL" \
        --run-name "$RUN_NAME-$tag" \
        --max-tokens "$MAX_TOKENS" --extra-body "$EXTRA_BODY" \
        --concurrency "$CONCURRENCY" --timeout "$EVAL_TIMEOUT" "${dir_args[@]}" $EVAL_ARGS 2>&1 | tee "$LOG_DIR/eval-$tag.log" || status=$?
done
echo "[$(date +%T)] done (status=$status)"
exit "$status"
