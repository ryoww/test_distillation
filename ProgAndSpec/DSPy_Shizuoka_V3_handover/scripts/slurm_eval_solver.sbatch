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
# GEPA で進化させた指示文を system に使うとき（交互最適化の評価）。
INSTRUCTION_FILE="${INSTRUCTION_FILE:-}"
# 生成→検証→修復の 2 段プログラムで測るとき（scripts/evaluate_student_program.py）。修復段の指示文を指す。
# "default" を渡すと既定の修復指示文で、修復ループだけの効果を測る。
REPAIR_INSTRUCTION_FILE="${REPAIR_INSTRUCTION_FILE:-}"
MAX_REPAIRS="${MAX_REPAIRS:-1}"
# 学習した種別の対応表（scripts/list_supported_kinds.py）。渡すと対応外の種別は解かずに unsupported で返す。
SUPPORTED_KINDS_FILE="${SUPPORTED_KINDS_FILE:-}"
# フォールバック LM（student が通らないときに同じ手順で試す大きいモデル）。同じ GPU に 2 つ目の vLLM を立てる。
FALLBACK_PATH="${FALLBACK_PATH:-}"
FALLBACK_SERVED="${FALLBACK_SERVED:-fallback}"
FALLBACK_REASONING_PARSER="${FALLBACK_REASONING_PARSER:-gemma4}"
FALLBACK_THINKING="${FALLBACK_THINKING:-off}"
# フォールバック側の vLLM に足す引数（dsh のエージェントに使うときのツール呼び出しの解釈器など）。
FALLBACK_VLLM_EXTRA_ARGS="${FALLBACK_VLLM_EXTRA_ARGS:-}"
# 3 月の構成を段階式で通して動かすとき（scripts/run_cascade.py）。student の対応表を指す。student は本体の vLLM、
# 汎用エージェント（dsh）はフォールバックの vLLM を使う。DSH・DSH_HOME・DSH_WORK_ROOT も要る。
CASCADE_KINDS_FILE="${CASCADE_KINDS_FILE:-}"
# Why not 常に 0.85: フォールバックと同居させるときは、4B（約 10 GB）と 12B（約 24 GB）に KV の余地を足して分ける。
if [[ -n "$FALLBACK_PATH" ]]; then GPU_UTIL="${GPU_UTIL:-0.30}"; FALLBACK_GPU_UTIL="${FALLBACK_GPU_UTIL:-0.50}"; else GPU_UTIL="${GPU_UTIL:-0.85}"; fi
EVAL_TIMEOUT="${EVAL_TIMEOUT:-600}"
EXEC_TIMEOUT="${EXEC_TIMEOUT:-60}"
HANDOVER_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
ROOT_DIR="$(cd "$HANDOVER_DIR/../.." && pwd)"
RUN_NAME="${RUN_NAME:-solver-eval-$(date +%Y%m%d)}"
DATA_DIRS="${DATA_DIRS:-data/sft/problems_test data/problems_generated}"
MAX_TOKENS="${MAX_TOKENS:-6144}"
# 入力（約 3k tokens）と MAX_TOKENS の合計が収まる長さにする。
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
# Why not 思考あり: SFT データの assistant 冒頭は空の思考ブロックで描画されているので、推論も
# 同じ描画（enable_thinking=false）にする。素のモデルを思考ありで測るときだけ上書きする。
# Why not 既定値を ${VAR:-...} に直書き: 波括弧を含む既定値は bash の展開が乱れ、値を渡したときに
# 余分な "}}" が付いて JSON が壊れる。既定は別変数に置く。
DEFAULT_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'
EXTRA_BODY="${EXTRA_BODY:-$DEFAULT_EXTRA_BODY}"
CONCURRENCY="${CONCURRENCY:-8}"
# Why not 生成と同じ並列度で実行: 時間制限付きソルバーは CPU の混み具合で解の質が変わり、8 並列では教師と同一の
# コードが参照より 19〜187% 悪く判定された（RESCORE_REPORT 34.2 節）。生成は CONCURRENCY 並列、実行は EXEC_CONCURRENCY 本ずつ。
EXEC_CONCURRENCY="${EXEC_CONCURRENCY:-4}"
# 例: EVAL_ARGS=--exclude-templated で雛形化済みの問題を除き、汎化だけを測る。
EVAL_ARGS="${EVAL_ARGS:-}"
# DeepSeek Harness（dsh）の非対話モードに解かせるとき（scripts/evaluate_with_dsh.py）。dsh の実行ファイルを指す。
# DSH_HOME（プロファイル）と DSH_WORK_ROOT（作業ディレクトリの置き場）も渡す。
DSH="${DSH:-}"
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

SERVE_ARGS=(--vllm-env-dir "$VLLM_ENV" --served-model-name "$SERVED" --port "$PORT" --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_UTIL")
if [[ -n "$MODEL_PATH" ]]; then
    # Why not --adapter-path: マージ済みなら通常モデルと同じ経路で配信でき、LoRA 対応の差に左右されない。
    SERVE_ARGS+=(--model-path "$MODEL_PATH" --model-revision "$MODEL_REVISION")
fi
# Why: vLLM 0.28 の FlashInfer は sampling カーネルを起動時に nvcc で JIT する。PATH の nvcc は
# ヘッダを持たないラッパーなので、ルートに入れた CUDA 13 toolkit を明示する（学習ジョブと同じ）。
export CUDA_HOME="$ROOT_DIR/.runtime/cuda-13.0"
export PATH="$CUDA_HOME/bin:$PATH"
# Why not conda で libcurand-dev: cuda-13.0 の channel label に dev 版がなく解決できない。
# Why not pip の include を丸ごと -I: 新しい runtime の crt/ が nvcc 13.0 の stub と食い違って壊れる。
# torch が持ち込む pip の CUDA 13 ヘッダから curand*.h だけをリンクした dir を nvcc に足す。
NVIDIA_INC="$(ls -d "$VLLM_ENV"/lib/python3*/site-packages/nvidia/cu13/include 2>/dev/null | head -1)"
CURAND_INC="$CUDA_HOME/curand-include"
if [[ -n "$NVIDIA_INC" && ! -f "$CURAND_INC/curand_kernel.h" ]]; then
    mkdir -p "$CURAND_INC" && ln -sf "$NVIDIA_INC"/curand*.h "$CURAND_INC"/
fi
if [[ -f "$CURAND_INC/curand_kernel.h" ]]; then export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-} -I$CURAND_INC"; fi
# FlashInfer は -L$CUDA_HOME/lib64 でリンクするが、conda 版 toolkit は lib にしか置かない。
[[ -e "$CUDA_HOME/lib64" ]] || ln -s lib "$CUDA_HOME/lib64"
(
    cd "$ROOT_DIR"
    # Why setsid: vLLM は EngineCore を子プロセスとして起こす。親だけ kill すると EngineCore が
    # GPU メモリを握ったまま残るので、独立したプロセスグループにしてグループごと止める。
    # Why --reasoning-parser: 思考を開くモデルの <think> を content から分離し、コード抽出を汚さない。
    exec setsid uv run scripts/serve_vllm.py "${SERVE_ARGS[@]}" --max-num-seqs 16 --disable-log-stats --disable-uvicorn-access-log --reasoning-parser "$REASONING_PARSER" $VLLM_EXTRA_ARGS
) > "$LOG_DIR/vllm_$PORT.log" 2>&1 &
SERVER_PID=$!
stop_server() {
    echo "[$(date +%T)] stopping server group $SERVER_PID"
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    sleep 3
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
}
trap stop_server EXIT

waited=0
until curl -sf -H "Authorization: Bearer local" "http://127.0.0.1:$PORT/v1/models" > /dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[$(date +%T)] server process exited before becoming ready; see $LOG_DIR/vllm_$PORT.log" >&2
        exit 1
    fi
    if (( waited >= READY_TIMEOUT_SEC )); then
        echo "[$(date +%T)] server on port $PORT did not come up within ${READY_TIMEOUT_SEC}s" >&2
        exit 1
    fi
    sleep 15
    waited=$((waited + 15))
done
echo "[$(date +%T)] server ready on port $PORT after ${waited}s"
FALLBACK_PORT=$((PORT + 50))
if [[ -n "$FALLBACK_PATH" ]]; then
    (
        cd "$ROOT_DIR"
        exec setsid uv run scripts/serve_vllm.py --vllm-env-dir "$VLLM_ENV" --model-path "$FALLBACK_PATH" --model-revision "" \
            --served-model-name "$FALLBACK_SERVED" --port "$FALLBACK_PORT" --max-model-len "$MAX_MODEL_LEN" \
            --gpu-memory-utilization "$FALLBACK_GPU_UTIL" --max-num-seqs 16 --disable-log-stats --disable-uvicorn-access-log \
            --reasoning-parser "$FALLBACK_REASONING_PARSER" $FALLBACK_VLLM_EXTRA_ARGS
    ) > "$LOG_DIR/vllm_fallback_$FALLBACK_PORT.log" 2>&1 &
    FALLBACK_PID=$!
    trap 'stop_server; kill -TERM -- "-$FALLBACK_PID" 2>/dev/null || true' EXIT
    waited=0
    until curl -sf -H "Authorization: Bearer local" "http://127.0.0.1:$FALLBACK_PORT/v1/models" > /dev/null; do
        if ! kill -0 "$FALLBACK_PID" 2>/dev/null; then echo "fallback server exited; see $LOG_DIR/vllm_fallback_$FALLBACK_PORT.log" >&2; exit 1; fi
        if (( waited >= READY_TIMEOUT_SEC )); then echo "fallback server not ready" >&2; exit 1; fi
        sleep 15; waited=$((waited + 15))
    done
    echo "[$(date +%T)] fallback server ready on port $FALLBACK_PORT after ${waited}s"
fi

cd "$HANDOVER_DIR"
status=0
for data_dir in $DATA_DIRS; do
    tag="$(basename "$data_dir")"
    dir_args=()
    for excluded in $EXCLUDE_TEMPLATED_DIRS; do
        [[ "$excluded" == "$data_dir" ]] && dir_args+=(--exclude-templated)
    done
    if [[ -n "$SYSTEM_PREFIX_FILE" ]]; then dir_args+=(--system-prefix-file "$SYSTEM_PREFIX_FILE"); fi
    if [[ -n "$INSTRUCTION_FILE" ]]; then dir_args+=(--instruction-file "$INSTRUCTION_FILE"); fi
    echo "[$(date +%T)] evaluating $tag ${dir_args[*]:-}"
    if [[ -n "$CASCADE_KINDS_FILE" ]]; then
        : "${FALLBACK_PATH:?set FALLBACK_PATH to the agent model served for dsh}"
        : "${DSH:?set DSH to the dsh executable}"
        : "${DSH_HOME:?set DSH_HOME to the dsh home holding the headless profile}"
        : "${DSH_WORK_ROOT:?set DSH_WORK_ROOT to an empty directory for per-problem workspaces}"
        cascade_args=()
        for excluded in $EXCLUDE_TEMPLATED_DIRS; do [[ "$excluded" == "$data_dir" ]] && cascade_args+=(--exclude-templated); done
        DSH_LOCAL_BASE_URL="http://127.0.0.1:$FALLBACK_PORT/v1" DSH_LOCAL_API_KEY=local DSH_LOCAL_MODEL="$FALLBACK_SERVED" \
        DSH_LOCAL_CONTEXT="$MAX_MODEL_LEN" \
        uv run python scripts/run_cascade.py --data-dir "$data_dir" --run-name "$RUN_NAME-$tag" --label "$LABEL" \
            --student-api-base "http://127.0.0.1:$PORT/v1" --student-model "$SERVED" \
            --student-max-tokens "$MAX_TOKENS" --student-extra-body "$EXTRA_BODY" \
            --supported-kinds-file "$CASCADE_KINDS_FILE" --dsh "$DSH" --work-root "$DSH_WORK_ROOT" \
            --concurrency "$CONCURRENCY" --exec-concurrency "$EXEC_CONCURRENCY" --exec-timeout "$EXEC_TIMEOUT" \
            "${cascade_args[@]}" $EVAL_ARGS 2>&1 | tee "$LOG_DIR/eval-$tag.log" || status=$?
        continue
    fi
    if [[ -n "$DSH" ]]; then
        : "${DSH_HOME:?set DSH_HOME to the dsh home holding the headless profile}"
        : "${DSH_WORK_ROOT:?set DSH_WORK_ROOT to an empty directory for per-problem workspaces}"
        dsh_args=()
        for excluded in $EXCLUDE_TEMPLATED_DIRS; do [[ "$excluded" == "$data_dir" ]] && dsh_args+=(--exclude-templated); done
        DSH_LOCAL_BASE_URL="http://127.0.0.1:$PORT/v1" DSH_LOCAL_API_KEY=local DSH_LOCAL_MODEL="$SERVED" \
        DSH_LOCAL_CONTEXT="$MAX_MODEL_LEN" \
        uv run python scripts/evaluate_with_dsh.py --dsh "$DSH" --work-root "$DSH_WORK_ROOT" \
            --data-dir "$data_dir" --label "$LABEL" --run-name "$RUN_NAME-$tag" \
            --concurrency "$CONCURRENCY" --exec-concurrency "$EXEC_CONCURRENCY" --exec-timeout "$EXEC_TIMEOUT" \
            "${dsh_args[@]}" $EVAL_ARGS 2>&1 | tee "$LOG_DIR/eval-$tag.log" || status=$?
        continue
    fi
    if [[ -n "$REPAIR_INSTRUCTION_FILE" ]]; then
        repair_args=()
        [[ "$REPAIR_INSTRUCTION_FILE" != "default" ]] && repair_args+=(--repair-instruction-file "$REPAIR_INSTRUCTION_FILE")
        [[ -n "$INSTRUCTION_FILE" ]] && repair_args+=(--generate-instruction-file "$INSTRUCTION_FILE")
        for excluded in $EXCLUDE_TEMPLATED_DIRS; do [[ "$excluded" == "$data_dir" ]] && repair_args+=(--exclude-templated); done
        thinking=on; [[ "$EXTRA_BODY" == *'"enable_thinking": false'* ]] && thinking=off
        repair_args+=(--max-repairs "$MAX_REPAIRS")
        if [[ -n "$SUPPORTED_KINDS_FILE" ]]; then repair_args+=(--supported-kinds-file "$SUPPORTED_KINDS_FILE"); fi
        if [[ -n "$FALLBACK_PATH" ]]; then
            repair_args+=(--fallback-model "$FALLBACK_SERVED" --fallback-api-base "http://127.0.0.1:$FALLBACK_PORT/v1" --fallback-thinking "$FALLBACK_THINKING")
        fi
        uv run python scripts/evaluate_student_program.py \
            --data-dir "$data_dir" --api-base "http://127.0.0.1:$PORT/v1" --model "$SERVED" \
            --label "$LABEL" --run-name "$RUN_NAME-$tag" --max-tokens "$MAX_TOKENS" --thinking "$thinking" \
            --concurrency "$CONCURRENCY" --exec-concurrency "$EXEC_CONCURRENCY" --timeout "$EVAL_TIMEOUT" --exec-timeout "$EXEC_TIMEOUT" \
            "${repair_args[@]}" $EVAL_ARGS 2>&1 | tee "$LOG_DIR/eval-$tag.log" || status=$?
        continue
    fi
    uv run python scripts/evaluate_solver_model.py \
        --data-dir "$data_dir" \
        --api-base "http://127.0.0.1:$PORT/v1" \
        --model "$SERVED" \
        --label "$LABEL" \
        --run-name "$RUN_NAME-$tag" \
        --max-tokens "$MAX_TOKENS" --extra-body "$EXTRA_BODY" \
        --concurrency "$CONCURRENCY" --exec-concurrency "$EXEC_CONCURRENCY" --timeout "$EVAL_TIMEOUT" --exec-timeout "$EXEC_TIMEOUT" "${dir_args[@]}" $EVAL_ARGS 2>&1 | tee "$LOG_DIR/eval-$tag.log" || status=$?
done
echo "[$(date +%T)] done (status=$status)"
exit "$status"
