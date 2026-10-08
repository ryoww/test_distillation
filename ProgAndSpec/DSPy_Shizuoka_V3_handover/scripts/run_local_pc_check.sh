#!/bin/bash
# GPU なし・CPU N コアだけの環境（ノート PC 相当）で、GGUF にした student を llama.cpp で配信し、問題集を 1 問ずつ解かせる。
# 生成とソルバーが同じコアを順に使う。RESCORE_REPORT 35 章。
#
#   GGUF=/var/tmp/yy-lab-ft/agents-a1-4b-merged-20261006-q8_0.gguf CORES=0-7 THREADS=8 \
#     DATA_DIR=data/problems_hard_gen/subsets/local_pc_17 RUN_NAME=local-pc-20261008 scripts/run_local_pc_check.sh
#
# GGUF の作り方（焼き込み済みの student から。MTP 層は焼き込みで落ちているので --no-mtp）:
#   PYTHONPATH=.runtime/llama.cpp/gguf-py .runtime/train/bin/python .runtime/llama.cpp/convert_hf_to_gguf.py \
#     <merged dir> --no-mtp --outtype q8_0 --outfile <out.gguf>        # リポジトリルートで実行
set -euo pipefail
: "${GGUF:?set GGUF}" "${DATA_DIR:?set DATA_DIR}" "${RUN_NAME:?set RUN_NAME}"
CORES="${CORES:-0-7}"; THREADS="${THREADS:-8}"; PORT="${PORT:-7680}"; LABEL="${LABEL:-student_q8_cpu}"
HANDOVER_DIR="$(cd "$(dirname "$0")/.." && pwd)"; ROOT="$(cd "$HANDOVER_DIR/../.." && pwd)"; B=$ROOT/.runtime/llama.cpp/build/bin
export LD_LIBRARY_PATH=$B:$ROOT/.runtime/cuda-13.0-llamacpp/lib:${LD_LIBRARY_PATH:-}
LOG=$HANDOVER_DIR/outputs/local_pc/$RUN_NAME; mkdir -p "$LOG"
CUDA_VISIBLE_DEVICES="" taskset -c "$CORES" "$B/llama-server" -m "$GGUF" -ngl 0 -t "$THREADS" -c 24576 -np 1 \
  --port "$PORT" --host 127.0.0.1 --jinja --chat-template-kwargs '{"enable_thinking": false}' > "$LOG/llama_server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
until curl -sf "http://127.0.0.1:$PORT/health" | grep -q ok; do kill -0 $SRV || exit 1; sleep 5; done
cd "$HANDOVER_DIR"
taskset -c "$CORES" uv run python scripts/evaluate_solver_model.py --data-dir "$DATA_DIR" \
  --api-base "http://127.0.0.1:$PORT/v1" --model student-gguf --label "$LABEL" --run-name "$RUN_NAME" \
  --max-tokens 8192 --timeout 3600 --exec-timeout 900 --concurrency 1 --exec-concurrency 1 2>&1 | tee "$LOG/eval.log"
