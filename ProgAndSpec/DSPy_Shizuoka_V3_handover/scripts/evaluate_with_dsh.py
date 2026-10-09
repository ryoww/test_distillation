#!/usr/bin/env python3
"""DeepSeek Harness（dsh）の非対話モードに問題を解かせ、残った solve.py を既存の採点系で採点する。

問題ごとに作業ディレクトリ（problem.md、instance.json、meta.json）を作り、その中で
`dsh --profile headless --json "<依頼>"` を起動する。モデルはサンドボックス（作業ディレクトリだけ書ける）で
コマンドを実行でき、`scripts/verify_solve.py` で自分の解を確かめられる（参照値は見せない）。
採点は evaluate_solver_model.py と同じで、結果も同じ shard 形式で書くので summarize_solver_runs.py で集計できる。

  DSH_HOME=/var/tmp/yy-lab-dsh/home uv run python scripts/evaluate_with_dsh.py \\
      --dsh /var/tmp/yy-lab-dsh/node_modules/.bin/dsh --work-root /var/tmp/yy-lab-dsh/ws \\
      --data-dir data/problems_hard --run-name dsh-20261010 --label dsh_gemma4_12b_it
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.evaluate_solver_model import RESULT_FILENAME, _evaluate
from src import best_known as _best_known
from src.data_loader import convert_to_dspy_example, load_v3_data
from src.exec_gate import exec_slot, set_exec_concurrency

TASK = """Solve the optimization problem described in problem.md. The instance data is in instance.json.
Write a Python file solve.py that defines solve(instance: dict) -> dict and returns the solution in the
required return schema from problem.md. Use only the libraries problem.md allows.
Check your solution by running: {python} {verify} .
It runs solve(instance) on instance.json in a sandbox and reports whether the solution is feasible,
the violated constraints, and the objective recomputed from your solution.
Fix solve.py until the check reports a feasible solution, try to improve the objective, keep solve()
under {limit} seconds, and reply DONE when solve.py is final."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--dsh", type=Path, required=True, help="dsh の実行ファイル")
    parser.add_argument("--work-root", type=Path, required=True, help="作業ディレクトリを作る場所")
    parser.add_argument(
        "--output-dir", type=Path, default=BASE_DIR / "outputs" / "prompt_model_comparisons"
    )
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--exec-concurrency", type=int, default=4)
    parser.add_argument("--task-timeout", type=float, default=3600.0, help="1 問の dsh の打ち切り秒数")
    parser.add_argument("--solve-limit", type=int, default=300, help="solve() に許す秒数（依頼文に書く）")
    parser.add_argument("--exec-timeout", type=float, default=900.0)
    parser.add_argument("--ids", help="カンマ区切りの instance_id だけ解く（動作確認用）")
    parser.add_argument(
        "--exclude-templated", action="store_true", help="雛形化済みの問題を除く（汎化の測定用）"
    )
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def prepare(workspace: Path, example: dict, solve_limit: int) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    problem = convert_to_dspy_example(example["record"], use_reference=False)["requirement"]
    (workspace / "problem.md").write_text(problem, encoding="utf-8")
    (workspace / "instance.json").write_text(
        json.dumps(example["instance"], ensure_ascii=False), encoding="utf-8"
    )
    meta = {"core_type": example["core_type"], "timeout": solve_limit}
    (workspace / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def count_events(stdout: str) -> dict:
    counts: dict[str, int] = {}
    for line in stdout.splitlines():
        try:
            kind = json.loads(line).get("type", "?")
        except json.JSONDecodeError:
            continue
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def solve_one(args: argparse.Namespace, example: dict) -> dict:
    workspace = args.work_root / args.label / example["instance_id"]
    if workspace.exists() and any(workspace.iterdir()):
        # Why not 上書き: 前の run の solve.py が残ると、dsh が書かなかった問題まで採点してしまう。
        raise RuntimeError(f"{workspace} is not empty; use another --work-root or --label")
    prepare(workspace, example, args.solve_limit)
    task = TASK.format(
        python=sys.executable,
        verify=BASE_DIR / "scripts" / "verify_solve.py",
        limit=args.solve_limit,
    )
    started = time.monotonic()
    try:
        proc = subprocess.run(
            [str(args.dsh), "--profile", "headless", "--json", task],
            cwd=workspace,
            check=False,
            capture_output=True,
            text=True,
            timeout=args.task_timeout,
            env=os.environ.copy(),
        )
        returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        returncode, stdout, stderr = -1, exc.stdout or "", "dsh timed out"
        stdout = stdout.decode() if isinstance(stdout, bytes) else stdout
    elapsed = time.monotonic() - started
    (workspace / "dsh_events.jsonl").write_text(stdout, encoding="utf-8")
    (workspace / "dsh_stderr.txt").write_text(stderr or "", encoding="utf-8")
    row = {
        "instance_id": example["instance_id"],
        "name": example.get("name", ""),
        "core_type": example["core_type"],
        "dsh_returncode": returncode,
        "dsh_events": count_events(stdout),
        "elapsed": elapsed,
    }
    solve_path = workspace / "solve.py"
    if not solve_path.exists():
        return {**row, "code": None, "status": "gen_error", "score": -0.5,
                "detail": f"no solve.py (dsh exit {returncode}): {(stderr or '')[-300:]}"}
    code = solve_path.read_text(encoding="utf-8")
    registry = _best_known.BestKnownRegistry()
    if example.get("reference_value") is not None:
        registry.register(example["instance_id"], example["reference_value"])
    with exec_slot():
        result = _evaluate(args, example, code, registry)
    return {
        **row,
        "code": code,
        "status": result["status"],
        "score": result["score"],
        "cost": result.get("cost"),
        "reference_value": example.get("reference_value"),
        "detail": result.get("detail", ""),
        "feasibility_verified": result.get("feasibility_verified"),
    }


def main() -> int:
    args = parse_args()
    set_exec_concurrency(args.exec_concurrency)
    examples = []
    for record in load_v3_data(str(args.data_dir)):
        example = convert_to_dspy_example(record, use_reference=True)
        example["record"] = record
        examples.append(example)
    examples.sort(key=lambda e: e["instance_id"])
    if args.exclude_templated:
        from src.datagen import TEMPLATES

        templated = {f"prob_{k:03d}" for k in TEMPLATES}
        examples = [ex for ex in examples if ex["instance_id"] not in templated]
    if args.ids:
        wanted = set(args.ids.split(","))
        examples = [ex for ex in examples if ex["instance_id"] in wanted]
    if args.limit:
        examples = examples[: args.limit]
    shard_dir = args.output_dir / args.run_name / f"{args.label}__shard01of01"
    shard_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(solve_one, args, ex): ex["instance_id"] for ex in examples}
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(
                f"[{index}/{len(examples)}] {row['instance_id']}: {row['status']} "
                f"score={row['score']:.2f} dsh_exit={row['dsh_returncode']} "
                f"events={row['dsh_events']} ({row['elapsed']:.0f}s)",
                flush=True,
            )
    rows.sort(key=lambda r: r["instance_id"])
    payload = {
        "test": {
            "results": rows,
            "total_count": len(rows),
            "mean_score": sum(r["score"] for r in rows) / max(len(rows), 1),
        },
        "config": {
            "label": args.label,
            "data_dir": str(args.data_dir.resolve()),
            "harness": "deepseek-harness headless",
            "dsh": str(args.dsh),
            "task_timeout": args.task_timeout,
            "solve_limit": args.solve_limit,
        },
    }
    (shard_dir / RESULT_FILENAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"results: {shard_dir / RESULT_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
