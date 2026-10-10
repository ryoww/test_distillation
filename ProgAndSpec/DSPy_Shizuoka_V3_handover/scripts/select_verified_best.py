#!/usr/bin/env python3
"""複数の run の解から、参照値を使わない検証器で問題ごとに 1 つを選び、選んだ行で shard を作る（LLM は呼ばない）。

各 run の保存済みコードを `verify_solution`（src/agent.py と同じ、参照値なし）で走らせ、
「可行で目的値あり（小さい順、検証器は最小化の向きにそろえる）＞ 可行で目的値なし ＞ 検証器が形を読めない ＞ 失敗」
の順で 1 つ採り、同順位は --run の順。採った行の判定（参照値を使った採点）はその run のものをそのまま使う。

  uv run python scripts/select_verified_best.py --data-dir data/problems_hard \\
      --run fft=outputs/prompt_model_comparisons/v3-fft-20261009-problems_hard/fft12b_v3__shard01of01 \\
      --run dsh=outputs/prompt_model_comparisons/dsh-draft2-20261010-problems_hard/dsh_it_on_fft_draft__shard01of01 \\
      --out-run select-20261010-problems_hard --label fft_or_dsh
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.summarize_solver_runs import RESULT_FILENAME
from src.data_loader import convert_to_dspy_example, load_v3_data

OUT_ROOT = BASE_DIR / "outputs" / "prompt_model_comparisons"


def _verify(job: tuple[str, dict, str, float]) -> tuple[str, float | None]:
    """子プロセスで 1 本を検証し、(検証器の判定, 再計算した目的値) を返す。"""
    from src.utils.feasibility import check_feasibility_detailed
    from src.verify_loop import verify_solution

    code, instance, core_type, timeout = job
    verdict = verify_solution(code, instance, core_type, timeout=timeout)
    if verdict.kind != "feasible" or not isinstance(verdict.solution, dict):
        return verdict.kind, None
    return "feasible", check_feasibility_detailed(core_type, instance, verdict.solution).get("cost")


def _rank(kind: str, cost: float | None) -> int:
    if kind == "feasible":
        return 0 if cost is not None else 1
    return 2 if kind == "unverified" else 3


def pick(candidates: list[tuple[str, str, float | None]]) -> str:
    """(run 名, 検証器の判定, 目的値) の並びから採る run 名を返す。同順位は並びの先。"""
    order = range(len(candidates))
    best = min(
        order,
        key=lambda i: (_rank(*candidates[i][1:]), candidates[i][2] or 0.0, i),
    )
    return candidates[best][0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="label=shard_dir（先頭が既定）")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-run", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    examples = {}
    for record in load_v3_data(str(args.data_dir)):
        example = convert_to_dspy_example(record)
        examples[example["instance_id"]] = example
    runs: dict[str, dict[str, dict]] = {}
    for spec in args.run:
        label, _, path = spec.partition("=")
        rows = json.loads((Path(path) / RESULT_FILENAME).read_text(encoding="utf-8"))["test"]["results"]
        runs[label] = {row["instance_id"]: row for row in rows}

    ids = sorted(next(iter(runs.values())))
    jobs, keys = [], []
    for instance_id in ids:
        example = examples[instance_id]
        for label, rows in runs.items():
            code = (rows.get(instance_id) or {}).get("code")
            if code:
                jobs.append((code, example["instance"], example["core_type"], args.timeout))
                keys.append((instance_id, label))
    with ProcessPoolExecutor(args.workers) as pool:
        verdicts = dict(zip(keys, pool.map(_verify, jobs)))

    picked_rows = []
    for instance_id in ids:
        candidates = [
            (label, *verdicts.get((instance_id, label), ("missing", None))) for label in runs
        ]
        chosen = pick(candidates)
        row = dict(runs[chosen].get(instance_id) or next(iter(runs.values()))[instance_id])
        row["picked_from"] = chosen
        row["candidates"] = [{"run": c[0], "verdict": c[1], "cost": c[2]} for c in candidates]
        picked_rows.append(row)
        print(f"{instance_id}: {chosen} {row['status']} {row['candidates']}", flush=True)

    shard_dir = OUT_ROOT / args.out_run / f"{args.label}__shard01of01"
    shard_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "test": {
            "results": picked_rows,
            "total_count": len(picked_rows),
            "mean_score": sum(r["score"] for r in picked_rows) / max(len(picked_rows), 1),
        },
        "config": {"label": args.label, "runs": args.run, "rule": "feasible with cost (lowest) > feasible > unverified > failed; ties by run order"},
    }
    (shard_dir / RESULT_FILENAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"results: {shard_dir / RESULT_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
