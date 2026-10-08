#!/usr/bin/env python3
"""評価で未正解だった行の保存済みコードを、並列度を絞って再実行し採点し直す（LLM は呼ばない）。

時間制限付きのソルバーは CPU の混み具合で解の質が変わる。評価は 8 問並列で他のジョブとも重なるので、
同じコードが単独では正解でも評価では「参照より 10% 超悪い」や時間超過になる（RESCORE_REPORT 34 章）。
未正解の行だけを `--workers` 並列で再実行し、正解だった行はそのまま残した shard を `--out-run` に書く。

  uv run python scripts/replay_failed.py \\
      --run fft=outputs/prompt_model_comparisons/gemma4-ft-20261004-test/sft_gemma4_12b_fft_fp32__shard01of01 \\
      --data-dir data/problems_hard_gen/test --out-run replay-20261008-test --workers 4
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

from scripts.summarize_solver_runs import RESULT_FILENAME, is_correct

OUT_ROOT = BASE_DIR / "outputs" / "prompt_model_comparisons"


def _rescore(job: tuple[dict, dict, float]) -> dict:
    """子プロセスで 1 行を再採点する。評価時と同じく参照値で best_known を初期化する。"""
    from src import best_known as _best_known
    from src.data_loader import convert_to_dspy_example
    from src.metrics_v3 import evaluate_algorithm_v3

    row, record, timeout = job
    example = convert_to_dspy_example(record, use_reference=True)
    registry = _best_known.BestKnownRegistry()
    if example.get("reference_value") is not None:
        registry.register(example["instance_id"], example["reference_value"])
    result = evaluate_algorithm_v3(
        code=row["code"],
        instance=example["instance"],
        core_type=example["core_type"],
        instance_id=example["instance_id"],
        registry=registry,
        timeout=timeout,
        reference_value=example.get("reference_value"),
        reference_solution=example.get("reference_solution", {}),
        objective_text=example.get("objective", ""),
        use_reference=True,
    )
    return {
        **row,
        "status": result["status"],
        "score": result["score"],
        "cost": result.get("cost"),
        "detail": result.get("detail", ""),
        "feasibility_verified": result.get("feasibility_verified"),
        "replayed_from": {"status": row["status"], "cost": row.get("cost")},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="label=shard_dir")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-run", required=True, help="outputs/prompt_model_comparisons/<この名前>/")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--max-gap", type=float, default=0.10)
    args = parser.parse_args()

    records = {}
    for path in args.data_dir.glob("prob_*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        records[convert_id(record["id"])] = record

    plans = []
    for spec in args.run:
        label, _, path = spec.partition("=")
        payload = json.loads((Path(path) / RESULT_FILENAME).read_text(encoding="utf-8"))
        rows = payload["test"]["results"]
        failed = [
            i for i, r in enumerate(rows)
            if not is_correct(r, args.max_gap) and r.get("code") and "def solve" in r["code"]
        ]
        plans.append((label, payload, rows, failed))
    jobs = [(rows[i], records[rows[i]["instance_id"]], args.timeout) for _, _, rows, failed in plans for i in failed]
    print(f"replaying {len(jobs)} failed rows with {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_rescore, jobs))

    cursor = 0
    for label, payload, rows, failed in plans:
        for i in failed:
            rows[i] = results[cursor]
            cursor += 1
        payload["test"]["mean_score"] = sum(r["score"] for r in rows) / max(len(rows), 1)
        payload["config"] = {**payload.get("config", {}), "replayed_failed_with_workers": args.workers}
        shard = OUT_ROOT / args.out_run / f"{label}__shard01of01"
        shard.mkdir(parents=True, exist_ok=True)
        (shard / RESULT_FILENAME).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        fixed = sum(is_correct(rows[i], args.max_gap) for i in failed)
        print(f"{label}: replayed {len(failed)}, now correct {fixed} -> {shard}", flush=True)
    return 0


def convert_id(raw: int) -> str:
    """convert_to_dspy_example と同じ instance_id の書式。"""
    return f"prob_{raw:03d}"


if __name__ == "__main__":
    raise SystemExit(main())
