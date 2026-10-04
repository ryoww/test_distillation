#!/usr/bin/env python3
"""evaluate_solver_model.py の結果を、学習データと同じ「正解」の基準で並べて数える。

正解 = 検証器が解を読めて違反 0（metrics の可行系の状態）、かつ参照解からの gap が `--max-gap` 以内。
gap は metrics と同じく最小化の向きにそろえた目的値で (cost - ref) / |ref|。
最初に渡した条件を基準に、問題ごとの「両方正解 / 基準のみ / 比較側のみ」も出す。

  uv run python scripts/summarize_solver_runs.py \
      --run base=outputs/prompt_model_comparisons/gemma4-ft-20261004-test/gemma4_12b_base_nothink__shard01of01 \
      --run lora=... --run fft=...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULT_FILENAME = "evaluation_results_v3_gepa_phaseE.json"
FEASIBLE = {
    "exact_match",
    "beat_reference",
    "new_best",
    "improved",
    "worse",
    "similar",
    "first_valid",
}
AT_LEAST_REFERENCE = {"exact_match", "beat_reference", "new_best"}


def gap_of(row: dict) -> float | None:
    cost, ref = row.get("cost"), row.get("reference_value")
    if row.get("status") not in FEASIBLE or cost is None or ref is None:
        return None
    if ref == 0:
        return 0.0 if abs(cost) < 1e-6 else float("inf")
    return (cost - ref) / abs(ref)


def is_correct(row: dict, max_gap: float) -> bool:
    gap = gap_of(row)
    return gap is not None and gap <= max_gap


def summarize(rows: list[dict], max_gap: float) -> dict:
    gaps = [gap_of(r) for r in rows]
    feasible = [g for g in gaps if g is not None]
    tokens = [r["usage"].get("completion_tokens", 0) for r in rows if r.get("usage")]
    return {
        "problems": len(rows),
        "correct": sum(g <= max_gap for g in feasible),
        # Why not gap <= 0: metrics の exact_match は丸め誤差を許すので、状態で数える。
        "at_least_reference": sum(r["status"] in AT_LEAST_REFERENCE for r in rows),
        "feasible": len(feasible),
        "mean_score": sum(r["score"] for r in rows) / max(len(rows), 1),
        "mean_output_tokens": sum(tokens) / max(len(tokens), 1),
    }


def load(spec: str) -> tuple[str, dict[str, dict]]:
    label, _, path = spec.partition("=")
    path = Path(path)
    if path.is_dir():
        path = path / RESULT_FILENAME
    rows = json.loads(path.read_text(encoding="utf-8"))["test"]["results"]
    return label, {r["instance_id"]: r for r in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="label=shard_dir|json")
    parser.add_argument("--max-gap", type=float, default=0.10)
    args = parser.parse_args()

    runs = [load(spec) for spec in args.run]
    base_label, base_rows = runs[0]
    print("| 条件 | 問題数 | 正解 | 参照以上 | 可行 | 平均スコア | 平均出力 token | 両方正解 / 基準のみ / 比較側のみ |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    for label, rows in runs:
        s = summarize(list(rows.values()), args.max_gap)
        pair = "—"
        if label != base_label:
            shared = sorted(set(rows) & set(base_rows))
            ok = {iid: is_correct(rows[iid], args.max_gap) for iid in shared}
            ok_base = {iid: is_correct(base_rows[iid], args.max_gap) for iid in shared}
            pair = (
                f"{sum(ok[i] and ok_base[i] for i in shared)} / "
                f"{sum(ok_base[i] and not ok[i] for i in shared)} / "
                f"{sum(ok[i] and not ok_base[i] for i in shared)}"
            )
        print(
            f"| {label} | {s['problems']} | {s['correct']} | {s['at_least_reference']} | "
            f"{s['feasible']} | {s['mean_score']:.3f} | {s['mean_output_tokens']:.0f} | {pair} |"
        )


if __name__ == "__main__":
    main()
