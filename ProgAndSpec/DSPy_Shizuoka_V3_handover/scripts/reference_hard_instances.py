#!/usr/bin/env python3
"""生成した大規模 instance に、教師コードの最良解を参照解として付ける。

厳密解が作れない規模なので、参照解は「検証器を通った既存の solve() の中で最も良い目的値」。
採点は検証器が目的値を再計算するので、参照値は gap を読むための目安になる。

  collect: 7 条件の保存結果から、元 28 問で検証済み可行だった solve() を種別ごとに取り出す
           → outputs/hard_teachers/<kind>/<condition>__<prob_id>.py
  run:     instances/ の各 instance に種別の教師コードを全部走らせ、可行な最良解を参照にして
           <output-dir>/<split>/prob_XXXX.json へ書く。可行解が 1 つもない instance は
           <output-dir>/unreferenced.json に理由つきで残す。

  uv run python scripts/reference_hard_instances.py collect
  uv run python scripts/reference_hard_instances.py run --workers 48 --timeout 900
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.utils.feasibility import check_feasibility_detailed
from src.utils.hard import find_kind
from src.utils.safe_exec import safe_run

RESULT_FILENAME = "evaluation_results_v3_gepa_phaseE.json"
COMPARISONS = BASE_DIR / "outputs" / "prompt_model_comparisons"
TEACHER_DIR = BASE_DIR / "outputs" / "hard_teachers"
GOOD = {"exact_match", "beat_reference", "new_best", "improved", "similar", "worse", "first_valid"}
# 条件名 → 保存コードのある run ディレクトリ（最終再採点 JSON の条件名と対応）
RUNS = {
    "compact__qwen3_6_27b": "hard28-20261001-qwen3_6_27b-compact",
    "compact__qwen3_8_27b": "hard28-20261001-qwen3_8_27b-compact",
    "gemma4_12b_nothink": "hard28-20261001b-problems_hard",
    "fable__claude": "hard28-fable-20261001",
    "repair_compact__qwen3_6_27b": "hard28-repair-20261001-qwen3_6_27b-compact",
    "repair_compact__qwen3_8_27b": "hard28-repair-20261001-qwen3_8_27b-compact",
    "repair_gepa__qwen3_8_27b": "hard28-repair-20261001-qwen3_8_27b-gepa_compact",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    col = sub.add_parser("collect")
    col.add_argument(
        "--rescored",
        type=Path,
        default=COMPARISONS / "rescored-hard28-final-20261001.json",
        help="どの保存コードが検証済み可行だったかを読む再採点 JSON",
    )
    col.add_argument("--problem-dir", type=Path, default=BASE_DIR / "data" / "problems_hard")
    col.add_argument("--teacher-dir", type=Path, default=TEACHER_DIR)
    col.add_argument(
        "--solutions-dirs",
        nargs="*",
        type=Path,
        default=[
            BASE_DIR / "outputs/rescore_hard/solutions",
            BASE_DIR / "outputs/rescore_hard/solutions_repair",
        ],
        help="保存した返り値。申告値のずれだけで落ちた解（構造は可行）も教師に含めるために読む",
    )
    run = sub.add_parser("run")
    run.add_argument(
        "--instances-dir", type=Path, default=BASE_DIR / "data" / "problems_hard_gen" / "instances"
    )
    run.add_argument("--output-dir", type=Path, default=BASE_DIR / "data" / "problems_hard_gen")
    run.add_argument("--teacher-dir", type=Path, default=TEACHER_DIR)
    run.add_argument("--workers", type=int, default=32)
    run.add_argument("--timeout", type=float, default=900.0)
    run.add_argument("--kinds", default="all")
    run.add_argument("--limit", type=int, help="先頭 N instance だけ（動作確認用）")
    return parser.parse_args()


# 保存した返り値のディレクトリ名は条件名と揃っていないので、候補を順に探す。
DUMP_DIR_ALIASES = {
    "gemma4_12b_nothink": ["gemma4_12b_nothink__shard01of01"],
    "fable__claude": ["fable__claude__shard01of01"],
    "repair_compact__qwen3_8_27b": ["compact__qwen3_8_27b"],
    "repair_gepa__qwen3_8_27b": ["gepa_compact__qwen3_8_27b"],
}


def _declared_only(
    solutions_dirs: list[Path], cond: str, iid: str, core_type: str, instance: dict
) -> bool:
    """検証器の違反が「申告値のずれ」1 件だけなら、構造は可行なので教師にできる。"""
    names = [cond, *DUMP_DIR_ALIASES.get(cond, [])]
    # Why not 最初に見つかった dump だけ: 同じ条件名のディレクトリが solutions/ と solutions_repair/ の
    # 両方にあり（compact__qwen3_8_27b）、片方は別条件の解なので、両方を見て判定が一致するものだけ採る。
    for base in solutions_dirs:
        for name in names:
            path = base / name / f"{iid}.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not payload.get("ok") or payload.get("solution") is None:
                continue
            checked = check_feasibility_detailed(core_type, instance, payload["solution"])
            violations = [str(v) for v in checked.get("violations", [])]
            if (
                checked.get("verified")
                and len(violations) == 1
                and violations[0].startswith("declared")
            ):
                return True
    return False


def collect(args: argparse.Namespace) -> None:
    verdict: dict[tuple[str, str], dict] = {}
    for cond, recs in json.loads(args.rescored.read_text(encoding="utf-8"))["records"].items():
        for row in recs:
            verdict[(cond, row["instance_id"])] = row
    kind_of = {}
    base_records = {}
    for path in args.problem_dir.glob("prob_*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        kind = find_kind(record["instance"])
        if kind is not None:
            kind_of[f"prob_{record['id']:03d}"] = kind.name
            base_records[f"prob_{record['id']:03d}"] = record
    written = 0
    declared_only = 0
    for cond, run_name in RUNS.items():
        for shard in sorted((COMPARISONS / run_name).glob(f"*/{RESULT_FILENAME}")):
            for row in json.loads(shard.read_text(encoding="utf-8"))["test"]["results"]:
                iid = row["instance_id"]
                graded = verdict.get((cond, iid))
                if not row.get("code") or graded is None:
                    continue
                strictly_ok = graded["status"] in GOOD and graded.get("feasibility_verified")
                if not strictly_ok:
                    base = base_records[iid]
                    if not _declared_only(
                        args.solutions_dirs,
                        cond,
                        iid,
                        f"{base['domain']}_{base['math_type']}",
                        base["instance"],
                    ):
                        continue
                    declared_only += 1
                target = args.teacher_dir / kind_of[iid] / f"{cond}__{iid}.py"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(row["code"].strip() + "\n", encoding="utf-8")
                written += 1
    per_kind = {d.name: len(list(d.glob("*.py"))) for d in sorted(args.teacher_dir.iterdir())}
    print(
        f"collected {written} teacher codes ({declared_only} with a declared-value mismatch only): {per_kind}"
    )


def _try_teacher(job: tuple[str, str, str, float]) -> tuple[str, str, dict]:
    """1 (instance, 教師) 対を実行して検証する。子プロセスで動くので素の値だけ受け渡す。"""
    instance_path, teacher_path, core_type, timeout = job
    record = json.loads(Path(instance_path).read_text(encoding="utf-8"))
    code = Path(teacher_path).read_text(encoding="utf-8")
    ok, result = safe_run(code, record["instance"], timeout=timeout)
    if not ok:
        return instance_path, teacher_path, {"ok": False, "error": str(result)[:300]}
    checked = check_feasibility_detailed(core_type, record["instance"], result)
    violations = [str(v) for v in checked.get("violations", [])]
    verified = bool(checked.get("verified"))
    feasible = verified and checked["feasible"] and not violations
    # 申告した目的値だけがずれている解は構造としては可行。申告欄を再計算値で上書きして使う。
    declared_key = None
    if verified and len(violations) == 1 and violations[0].startswith("declared "):
        declared_key = violations[0].split()[1]
        feasible = True
    if feasible and declared_key and isinstance(result, dict) and checked.get("cost") is not None:
        result = {**result, declared_key: checked["cost"]}
        if "objective_value" in result:
            result["objective_value"] = checked["cost"]
    return (
        instance_path,
        teacher_path,
        {
            "ok": True,
            "feasible": feasible,
            "declared_fixed": declared_key,
            "cost": checked.get("cost"),
            "violations": [v[:120] for v in violations[:3]],
            "solution": result if feasible else None,
        },
    )


def run(args: argparse.Namespace) -> None:
    kinds = None if args.kinds == "all" else set(args.kinds.split(","))
    instances = sorted(args.instances_dir.glob("prob_*.json"))
    if args.limit:
        instances = instances[: args.limit]
    jobs = []
    meta: dict[str, dict] = {}
    for path in instances:
        record = json.loads(path.read_text(encoding="utf-8"))
        kind = record["provenance"]["kind"]
        if kinds and kind not in kinds:
            continue
        teachers = sorted((args.teacher_dir / kind).glob("*.py"))
        core_type = f"{record['domain']}_{record['math_type']}"
        meta[str(path)] = {"record": record, "results": {}}
        for teacher in teachers:
            jobs.append((str(path), str(teacher), core_type, args.timeout))
    print(f"{len(meta)} instances, {len(jobs)} (instance, teacher) pairs, {args.workers} workers")
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_try_teacher, job) for job in jobs]
        for future in as_completed(futures):
            instance_path, teacher_path, result = future.result()
            meta[instance_path]["results"][Path(teacher_path).stem] = result
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)} pairs done", flush=True)

    unreferenced = []
    counts: dict[str, int] = defaultdict(int)
    for instance_path, entry in meta.items():
        record = entry["record"]
        feasible = {
            name: r
            for name, r in entry["results"].items()
            if r.get("feasible") and r.get("cost") is not None
        }
        if not feasible:
            unreferenced.append(
                {
                    "instance_id": f"prob_{record['id']}",
                    "kind": record["provenance"]["kind"],
                    "teachers": {
                        name: (r.get("error") or r.get("violations"))
                        for name, r in entry["results"].items()
                    },
                }
            )
            continue
        best_name = min(feasible, key=lambda n: feasible[n]["cost"])
        best = feasible[best_name]
        solution = {
            k: v for k, v in best["solution"].items() if k not in ("objective_value", "note")
        }
        record["reference_solution"] = {
            "objective_value": best["cost"],
            **solution,
            "note": f"教師コード {best_name} の解（検証器で可行確認、目的値は再計算値）。最適性は未証明。",
        }
        record["reference_meta"] = {
            "status": "teacher",
            "best_teacher": best_name,
            "teacher_costs": {n: r["cost"] for n, r in feasible.items()},
            "teachers_tried": len(entry["results"]),
            "teachers_feasible": len(feasible),
            "bound": None,
            "is_optimal": False,
        }
        split_dir = args.output_dir / record["split"]
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / f"prob_{record['id']}.json").write_text(
            json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        counts[record["split"]] += 1
    (args.output_dir / "unreferenced.json").write_text(
        json.dumps(unreferenced, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(
        f"referenced: {dict(counts)}; unreferenced: {len(unreferenced)} -> {args.output_dir / 'unreferenced.json'}"
    )


def main() -> None:
    args = parse_args()
    if args.command == "collect":
        collect(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
