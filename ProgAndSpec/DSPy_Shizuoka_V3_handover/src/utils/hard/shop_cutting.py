"""大規模問題集の検証器（shop_cutting 群）。register_kind で種別を登録する。

対象種別:
- fjsp / fjsp_setup: フレキシブルジョブショップ（prob_305, prob_315, prob_322）。解は
  `schedule` に工程ごとの機械と開始・終了時刻を持ち、目的値はメイクスパン。
- cutting_1d: パターン種類数制限付き 1 次元カッティングストック（prob_307, prob_317）。
- cutting_2d: 2 段ギヨシュのストリップパターン切断（prob_324）。
目的値は申告値を読まず、instance の加工時間・原材コストから再計算する。
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

from ..feasibility_v3_ext import _result, _unverified
from . import register_kind

_EPS = 1e-6
# 申告値と再計算値の許容ずれ（相対 0.5%）。
_DECLARED_TOL = 5e-3


# ---------------------------------------------------------------- 共通ヘルパー
def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _sid(x: Any) -> str:
    """ID を比較用の文字列へ寄せる（JSON 経由で "1" / 1 / 1.0 が混在するため）。"""
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    return str(x)


def _pick(d: dict, *names: str) -> Any:
    """別名のうち最初に存在するキーの値を返す（解の形式ゆれに対応）。"""
    for name in names:
        if name in d:
            return d[name]
    return None


def _lookup(d: Any, key: Any) -> Any:
    """setups[m][a][b] のように文字列/整数どちらのキーでも引けるようにする。"""
    if not isinstance(d, dict):
        return None
    for k in (key, _sid(key)):
        if k in d:
            return d[k]
    try:
        return d.get(int(key))
    except (TypeError, ValueError):
        return None


def _is_count(x: Any) -> bool:
    return _num(x) and abs(x - round(x)) < _EPS


def _declared_mismatch(solution: dict, cost: float, *names: str) -> list[str]:
    """申告目的値が再計算値と 0.5% 以上ずれていれば違反として返す。"""
    declared = _pick(solution, *names)
    if not _num(declared):
        return []
    if abs(declared - cost) > _DECLARED_TOL * max(1.0, abs(cost)):
        return [f"declared objective {declared} differs from recomputed {cost}"]
    return []


# ---------------------------------------------------------------- FJSP
def _has_fjsp_jobs(instance: dict) -> bool:
    jobs = instance.get("jobs")
    if not isinstance(jobs, list) or not jobs or not isinstance(instance.get("machines"), list):
        return False
    job = jobs[0]
    ops = job.get("operations") if isinstance(job, dict) else None
    # 同梱の prob_004 は operations に machine を直書きするので machine_options で区別する。
    return bool(ops) and isinstance(ops[0], dict) and "machine_options" in ops[0]


def _detect_fjsp(instance: dict) -> bool:
    return _has_fjsp_jobs(instance) and "setups" not in instance and "breakdowns" not in instance


def _detect_fjsp_setup(instance: dict) -> bool:
    return _has_fjsp_jobs(instance) and ("setups" in instance or "breakdowns" in instance)


def _parse_schedule(solution: Any) -> list[dict] | None:
    """解を {job, op, machine, start, end} の平坦なリストへ寄せる。読めなければ None。"""
    if not isinstance(solution, dict):
        return None
    schedule = solution.get("schedule")
    if isinstance(schedule, dict):
        rows = [
            dict(row, _job=job) for job, ops in schedule.items()
            if isinstance(ops, list) for row in ops
        ]
    elif isinstance(schedule, list):
        rows = list(schedule)
    else:
        return None
    parsed = []
    for row in rows:
        if not isinstance(row, dict):
            return None
        job = row["_job"] if "_job" in row else _pick(row, "job_id", "job")
        op = _pick(row, "op_index", "operation", "op")
        machine = _pick(row, "machine", "machine_id")
        start = _pick(row, "start", "start_time")
        end = _pick(row, "end", "end_time")
        if job is None or op is None or machine is None or not _num(start) or not _num(end):
            return None
        parsed.append({"job": _sid(job), "op": op, "machine": _sid(machine),
                       "start": start, "end": end})
    return parsed


def _check_fjsp(instance: dict, solution: Any, *, with_setup: bool) -> dict:
    rows = _parse_schedule(solution)
    if rows is None:
        return _unverified("fjsp solution without a readable 'schedule'")
    jobs = {_sid(j["id"]): j for j in instance["jobs"]}
    machines = [_sid(m["id"]) if isinstance(m, dict) else _sid(m) for m in instance["machines"]]
    violations: list[str] = []

    # 工程ごとに 1 行だけ割り当てられていること。
    seen: dict[tuple[str, int], dict] = {}
    for row in rows:
        try:
            key = (row["job"], int(_sid(row["op"])))
        except ValueError:
            key = (row["job"], -1)
        if key[0] not in jobs or not 1 <= key[1] <= len(jobs[key[0]]["operations"]):
            violations.append(f"unknown operation job={row['job']} op={row['op']}")
            continue
        if key in seen:
            violations.append(f"operation job={key[0]} op={key[1]} scheduled twice")
            continue
        seen[key] = row
    n_ops = sum(len(j["operations"]) for j in jobs.values())
    for job_id, job in jobs.items():
        release = job.get("release_time", 0) or 0
        prev_end = release
        for op in job["operations"]:
            key = (job_id, int(op["op_index"]))
            row = seen.get(key)
            if row is None:
                violations.append(f"operation job={job_id} op={key[1]} not scheduled")
                continue
            duration = _lookup(op["machine_options"], row["machine"])
            if duration is None:
                violations.append(
                    f"job={job_id} op={key[1]} uses machine {row['machine']} not in its options"
                )
            elif abs(row["end"] - row["start"] - duration) > _EPS:
                violations.append(
                    f"job={job_id} op={key[1]} span {row['end'] - row['start']} != {duration}"
                )
            # 先行工程の完了と release_time の両方を待つ（release は初工程で効く）。
            if row["start"] < prev_end - _EPS:
                violations.append(
                    f"job={job_id} op={key[1]} starts {row['start']} before {prev_end}"
                )
            prev_end = max(prev_end, row["end"])

    # 機械ごとの排他（段取り・故障を含む）。
    by_machine: dict[str, list[dict]] = {m: [] for m in machines}
    for row in seen.values():
        if row["machine"] in by_machine:
            by_machine[row["machine"]].append(row)
        else:
            violations.append(f"unknown machine {row['machine']}")
    setups = instance.get("setups") if with_setup else None
    breakdowns = instance.get("breakdowns") if with_setup else None
    for m, ops in by_machine.items():
        ops.sort(key=lambda r: (r["start"], r["end"]))
        for a, b in pairwise(ops):
            setup = 0
            if setups is not None:
                setup = _lookup(_lookup(_lookup(setups, m), a["job"]), b["job"]) or 0
            if b["start"] < a["end"] + setup - _EPS:
                what = f"setup {setup}" if setup else "no gap"
                violations.append(
                    f"machine {m}: job={b['job']} op={b['op']} starts {b['start']} "
                    f"before job={a['job']} op={a['op']} ends {a['end']} ({what})"
                )
        for br in _lookup(breakdowns, m) or []:
            bs = br.get("start", 0)
            be = br["end"] if "end" in br else bs + br.get("duration", 0)
            for row in ops:
                if row["start"] < be - _EPS and row["end"] > bs + _EPS:
                    violations.append(
                        f"machine {m}: job={row['job']} op={row['op']} overlaps "
                        f"breakdown [{bs}, {be}]"
                    )
    cost = max((r["end"] for r in seen.values()), default=0.0)
    violations += _declared_mismatch(solution, cost, "makespan", "objective_value")
    per_machine = 3 if with_setup else 1
    return _result(violations, 3 * n_ops + per_machine * len(machines) + 1, cost=float(cost))


# ---------------------------------------------------------------- 1 次元カッティング
def _detect_cutting_1d(instance: dict) -> bool:
    return (
        isinstance(instance.get("items"), list)
        and isinstance(instance.get("stocks"), list)
        and "max_distinct_patterns" in instance
        and "plates" not in instance
    )


def _detect_cutting_2d(instance: dict) -> bool:
    return (
        isinstance(instance.get("items"), list)
        and isinstance(instance.get("plates"), list)
        and "max_distinct_patterns" in instance
    )


def _parse_counts(raw: Any) -> dict[str, float] | None:
    """cuts / items を {item_id: 本数} へ寄せる。ID のリストなら出現回数を数える。"""
    if isinstance(raw, dict):
        if not all(_num(v) for v in raw.values()):
            return None
        return {_sid(k): v for k, v in raw.items()}
    if isinstance(raw, list):
        counts: dict[str, float] = {}
        for item in raw:
            if isinstance(item, (dict, list)):
                return None
            counts[_sid(item)] = counts.get(_sid(item), 0) + 1
        return counts
    return None


def _parse_patterns(solution: Any) -> list[dict] | None:
    """解の patterns を {stock, count, raw} の並びへ寄せる。読めなければ None。"""
    if not isinstance(solution, dict) or not isinstance(solution.get("patterns"), list):
        return None
    parsed = []
    for pat in solution["patterns"]:
        if not isinstance(pat, dict):
            return None
        stock = _pick(pat, "stock_id", "stock", "plate_id", "plate")
        count = _pick(pat, "runs", "count", "uses", "quantity", "num")
        if stock is None or not _num(count):
            return None
        parsed.append({"stock": _sid(stock), "count": count, "raw": pat})
    return parsed


def _check_cutting_common(
    items: dict[str, dict], stocks: dict[str, dict], solution: dict, patterns: list[dict],
    pattern_sigs: list, max_patterns: Any, min_lot: float,
) -> tuple[list[str], float]:
    """需要充足・使用本数・種類数・申告値のように 1D/2D で共通の検査を行う。"""
    violations: list[str] = []
    produced: dict[str, float] = {}
    for pat, sig in zip(patterns, pattern_sigs):
        stock = stocks.get(pat["stock"])
        if stock is None:
            violations.append(f"pattern uses unknown stock {pat['stock']}")
        if not _is_count(pat["count"]) or pat["count"] < 0:
            violations.append(f"pattern count {pat['count']} is not a non-negative integer")
        elif 0 < pat["count"] < min_lot:
            violations.append(f"pattern used {pat['count']} times, fewer than min_lot {min_lot}")
        for item_id, cnt in sig["counts"].items():
            produced[item_id] = produced.get(item_id, 0) + cnt * max(pat["count"], 0)
    for item_id, item in items.items():
        if produced.get(item_id, 0) < item["demand"] - _EPS:
            violations.append(
                f"item {item_id}: produced {produced.get(item_id, 0)} < demand {item['demand']}"
            )
    # 同じ (原材, 内容) を別行に分けても種類数は増えないので、署名で数える。
    distinct = {sig["key"] for pat, sig in zip(patterns, pattern_sigs) if pat["count"] > 0}
    if _num(max_patterns) and len(distinct) > max_patterns:
        violations.append(f"{len(distinct)} distinct patterns exceed limit {max_patterns}")
    cost = float(sum(
        pat["count"] * stocks[pat["stock"]]["cost"]
        for pat in patterns if pat["stock"] in stocks and pat["count"] > 0
    ))
    violations += _declared_mismatch(solution, cost, "total_cost", "objective_value", "cost")
    return violations, cost


def _check_cutting_1d(instance: dict, solution: Any) -> dict:
    patterns = _parse_patterns(solution)
    if patterns is None:
        return _unverified("cutting solution without a readable 'patterns' list")
    items = {_sid(i["id"]): i for i in instance["items"]}
    stocks = {_sid(s["id"]): s for s in instance["stocks"]}
    violations: list[str] = []
    sigs = []
    for idx, pat in enumerate(patterns):
        counts = _parse_counts(_pick(pat["raw"], "cuts", "items"))
        if counts is None:
            return _unverified(f"pattern {idx} has no readable 'cuts'")
        width = 0.0
        for item_id, cnt in counts.items():
            if item_id not in items:
                violations.append(f"pattern {idx} cuts unknown item {item_id}")
            elif not _is_count(cnt) or cnt < 0:
                violations.append(f"pattern {idx}: item {item_id} count {cnt} is not an integer")
            else:
                width += items[item_id]["width"] * cnt
        stock = stocks.get(pat["stock"])
        if stock is not None and width > stock["length"] + _EPS:
            violations.append(
                f"pattern {idx}: total width {width} exceeds stock {pat['stock']} "
                f"length {stock['length']}"
            )
        sigs.append({"counts": counts,
                     "key": (pat["stock"], tuple(sorted(counts.items())))})
    common, cost = _check_cutting_common(
        items, stocks, solution, patterns, sigs, instance.get("max_distinct_patterns"), 0,
    )
    violations += common
    total = 3 * len(patterns) + len(items) + 2
    return _result(violations, total, cost=cost)


# ---------------------------------------------------------------- 2 次元カッティング
def _check_cutting_2d(instance: dict, solution: Any) -> dict:
    patterns = _parse_patterns(solution)
    if patterns is None:
        return _unverified("cutting solution without a readable 'patterns' list")
    items = {_sid(i["id"]): i for i in instance["items"]}
    plates = {_sid(p["id"]): p for p in instance["plates"]}
    min_lot = instance.get("min_lot", 0) or 0
    violations: list[str] = []
    sigs = []
    for idx, pat in enumerate(patterns):
        strips = pat["raw"].get("strips")
        if not isinstance(strips, list):
            return _unverified(f"pattern {idx} has no 'strips' list")
        plate = plates.get(pat["stock"])
        counts: dict[str, float] = {}
        strip_keys = []
        stacked = 0.0
        for s_idx, strip in enumerate(strips):
            if not isinstance(strip, dict):
                return _unverified(f"pattern {idx} strip {s_idx} is not a mapping")
            strip_items = _parse_counts(_pick(strip, "items", "cuts"))
            if strip_items is None:
                return _unverified(f"pattern {idx} strip {s_idx} has no readable 'items'")
            width = 0.0
            tallest = 0.0
            for item_id, cnt in strip_items.items():
                if item_id not in items:
                    violations.append(f"pattern {idx} strip {s_idx} cuts unknown item {item_id}")
                    continue
                if not _is_count(cnt) or cnt < 0:
                    violations.append(
                        f"pattern {idx} strip {s_idx}: item {item_id} count {cnt} is not an integer"
                    )
                    continue
                width += items[item_id]["width"] * cnt
                tallest = max(tallest, items[item_id]["height"])
                counts[item_id] = counts.get(item_id, 0) + cnt
            # 高さが省略されたストリップは、載る品目の最大高さで切ったとみなす。
            height = strip.get("height", tallest)
            if not _num(height):
                return _unverified(f"pattern {idx} strip {s_idx} height is not numeric")
            if tallest > height + _EPS:
                violations.append(
                    f"pattern {idx} strip {s_idx}: item height {tallest} exceeds strip height {height}"
                )
            if plate is not None and width > plate["width"] + _EPS:
                violations.append(
                    f"pattern {idx} strip {s_idx}: width {width} exceeds plate width {plate['width']}"
                )
            stacked += height
            strip_keys.append((height, tuple(sorted(strip_items.items()))))
        if plate is not None and stacked > plate["height"] + _EPS:
            violations.append(
                f"pattern {idx}: stacked strip height {stacked} exceeds plate height {plate['height']}"
            )
        sigs.append({"counts": counts, "key": (pat["stock"], tuple(sorted(strip_keys)))})
    common, cost = _check_cutting_common(
        items, plates, solution, patterns, sigs, instance.get("max_distinct_patterns"), min_lot,
    )
    violations += common
    total = 4 * len(patterns) + len(items) + 2
    return _result(violations, total, cost=cost)


register_kind(
    "fjsp", _detect_fjsp, lambda inst, sol: _check_fjsp(inst, sol, with_setup=False)
)
register_kind(
    "fjsp_setup", _detect_fjsp_setup, lambda inst, sol: _check_fjsp(inst, sol, with_setup=True)
)
register_kind("cutting_1d", _detect_cutting_1d, _check_cutting_1d)
register_kind("cutting_2d", _detect_cutting_2d, _check_cutting_2d)
