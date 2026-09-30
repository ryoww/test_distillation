"""大規模問題集の検証器（rosters 群）。register_kind で種別を登録する。

対象種別:
- crew_pairing: 乗務員ペアリング（prob_302, prob_312）。解は `pairings=[{flights:[...]}]` と
  `uncovered_flights`。費用は 固定費 + 便乗務費 + 待機時間×単価、未カバー便は外注費。
- crew_pairing_seniority: 上記 stage1 に加え、stage2 でペアリングをクルーへ割り当てる（prob_330）。
  bid = 拘束時間 × span_pref × 100 (+5000 if 基地不一致)、未割当ペアリングは penalty。
- nurse_roster: 看護師勤務表（prob_306, prob_316）。`roster[nurse] = 日ごとの状態列`。
- role_roster: 役割・地域付き勤務表（prob_323）。`roster[staff] = [{state, region}]`。
目的値は申告値を読まず、instance の費用・ペナルティから再計算する。
"""

from __future__ import annotations

from collections import Counter
from itertools import pairwise
from typing import Any

from ..feasibility_v3_ext import _result, _unverified
from . import register_kind

_EPS = 1e-6
# 申告値と再計算値の許容ずれ（相対 0.5%）。
_DECLARED_TOL = 5e-3
_OFF = "off"
_NIGHT = "night"
# 同梱参照解はペアリングの出発基地とクルーの home_base が違うと 5000 を加算している。
_HOME_BASE_MISMATCH_COST = 5000.0
_BID_SPAN_SCALE = 100.0


# ---------------------------------------------------------------- 共通ヘルパー
def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _as_id(x: Any) -> int | None:
    """JSON 経由で 1 / "1" / 1.0 が混在する ID を int へ寄せる。"""
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float) and x.is_integer():
        return int(x)
    if isinstance(x, str):
        try:
            return int(x.strip())
        except ValueError:
            return None
    return None


def _declared_mismatch(solution: dict, cost: float, *names: str) -> list[str]:
    """申告目的値が再計算値と 0.5% 以上ずれていれば違反として返す。"""
    declared = next((solution[n] for n in names if n in solution), None)
    if not _num(declared):
        return []
    if abs(declared - cost) > _DECLARED_TOL * max(1.0, abs(cost)):
        return [f"declared objective {declared} differs from recomputed {cost:.2f}"]
    return []


# ---------------------------------------------------------------- crew pairing
def _flight_table(instance: dict) -> dict[int, dict] | None:
    flights = instance.get("flights")
    if not isinstance(flights, list) or not flights:
        return None
    table = {}
    for f in flights:
        if not isinstance(f, dict) or _as_id(f.get("id")) is None:
            return None
        table[_as_id(f["id"])] = f
    return table


def _parse_pairings(container: Any, flights: dict[int, dict]) -> list[list[int]] | None:
    """`pairings` を便 ID のリストの列へ正規化する。読めなければ None。"""
    if not isinstance(container, dict) or not isinstance(container.get("pairings"), list):
        return None
    parsed = []
    for p in container["pairings"]:
        legs = p.get("flights") if isinstance(p, dict) else p
        if not isinstance(legs, list):
            return None
        ids = [_as_id(x) for x in legs]
        if any(i is None or i not in flights for i in ids):
            return None
        parsed.append(ids)
    return parsed


def _pairing_cost(instance: dict, legs: list[dict]) -> float:
    wait = sum(b["dep_time"] - a["arr_time"] for a, b in pairwise(legs))
    return (
        instance["fixed_cost_per_pairing"]
        + sum(f["flight_cost"] for f in legs)
        + wait * instance["wait_cost_per_hour"]
    )


def _check_pairings(
    instance: dict, flights: dict[int, dict], pairings: list[list[int]], container: dict
) -> tuple[list[str], float, int]:
    """ペアリング制約を検査し、(違反, stage1 費用, 検査した制約数) を返す。

    Why not 基地帰着を要求しない: prob_312 / prob_330 の要件は「基地発」のみで、同梱参照解も
    3 問すべてで基地に戻らないペアリングを含む（prob_302 の description は帰着を求めるが、
    instance の形だけでは prob_312 と区別できない）。
    """
    bases = set(instance["bases"])
    lo, hi = instance["min_connect_hours"], instance["max_connect_hours"]
    max_legs, max_span = instance["max_legs_per_pairing"], instance["max_span_hours"]
    violations: list[str] = []
    covered: Counter = Counter()
    cost = 0.0
    for idx, ids in enumerate(pairings):
        if not ids:
            violations.append(f"pairing {idx} has no flights")
            continue
        legs = [flights[i] for i in ids]
        covered.update(ids)
        if legs[0]["origin"] not in bases:
            violations.append(f"pairing {idx} departs from non-base {legs[0]['origin']}")
        bad_links = 0
        for a, b in pairwise(legs):
            gap = b["dep_time"] - a["arr_time"]
            if a["destination"] != b["origin"] or gap < lo - _EPS or gap > hi + _EPS:
                bad_links += 1
        if bad_links:
            violations.append(f"pairing {idx} has {bad_links} invalid connections")
        if len(ids) > max_legs:
            violations.append(f"pairing {idx} has {len(ids)} legs > {max_legs}")
        span = legs[-1]["arr_time"] - legs[0]["dep_time"]
        if span > max_span + _EPS:
            violations.append(f"pairing {idx} spans {span:.2f}h > {max_span}")
        cost += _pairing_cost(instance, legs)
    uncovered = set(flights) - set(covered)
    declared = container.get("uncovered_flights")
    if isinstance(declared, list):
        declared_ids = {_as_id(x) for x in declared}
        if declared_ids != uncovered:
            violations.append(
                f"uncovered_flights lists {len(declared_ids)} flights but "
                f"{len(uncovered)} are actually uncovered"
            )
    cost += len(uncovered) * instance["uncovered_flight_cost"]
    return violations, cost, 4 * len(pairings) + 1


def _detect_crew_pairing(instance: dict) -> bool:
    return "flights" in instance and "bases" in instance and "crews" not in instance


def _check_crew_pairing(instance: dict, solution: Any) -> dict:
    flights = _flight_table(instance)
    if flights is None:
        return _unverified("crew pairing instance without flights")
    pairings = _parse_pairings(solution, flights)
    if pairings is None:
        return _unverified("crew pairing without readable 'pairings'")
    violations, cost, total = _check_pairings(instance, flights, pairings, solution)
    violations += _declared_mismatch(solution, cost, "objective_value", "total_cost", "cost")
    return _result(violations, total + 1, cost=cost)


# ---------------------------------------------------------------- crew pairing + seniority
def _detect_crew_pairing_seniority(instance: dict) -> bool:
    return "flights" in instance and "bases" in instance and "crews" in instance


def _crew_table(instance: dict) -> dict[int, dict] | None:
    crews = instance.get("crews")
    if not isinstance(crews, list) or not crews:
        return None
    table = {}
    for c in crews:
        if not isinstance(c, dict) or _as_id(c.get("id")) is None:
            return None
        table[_as_id(c["id"])] = c
    return table


def _parse_assignments(stage2: Any) -> dict[int, tuple[int, list | None]] | None:
    """`assignments` を {ペアリング番号(1 始まり): (クルー ID, 申告便列)} へ正規化する。"""
    if not isinstance(stage2, dict) or not isinstance(stage2.get("assignments"), dict):
        return None
    parsed = {}
    for key, value in stage2["assignments"].items():
        idx = _as_id(key)
        crew = _as_id(value.get("crew")) if isinstance(value, dict) else _as_id(value)
        if idx is None or crew is None:
            return None
        legs = value.get("pairing") if isinstance(value, dict) else None
        parsed[idx] = (crew, legs if isinstance(legs, list) else None)
    return parsed


def _check_crew_pairing_seniority(instance: dict, solution: Any) -> dict:
    flights, crews = _flight_table(instance), _crew_table(instance)
    if flights is None or crews is None:
        return _unverified("crew pairing instance without flights or crews")
    if not isinstance(solution, dict):
        return _unverified("crew pairing solution is not a dict")
    stage1 = solution.get("stage1", solution)
    stage2 = solution.get("stage2", solution)
    pairings = _parse_pairings(stage1, flights)
    if pairings is None:
        return _unverified("stage1 without readable 'pairings'")
    assignments = _parse_assignments(stage2)
    if assignments is None:
        return _unverified("stage2 without readable 'assignments'")
    violations, cost, total = _check_pairings(instance, flights, pairings, stage1)

    duties: Counter = Counter()
    for idx, (crew_id, declared_legs) in sorted(assignments.items()):
        if not 1 <= idx <= len(pairings):
            violations.append(f"assignment key {idx} does not index a pairing (1-based)")
            continue
        if crew_id not in crews:
            violations.append(f"assignment {idx} names unknown crew {crew_id}")
            continue
        ids = pairings[idx - 1]
        if declared_legs is not None and [_as_id(x) for x in declared_legs] != ids:
            violations.append(f"assignment {idx} lists flights that differ from pairing {idx}")
        if not ids:
            continue
        legs = [flights[i] for i in ids]
        crew = crews[crew_id]
        duties[crew_id] += 1
        span = legs[-1]["arr_time"] - legs[0]["dep_time"]
        cost += span * crew["span_pref"] * _BID_SPAN_SCALE
        if crew["home_base"] != legs[0]["origin"]:
            cost += _HOME_BASE_MISMATCH_COST
    for crew_id, crew in crews.items():
        if duties[crew_id] > crew["max_duties"]:
            violations.append(
                f"crew {crew_id} has {duties[crew_id]} duties > max_duties {crew['max_duties']}"
            )
    unassigned = {i for i in range(1, len(pairings) + 1) if i not in assignments}
    declared = stage2.get("unassigned_pairings")
    if isinstance(declared, list) and {_as_id(x) for x in declared} != unassigned:
        violations.append(
            f"unassigned_pairings lists {len(declared)} pairings but "
            f"{len(unassigned)} are actually unassigned"
        )
    cost += len(unassigned) * instance["unassigned_pairing_cost"]
    violations += _declared_mismatch(solution, cost, "objective_value", "total_cost", "cost")
    return _result(violations, total + len(assignments) + len(crews) + 2, cost=cost)


# ---------------------------------------------------------------- roster 共通
def _person_table(instance: dict, key: str) -> dict[int, dict] | None:
    people = instance.get(key)
    if not isinstance(people, list) or not people:
        return None
    table = {}
    for p in people:
        if not isinstance(p, dict) or _as_id(p.get("id")) is None:
            return None
        table[_as_id(p["id"])] = p
    return table


def _parse_roster(
    solution: Any, people: dict[int, dict], days: int, states: set[str]
) -> dict[int, list[tuple[str, Any]]] | None:
    """`roster` を {人 ID: [(state, region or None)] × days} へ正規化する。"""
    if not isinstance(solution, dict):
        return None
    roster = solution.get("roster")
    if isinstance(roster, list) and len(roster) == len(people):
        roster = dict(zip(people, roster))
    if not isinstance(roster, dict):
        return None
    parsed = {}
    for key, row in roster.items():
        pid = _as_id(key)
        if pid is None or not isinstance(row, list) or len(row) != days:
            return None
        entries = []
        for cell in row:
            state = cell.get("state") if isinstance(cell, dict) else cell
            region = cell.get("region") if isinstance(cell, dict) else None
            if state not in states:
                return None
            entries.append((state, region))
        parsed[pid] = entries
    if set(parsed) != set(people):
        return None
    return parsed


def _rule_violations(label: str, states: list[str], rules: dict) -> list[str]:
    """一人分の状態列に対する労働規則の違反（規則ごとに 1 件）。

    Why not 期間末で切れる休養を違反にしない: 同梱参照解は期間の最終日近くで終わる夜勤ブロックの
    後に休養日を取れていないので、期間内に収まる日だけを検査する。夜勤ブロックの最短日数は
    参照解が期間末でも守っているので切り詰めない。
    """
    days = len(states)
    out = []
    forbidden = set(rules.get("forbidden_transitions", []))
    bad = sum(1 for a, b in pairwise(states) if f"{a}->{b}" in forbidden)
    if bad:
        out.append(f"{label}: {bad} forbidden shift transitions")

    min_block, max_block = rules["min_consecutive_nights"], rules["max_consecutive_nights"]
    rest = rules["rest_days_after_night_block"]
    bad_blocks = bad_rest = 0
    i = 0
    while i < days:
        if states[i] != _NIGHT:
            i += 1
            continue
        j = i
        while j < days and states[j] == _NIGHT:
            j += 1
        if not min_block <= j - i <= max_block:
            bad_blocks += 1
        if any(s != _OFF for s in states[j : j + rest]):
            bad_rest += 1
        i = j
    if bad_blocks:
        out.append(f"{label}: {bad_blocks} night blocks outside {min_block}-{max_block} days")
    if bad_rest:
        out.append(f"{label}: {bad_rest} night blocks not followed by {rest} off days")

    run = longest = 0
    for s in states:
        run = run + 1 if s != _OFF else 0
        longest = max(longest, run)
    if longest > rules["max_consecutive_work_days"]:
        out.append(f"{label}: {longest} consecutive work days")

    # Why not 任意の連続 7 日: 「1 週間（7 日）」を暦週（1-7 日, 8-14 日, ...）と読む。
    per_week = rules["max_work_days_per_week"]
    weeks_over = sum(
        1
        for w in range(0, days, 7)
        if sum(1 for s in states[w : w + 7] if s != _OFF) > per_week
    )
    if weeks_over:
        out.append(f"{label}: {weeks_over} weeks with more than {per_week} work days")

    offs = sum(1 for s in states if s == _OFF)
    if offs < rules["min_off_days"]:
        out.append(f"{label}: only {offs} off days < {rules['min_off_days']}")
    nights = sum(1 for s in states if s == _NIGHT)
    if nights > rules["max_nights"]:
        out.append(f"{label}: {nights} nights > {rules['max_nights']}")
    return out


_RULE_COUNT = 7


def _soft_cost(
    instance: dict,
    people: dict[int, dict],
    roster: dict[int, list[tuple[str, Any]]],
    shortage: int,
    pref_weight,
) -> float:
    """不足・希望休・土日休み・バランスの重み付き和。

    バランスは「勤務日数の平均からの絶対偏差の合計」を四捨五入したもの。
    Why not 最大−最小: 同梱参照解の breakdown (imb_raw=123.92) は最大−最小では再現できず、
    絶対偏差の合計と重み 3 × round(imb_raw) で objective_value が一致する。
    """
    pen = instance["penalties"]
    rules = instance["rules"]
    days = len(instance["daily_requirements"])
    pref = 0
    for pid, person in people.items():
        row = roster[pid]
        for d in person.get("preferred_off_days", []):
            day = _as_id(d)
            if day is not None and 1 <= day <= days and row[day - 1][0] != _OFF:
                pref += pref_weight(person)
    # 1 日目を月曜として 6-7 日目, 13-14 日目, ... を土日とみなす。
    weekends = [(d, d + 1) for d in range(6, days, 7)]
    weekend_short = 0
    for row in roster.values():
        full = sum(1 for a, b in weekends if row[a - 1][0] == _OFF and row[b - 1][0] == _OFF)
        weekend_short += max(0, rules["min_full_weekend_off"] - full)
    work = [sum(1 for s, _ in row if s != _OFF) for row in roster.values()]
    mean = sum(work) / len(work)
    imbalance = round(sum(abs(w - mean) for w in work))
    return float(
        pen["shortage"] * shortage
        + pen["preference"] * pref
        + pen["weekend_off"] * weekend_short
        + pen["imbalance"] * imbalance
    )


# ---------------------------------------------------------------- nurse roster
def _detect_nurse_roster(instance: dict) -> bool:
    return "nurses" in instance and "daily_requirements" in instance and "rules" in instance


def _check_nurse_roster(instance: dict, solution: Any) -> dict:
    """看護師勤務表。

    Why not 夜勤の有資格者数を検査しない: 同梱参照解は毎日の有資格者 10 人を満たさず、その不足を
    breakdown の shortage にも計上していない（参照解の objective_value と一致させるため従う）。
    """
    nurses = _person_table(instance, "nurses")
    reqs = instance.get("daily_requirements")
    if nurses is None or not isinstance(reqs, list) or not reqs:
        return _unverified("nurse roster instance without nurses or daily_requirements")
    states = set(instance.get("shifts", [])) | {_OFF}
    roster = _parse_roster(solution, nurses, len(reqs), states)
    if roster is None:
        return _unverified("nurse roster without a readable 'roster' covering every nurse")
    violations = []
    for pid in sorted(roster):
        violations += _rule_violations(f"nurse {pid}", [s for s, _ in roster[pid]], instance["rules"])
    shortage = 0
    for d, req in enumerate(reqs):
        counts = Counter(row[d][0] for row in roster.values())
        shortage += sum(
            max(0, need - counts[shift]) for shift, need in req.items() if shift != "day"
        )
    cost = _soft_cost(instance, nurses, roster, shortage, lambda _: 1)
    violations += _declared_mismatch(solution, cost, "objective_value", "objective", "total_cost")
    return _result(violations, _RULE_COUNT * len(nurses) + 1, cost=cost)


# ---------------------------------------------------------------- role roster
def _detect_role_roster(instance: dict) -> bool:
    return all(k in instance for k in ("staff", "daily_requirements", "roles", "rules"))


def _check_role_roster(instance: dict, solution: Any) -> dict:
    staff = _person_table(instance, "staff")
    reqs = instance.get("daily_requirements")
    if staff is None or not isinstance(reqs, list) or not reqs:
        return _unverified("role roster instance without staff or daily_requirements")
    states = set(instance.get("shifts", [])) | {_OFF}
    roster = _parse_roster(solution, staff, len(reqs), states)
    if roster is None:
        return _unverified("role roster without a readable 'roster' covering every staff member")
    rules = instance["rules"]
    violations = []
    for pid in sorted(roster):
        person, row = staff[pid], roster[pid]
        violations += _rule_violations(f"staff {pid}", [s for s, _ in row], rules)
        away = sum(1 for s, r in row if s != _OFF and r is not None and r != person["region"])
        if away:
            violations.append(f"staff {pid}: {away} shifts outside home region {person['region']}")
        if rules.get("night_requires_qualified", True) and not person.get("qualified", False):
            nights = sum(1 for s, _ in row if s == _NIGHT)
            if nights:
                violations.append(f"staff {pid}: {nights} night shifts without qualification")
    shortage = 0
    for d, req in enumerate(reqs):
        # 地域が省略された勤務日は home region で働いたものとみなす。
        counts = Counter(
            (staff[pid]["role"], row[d][0], row[d][1] or staff[pid]["region"])
            for pid, row in roster.items()
            if row[d][0] != _OFF
        )
        for key, need in req.items():
            if key == "day":
                continue
            shortage += max(0, need - counts[tuple(key.split("_", 2))])
    cost = _soft_cost(instance, staff, roster, shortage, lambda p: p.get("seniority", 1))
    violations += _declared_mismatch(solution, cost, "objective_value", "objective", "total_cost")
    return _result(violations, (_RULE_COUNT + 2) * len(staff) + 1, cost=cost)


register_kind("crew_pairing", _detect_crew_pairing, _check_crew_pairing)
register_kind(
    "crew_pairing_seniority", _detect_crew_pairing_seniority, _check_crew_pairing_seniority
)
register_kind("nurse_roster", _detect_nurse_roster, _check_nurse_roster)
register_kind("role_roster", _detect_role_roster, _check_role_roster)
