"""大規模問題集の検証器（rosters 群）。register_kind で種別を登録する。

対象種別:
- crew_pairing: 乗務員ペアリング（prob_302, prob_312）。解は `pairings=[{flights:[...]}]` と
  `uncovered_flights`。費用は 固定費 + 便乗務費 + 待機時間×単価、未カバー便は外注費。
- crew_pairing_seniority: 上記 stage1 に加え、stage2 でペアリングをクルーへ割り当てる（prob_330）。
  bid = 拘束時間 × span_pref × 100 (+5000 if 基地不一致)、未割当ペアリングは penalty。
- nurse_roster: 看護師勤務表（prob_306, prob_316）。`roster[nurse] = 日ごとの状態列`。
- role_roster: 役割・地域付き勤務表（prob_323）。`roster[staff] = [{state, region}]`。
目的値は申告値を読まず、instance の費用・ペナルティから再計算する。

解析は「意味が一意に取れる形」まで広げる: record のリスト（`{"nurse_id", "schedule"}` /
`{"staff_id", "day", "shift"}`）、ラベル付きペアリングと `{"pairing_id", "crew_id"}` 型の割当、
`{"p0": [便列]}` / `{"p0": クルー}` 型の stage、`outsourced_flights` などの同義キー。
載っていない人は全日 off として読む。凡例の無い整数コードの勤務表は状態が決まらないので unverified のまま。
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


def _first(record: dict, keys: tuple[str, ...]) -> Any:
    """同義キーのうち最初に存在する値（無ければ None）。"""
    return next((record[k] for k in keys if k in record), None)


def _label(x: Any) -> Any:
    """ペアリング名などのラベル。数値なら int、それ以外は空白を除いた str に寄せる。"""
    as_int = _as_id(x)
    if as_int is not None:
        return as_int
    return x.strip() if isinstance(x, str) and x.strip() else None


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


_PAIRING_LABEL_KEYS = ("id", "pairing_id", "pairing_index", "name")
_UNCOVERED_KEYS = ("uncovered_flights", "outsourced_flights", "uncovered")
# stage1 / stage2 の {ラベル: ...} 形を読むとき、ペアリング名とは見なさない予約キー。
_RESERVED_KEYS = frozenset(_UNCOVERED_KEYS) | {
    "pairings",
    "assignments",
    "unassigned_pairings",
    "generated_columns",
    "cost",
    "num_pairings",
    "num_uncovered",
}


def _pairing_items(container: Any) -> list[tuple[Any, Any]] | None:
    """ペアリング集合を [(ラベル, 生のペアリング)] に寄せる。

    `pairings` がリストなら位置 (1 始まり) をラベルにし、要素 dict に id があればそれを優先する。
    `pairings` が dict、または `pairings` キーが無く全値が便列の dict ({"p0": [...]}) も読む。
    """
    if not isinstance(container, dict):
        return None
    pairings = container.get("pairings")
    if isinstance(pairings, dict):
        return list(pairings.items())
    if isinstance(pairings, list):
        items = []
        for pos, p in enumerate(pairings, 1):
            explicit = _first(p, _PAIRING_LABEL_KEYS) if isinstance(p, dict) else None
            items.append((pos if explicit is None else explicit, p))
        return items
    plain = pairings is None and container and not (set(container) & _RESERVED_KEYS)
    if plain and all(isinstance(v, list) for v in container.values()):
        return list(container.items())
    return None


def _parse_pairings(
    container: Any, flights: dict[int, dict]
) -> list[tuple[Any, list[int]]] | None:
    """ペアリング集合を [(ラベル, 便 ID 列)] へ正規化する。読めなければ None。"""
    items = _pairing_items(container)
    if items is None:
        return None
    parsed = []
    for raw_label, p in items:
        label = _label(raw_label)
        legs = p.get("flights") if isinstance(p, dict) else p
        if label is None or not isinstance(legs, list):
            return None
        ids = [_as_id(x) for x in legs]
        if any(i is None or i not in flights for i in ids):
            return None
        parsed.append((label, ids))
    if len({label for label, _ in parsed}) != len(parsed):
        return None
    return parsed


def _pairing_cost(instance: dict, legs: list[dict]) -> float:
    wait = sum(b["dep_time"] - a["arr_time"] for a, b in pairwise(legs))
    return (
        instance["fixed_cost_per_pairing"]
        + sum(f["flight_cost"] for f in legs)
        + wait * instance["wait_cost_per_hour"]
    )


def _check_pairings(
    instance: dict,
    flights: dict[int, dict],
    pairings: list[list[int]],
    container: dict,
    *,
    strict: bool = False,
) -> tuple[list[str], float, int]:
    """ペアリング制約を検査し、(違反, stage1 費用, 検査した制約数) を返す。

    strict は prob_302 系の規則（各便はちょうど 1 つのペアリングか外注、ペアリングは基地に戻る）。
    prob_312 / prob_330 は「基地発」「各便 1 つ以上」だけなので既定は非 strict。instance の形は
    両者で同じなので、呼び出し側が core_type で切り替える。
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
        if strict and legs[-1]["destination"] not in bases:
            violations.append(f"pairing {idx} ends at non-base {legs[-1]['destination']}")
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
    if strict:
        for fid, times in sorted(covered.items()):
            if times > 1:
                violations.append(f"flight {fid} covered {times} times (each flight exactly once)")
    uncovered = set(flights) - set(covered)
    declared = _first(container, _UNCOVERED_KEYS)
    if isinstance(declared, list):
        declared_ids = {_as_id(x) for x in declared}
        if declared_ids != uncovered:
            violations.append(
                f"uncovered_flights lists {len(declared_ids)} flights but "
                f"{len(uncovered)} are actually uncovered"
            )
    cost += len(uncovered) * instance["uncovered_flight_cost"]
    return violations, cost, (6 if strict else 4) * len(pairings) + 1


def _detect_crew_pairing(instance: dict) -> bool:
    return "flights" in instance and "bases" in instance and "crews" not in instance


def _check_crew_pairing(instance: dict, solution: Any, core_type: str = "") -> dict:
    # prob_302 系は math_type が全角括弧「集合分割（列生成）」、prob_312 系は半角括弧。
    # 規則が違う（302: ちょうど 1 回・基地帰着）のに instance の形は同じなので、ここだけで見分ける。
    strict = "（" in core_type
    flights = _flight_table(instance)
    if flights is None:
        return _unverified("crew pairing instance without flights")
    pairings = _parse_pairings(solution, flights)
    if pairings is None:
        return _unverified("crew pairing without readable 'pairings'")
    violations, cost, total = _check_pairings(
        instance, flights, [ids for _, ids in pairings], solution, strict=strict
    )
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


_CREW_KEYS = ("crew", "crew_id")


def _assignment_items(stage2: Any, labels: set) -> list[tuple[Any, Any]] | None:
    """割当を [(ペアリングラベル, 生の割当)] に寄せる。

    `assignments` が dict ならキーがラベル。record のリストなら pairing_id / id / pairing
    (スカラーの場合) がラベル。`assignments` キーが無く、全キーが stage1 のペアリングラベルで
    値がスカラーの dict ({"p0": 55}) も読む。
    """
    if not isinstance(stage2, dict):
        return None
    raw = stage2.get("assignments")
    if isinstance(raw, dict):
        return list(raw.items())
    if isinstance(raw, list):
        items = []
        for a in raw:
            if not isinstance(a, dict):
                return None
            label = _first(a, ("pairing_id", "pairing_index", "id"))
            if label is None and not isinstance(a.get("pairing"), list):
                label = a.get("pairing")
            items.append((label, a))
        return items
    plain = raw is None and stage2 and not (set(stage2) & _RESERVED_KEYS)
    scalar = all(not isinstance(v, (list, dict)) for v in stage2.values())
    if plain and scalar and {_label(k) for k in stage2} <= labels:
        return list(stage2.items())
    return None


def _parse_assignments(stage2: Any, labels: set) -> dict[Any, tuple[int, list | None]] | None:
    """割当を {ペアリングラベル: (クルー ID, 申告便列)} へ正規化する。"""
    items = _assignment_items(stage2, labels)
    if items is None:
        return None
    parsed = {}
    for raw_label, value in items:
        label = _label(raw_label)
        crew = _as_id(_first(value, _CREW_KEYS)) if isinstance(value, dict) else _as_id(value)
        if label is None or crew is None or label in parsed:
            return None
        legs = _first(value, ("pairing", "flights")) if isinstance(value, dict) else None
        parsed[label] = (crew, legs if isinstance(legs, list) else None)
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
    by_label = dict(pairings)
    assignments = _parse_assignments(stage2, set(by_label))
    if assignments is None:
        return _unverified("stage2 without readable 'assignments'")
    violations, cost, total = _check_pairings(
        instance, flights, [ids for _, ids in pairings], stage1
    )

    duties: Counter = Counter()
    for idx, (crew_id, declared_legs) in sorted(assignments.items(), key=lambda kv: str(kv[0])):
        if idx not in by_label:
            violations.append(f"assignment key {idx} does not name a stage1 pairing")
            continue
        if crew_id not in crews:
            violations.append(f"assignment {idx} names unknown crew {crew_id}")
            continue
        ids = by_label[idx]
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
    unassigned = set(by_label) - set(assignments)
    declared = stage2.get("unassigned_pairings")
    if isinstance(declared, list) and {_label(x) for x in declared} != unassigned:
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


_ROSTER_KEYS = ("roster", "roster_by_staff", "roster_by_nurse", "schedule")
_PERSON_KEYS = (
    "nurse_id",
    "nurse",
    "staff_id",
    "staff",
    "person_id",
    "person",
    "employee_id",
    "employee",
    "id",
)
_ROW_KEYS = ("schedule", "shifts", "days", "states", "roster")
_STATE_KEYS = ("state", "shift")


def _roster_rows_from_records(records: list, days: int) -> dict[Any, list] | None:
    """record のリストを {人キー: 日ごとの行} に組み直す。

    人ごとの record (`{"nurse_id": 1, "schedule": [...]}`) と、(人, 日) ごとの record
    (`{"staff_id": 1, "day": 3, "shift": "late"}`) を読む。日は 1 始まりで全日揃っている
    ことを要求し、欠けがあれば None。
    """
    if not records or not all(isinstance(r, dict) for r in records):
        return None
    if _first(records[0], _ROW_KEYS) is not None:
        rows: dict[Any, list] = {}
        for r in records:
            pid, row = _first(r, _PERSON_KEYS), _first(r, _ROW_KEYS)
            if pid is None or not isinstance(row, list) or pid in rows:
                return None
            rows[pid] = row
        return rows
    if "day" in records[0]:
        cells: dict[tuple[Any, int], Any] = {}
        for r in records:
            pid, day = _first(r, _PERSON_KEYS), _as_id(r.get("day"))
            if pid is None or day is None or (pid, day) in cells:
                return None
            cells[(pid, day)] = r
        people = {pid for pid, _ in cells}
        if len(cells) != len(people) * days:
            return None
        rows = {pid: [cells.get((pid, d)) for d in range(1, days + 1)] for pid in people}
        return None if any(None in row for row in rows.values()) else rows
    return None


def _parse_roster(
    solution: Any, people: dict[int, dict], days: int, states: set[str]
) -> dict[int, list[tuple[str, Any]]] | None:
    """`roster` を {人 ID: [(state, region or None)] × days} へ正規化する。

    載っていない人は全日 off として読む（record 0 件も同じ）。Why not unverified: 状態文字列は
    参照解と同じ語彙なので「勤務が無い」以外に読みようがなく、人員不足は検査で違反と費用に積める。
    instance に居ない人 ID だけは意味が取れないので None。
    """
    if not isinstance(solution, dict):
        return None
    roster = _first(solution, _ROSTER_KEYS)
    if isinstance(roster, list):
        rows = {} if not roster else _roster_rows_from_records(roster, days)
        if rows is None and len(roster) == len(people):
            rows = dict(zip(people, roster))
        roster = rows
    if not isinstance(roster, dict):
        return None
    parsed = {}
    for key, row in roster.items():
        pid = _as_id(key)
        if pid is None or not isinstance(row, list) or len(row) != days:
            return None
        entries = []
        for cell in row:
            state = _first(cell, _STATE_KEYS) if isinstance(cell, dict) else cell
            region = cell.get("region") if isinstance(cell, dict) else None
            if state not in states:
                return None
            entries.append((state, region))
        parsed[pid] = entries
    if not set(parsed) <= set(people):
        return None
    for pid in people:
        parsed.setdefault(pid, [(_OFF, None)] * days)
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
        return _unverified("nurse roster without a readable 'roster'")
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
        return _unverified("role roster without a readable 'roster'")
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
