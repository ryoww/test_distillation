"""大規模問題集の検証器（routing 群）。register_kind で種別を登録する。

- vrptw_md（prob_303 / prob_313）: 複数デポ・複数車種の時間枠付き配送。
- pdptw（prob_321）: 時間枠付き受取・配送ペア。

両種別とも費用は「車両固定費 + 距離単価 × ユークリッド距離 + 未配送ペナルティ」、所要時間は
距離 / 車種速度（時間）。同梱参照解はこの定義で objective_value を 1e-3 未満の差で再現する。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from ..feasibility_v3_ext import _result, _unverified
from . import register_kind

_EPS = 1e-6
# 申告値（total_cost / objective_value）と再計算値のずれをこの相対差まで許す。
_DECLARED_TOL = 5e-3


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _dist(a: dict, b: dict) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def _parse_routes(solution: Any, stop_keys: tuple[str, ...]) -> list[tuple[dict, list]] | str:
    """routes を (route, stops) の列に解析する。解析できなければ理由の文字列を返す。"""
    if not isinstance(solution, dict) or not isinstance(solution.get("routes"), list):
        return "solution must be a dict with a 'routes' list"
    parsed = []
    for index, route in enumerate(solution["routes"]):
        if not isinstance(route, dict) or "depot" not in route or "vehicle_type" not in route:
            return f"route {index} must be a dict with 'depot' and 'vehicle_type'"
        stops = next((route[k] for k in stop_keys if isinstance(route.get(k), list)), None)
        if stops is None:
            return f"route {index} must have a list under one of {stop_keys}"
        parsed.append((route, stops))
    return parsed


def _check_route_timing(
    depot: dict, vehicle: dict, stops: list[dict], label: str, violations: list[str]
) -> float:
    """1 ルートの距離を返しつつ、時間枠・最大稼働時間・営業時間内帰着を検査する。

    出発は営業開始以降で遅らせてよいので、最早スケジュールで計算した所要時間から
    「待機で吸収できる遅延量」を引いたものを稼働時間とする（前方時間スラック）。
    """
    speed = vehicle["speed_kmh"]
    time = depot["open_time"]
    waited = 0.0
    max_delay = math.inf
    distance = 0.0
    previous = depot
    late = 0
    for stop in stops:
        leg = _dist(previous, stop)
        distance += leg
        time += leg / speed
        wait = max(0.0, stop["tw_start"] - time)
        waited += wait
        time += wait
        if time > stop["tw_end"] + _EPS:
            late += 1
        # 出発を delay 遅らせるとこの停留所の開始は max(0, delay - waited) だけ遅れる。
        max_delay = min(max_delay, waited + stop["tw_end"] - time)
        time += stop.get("service_time", 0.0)
        previous = stop
    leg = _dist(previous, depot)
    distance += leg
    time += leg / speed
    if late:
        violations.append(f"{label}: service starts after tw_end at {late} stop(s)")
    duration = (time - depot["open_time"]) - max(0.0, min(waited, max_delay))
    if duration > vehicle["max_route_hours"] + _EPS:
        violations.append(
            f"{label}: route duration {duration:.2f}h exceeds {vehicle['max_route_hours']}h"
        )
    if time > depot["close_time"] + _EPS:
        violations.append(f"{label}: returns at {time:.2f} after depot close {depot['close_time']}")
    return distance


def _check_load(vehicle: dict, stops: list[dict], label: str, violations: list[str]) -> None:
    """累計需要の最大値が積載量以下であることを検査する（vrptw では需要合計と一致）。"""
    load = 0.0
    peak = 0.0
    for stop in stops:
        load += stop["demand"]
        peak = max(peak, load)
    if peak > vehicle["capacity"] + _EPS:
        violations.append(f"{label}: load {peak:g} exceeds capacity {vehicle['capacity']}")


def _check_fleet(used: Counter, depots: dict, violations: list[str]) -> None:
    for (depot_id, vtype), count in sorted(used.items(), key=str):
        available = depots[depot_id].get("fleet", {}).get(vtype, 0)
        if count > available:
            violations.append(
                f"depot {depot_id} uses {count} '{vtype}' vehicles but holds {available}"
            )


def _check_declared(
    solution: dict, cost: float, unserved: set, unserved_key: str, violations: list[str]
) -> None:
    """申告された目的値と未配送リストが再計算と食い違えば違反にする。"""
    for key in ("total_cost", "objective_value"):
        declared = solution.get(key)
        if _is_num(declared) and abs(declared - cost) > _DECLARED_TOL * max(1.0, abs(cost)):
            violations.append(f"declared {key} {declared} differs from recomputed {cost:.2f}")
            break  # 同じずれを二重に数えない
    declared_unserved = solution.get(unserved_key)
    if isinstance(declared_unserved, list) and set(declared_unserved) != unserved:
        violations.append(f"declared {unserved_key} disagrees with the routes")


# ----------------------------------------------------------------------------
# vrptw_md
# ----------------------------------------------------------------------------


def _detect_vrptw_md(instance: dict) -> bool:
    keys = ("depots", "customers", "vehicle_types")
    return all(isinstance(instance.get(k), list) for k in keys) and "pairs" not in instance


def _check_vrptw_md(instance: dict, solution: Any) -> dict:
    parsed = _parse_routes(solution, ("customers", "sequence"))
    if isinstance(parsed, str):
        return _unverified(parsed)
    depots = {d["id"]: d for d in instance["depots"]}
    customers = {c["id"]: c for c in instance["customers"]}
    vehicles = {v["type"]: v for v in instance["vehicle_types"]}
    violations: list[str] = []
    visits: Counter = Counter()
    used: Counter = Counter()
    cost = 0.0
    for index, (route, stops) in enumerate(parsed):
        label = f"route {index}"
        depot = depots.get(route["depot"]) if _is_num(route["depot"]) else None
        vehicle = vehicles.get(route["vehicle_type"]) if isinstance(route["vehicle_type"], str) else None
        if depot is None or vehicle is None:
            violations.append(f"{label}: unknown depot or vehicle_type")
            continue
        used[(depot["id"], vehicle["type"])] += 1
        known = []
        for cid in stops:
            if _is_num(cid) and cid in customers:
                known.append(customers[cid])
                visits[cid] += 1
            else:
                violations.append(f"{label}: unknown customer {cid!r}")
        _check_load(vehicle, known, label, violations)
        distance = _check_route_timing(depot, vehicle, known, label, violations)
        cost += vehicle["fixed_cost"] + vehicle["cost_per_km"] * distance
    repeated = sorted(cid for cid, n in visits.items() if n > 1)
    if repeated:
        violations.append(f"customers visited more than once: {repeated[:10]}")
    unserved = set(customers) - set(visits)
    cost += instance["unserved_penalty"] * len(unserved)
    _check_fleet(used, depots, violations)
    _check_declared(solution, cost, unserved, "unserved_customers", violations)
    return _result(violations, len(parsed) * 5 + 3, cost=cost)


# ----------------------------------------------------------------------------
# pdptw
# ----------------------------------------------------------------------------


def _detect_pdptw(instance: dict) -> bool:
    keys = ("depots", "pairs", "vehicle_types")
    return all(isinstance(instance.get(k), list) for k in keys) and "customers" not in instance


# "P12" / "D12" / "pickup_12" / "delivery-12" のような文字列停留所（大文字小文字は区別しない）。
_STOP_TEXT = re.compile(r"^(p|d|pickup|delivery|dropoff)[ _:-]?(\d+)$", re.IGNORECASE)
_KIND_ALIASES = {"p": "pickup", "pickup": "pickup", "d": "delivery", "delivery": "delivery",
                 "dropoff": "delivery"}


def _as_id(value: Any) -> Any:
    """数値 id が "12" のように文字列化されていれば int に戻す。それ以外はそのまま返す。"""
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return value


def _parse_stop(stop: Any) -> tuple[Any, Any] | None:
    """stop を (pair_id, kind) に揃える。読めない形は None。

    [id, kind]、{"pair"/"pair_id"/"id", "type"/"kind"}（座標などの付随キーは無視）、"P12"/"D12" を受ける。
    kind の無い裸の id は受取か配送か決まらないので読まない。
    """
    if isinstance(stop, str):
        match = _STOP_TEXT.match(stop.strip())
        return (int(match.group(2)), _KIND_ALIASES[match.group(1).lower()]) if match else None
    if isinstance(stop, (list, tuple)) and len(stop) == 2:
        pair_id, kind = stop
    elif isinstance(stop, dict):
        pair_id = next((stop[k] for k in ("pair", "pair_id", "id") if k in stop), None)
        kind = next((stop[k] for k in ("type", "kind") if k in stop), None)
    else:
        return None
    if pair_id is None or not isinstance(kind, str):
        return None
    return _as_id(pair_id), _KIND_ALIASES.get(kind.strip().lower(), kind)


def _strip_depot_ends(stops: list, depot_id: Any) -> list:
    """先頭・末尾に置かれたデポ id（裸の数値）は出発・帰着の印として読み飛ばす。

    裸の数値は kind を持たず pair 停留所にはなり得ないので、デポ id と一致する端の要素だけ外す。
    途中や、デポ id と一致しない裸の数値は _parse_stop で None になり unverified のまま。
    """
    if stops and _is_num(stops[0]) and stops[0] == depot_id:
        stops = stops[1:]
    if stops and _is_num(stops[-1]) and stops[-1] == depot_id:
        stops = stops[:-1]
    return stops


def _check_pdptw(instance: dict, solution: Any) -> dict:
    parsed = _parse_routes(solution, ("stops",))
    if isinstance(parsed, str):
        return _unverified(parsed)
    depots = {d["id"]: d for d in instance["depots"]}
    pairs = {p["id"]: p for p in instance["pairs"]}
    vehicles = {v["type"]: v for v in instance["vehicle_types"]}
    violations: list[str] = []
    seen: dict[Any, list[tuple[int, str]]] = {}  # pair_id -> [(route index, kind)]
    used: Counter = Counter()
    cost = 0.0
    for index, (route, stops) in enumerate(parsed):
        label = f"route {index}"
        depot = depots.get(route["depot"]) if _is_num(route["depot"]) else None
        vehicle = vehicles.get(route["vehicle_type"]) if isinstance(route["vehicle_type"], str) else None
        if depot is None or vehicle is None:
            violations.append(f"{label}: unknown depot or vehicle_type")
            continue
        used[(depot["id"], vehicle["type"])] += 1
        known = []
        order: dict[Any, dict[str, int]] = {}
        for position, raw in enumerate(_strip_depot_ends(stops, depot["id"])):
            stop = _parse_stop(raw)
            if stop is None:
                return _unverified(f"{label}: stop {raw!r} is not [pair_id, kind]")
            pair_id, kind = stop
            if pair_id not in pairs or kind not in ("pickup", "delivery"):
                violations.append(f"{label}: unknown stop {raw!r}")
                continue
            known.append(pairs[pair_id][kind])
            order.setdefault(pair_id, {})[kind] = position
            seen.setdefault(pair_id, []).append((index, kind))
        broken = [
            pid for pid, pos in order.items()
            if "pickup" not in pos or "delivery" not in pos or pos["pickup"] > pos["delivery"]
        ]
        if broken:
            violations.append(f"{label}: pickup must precede delivery in the same route: {broken}")
        _check_load(vehicle, known, label, violations)
        distance = _check_route_timing(depot, vehicle, known, label, violations)
        cost += vehicle["fixed_cost"] + vehicle["cost_per_km"] * distance
    repeated = sorted(
        pid for pid, stops in seen.items()
        if len({r for r, _ in stops}) > 1 or any(n > 1 for n in Counter(k for _, k in stops).values())
    )
    if repeated:
        violations.append(f"pairs served more than once or split across routes: {repeated[:10]}")
    unserved = set(pairs) - set(seen)
    cost += instance["unserved_penalty"] * len(unserved)
    _check_fleet(used, depots, violations)
    _check_declared(solution, cost, unserved, "unserved_pairs", violations)
    return _result(violations, len(parsed) * 6 + 3, cost=cost)


register_kind("vrptw_md", _detect_vrptw_md, _check_vrptw_md)
register_kind("pdptw", _detect_pdptw, _check_pdptw)
