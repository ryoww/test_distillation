"""大規模問題集の検証器（network_finance 群）。register_kind で種別を登録する。

種別と判定キー:

- mcnd: ``arcs`` + ``od_demands``（``disjoint_paths`` なし）。容量制約付き多品種網設計。
- mcnd_surv: ``arcs`` + ``od_demands`` + ``disjoint_paths``。各 OD を edge-disjoint な
  複数経路に等分し、各経路の距離合計に上限がある網設計。
- portfolio: ``assets`` + ``scenario_returns``（``risk.measure`` が cvar でない）。
  MAD 最小化のカーディナリティ制約ポートフォリオ。
- portfolio_cvar: ``assets`` + ``scenario_returns`` + ``risk.measure == "cvar"``。

目的値の定義は同梱参照解の objective_value を再現するものを採用した:

- 網設計の費用 = opened_arc_ids の固定費合計 + Σ(flow_cost × アーク流量)。流量は od_paths から
  再構成する（arc_flows は申告値なので整合だけ見る）。
- MAD = (1/S) Σ_s |r_s − μ_w|。r_s はシナリオ s のポートフォリオ収益、μ_w = Σ w_i expected_return_i。
  Why not シナリオ平均からの偏差: scenario_returns は平均がほぼ 0 の擾乱で、参照解は
  銘柄の expected_return を中心に偏差を取っている。
- CVaR_α = 損失 L_s = μ_w − r_s の上位 (1−α)S 個の平均（(1−α)S が整数でなければ端数按分）。
- 回転率 = Σ_i |w_i − w0_i|（半分にしない）。
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from ..feasibility_v3_ext import _num, _result, _unverified
from . import register_kind

_FLOW_TOL = 1e-6
_DECLARED_REL_TOL = 5e-3
# 参照解の weights は小数 4〜5 桁に丸められており、半連続の上下限を 1e-4 だけ超えている。
# 丸め由来の超過だけを許すため、重み系の制約は絶対許容差 5e-4 で比較する。
_PORT_TOL = 5e-4
_HOLD_EPS = 1e-9


# ---------------------------------------------------------------------------
# 共通
# ---------------------------------------------------------------------------


def _declared_mismatch(solution: dict, cost: float, *names: str) -> str | None:
    """申告目的値があり、再計算値と 0.5% 以上ずれていれば違反文を返す。"""
    for name in names:
        declared = _num(solution.get(name))
        if declared is None:
            continue
        # Why not max(1, |cost|): ポートフォリオの目的値は 0.1 程度なので、絶対 0.005 の許容では
        # 数 % のずれを見逃す。相対 0.5% に微小な絶対下限だけを付ける。
        if abs(declared - cost) > max(_DECLARED_REL_TOL * abs(cost), 1e-6):
            return f"declared {name} {declared:.6g} differs from recomputed {cost:.6g}"
        return None
    return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def _id_list(value: Any) -> list[int] | None:
    """整数 id のリストとして読めるなら返す。"""
    if not isinstance(value, (list, tuple)):
        return None
    out = []
    for item in value:
        parsed = _int_or_none(item)
        if parsed is None:
            return None
        out.append(parsed)
    return out


def _indexed_entries(value: Any) -> dict[int, Any] | None:
    """OD 番号 → 解の要素。dict（キーは番号）でも list（位置が番号）でも読む。"""
    if isinstance(value, list):
        return dict(enumerate(value))
    if not isinstance(value, dict):
        return None
    out: dict[int, Any] = {}
    for key, item in value.items():
        index = _int_or_none(key)
        if index is None:
            return None
        out[index] = item
    return out


# ---------------------------------------------------------------------------
# 網設計（mcnd / mcnd_surv）
# ---------------------------------------------------------------------------


def _detect_network(instance: dict) -> bool:
    return isinstance(instance.get("arcs"), list) and isinstance(instance.get("od_demands"), list)


def _detect_mcnd(instance: dict) -> bool:
    return _detect_network(instance) and "disjoint_paths" not in instance


def _detect_mcnd_surv(instance: dict) -> bool:
    return _detect_network(instance) and "disjoint_paths" in instance


def _arc_table(instance: dict) -> dict[int, dict]:
    return {int(arc["id"]): arc for arc in instance["arcs"]}


def _walk(
    arcs: dict[int, dict], path: list[int], origin: int, destination: int, label: str
) -> tuple[list[str], float]:
    """経路を辿り、違反文と距離合計を返す。未知アークは距離に数えない。"""
    violations: list[str] = []
    node = origin
    distance = 0.0
    for arc_id in path:
        arc = arcs.get(arc_id)
        if arc is None:
            violations.append(f"{label} uses unknown arc {arc_id}")
            continue
        if arc["from"] != node:
            violations.append(f"{label} is disconnected at arc {arc_id}")
        node = arc["to"]
        distance += float(arc.get("distance", 0.0))
    if node != destination:
        violations.append(f"{label} ends at node {node}, not destination {destination}")
    return violations, distance


def _network_cost_checks(
    instance: dict,
    solution: dict,
    opened: set[int],
    flows: dict[int, float],
    violations: list[str],
) -> tuple[int, float]:
    """開設アークの妥当性、容量、費用、申告値の整合を検査し、検査数と費用を返す。"""
    arcs = _arc_table(instance)
    checks = 0
    unknown = sorted(a for a in opened if a not in arcs)
    checks += 1
    if unknown:
        violations.append(f"opened_arc_ids contains unknown arcs {unknown[:5]}")
    for arc_id, arc in arcs.items():
        checks += 1
        flow = flows.get(arc_id, 0.0)
        capacity = float(arc["capacity"])
        if flow > capacity + _FLOW_TOL:
            violations.append(f"arc {arc_id} carries {flow:.6g} over capacity {capacity:.6g}")
    declared_flows = solution.get("arc_flows")
    if isinstance(declared_flows, dict):
        checks += 1
        mismatched = 0
        for arc_id in set(flows) | {_int_or_none(k) for k in declared_flows}:
            if arc_id is None:
                mismatched += 1
                continue
            declared = _num(declared_flows.get(str(arc_id), declared_flows.get(arc_id, 0.0)))
            if declared is None or abs(declared - flows.get(arc_id, 0.0)) > _FLOW_TOL:
                mismatched += 1
        if mismatched:
            violations.append(f"arc_flows disagrees with od_paths on {mismatched} arcs")
    fixed = sum(float(arcs[a]["fixed_cost"]) for a in opened if a in arcs)
    variable = sum(float(arcs[a]["flow_cost"]) * f for a, f in flows.items() if a in arcs)
    cost = fixed + variable
    checks += 1
    mismatch = _declared_mismatch(solution, cost, "objective_value", "total_cost")
    if mismatch:
        violations.append(mismatch)
    return checks, cost


def _parse_split_paths(entry: Any) -> list[tuple[list[int], float]] | None:
    """mcnd の OD 要素（{path, volume} のリスト）を読む。"""
    if isinstance(entry, dict):
        entry = [entry]
    if not isinstance(entry, list):
        return None
    out = []
    for item in entry:
        if not isinstance(item, dict):
            return None
        path = _id_list(item.get("path"))
        volume = _num(item.get("volume"))
        if path is None or volume is None:
            return None
        out.append((path, volume))
    return out


def _check_mcnd(instance: dict, solution: Any) -> dict:
    if not isinstance(solution, dict):
        return _unverified("solution is not a dict")
    opened_list = _id_list(solution.get("opened_arc_ids"))
    demands = instance["od_demands"]
    entries = _indexed_entries(solution.get("od_paths"))
    if opened_list is None or entries is None:
        return _unverified("opened_arc_ids or od_paths missing")
    parsed: dict[int, list[tuple[list[int], float]]] = {}
    for index, entry in entries.items():
        paths = _parse_split_paths(entry)
        if paths is None:
            return _unverified(f"od_paths[{index}] is not a list of {{path, volume}}")
        parsed[index] = paths

    arcs = _arc_table(instance)
    opened = set(opened_list)
    violations: list[str] = []
    flows: dict[int, float] = defaultdict(float)
    checks = 0
    checks += 1
    stray = sorted(i for i in parsed if i < 0 or i >= len(demands))
    if stray:
        violations.append(f"od_paths has unknown OD indices {stray[:5]}")
    for index, od in enumerate(demands):
        checks += 2
        carried = 0.0
        for path, volume in parsed.get(index, []):
            if volume < 0:
                violations.append(f"OD {index} has negative volume {volume:.6g}")
                continue
            carried += volume
            walk_violations, _ = _walk(arcs, path, od["origin"], od["destination"], f"OD {index}")
            violations.extend(walk_violations)
            for arc_id in path:
                if arc_id in arcs and arc_id not in opened:
                    violations.append(f"OD {index} flows on closed arc {arc_id}")
                flows[arc_id] += volume
        demand = float(od["volume"])
        if abs(carried - demand) > _FLOW_TOL * max(1.0, demand):
            violations.append(f"OD {index} carries {carried:.6g} of demand {demand:.6g}")
    extra, cost = _network_cost_checks(instance, solution, opened, flows, violations)
    return _result(violations, checks + extra, cost=cost)


def _parse_disjoint_entry(entry: Any, count: int) -> tuple[list[list[int]], float | None] | None:
    """mcnd_surv の OD 要素（path1..pathK と volume_each）を読む。"""
    if not isinstance(entry, dict):
        return None
    paths = []
    for k in range(1, count + 1):
        path = _id_list(entry.get(f"path{k}"))
        if path is None:
            return None
        paths.append(path)
    volume_each = entry.get("volume_each")
    if volume_each is None:
        return paths, None
    parsed = _num(volume_each)
    if parsed is None:
        return None
    return paths, parsed


def _check_mcnd_surv(instance: dict, solution: Any) -> dict:
    if not isinstance(solution, dict):
        return _unverified("solution is not a dict")
    count = int(instance["disjoint_paths"])
    opened_list = _id_list(solution.get("opened_arc_ids"))
    demands = instance["od_demands"]
    entries = _indexed_entries(solution.get("od_paths"))
    if opened_list is None or entries is None:
        return _unverified("opened_arc_ids or od_paths missing")
    parsed: dict[int, tuple[list[list[int]], float | None]] = {}
    for index, entry in entries.items():
        item = _parse_disjoint_entry(entry, count)
        if item is None:
            return _unverified(f"od_paths[{index}] lacks path1..path{count}")
        parsed[index] = item

    arcs = _arc_table(instance)
    opened = set(opened_list)
    violations: list[str] = []
    flows: dict[int, float] = defaultdict(float)
    checks = 0
    checks += 1
    stray = sorted(i for i in parsed if i < 0 or i >= len(demands))
    if stray:
        violations.append(f"od_paths has unknown OD indices {stray[:5]}")
    for index, od in enumerate(demands):
        # 経路の妥当性、相違性、遅延、分流量の 4 項目を OD ごとに検査する。
        checks += 4
        demand = float(od["volume"])
        if index not in parsed:
            violations.append(f"OD {index} has no paths")
            continue
        paths, volume_each = parsed[index]
        if volume_each is None:
            volume_each = demand / count
        elif abs(volume_each * count - demand) > _FLOW_TOL * max(1.0, demand):
            violations.append(
                f"OD {index} splits {volume_each:.6g}x{count} instead of demand {demand:.6g}"
            )
        if volume_each < 0:
            violations.append(f"OD {index} has negative volume_each {volume_each:.6g}")
            continue
        used: set[int] = set()
        for k, path in enumerate(paths, start=1):
            label = f"OD {index} path{k}"
            walk_violations, distance = _walk(arcs, path, od["origin"], od["destination"], label)
            violations.extend(walk_violations)
            max_delay = _num(od.get("max_delay"))
            if max_delay is not None and distance > max_delay + _FLOW_TOL:
                violations.append(f"{label} length {distance:.6g} exceeds max_delay {max_delay:.6g}")
            for arc_id in path:
                if arc_id in used:
                    violations.append(f"OD {index} shares arc {arc_id} between paths")
                used.add(arc_id)
                if arc_id in arcs and arc_id not in opened:
                    violations.append(f"{label} flows on closed arc {arc_id}")
                flows[arc_id] += volume_each
    extra, cost = _network_cost_checks(instance, solution, opened, flows, violations)
    return _result(violations, checks + extra, cost=cost)


# ---------------------------------------------------------------------------
# ポートフォリオ（portfolio / portfolio_cvar）
# ---------------------------------------------------------------------------


def _detect_portfolio_any(instance: dict) -> bool:
    return isinstance(instance.get("assets"), list) and isinstance(
        instance.get("scenario_returns"), list
    )


def _is_cvar(instance: dict) -> bool:
    risk = instance.get("risk")
    return isinstance(risk, dict) and str(risk.get("measure", "")).lower() == "cvar"


def _detect_portfolio(instance: dict) -> bool:
    return _detect_portfolio_any(instance) and not _is_cvar(instance)


def _detect_portfolio_cvar(instance: dict) -> bool:
    return _detect_portfolio_any(instance) and _is_cvar(instance)


def _parse_weights(solution: dict, assets: list[dict]) -> dict[int, float] | None:
    """weights を 銘柄 id → 重み に読む。dict でも銘柄数と同じ長さの list でもよい。"""
    raw = solution.get("weights")
    if isinstance(raw, list):
        if len(raw) != len(assets):
            return None
        raw = {asset["id"]: value for asset, value in zip(assets, raw)}
    if not isinstance(raw, dict):
        return None
    out: dict[int, float] = {}
    for key, value in raw.items():
        asset_id = _int_or_none(key)
        weight = _num(value)
        if asset_id is None or weight is None:
            return None
        out[asset_id] = weight
    return out


def _mad(portfolio: list[float], expected: float) -> float:
    return sum(abs(r - expected) for r in portfolio) / len(portfolio)


def _cvar(portfolio: list[float], expected: float, alpha: float) -> float:
    losses = sorted((expected - r for r in portfolio), reverse=True)
    tail = (1.0 - alpha) * len(losses)
    if tail <= 0:
        return losses[0]
    whole = math.floor(tail)
    total = sum(losses[:whole])
    if whole < len(losses):
        total += (tail - whole) * losses[whole]
    return total / tail


def _check_portfolio(instance: dict, solution: Any) -> dict:
    if not isinstance(solution, dict):
        return _unverified("solution is not a dict")
    assets = instance["assets"]
    weights = _parse_weights(solution, assets)
    if weights is None:
        return _unverified("weights missing or not id -> number")
    column = {int(asset["id"]): idx for idx, asset in enumerate(assets)}
    asset_by_id = {int(asset["id"]): asset for asset in assets}
    limits = instance.get("limits") if isinstance(instance.get("limits"), dict) else {}
    violations: list[str] = []
    checks = 0

    checks += 1
    unknown = sorted(a for a in weights if a not in column)
    if unknown:
        violations.append(f"weights contain unknown asset ids {unknown[:5]}")
    held = {a: w for a, w in weights.items() if a in column and w > _HOLD_EPS}

    checks += 1
    negative = sorted(a for a, w in weights.items() if w < -_PORT_TOL)
    if negative:
        violations.append(f"negative weights on assets {negative[:5]}")

    checks += 1
    total = sum(weights.values())
    if abs(total - 1.0) > _PORT_TOL:
        violations.append(f"weights sum to {total:.6g}, not 1")

    checks += 1
    min_hold, max_hold = limits.get("min_holdings"), limits.get("max_holdings")
    if min_hold is not None and len(held) < int(min_hold):
        violations.append(f"{len(held)} holdings below minimum {min_hold}")
    if max_hold is not None and len(held) > int(max_hold):
        violations.append(f"{len(held)} holdings above maximum {max_hold}")

    min_w, max_w = _num(limits.get("min_weight")), _num(limits.get("max_weight"))
    for asset_id, w in held.items():
        checks += 1
        if min_w is not None and w < min_w - _PORT_TOL:
            violations.append(f"asset {asset_id} weight {w:.6g} below min_weight {min_w:.6g}")
        if max_w is not None and w > max_w + _PORT_TOL:
            violations.append(f"asset {asset_id} weight {w:.6g} above max_weight {max_w:.6g}")

    sector_weight: dict[Any, float] = defaultdict(float)
    sector_count: dict[Any, int] = defaultdict(int)
    for asset_id, w in held.items():
        sector = asset_by_id[asset_id].get("sector")
        sector_weight[sector] += w
        sector_count[sector] += 1
    max_sector = _num(limits.get("max_sector_weight"))
    if max_sector is not None:
        for sector, w in sector_weight.items():
            checks += 1
            if w > max_sector + _PORT_TOL:
                violations.append(f"sector {sector} weight {w:.6g} above {max_sector:.6g}")
    min_sectors = limits.get("min_sectors")
    if min_sectors is not None:
        checks += 1
        if len(sector_count) < int(min_sectors):
            violations.append(f"{len(sector_count)} sectors held, below {min_sectors}")

    expected = sum(
        w * float(asset_by_id[a].get("expected_return", asset_by_id[a].get("mean_return", 0.0)))
        for a, w in held.items()
    )
    target = _num(instance.get("target_return"))
    if target is not None:
        checks += 1
        if expected < target - _PORT_TOL:
            violations.append(f"expected return {expected:.6g} below target {target:.6g}")

    current = instance.get("current_portfolio")
    current = current if isinstance(current, dict) else {}
    w0 = {int(k): float(v) for k, v in current.items()}
    max_turnover = _num(limits.get("max_turnover"))
    if max_turnover is not None:
        checks += 1
        turnover = sum(abs(weights.get(a, 0.0) - w0.get(a, 0.0)) for a in set(weights) | set(w0))
        if turnover > max_turnover + _PORT_TOL:
            violations.append(f"turnover {turnover:.6g} above {max_turnover:.6g}")

    if _is_cvar(instance):
        cardinality = instance.get("sector_cardinality")
        if isinstance(cardinality, dict):
            for sector_key, bounds in cardinality.items():
                checks += 1
                sector = _int_or_none(sector_key)
                count = sector_count.get(sector, 0)
                lo, hi = bounds
                if count < int(lo) or count > int(hi):
                    violations.append(f"sector {sector} holds {count}, outside [{lo}, {hi}]")
        for asset_id, w in w0.items():
            checks += 1
            if weights.get(asset_id, 0.0) < w - _PORT_TOL:
                violations.append(f"asset {asset_id} sold below current weight {w:.6g}")

    scenarios = instance["scenario_returns"]
    portfolio = [sum(w * row[column[a]] for a, w in held.items()) for row in scenarios]
    if _is_cvar(instance):
        alpha = _num(instance["risk"].get("alpha"))
        cost = _cvar(portfolio, expected, 0.9 if alpha is None else alpha)
        names = ("objective_value", "cvar")
    else:
        cost = _mad(portfolio, expected)
        names = ("objective_value", "mad_risk", "mad")
    checks += 1
    mismatch = _declared_mismatch(solution, cost, *names)
    if mismatch:
        violations.append(mismatch)
    return _result(violations, checks, cost=cost)


register_kind("mcnd", _detect_mcnd, _check_mcnd)
register_kind("mcnd_surv", _detect_mcnd_surv, _check_mcnd_surv)
register_kind("portfolio", _detect_portfolio, _check_portfolio)
register_kind("portfolio_cvar", _detect_portfolio_cvar, _check_portfolio)
