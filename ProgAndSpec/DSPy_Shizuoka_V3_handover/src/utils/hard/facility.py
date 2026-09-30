"""大規模問題集の検証器（facility 群）。register_kind で種別を登録する。

facility_multi  : 多期間・単一供給元の施設配置・在庫統合（prob_304 / prob_314）
facility_2ech   : 工場→DC→顧客の 2 階層（prob_328）
facility_robust : シナリオ min-max（prob_329）

facility_multi は解に調達・在庫計画が含まれるのでそれを検査し、目的値をそこから再計算する。
facility_2ech と facility_robust は参照解に運用計画が含まれず、開設・割当を固定すれば残りは
線形計画で一意に最適化できるので、検証器が LP を解いて計画を導出し目的値を求める。
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
from scipy.optimize import linprog

from ..feasibility_v3_ext import _result, _unverified
from . import register_kind

# 参照解の計画は小数 1 桁へ丸められており、収支の残差が 0.1 程度出るので、その分を許す。
_PLAN_TOL = 0.15
# 申告目的値と再計算値の許容相対差。
_OBJ_REL_TOL = 5e-3


# ---------------------------------------------------------------- 共通ヘルパ
def _dist(a: dict, b: dict) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def _coef(formula: Any, default: float) -> float:
    """"0.05 * 距離 * 数量" のような式文字列から先頭の係数を読む。"""
    match = re.match(r"\s*([0-9]*\.?[0-9]+)", str(formula or ""))
    return float(match.group(1)) if match else default


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _id_list(value: Any, valid: set[int]) -> set[int] | None:
    """id のリストを集合に読む。未知の id や型違いは None。"""
    if not isinstance(value, list):
        return None
    ids = set()
    for item in value:
        if not _is_num(item) or int(item) != item or int(item) not in valid:
            return None
        ids.add(int(item))
    return ids


def _id_map(value: Any, keys: set[int], values: set[int]) -> dict[int, int] | None:
    """{顧客 id: 施設 id} 形式の割当を読む。キー・値とも既知 id でなければ None。"""
    if not isinstance(value, dict):
        return None
    out: dict[int, int] = {}
    for key, item in value.items():
        try:
            k = int(key)
        except (TypeError, ValueError):
            return None
        if k not in keys or not _is_num(item) or int(item) != item or int(item) not in values:
            return None
        out[k] = int(item)
    return out


def _plan(value: Any, valid: set[int], periods: int) -> dict[int, list[float]] | None:
    """{施設 id: [期別の量]} 形式の計画を読む。長さや型が合わなければ None。"""
    if not isinstance(value, dict):
        return None
    out: dict[int, list[float]] = {}
    for key, item in value.items():
        try:
            k = int(key)
        except (TypeError, ValueError):
            return None
        if k not in valid or not isinstance(item, list) or len(item) != periods:
            return None
        if not all(_is_num(q) for q in item):
            return None
        out[k] = [float(q) for q in item]
    return out


def _outflows(
    customers: list[dict], assignment: dict[int, int], periods: int, factors: dict | None = None
) -> dict[int, list[float]]:
    """割当から施設ごとの期別出荷量を集計する。factors はシナリオの需要乗数。"""
    out: dict[int, list[float]] = {}
    for c in customers:
        f = assignment.get(c["id"])
        if f is None:
            continue
        scale = 1.0 if factors is None else float(factors.get(str(c["id"]), 1.0))
        row = out.setdefault(f, [0.0] * periods)
        for t in range(periods):
            row[t] += c["demand"][t] * scale
    return out


def _check_assignment(
    customers: list[dict],
    assignment: dict[int, int],
    opened: set[int],
    cand_key: str,
    violations: list[str],
) -> None:
    """全顧客がちょうど 1 つの開設済み候補施設へ割り当てられているかを見る。"""
    for c in customers:
        f = assignment.get(c["id"])
        if f is None:
            violations.append(f"customer {c['id']} is not assigned")
        elif f not in c[cand_key]:
            violations.append(f"customer {c['id']} assigned to non-candidate {f}")
        elif f not in opened:
            violations.append(f"customer {c['id']} assigned to closed facility {f}")


def _declared_mismatch(solution: dict, cost: float, violations: list[str]) -> None:
    """申告目的値があれば再計算値との差を検査する（申告値自体は cost に使わない）。"""
    for key in ("objective_value", "total_cost"):
        declared = solution.get(key)
        if _is_num(declared) and abs(declared - cost) > _OBJ_REL_TOL * max(1.0, abs(cost)):
            violations.append(f"declared {key} {declared} differs from recomputed {cost:.2f}")
            return


def _min_cost_plan(nodes: list[dict], periods: int, plant: dict | None = None) -> float | None:
    """出荷量を固定した調達・在庫の最小費用を LP で求める。実行不能なら None。

    nodes の各要素: out(期別出荷), rate, storage, holding, unit_cost(調達 1 単位の費用),
    proc_cap(期あたり調達上限、None なら無制限), init。plant を渡すと各 node の調達を
    工場の生産(cap, prod_cost)と工場在庫(inv_cost, max_inv, init)でまかなう制約が加わる。
    Why not 逐次計算: 安全在庫と調達上限が期をまたいで絡むので閉形式にならない。
    """
    n = len(nodes)
    width = 2 * n * periods + (2 * periods if plant else 0)
    cost = np.zeros(width)
    lower = np.zeros(width)
    upper = np.full(width, np.inf)
    rows = n * periods + (periods if plant else 0)
    a_eq = np.zeros((rows, width))
    b_eq = np.zeros(rows)
    for i, node in enumerate(nodes):
        proc, inv = 2 * i * periods, (2 * i + 1) * periods
        cost[proc : proc + periods] = node["unit_cost"]
        cost[inv : inv + periods] = node["holding"]
        if node["proc_cap"] is not None:
            upper[proc : proc + periods] = node["proc_cap"]
        upper[inv : inv + periods] = node["storage"]
        for t in range(periods):
            lower[inv + t] = node["rate"] * node["out"][t]
            row = i * periods + t
            a_eq[row, proc + t] = 1.0
            a_eq[row, inv + t] = -1.0
            if t:
                a_eq[row, inv + t - 1] = 1.0
            b_eq[row] = node["out"][t] - (node["init"] if t == 0 else 0.0)
    if plant:
        prod, pinv = 2 * n * periods, 2 * n * periods + periods
        cost[prod : prod + periods] = plant["prod_cost"]
        cost[pinv : pinv + periods] = plant["inv_cost"]
        upper[prod : prod + periods] = plant["cap"]
        upper[pinv : pinv + periods] = plant["max_inv"]
        for t in range(periods):
            row = n * periods + t
            a_eq[row, prod + t] = 1.0
            a_eq[row, pinv + t] = -1.0
            if t:
                a_eq[row, pinv + t - 1] = 1.0
            for i in range(n):
                a_eq[row, 2 * i * periods + t] = -1.0
            b_eq[row] = -(plant["init"] if t == 0 else 0.0)
    if np.any(lower > upper):
        return None
    res = linprog(cost, A_eq=a_eq, b_eq=b_eq, bounds=list(zip(lower, upper)), method="highs")
    return float(res.fun) if res.success else None


# ---------------------------------------------------------------- facility_multi
def _detect_multi(instance: dict) -> bool:
    keys = set(instance)
    return {"facilities", "customers", "num_periods", "safety_stock_rate"} <= keys and not (
        keys & {"scenarios", "plants", "dcs"}
    )


def _check_multi(instance: dict, solution: Any) -> dict:
    if not isinstance(solution, dict):
        return _unverified("solution is not a dict")
    facilities = {f["id"]: f for f in instance["facilities"]}
    customers = instance["customers"]
    periods = int(instance["num_periods"])
    rate = float(instance["safety_stock_rate"])
    opened = _id_list(solution.get("opened_facilities"), set(facilities))
    assignment = _id_map(
        solution.get("customer_assignment"), {c["id"] for c in customers}, set(facilities)
    )
    procurement = _plan(solution.get("procurement_plan"), set(facilities), periods)
    inventory = _plan(solution.get("inventory_plan"), set(facilities), periods)
    if opened is None or assignment is None:
        return _unverified("opened_facilities / customer_assignment are not readable")
    if procurement is None or inventory is None:
        return _unverified("procurement_plan / inventory_plan are not readable")

    violations: list[str] = []
    _check_assignment(customers, assignment, opened, "candidate_facilities", violations)
    out = _outflows(customers, assignment, periods)
    # 計画の各期を検査する。計画がない開設拠点は調達も在庫も 0 として扱う。
    for fid in sorted(set(procurement) | set(inventory) | opened):
        f = facilities[fid]
        proc = procurement.get(fid, [0.0] * periods)
        inv = inventory.get(fid, [0.0] * periods)
        ship = out.get(fid, [0.0] * periods)
        if fid not in opened and (sum(proc) > _PLAN_TOL or sum(inv) > _PLAN_TOL):
            violations.append(f"facility {fid} has a plan but is not opened")
        prev = 0.0
        for t in range(periods):
            if proc[t] < -_PLAN_TOL or inv[t] < -_PLAN_TOL:
                violations.append(f"facility {fid} period {t + 1}: negative quantity")
            if abs(prev + proc[t] - ship[t] - inv[t]) > _PLAN_TOL:
                violations.append(f"facility {fid} period {t + 1}: inventory balance broken")
            if proc[t] > f["throughput_per_period"] + _PLAN_TOL:
                violations.append(f"facility {fid} period {t + 1}: procurement over throughput")
            if inv[t] > f["storage_capacity"] + _PLAN_TOL:
                violations.append(f"facility {fid} period {t + 1}: inventory over storage")
            if inv[t] < rate * ship[t] - _PLAN_TOL:
                violations.append(f"facility {fid} period {t + 1}: below safety stock")
            prev = inv[t]

    coef = _coef(instance.get("transport_cost_formula"), 0.05)
    cost = sum(facilities[fid]["open_cost"] for fid in opened)
    cost += sum(
        coef * _dist(c, facilities[assignment[c["id"]]]) * sum(c["demand"])
        for c in customers
        if c["id"] in assignment
    )
    cost += sum(facilities[fid]["procure_cost"] * sum(q) for fid, q in procurement.items())
    cost += sum(facilities[fid]["holding_cost"] * sum(q) for fid, q in inventory.items())
    _declared_mismatch(solution, cost, violations)
    total = len(customers) + 5 * periods * max(len(opened), 1) + 1
    return _result(violations, total, cost=cost)


# ---------------------------------------------------------------- facility_robust
def _detect_robust(instance: dict) -> bool:
    return {"facilities", "customers", "scenarios", "safety_stock_rate"} <= set(instance)


def _check_robust(instance: dict, solution: Any) -> dict:
    if not isinstance(solution, dict):
        return _unverified("solution is not a dict")
    facilities = {f["id"]: f for f in instance["facilities"]}
    customers = instance["customers"]
    periods = int(instance["num_periods"])
    rate = float(instance["safety_stock_rate"])
    opened = _id_list(solution.get("opened_facilities"), set(facilities))
    assignment = _id_map(
        solution.get("customer_assignment"), {c["id"] for c in customers}, set(facilities)
    )
    if opened is None or assignment is None:
        return _unverified("opened_facilities / customer_assignment are not readable")

    violations: list[str] = []
    _check_assignment(customers, assignment, opened, "candidate_facilities", violations)
    coef = _coef(instance.get("transport_cost_formula"), 0.05)
    open_cost = sum(facilities[fid]["open_cost"] for fid in opened)
    scenario_costs = []
    for scenario in instance["scenarios"]:
        factors = scenario["factors"]
        out = _outflows(customers, assignment, periods, factors)
        cost = open_cost + sum(
            coef
            * _dist(c, facilities[assignment[c["id"]]])
            * sum(c["demand"])
            * float(factors.get(str(c["id"]), 1.0))
            for c in customers
            if c["id"] in assignment
        )
        for fid, ship in out.items():
            f = facilities[fid]
            node = {
                "out": ship,
                "rate": rate,
                "storage": f["storage_capacity"],
                "holding": f["holding_cost"],
                "unit_cost": f["procure_cost"],
                "proc_cap": f["throughput_per_period"],
                "init": 0.0,
            }
            plan_cost = _min_cost_plan([node], periods)
            if plan_cost is None:
                violations.append(
                    f"scenario {scenario['id']}: facility {fid} cannot meet demand within "
                    "throughput/storage/safety stock"
                )
                cost = math.inf
                continue
            cost += plan_cost
        scenario_costs.append(cost)
    cost = max(scenario_costs)
    total = len(customers) + len(instance["scenarios"]) * max(len(opened), 1) + 1
    if math.isinf(cost):
        return _result(violations, total, cost=None)  # 計画が組めなければ目的値は未定義
    _declared_mismatch(solution, cost, violations)
    return _result(violations, total, cost=cost)


# ---------------------------------------------------------------- facility_2ech
def _detect_2ech(instance: dict) -> bool:
    return {"plants", "dcs", "customers", "num_periods"} <= set(instance)


def _check_2ech(instance: dict, solution: Any) -> dict:
    if not isinstance(solution, dict):
        return _unverified("solution is not a dict")
    plants = {p["id"]: p for p in instance["plants"]}
    dcs = {d["id"]: d for d in instance["dcs"]}
    customers = instance["customers"]
    periods = int(instance["num_periods"])
    rate = float(instance["safety_stock_rate"])
    opened_plants = _id_list(solution.get("opened_plants"), set(plants))
    opened_dcs = _id_list(solution.get("opened_dcs"), set(dcs))
    dc_plant = _id_map(solution.get("dc_plant_assignment"), set(dcs), set(plants))
    assignment = _id_map(
        solution.get("customer_assignment"), {c["id"] for c in customers}, set(dcs)
    )
    if opened_plants is None or opened_dcs is None:
        return _unverified("opened_plants / opened_dcs are not readable")
    if dc_plant is None or assignment is None:
        return _unverified("dc_plant_assignment / customer_assignment are not readable")

    violations: list[str] = []
    _check_assignment(customers, assignment, opened_dcs, "candidate_dcs", violations)
    out = _outflows(customers, assignment, periods)
    for did in sorted(opened_dcs):
        if did not in dc_plant:
            violations.append(f"DC {did} is opened but has no plant")
        elif dc_plant[did] not in opened_plants:
            violations.append(f"DC {did} is assigned to closed plant {dc_plant[did]}")
        ship = out.get(did, [0.0] * periods)
        if max(ship) > dcs[did]["throughput_per_period"] + 1e-6:
            violations.append(f"DC {did}: outflow over throughput")
    for did in sorted(set(dc_plant) - opened_dcs):
        violations.append(f"DC {did} is assigned to a plant but not opened")

    dc_coef = _coef(instance.get("transport_cost_plant_dc"), 0.04)
    cust_coef = _coef(instance.get("transport_cost_dc_customer"), 0.08)
    cost = sum(plants[pid]["open_cost"] for pid in opened_plants)
    cost += sum(dcs[did]["open_cost"] for did in opened_dcs)
    cost += sum(
        cust_coef * _dist(c, dcs[assignment[c["id"]]]) * sum(c["demand"])
        for c in customers
        if c["id"] in assignment
    )
    # 工場ごとに、配下 DC の調達と工場の生産・在庫をまとめて LP で最適化する。
    for pid in sorted(opened_plants):
        plant = plants[pid]
        nodes = [
            {
                "out": out.get(did, [0.0] * periods),
                "rate": rate,
                "storage": dcs[did]["storage_capacity"],
                "holding": dcs[did]["holding_cost"],
                "unit_cost": dc_coef * _dist(plant, dcs[did]),
                "proc_cap": None,
                "init": float(dcs[did].get("initial_inventory", 0.0)),
            }
            for did in sorted(opened_dcs)
            if dc_plant.get(did) == pid
        ]
        if not nodes:
            continue
        plan_cost = _min_cost_plan(
            nodes,
            periods,
            {
                "cap": plant["capacity_per_period"],
                "prod_cost": plant["production_cost"],
                "inv_cost": plant["inventory_cost"],
                "max_inv": plant["max_inventory"],
                "init": float(plant.get("initial_inventory", 0.0)),
            },
        )
        if plan_cost is None:
            violations.append(f"plant {pid}: no feasible production plan for its DCs")
            cost = math.inf
            continue
        cost += plan_cost
    total = len(customers) + 2 * len(dcs) + len(plants) + 1
    if math.isinf(cost):
        return _result(violations, total, cost=None)  # 生産計画が組めなければ目的値は未定義
    _declared_mismatch(solution, cost, violations)
    return _result(violations, total, cost=cost)


register_kind("facility_multi", _detect_multi, _check_multi)
register_kind("facility_robust", _detect_robust, _check_robust)
register_kind("facility_2ech", _detect_2ech, _check_2ech)
