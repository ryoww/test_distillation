"""composite 系の雛形。templates.py と同じ規約で (generate, solve) を登録する。

複合・グラフ最適化の雛形を置く。工場数・顧客数・期間数・発電機台数は問題文に
あるので雛形の値を保ち、能力・費用・需要だけを乱数で置き換える。
"""

from __future__ import annotations

import itertools
import random

from .base import cp_sat_solver, int_list, int_matrix, register, require_optimal, retry


@register(94, "min_total_cost")
def production_distribution():
    """prob_094: 生産流通複合。工場数と顧客数は問題文にあるので雛形の値を保つ。"""

    def generate(rng: random.Random, base: dict) -> dict:
        num_plants = base["num_plants"]
        num_customers = base["num_customers"]

        def make() -> dict:
            return {
                "num_plants": num_plants,
                "num_customers": num_customers,
                "capacity": int_list(rng, num_plants, 30, 60),
                "production_cost": int_list(rng, num_plants, 5, 10),
                "demand": int_list(rng, num_customers, 8, 25),
                "transport_cost": int_matrix(rng, num_plants, num_customers, 1, 8),
            }

        def ok(instance: dict) -> bool:
            total_demand = sum(instance["demand"])
            # 総能力が需要を上回り、かつ 1 工場では賄えない（能力制約が効く）instance だけ使う。
            return (
                sum(instance["capacity"]) > total_demand
                and max(instance["capacity"]) < total_demand
            )

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        from scipy.optimize import linprog

        plants = range(instance["num_plants"])
        customers = range(instance["num_customers"])
        transport = instance["transport_cost"]
        production = instance["production_cost"]
        # 変数 x[p][c] を p 優先で並べる。目的係数は生産費＋輸送費。
        cost = [production[p] + transport[p][c] for p in plants for c in customers]
        a_ub = []
        for p in plants:
            row = [0.0] * len(cost)
            for c in customers:
                row[p * len(customers) + c] = 1.0
            a_ub.append(row)
        a_eq = []
        for c in customers:
            row = [0.0] * len(cost)
            for p in plants:
                row[p * len(customers) + c] = 1.0
            a_eq.append(row)
        result = linprog(
            c=cost,
            A_ub=a_ub,
            b_ub=instance["capacity"],
            A_eq=a_eq,
            b_eq=instance["demand"],
            bounds=[(0, None)] * len(cost),
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"LP failed: {result.message}")
        # Why not 小数2桁に丸める: 丸めた出荷量は需要や容量を 0.01 だけ外しうる。
        # 保存する解は LP の値のまま、費用もその解から計算する。
        shipments: dict[str, dict[str, float]] = {}
        total_cost = 0.0
        for p in plants:
            row = {}
            for c in customers:
                index = p * len(customers) + c
                quantity = float(result.x[index])
                if quantity > 1e-9:
                    row[str(c)] = quantity
                    total_cost += cost[index] * quantity
            shipments[str(p)] = row
        return {
            "min_total_cost": total_cost,
            "shipments": shipments,
            "note": "LP（HiGHS、厳密最適解）",
        }

    return generate, solve


@register(100, "min_cost")
def unit_commitment():
    """prob_100: 発電機起動停止計画。期間数と台数は問題文にあるので雛形の値を保つ。"""

    def generate(rng: random.Random, base: dict) -> dict:
        periods = base["periods"]
        num_units = len(base["units"])

        def make() -> dict:
            units = []
            for i in range(num_units):
                low = rng.randint(10, 25)
                units.append(
                    {
                        "id": i,
                        "min": low,
                        "max": rng.randint(low + 30, 90),
                        "cost": rng.randint(5, 12),
                        "startup": rng.choice([40, 60, 80, 100, 120]),
                    }
                )
            return {"periods": periods, "demand": int_list(rng, periods, 60, 170), "units": units}

        def ok(instance: dict) -> bool:
            units = instance["units"]
            demand = instance["demand"]
            # 稼働させる部分集合のどれかで min..max の範囲に需要が入る期だけ許す。
            ranges = [
                (sum(u["min"] for u in subset), sum(u["max"] for u in subset))
                for k in range(1, len(units) + 1)
                for subset in itertools.combinations(units, k)
            ]
            if not all(any(low <= d <= high for low, high in ranges) for d in demand):
                return False
            largest = max(u["max"] for u in units)
            smallest = min(u["max"] for u in units)
            # 1 台で足りる期ばかりでも、全台が常に要る期ばかりでも起動停止の選択が無い。
            needs_two = any(d > largest for d in demand)
            can_idle = any(d <= sum(u["max"] for u in units) - smallest for d in demand)
            return needs_two and can_idle

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        cp_model, solver = cp_sat_solver()
        units = instance["units"]
        demand = instance["demand"]
        periods = range(instance["periods"])
        model = cp_model.CpModel()
        on = {(u, t): model.NewBoolVar(f"on{u}_{t}") for u in range(len(units)) for t in periods}
        output = {
            (u, t): model.NewIntVar(0, unit["max"], f"out{u}_{t}")
            for u, unit in enumerate(units)
            for t in periods
        }
        start = {(u, t): model.NewBoolVar(f"st{u}_{t}") for u in range(len(units)) for t in periods}
        for u, unit in enumerate(units):
            for t in periods:
                model.Add(output[u, t] >= unit["min"]).OnlyEnforceIf(on[u, t])
                model.Add(output[u, t] <= unit["max"]).OnlyEnforceIf(on[u, t])
                model.Add(output[u, t] == 0).OnlyEnforceIf(on[u, t].Not())
                # 期 0 は全台が停止状態から始まるので、稼働させれば起動費が掛かる。
                if t == 0:
                    model.Add(start[u, t] >= on[u, t])
                else:
                    model.Add(start[u, t] >= on[u, t] - on[u, t - 1])
        for t in periods:
            model.Add(sum(output[u, t] for u in range(len(units))) >= demand[t])
        model.Minimize(
            sum(
                unit["cost"] * output[u, t] + unit["startup"] * start[u, t]
                for u, unit in enumerate(units)
                for t in periods
            )
        )
        require_optimal(cp_model, solver.Solve(model))
        schedule = {
            str(u): {
                "on": [solver.Value(on[u, t]) for t in periods],
                "output": [float(solver.Value(output[u, t])) for t in periods],
            }
            for u in range(len(units))
        }
        return {
            "min_cost": round(solver.ObjectiveValue()),
            "schedule": schedule,
            "note": "CP-SAT（厳密最適解）",
        }

    return generate, solve
