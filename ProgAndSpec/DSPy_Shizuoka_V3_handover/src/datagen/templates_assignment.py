"""assignment 系の雛形。templates.py と同じ規約で (generate, solve) を登録する。"""

from __future__ import annotations

import itertools
import math
import random

from .base import cp_sat_solver, int_list, int_matrix, register, require_optimal, retry

# ============================================================
# 割当・マッチング
# ============================================================


@register(68, "min_cost")
def generalized_assignment():
    """prob_068: 一般化割当。容量が実際に効く実行可能 instance だけを採用する。"""

    def generate(rng: random.Random, base: dict) -> dict:
        nt, na = base["num_tasks"], base["num_agents"]

        def make() -> dict:
            return {
                "num_tasks": nt,
                "num_agents": na,
                "cost": int_matrix(rng, na, nt, 2, 15),
                "resource": int_matrix(rng, na, nt, 2, 8),
                "capacity": int_list(rng, na, 12, 20),
            }

        def ok(inst: dict) -> bool:
            # 費用だけで選んだ割当が容量に収まるなら容量制約は無意味なので捨てる。
            cheapest = [min(range(na), key=lambda a: inst["cost"][a][t]) for t in range(nt)]
            used = [0] * na
            for task, agent in enumerate(cheapest):
                used[agent] += inst["resource"][agent][task]
            binding = any(u > c for u, c in zip(used, inst["capacity"]))
            return binding and _gap_solve(inst) is not None

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        solved = _gap_solve(instance)
        if solved is None:
            raise ValueError("instance has no feasible assignment")
        cost, chosen = solved
        return {
            "min_cost": cost,
            "assignment": {str(task): agent for task, agent in enumerate(chosen)},
            "note": "CP-SAT（厳密最適解）",
        }

    return generate, solve


def _gap_solve(instance: dict) -> tuple[int, list[int]] | None:
    """最適費用と各タスクの担当エージェント。実行不能なら None。"""
    cp_model, solver = cp_sat_solver()
    nt, na = instance["num_tasks"], instance["num_agents"]
    cost, resource, capacity = instance["cost"], instance["resource"], instance["capacity"]
    model = cp_model.CpModel()
    x = [[model.NewBoolVar(f"x{a}_{t}") for t in range(nt)] for a in range(na)]
    for t in range(nt):
        model.AddExactlyOne(x[a][t] for a in range(na))
    for a in range(na):
        model.Add(sum(resource[a][t] * x[a][t] for t in range(nt)) <= capacity[a])
    model.Minimize(sum(cost[a][t] * x[a][t] for a in range(na) for t in range(nt)))
    status = solver.Solve(model)
    if status == cp_model.INFEASIBLE:
        return None
    require_optimal(cp_model, status)
    chosen = [next(a for a in range(na) if solver.Value(x[a][t])) for t in range(nt)]
    return round(solver.ObjectiveValue()), chosen


@register(71, "min_cost")
def quadratic_assignment():
    """prob_071: 二次割当。対角は 0、それ以外は乱数にして全順列を列挙する。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n = base["num_facilities"]

        def square() -> list[list[int]]:
            return [[0 if i == j else rng.randint(1, 9) for j in range(n)] for i in range(n)]

        return {"num_facilities": n, "flow": square(), "distance": square()}

    def solve(instance: dict) -> dict:
        n = instance["num_facilities"]
        if n > 8:
            raise ValueError("brute force is limited to 8 facilities")
        flow, distance = instance["flow"], instance["distance"]
        best = math.inf
        best_placement: list[int] = []
        # 辞書順で最初に見つかった最小値を採用するので、同点でも参照解は一意に決まる。
        for placement in itertools.permutations(range(n)):
            cost = sum(
                flow[i][j] * distance[placement[i]][placement[j]]
                for i in range(n)
                for j in range(n)
            )
            if cost < best:
                best = cost
                best_placement = list(placement)
        return {
            "min_cost": int(best),
            "placement": best_placement,
            "note": "全順列列挙（厳密最適解）",
        }

    return generate, solve


@register(73, "min_bottleneck")
def bottleneck_assignment():
    """prob_073: ボトルネック割当。最大所要時間を CP-SAT で最小化する。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n = base["num_agents"]
        return {"num_agents": n, "time_matrix": int_matrix(rng, n, n, 3, 40)}

    def solve(instance: dict) -> dict:
        cp_model, solver = cp_sat_solver()
        n = instance["num_agents"]
        times = instance["time_matrix"]
        model = cp_model.CpModel()
        x = [[model.NewBoolVar(f"x{i}_{j}") for j in range(n)] for i in range(n)]
        for i in range(n):
            model.AddExactlyOne(x[i])
            model.AddExactlyOne(x[j][i] for j in range(n))
        bottleneck = model.NewIntVar(0, max(max(row) for row in times), "bottleneck")
        for i in range(n):
            for j in range(n):
                model.Add(bottleneck >= times[i][j]).OnlyEnforceIf(x[i][j])
        model.Minimize(bottleneck)
        require_optimal(cp_model, solver.Solve(model))
        # チェッカーは assignment を {作業員: 仕事} として読む。
        assignment = {str(i): next(j for j in range(n) if solver.Value(x[i][j])) for i in range(n)}
        return {
            "min_bottleneck": round(solver.ObjectiveValue()),
            "assignment": assignment,
            "note": "CP-SAT（厳密最適解）",
        }

    return generate, solve
