"""配送・輸送系の雛形。templates.py と同じ規約で (generate, solve) を登録する。

配送・輸送_混合整数計画 の同梱参照解は note が「ortools routing近似解」で、prob_021 は
容量 15 の車両 1 台に需要 25 を積む単一経路（総距離 245）である。厳密最適は約 144 なので、
この雛形は参照値の再現ではなく「参照値より良い厳密解」を検証する。
"""

from __future__ import annotations

import math
import random
from itertools import pairwise

from .base import register, retry

# Why not prob_026（動的 VRP）: 問題文は「初期時点で5人の顧客」だが instance の初期顧客は
# 4 人で、参照解 initial_routes も切り捨て距離の近似解（88、厳密には 73）。文章と
# instance が食い違う雛形は生成に使わない。


def _leg(a: dict, b: dict) -> float:
    """丸めなしのユークリッド距離。

    Why not 区間ごとの四捨五入: 同梱参照解はそうしているが、metrics_v3 の VRP スコアラーは
    経路から丸めなしで総距離を再計算する。参照解は採点側の規約に合わせないと、正答が
    参照値と 1% 弱ずれて exact_match にならない。
    """
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def _route_length(points: list[dict], order: list[int]) -> float:
    """経路順に区間距離を足す。スコアラーと同じ順序で足すと総距離が桁落ちなく一致する。"""
    closed = [0, *order, 0]
    return sum(_leg(points[u], points[v]) for u, v in pairwise(closed))


def _solve_cvrp(instance: dict) -> tuple[float, list[list[int]]]:
    """需要を満たす顧客部分集合ごとの Held-Karp と、部分集合分割 DP による厳密解。

    Returns:
        (総距離, 各経路の顧客インデックス列)。depot はインデックス 0。
    """
    depot = instance["depot"]
    customers = instance["customers"]
    points = [depot, *customers]
    n = len(customers)
    capacity = instance["vehicle_capacity"]
    vehicles = instance["num_vehicles"]
    dist = [[_leg(a, b) for b in points] for a in points]
    demand = [c["demand"] for c in customers]
    full = 1 << n
    inf = math.inf

    load = [0] * full
    for s in range(1, full):
        low = (s & -s).bit_length() - 1
        load[s] = load[s ^ (1 << low)] + demand[low]

    # path[s][j]: depot から s の顧客を全部回って顧客 j で終わる最短路（j は 0..n-1）。
    path = [[inf] * n for _ in range(full)]
    parent = [[-1] * n for _ in range(full)]
    for j in range(n):
        path[1 << j][j] = dist[0][j + 1]
    for s in range(1, full):
        if load[s] > capacity:
            continue
        for j in range(n):
            if not s & (1 << j) or path[s][j] == inf:
                continue
            rest = (~s) & (full - 1)
            k = rest
            while k:
                low = (k & -k).bit_length() - 1
                k ^= 1 << low
                t = s | (1 << low)
                if load[t] > capacity:
                    continue
                cand = path[s][j] + dist[j + 1][low + 1]
                if cand < path[t][low]:
                    path[t][low] = cand
                    parent[t][low] = j

    tour = [inf] * full
    tour_end = [-1] * full
    for s in range(1, full):
        if load[s] > capacity:
            continue
        for j in range(n):
            if path[s][j] < inf:
                cand = path[s][j] + dist[j + 1][0]
                if cand < tour[s]:
                    tour[s] = cand
                    tour_end[s] = j

    # best[k][s]: k 台以内で s を全部配送する最小総距離。最小番号の顧客を含む経路を固定して
    # 同じ分割を何度も数えないようにする。
    best = [[inf] * full for _ in range(vehicles + 1)]
    choice = [[0] * full for _ in range(vehicles + 1)]
    for k in range(vehicles + 1):
        best[k][0] = 0
    for k in range(1, vehicles + 1):
        for s in range(1, full):
            low_bit = s & -s
            sub = s
            while sub:
                if sub & low_bit and tour[sub] < inf:
                    cand = tour[sub] + best[k - 1][s ^ sub]
                    if cand < best[k][s]:
                        best[k][s] = cand
                        choice[k][s] = sub
                sub = (sub - 1) & s
    total = best[vehicles][full - 1]
    if total == inf:
        raise ValueError("no feasible CVRP solution")

    routes: list[list[int]] = []
    s, k = full - 1, vehicles
    while s:
        sub = choice[k][s]
        order: list[int] = []
        j = tour_end[sub]
        t = sub
        while j != -1:
            order.append(j + 1)
            j, t = parent[t][j], t ^ (1 << j)
        routes.append(order[::-1])
        s ^= sub
        k -= 1
    return total, routes


@register(21, "total_distance", shipped_reference_optimal=False)
def cvrp():
    """prob_021: 基本 CVRP。顧客数・車両数・容量は問題文どおりに固定する。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n = len(base["customers"])
        capacity = base["vehicle_capacity"]
        vehicles = base["num_vehicles"]

        def make() -> dict:
            customers = [
                {
                    "id": i + 1,
                    "x": rng.randint(-20, 20),
                    "y": rng.randint(-20, 20),
                    "demand": rng.randint(1, 5),
                }
                for i in range(n)
            ]
            return {
                "depot": {"id": 0, "x": 0, "y": 0},
                "customers": customers,
                "vehicle_capacity": capacity,
                "num_vehicles": vehicles,
            }

        def ok(inst: dict) -> bool:
            total = sum(c["demand"] for c in inst["customers"])
            # 1 台で運べる需要では経路分割が要らないので、2 台以上必要な instance に限る。
            return capacity < total <= capacity * vehicles

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        _, routes = _solve_cvrp(instance)
        points = [instance["depot"], *instance["customers"]]
        lengths = [_route_length(points, order) for order in routes]
        # Why not 小数2桁に丸める: 採点側は丸めなしで再計算するので、丸めると exact_match を外れる。
        return {
            "routes": {
                str(i + 1): {"route": [0, *order, 0], "distance": length}
                for i, (order, length) in enumerate(zip(routes, lengths, strict=True))
            },
            "total_distance": sum(lengths),
            "note": "部分集合 Held-Karp + 分割 DP（厳密最適解）。距離は丸めなしのユークリッド",
        }

    return generate, solve
