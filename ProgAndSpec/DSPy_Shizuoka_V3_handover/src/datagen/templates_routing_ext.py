"""routing 系の雛形（第 2 群: prob_022〜025）。 templates.py と同じ規約で (generate, solve) を登録する。

同梱参照解はいずれも note が「ortools routing近似解」で、距離を区間ごとに切り捨てた
上に、容量や訪問順の制約を無視した単一経路を返している（各雛形の docstring を参照）。
この群では参照値の再現ではなく「参照値より良い厳密解」を検証する。距離は
templates_routing と同じく丸めなしのユークリッドで、経路は [0, 顧客id..., 0] で返す。
"""

from __future__ import annotations

import math
import random

from .base import register, retry
from .templates_routing import _leg, _route_length, _solve_cvrp


def _subset_loads(demand: list[int]) -> list[int]:
    """部分集合ビットマスクごとの需要合計。最下位ビットを外した部分集合から積み上げる。"""
    full = 1 << len(demand)
    load = [0] * full
    for s in range(1, full):
        low = (s & -s).bit_length() - 1
        load[s] = load[s ^ (1 << low)] + demand[low]
    return load


def _partition(tour: list[float], n: int, vehicles: int) -> list[int]:
    """tour[s]（部分集合 s を 1 台で回る最短距離）から、vehicles 台以内で全顧客を覆う
    最小総距離の部分集合分割を返す。templates_routing の分割 DP と同じ手順で、
    最小番号の顧客を含む部分集合を固定して同じ分割の重複数えを避ける。

    Why not templates_routing の分割を呼ぶ: あちらは Held-Karp と一体の関数で、
    時間枠や訪問順つきの tour 表を差し込めない。
    """
    full = 1 << n
    inf = math.inf
    best = [[inf] * full for _ in range(vehicles + 1)]
    choice = [[0] * full for _ in range(vehicles + 1)]
    for k in range(vehicles + 1):
        best[k][0] = 0.0
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
    if best[vehicles][full - 1] == inf:
        raise ValueError("no feasible partition into routes")
    subsets: list[int] = []
    s, k = full - 1, vehicles
    while s:
        sub = choice[k][s]
        subsets.append(sub)
        s ^= sub
        k -= 1
    return subsets


def _routes_payload(points: list[dict], orders: list[list[int]]) -> tuple[dict, float]:
    """経路ごとの [0, 顧客id..., 0] と距離を、車両番号 1.. をキーにして組む。

    orders の要素は points のインデックス列（顧客 i は points[i]、id も i）。
    """
    lengths = [_route_length(points, order) for order in orders]
    routes = {
        str(i + 1): {"route": [0, *order, 0], "distance": length}
        for i, (order, length) in enumerate(zip(orders, lengths, strict=True))
    }
    return routes, sum(lengths)


def _solve_vrptw(instance: dict) -> list[list[int]]:
    """時間枠つき CVRP の厳密解。各経路の顧客インデックス列（depot は 0）を返す。

    移動時間 = 距離（速度 1）、到着が早ければ開始時刻まで待ち、各顧客で service_time
    だけ滞在し、depot の時間枠終了までに戻る。時間枠に間に合う経路を depot からの
    深さ優先で全列挙し、部分集合ごとの最短距離を分割 DP で組み合わせる。

    Why not Held-Karp: 時間枠があると (集合, 末尾) ごとに距離と時刻の 2 目的になり、
    最短距離の部分路が時間枠に間に合わないことがある。8 顧客なら実行可能な順列を
    全部たどっても 10 万本程度で足りる。
    """
    depot = instance["depot"]
    customers = instance["customers"]
    points = [depot, *customers]
    n = len(customers)
    capacity = instance["vehicle_capacity"]
    service = instance["service_time"]
    open_at, close_at = depot["time_window"]
    dist = [[_leg(a, b) for b in points] for a in points]
    windows = [c["time_window"] for c in customers]
    load = _subset_loads([c["demand"] for c in customers])
    full = 1 << n
    tour = [math.inf] * full
    tour_order: list[list[int]] = [[] for _ in range(full)]

    def extend(mask: int, last: int, clock: float, length: float, order: list[int]) -> None:
        if mask:
            back = length + dist[last][0]
            if clock + dist[last][0] <= close_at and back < tour[mask]:
                tour[mask] = back
                tour_order[mask] = list(order)
        for j in range(n):
            bit = 1 << j
            if mask & bit or load[mask | bit] > capacity:
                continue
            arrive = clock + dist[last][j + 1]
            # 遅刻する枝は、以降どう回しても遅刻したままなので刈る。
            if arrive > windows[j][1]:
                continue
            order.append(j + 1)
            extend(
                mask | bit,
                j + 1,
                max(arrive, windows[j][0]) + service,
                length + dist[last][j + 1],
                order,
            )
            order.pop()

    extend(0, 0, float(open_at), 0.0, [])
    return [tour_order[sub] for sub in _partition(tour, n, instance["num_vehicles"])]


@register(22, "total_distance", shipped_reference_optimal=False)
def vrp_time_windows():
    """prob_022: 時間枠つき VRP。顧客数 8、車両 3 台、容量 12 は問題文どおり固定する。

    同梱参照解は需要合計 23 を容量 12 の車両 1 台に積む単一経路で、容量制約に違反している
    （距離 154 は区間ごとの切り捨て値）。
    """

    def generate(rng: random.Random, base: dict) -> dict:
        n = len(base["customers"])
        capacity = base["vehicle_capacity"]
        vehicles = base["num_vehicles"]
        service = base["service_time"]
        horizon = base["depot"]["time_window"][1]

        def make() -> dict:
            customers = []
            for i in range(n):
                start = rng.choice([0, rng.randint(40, 200)])
                customers.append(
                    {
                        "id": i + 1,
                        "x": rng.randint(-15, 15),
                        "y": rng.randint(-15, 15),
                        "demand": rng.randint(1, 4),
                        "time_window": [start, start + rng.randint(100, 320)],
                    }
                )
            return {
                "depot": {"id": 0, "x": 0, "y": 0, "time_window": [0, horizon]},
                "customers": customers,
                "vehicle_capacity": capacity,
                "num_vehicles": vehicles,
                "service_time": service,
            }

        def ok(inst: dict) -> bool:
            total = sum(c["demand"] for c in inst["customers"])
            if not capacity < total <= capacity * vehicles:
                return False
            try:
                with_tw = solve(inst)["total_distance"]
            except ValueError:
                return False
            # 時間枠を外した CVRP と同じ距離なら時間枠が効いていないので捨てる。
            plain, _ = _solve_cvrp(inst)
            return with_tw > plain + 1e-9

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        points = [instance["depot"], *instance["customers"]]
        routes, total = _routes_payload(points, _solve_vrptw(instance))
        return {
            "routes": routes,
            "total_distance": total,
            "note": "時間枠つき経路の全列挙 + 分割 DP（厳密最適解）。距離は丸めなしのユークリッド",
        }

    return generate, solve


def _solve_vrp_backhaul(instance: dict) -> list[list[int]]:
    """出荷（linehaul）→回収（backhaul）の順序つき CVRP の厳密解。

    Held-Karp の遷移から backhaul→linehaul を除き、部分集合は出荷量と回収量が
    それぞれ容量以下のものだけ使う（出荷を全部済ませてから回収するので積載は
    別々に数えればよい）。経路は必ず出荷顧客から始める。

    Why not 回収顧客だけの経路を許す: 問題文は「まず出荷顧客を訪問し、その後回収顧客を
    訪問する」と各車両の動きを述べており、古典的な VRPB の定義（回収のみの経路は不可）に
    合わせた。出荷のみの経路は「その後回収する顧客がない」だけなので許す。
    """
    depot = instance["depot"]
    customers = instance["customers"]
    points = [depot, *customers]
    n = len(customers)
    capacity = instance["vehicle_capacity"]
    dist = [[_leg(a, b) for b in points] for a in points]
    linehaul = [c["type"] == "linehaul" for c in customers]
    load_out = _subset_loads([c["demand"] if linehaul[i] else 0 for i, c in enumerate(customers)])
    load_in = _subset_loads([0 if linehaul[i] else c["demand"] for i, c in enumerate(customers)])
    full = 1 << n
    inf = math.inf

    def fits(s: int) -> bool:
        return load_out[s] <= capacity and load_in[s] <= capacity

    path = [[inf] * n for _ in range(full)]
    parent = [[-1] * n for _ in range(full)]
    for j in range(n):
        if linehaul[j]:
            path[1 << j][j] = dist[0][j + 1]
    for s in range(1, full):
        if not fits(s):
            continue
        for j in range(n):
            if not s & (1 << j) or path[s][j] == inf:
                continue
            rest = (~s) & (full - 1)
            k = rest
            while k:
                low = (k & -k).bit_length() - 1
                k ^= 1 << low
                # 回収を始めたら出荷には戻れない。
                if linehaul[low] and not linehaul[j]:
                    continue
                t = s | (1 << low)
                if not fits(t):
                    continue
                cand = path[s][j] + dist[j + 1][low + 1]
                if cand < path[t][low]:
                    path[t][low] = cand
                    parent[t][low] = j

    tour = [inf] * full
    tour_end = [-1] * full
    for s in range(1, full):
        for j in range(n):
            if path[s][j] < inf:
                cand = path[s][j] + dist[j + 1][0]
                if cand < tour[s]:
                    tour[s] = cand
                    tour_end[s] = j

    routes: list[list[int]] = []
    for sub in _partition(tour, n, instance["num_vehicles"]):
        order: list[int] = []
        j, t = tour_end[sub], sub
        while j != -1:
            order.append(j + 1)
            j, t = parent[t][j], t ^ (1 << j)
        routes.append(order[::-1])
    return routes


@register(23, "total_distance", shipped_reference_optimal=False)
def vrp_backhaul():
    """prob_023: 出荷 5・回収 5 の VRPB。車両 3 台、容量 20 は問題文どおり固定する。

    同梱参照解は回収顧客を先に回ってから出荷顧客を回る単一経路で、訪問順の制約と
    容量（出荷量合計 23 > 20）の両方に違反している（距離 173 は区間ごとの切り捨て値）。
    """

    def generate(rng: random.Random, base: dict) -> dict:
        types = [c["type"] for c in base["customers"]]
        capacity = base["vehicle_capacity"]
        vehicles = base["num_vehicles"]

        def make() -> dict:
            customers = [
                {
                    "id": i + 1,
                    "x": rng.randint(-15, 15),
                    "y": rng.randint(-15, 15),
                    "demand": rng.randint(3, 6) if kind == "linehaul" else rng.randint(1, 4),
                    "type": kind,
                }
                for i, kind in enumerate(types)
            ]
            return {
                "depot": {"id": 0, "x": 0, "y": 0},
                "customers": customers,
                "vehicle_capacity": capacity,
                "num_vehicles": vehicles,
            }

        def ok(inst: dict) -> bool:
            out = sum(c["demand"] for c in inst["customers"] if c["type"] == "linehaul")
            # 出荷量が 1 台に収まると経路分割が要らないので、2 台以上必要な instance に限る。
            return capacity < out <= capacity * vehicles

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        points = [instance["depot"], *instance["customers"]]
        routes, total = _routes_payload(points, _solve_vrp_backhaul(instance))
        return {
            "routes": routes,
            "total_distance": total,
            "note": "訪問順つき Held-Karp + 分割 DP（厳密最適解）。距離は丸めなしのユークリッド",
        }

    return generate, solve


@register(24, "total_distance", shipped_reference_optimal=False)
def multi_depot_vrp():
    """prob_024: 顧客が配送センターに割り当て済みの複数デポ VRP。デポ 3、顧客 10、容量 15 は
    問題文どおり固定し、デポごとの車両数は乱数にする。

    割り当てが固定なのでデポごとに独立した CVRP に分かれ、templates_routing の厳密解を
    デポ単位で呼ぶ。同梱参照解はデポ 3 の経路 [8, 7, 3, 1] が遠い顧客 8 を先に回る近似解で、
    [1, 7, 3, 8] の順なら 114 → 約 76 に縮む（距離は区間ごとの切り捨て値）。
    """

    def generate(rng: random.Random, base: dict) -> dict:
        depot_ids = [d["id"] for d in base["depots"]]
        n = len(base["customers"])
        capacity = base["vehicle_capacity"]

        def make() -> dict:
            depots = [
                {
                    "id": d,
                    "x": rng.randint(-15, 15),
                    "y": rng.randint(-15, 15),
                    "num_vehicles": rng.randint(1, 3),
                }
                for d in depot_ids
            ]
            customers = [
                {
                    "id": i + 1,
                    "x": rng.randint(-20, 20),
                    "y": rng.randint(-20, 20),
                    "demand": rng.randint(1, 5),
                    "assigned_depot": rng.choice(depot_ids),
                }
                for i in range(n)
            ]
            return {"depots": depots, "customers": customers, "vehicle_capacity": capacity}

        def ok(inst: dict) -> bool:
            loads = {d["id"]: 0 for d in inst["depots"]}
            for c in inst["customers"]:
                loads[c["assigned_depot"]] += c["demand"]
            fleet = {d["id"]: d["num_vehicles"] for d in inst["depots"]}
            # 全デポに顧客があり、全デポの車両数で運べ、少なくとも 1 つのデポは 2 経路以上要る。
            return all(0 < loads[d] <= capacity * fleet[d] for d in loads) and any(
                loads[d] > capacity for d in loads
            )

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        routes: dict[str, dict] = {}
        total = 0.0
        for depot in instance["depots"]:
            mine = [c for c in instance["customers"] if c["assigned_depot"] == depot["id"]]
            if not mine:
                continue
            sub = {
                "depot": {"x": depot["x"], "y": depot["y"]},
                "customers": mine,
                "vehicle_capacity": instance["vehicle_capacity"],
                "num_vehicles": depot["num_vehicles"],
            }
            _, orders = _solve_cvrp(sub)
            points = [sub["depot"], *mine]
            for order in orders:
                length = _route_length(points, order)
                routes[str(len(routes) + 1)] = {
                    "depot": depot["id"],
                    # 経路の 0 は同梱参照解と同じくそのデポを指す。
                    "route": [0, *(mine[i - 1]["id"] for i in order), 0],
                    "distance": length,
                }
                total += length
        return {
            "routes": routes,
            "total_distance": total,
            "note": "デポ別の部分集合 Held-Karp + 分割 DP（厳密最適解）。距離は丸めなしのユークリッド",
        }

    return generate, solve


def _subset_tours(points: list[dict], max_load: int, demand: list[int]) -> tuple[list, list, list]:
    """需要合計が max_load 以下の各部分集合について、depot 発着で回る最短距離と順路。

    Returns:
        (tour, tour_end, parent): Held-Karp 表。順路は tour_end と parent からたどる。
    """
    n = len(demand)
    dist = [[_leg(a, b) for b in points] for a in points]
    load = _subset_loads(demand)
    full = 1 << n
    inf = math.inf
    path = [[inf] * n for _ in range(full)]
    parent = [[-1] * n for _ in range(full)]
    for j in range(n):
        path[1 << j][j] = dist[0][j + 1]
    for s in range(1, full):
        if load[s] > max_load:
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
                if load[t] > max_load:
                    continue
                cand = path[s][j] + dist[j + 1][low + 1]
                if cand < path[t][low]:
                    path[t][low] = cand
                    parent[t][low] = j
    tour = [inf] * full
    tour_end = [-1] * full
    for s in range(1, full):
        for j in range(n):
            if path[s][j] < inf:
                cand = path[s][j] + dist[j + 1][0]
                if cand < tour[s]:
                    tour[s] = cand
                    tour_end[s] = j
    return tour, tour_end, parent


def _solve_fleet_mix(instance: dict) -> list[tuple[int, list[int]]]:
    """異種車両の選択と経路の厳密解。(車両インデックス, 顧客インデックス列) を返す。

    部分集合ごとの最短巡回を Held-Karp で求め、車両を 1 台ずつ「使わない／どの部分集合を
    任せるか」で選ぶ DP を回す。目的は Σ(固定費 + 距離単価 × 経路長)。
    """
    customers = instance["customers"]
    vehicles = instance["vehicles"]
    points = [instance["depot"], *customers]
    n = len(customers)
    demand = [c["demand"] for c in customers]
    load = _subset_loads(demand)
    tour, tour_end, parent = _subset_tours(points, max(v["capacity"] for v in vehicles), demand)
    full = 1 << n
    inf = math.inf
    m = len(vehicles)
    # best[v][s]: 先頭 v 台だけで s を配送する最小費用。choice は車両 v が担う部分集合（0 は不使用）。
    best = [[inf] * full for _ in range(m + 1)]
    choice = [[0] * full for _ in range(m + 1)]
    best[0][0] = 0.0
    for v in range(1, m + 1):
        cap = vehicles[v - 1]["capacity"]
        fixed = vehicles[v - 1]["fixed_cost"]
        rate = vehicles[v - 1]["cost_per_km"]
        for s in range(full):
            best[v][s] = best[v - 1][s]
            sub = s
            while sub:
                if load[sub] <= cap and tour[sub] < inf and best[v - 1][s ^ sub] < inf:
                    cand = fixed + rate * tour[sub] + best[v - 1][s ^ sub]
                    if cand < best[v][s]:
                        best[v][s] = cand
                        choice[v][s] = sub
                sub = (sub - 1) & s
    if best[m][full - 1] == inf:
        raise ValueError("no feasible fleet assignment")
    plan: list[tuple[int, list[int]]] = []
    s = full - 1
    for v in range(m, 0, -1):
        sub = choice[v][s]
        if not sub:
            continue
        order: list[int] = []
        j, t = tour_end[sub], sub
        while j != -1:
            order.append(j + 1)
            j, t = parent[t][j], t ^ (1 << j)
        plan.append((v - 1, order[::-1]))
        s ^= sub
    return plan[::-1]


# 同梱参照解は容量 12 の車両 4 だけで需要合計 45 を運ぶ単一経路で容量制約に違反しており、
# 費用 178.8 は実行可能などの計画（需要 45 には 4 台全部が要り、固定費だけで 905）より安い。
# 実行可能解では到達できない値なので shipped_reference_feasible=False で大小比較を外す。
@register(25, "total_cost", shipped_reference_optimal=False, shipped_reference_feasible=False)
def fleet_size_and_mix():
    """prob_025: 異種 4 車両からの選択と経路決定。顧客 8、車両 4 台は問題文どおり固定する。

    同梱参照解は容量 12 の車両 4 だけで需要合計 45 を運ぶ単一経路で、容量制約に違反している
    （距離 59 は区間ごとの切り捨て値、費用 178.8 = 131 + 59 × 0.81）。
    """

    def generate(rng: random.Random, base: dict) -> dict:
        n = len(base["customers"])
        m = len(base["vehicles"])

        def make() -> dict:
            customers = [
                {
                    "id": i + 1,
                    "x": rng.randint(-15, 15),
                    "y": rng.randint(-15, 15),
                    "demand": rng.randint(3, 8),
                }
                for i in range(n)
            ]
            vehicles = [
                {
                    "id": v + 1,
                    "capacity": rng.choice([8, 10, 12, 15, 20]),
                    "fixed_cost": rng.randint(100, 300),
                    "cost_per_km": rng.uniform(0.5, 1.6),
                }
                for v in range(m)
            ]
            return {
                "depot": {"id": 0, "x": 0, "y": 0},
                "customers": customers,
                "vehicles": vehicles,
            }

        def ok(inst: dict) -> bool:
            total = sum(c["demand"] for c in inst["customers"])
            caps = [v["capacity"] for v in inst["vehicles"]]
            # 1 台で全部運べる instance は車両選択が自明なので捨てる。運べない instance も捨てる。
            if not max(caps) < total <= sum(caps):
                return False
            try:
                _solve_fleet_mix(inst)
            except ValueError:
                return False
            return True

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        points = [instance["depot"], *instance["customers"]]
        vehicles = instance["vehicles"]
        routes: dict[str, dict] = {}
        for v, order in _solve_fleet_mix(instance):
            length = _route_length(points, order)
            routes[str(vehicles[v]["id"])] = {
                "route": [0, *order, 0],
                "distance": length,
                "cost": vehicles[v]["fixed_cost"] + vehicles[v]["cost_per_km"] * length,
            }
        return {
            "routes": routes,
            # 保存した経路費用の和を目的値にし、DP 内部の値と桁落ちでずれないようにする。
            "total_cost": sum(r["cost"] for r in routes.values()),
            "note": "部分集合 Held-Karp + 車両選択 DP（厳密最適解）。距離は丸めなしのユークリッド",
        }

    return generate, solve
