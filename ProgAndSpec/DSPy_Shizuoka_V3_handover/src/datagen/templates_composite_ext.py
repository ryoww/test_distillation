"""複合・グラフ最適化 系の雛形（第 2 群: prob_090〜099）。

templates.py と同じ規約で (generate, solve) を登録する。ノード数・顧客数・拠点数・
期間数・容量・車両費用・半径は問題文にあるので雛形の値を保ち、座標・費用・需要・
辺・時間枠だけを乱数で置き換える。
"""

from __future__ import annotations

import itertools
import math
import random

from .base import SolveError, cp_sat_solver, int_list, int_matrix, register, require_optimal, retry


def _rounded_distance(a: dict, b: dict) -> int:
    """同梱 instance の距離行列と同じ、四捨五入したユークリッド距離。"""
    return round(math.hypot(a["x"] - b["x"], a["y"] - b["y"]))


def _random_points(rng: random.Random, n: int, span: int) -> list[dict]:
    return [
        {"id": i, "x": rng.randint(-span, span), "y": rng.randint(-span, span)} for i in range(n)
    ]


def _random_edges(rng: random.Random, num_nodes: int, num_edges: int) -> list[list[int]]:
    """重複のない無向辺を辺数ぶん選び、辞書順に並べる。"""
    pairs = list(itertools.combinations(range(num_nodes), 2))
    return [list(pair) for pair in sorted(rng.sample(pairs, num_edges))]


@register(90, "min_colors")
def graph_coloring():
    """prob_090: グラフ彩色。9 ノードは問題文にあり、辺数 16 は形状として保つ。"""

    def generate(rng: random.Random, base: dict) -> dict:
        num_nodes = base["num_nodes"]
        num_edges = len(base["edges"])

        def make() -> dict:
            return {"num_nodes": num_nodes, "edges": _random_edges(rng, num_nodes, num_edges)}

        def ok(instance: dict) -> bool:
            # 2 色で塗れる（二部グラフ）instance は色数の議論が要らないので捨てる。
            return solve(instance)["min_colors"] >= 3

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        num_nodes = instance["num_nodes"]
        neighbours: list[set[int]] = [set() for _ in range(num_nodes)]
        for u, v in instance["edges"]:
            neighbours[u].add(v)
            neighbours[v].add(u)
        colors = [-1] * num_nodes

        def paint(node: int, palette: int) -> bool:
            """ノード順・色番号順の深さ優先探索。最初に見つかる解が辞書順最小の塗り分け。"""
            if node == num_nodes:
                return True
            for color in range(palette):
                if all(colors[w] != color for w in neighbours[node]):
                    colors[node] = color
                    if paint(node + 1, palette):
                        return True
            colors[node] = -1
            return False

        # Why not CP-SAT: 色の入れ替えで同点解が大量にあり、色数の最小化と辞書順の
        # タイブレークを同時に扱うより、色数を 1 から増やす全探索のほうが単純で決定的。
        for palette in range(1, num_nodes + 1):
            if paint(0, palette):
                return {
                    "min_colors": palette,
                    "coloring": {str(v): colors[v] for v in range(num_nodes)},
                    "note": "バックトラック全探索（厳密最適解、ノード順に最小色）",
                }
        raise SolveError("no coloring found")

    return generate, solve


@register(92, "min_total_cost")
def location_routing():
    """prob_092: 立地配送複合。拠点数・顧客数・車両容量・車両費用は問題文どおり固定。"""

    def generate(rng: random.Random, base: dict) -> dict:
        num_facilities = len(base["facilities"])
        num_customers = len(base["customers"])

        def make() -> dict:
            facilities = _random_points(rng, num_facilities, 20)
            customers = _random_points(rng, num_customers, 25)
            return {
                "facilities": facilities,
                "customers": customers,
                "fixed_cost": int_list(rng, num_facilities, 40, 70),
                "demand": int_list(rng, num_customers, 3, 8),
                "vehicle_capacity": base["vehicle_capacity"],
                "vehicle_cost": base["vehicle_cost"],
                "distance": [[_rounded_distance(f, c) for c in customers] for f in facilities],
            }

        def ok(instance: dict) -> bool:
            # 1 拠点だけ開けば済む instance では立地の選択が無いので捨てる。
            return len(solve(instance)["opened_facilities"]) >= 2

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        cp_model, solver = cp_sat_solver()
        facilities = range(len(instance["facilities"]))
        customers = range(len(instance["customers"]))
        demand = instance["demand"]
        capacity = instance["vehicle_capacity"]
        model = cp_model.CpModel()
        opened = [model.NewBoolVar(f"y{f}") for f in facilities]
        assign = {(c, f): model.NewBoolVar(f"x{c}_{f}") for c in customers for f in facilities}
        vehicles = [model.NewIntVar(0, len(customers), f"v{f}") for f in facilities]
        for c in customers:
            model.AddExactlyOne(assign[c, f] for f in facilities)
            for f in facilities:
                model.AddImplication(assign[c, f], opened[f])
        for f in facilities:
            # 車両数は割当需要を容量で割った切り上げ。費用が正なので最小化で切り上げ値に落ちる。
            model.Add(capacity * vehicles[f] >= sum(demand[c] * assign[c, f] for c in customers))
        model.Minimize(
            sum(instance["fixed_cost"][f] * opened[f] for f in facilities)
            + instance["vehicle_cost"] * sum(vehicles)
            + 2
            * sum(instance["distance"][f][c] * assign[c, f] for c in customers for f in facilities)
        )
        require_optimal(cp_model, solver.Solve(model))
        return {
            "min_total_cost": round(solver.ObjectiveValue()),
            "opened_facilities": [f for f in facilities if solver.Value(opened[f])],
            "assignment": {
                str(c): next(f for f in facilities if solver.Value(assign[c, f])) for c in customers
            },
            "note": "CP-SAT（厳密最適解）",
        }

    return generate, solve


@register(93, "min_total_cost")
def inventory_distribution():
    """prob_093: 在庫配送複合。期間数・小売店数・生産能力・生産単価は問題文どおり固定。"""

    def generate(rng: random.Random, base: dict) -> dict:
        periods = base["periods"]
        retailers = base["num_retailers"]
        capacity = base["production_capacity"]

        def make() -> dict:
            holding_warehouse = rng.randint(1, 3)
            # Why not 同じ保持費を許す: 倉庫と小売店の保持費が同じだと在庫の置き場所が同点になり、
            # LP の解が一意でなくなる。費用が違えば置き場所は一意に決まる。
            holding_retailer = rng.choice([h for h in (1, 2, 3) if h != holding_warehouse])
            return {
                "periods": periods,
                "num_retailers": retailers,
                "demand": int_matrix(rng, retailers, periods, 15, 40),
                "prod_cost": base["prod_cost"],
                "ship_cost": int_list(rng, retailers, 2, 5),
                "holding_warehouse": holding_warehouse,
                "holding_retailer": holding_retailer,
                "production_capacity": capacity,
            }

        def ok(instance: dict) -> bool:
            totals = [sum(col) for col in zip(*instance["demand"], strict=True)]
            cumulative = list(itertools.accumulate(totals))
            # 能力を超える期があり（在庫が要る）、かつ累積需要は累積能力に収まる instance に限る。
            return max(totals) > capacity and all(
                need <= capacity * (t + 1) for t, need in enumerate(cumulative)
            )

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        from scipy.optimize import linprog

        periods = instance["periods"]
        retailers = instance["num_retailers"]
        demand = instance["demand"]
        # 変数の並び: 生産 p[t]、輸送 s[r][t]、倉庫在庫 w[t]、小売店在庫 i[r][t]。
        n_prod = periods
        n_ship = retailers * periods
        num_vars = n_prod + n_ship + periods + n_ship

        def prod(t: int) -> int:
            return t

        def ship(r: int, t: int) -> int:
            return n_prod + r * periods + t

        def stock_w(t: int) -> int:
            return n_prod + n_ship + t

        def stock_r(r: int, t: int) -> int:
            return n_prod + n_ship + periods + r * periods + t

        cost = [0.0] * num_vars
        for t in range(periods):
            cost[prod(t)] = instance["prod_cost"]
            cost[stock_w(t)] = instance["holding_warehouse"]
            for r in range(retailers):
                cost[ship(r, t)] = instance["ship_cost"][r]
                cost[stock_r(r, t)] = instance["holding_retailer"]
        a_eq: list[list[float]] = []
        b_eq: list[float] = []
        for t in range(periods):
            # 倉庫: 前期在庫 + 生産 - 出荷 = 今期在庫（期首在庫は 0）。
            row = [0.0] * num_vars
            row[prod(t)] = 1.0
            row[stock_w(t)] = -1.0
            if t:
                row[stock_w(t - 1)] = 1.0
            for r in range(retailers):
                row[ship(r, t)] = -1.0
            a_eq.append(row)
            b_eq.append(0.0)
            for r in range(retailers):
                # 小売店: 前期在庫 + 入荷 - 需要 = 今期在庫。
                row = [0.0] * num_vars
                row[ship(r, t)] = 1.0
                row[stock_r(r, t)] = -1.0
                if t:
                    row[stock_r(r, t - 1)] = 1.0
                a_eq.append(row)
                b_eq.append(float(demand[r][t]))
        bounds = [(0, None)] * num_vars
        for t in range(periods):
            bounds[prod(t)] = (0, instance["production_capacity"])
        result = linprog(c=cost, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if not result.success:
            raise SolveError(f"LP failed: {result.message}")
        # Why not 丸める: 丸めた生産・輸送量は在庫収支を 0.01 だけ外しうる。
        # 保存する解は LP の値のまま、費用もその解から計算する。
        x = [float(v) for v in result.x]
        return {
            "min_total_cost": sum(c * v for c, v in zip(cost, x, strict=True)),
            "plan": {
                "production": [x[prod(t)] for t in range(periods)],
                "shipments": {
                    str(r): [x[ship(r, t)] for t in range(periods)] for r in range(retailers)
                },
            },
            "note": "LP（HiGHS、厳密最適解）",
        }

    return generate, solve


@register(95, "min_total_cost")
def multi_echelon_supply_chain():
    """prob_095: 多段供給連鎖。工場・倉庫・顧客の数は問題文どおり固定。"""

    def generate(rng: random.Random, base: dict) -> dict:
        plants = base["num_plants"]
        warehouses = base["num_warehouses"]
        customers = base["num_customers"]

        def make() -> dict:
            return {
                "num_plants": plants,
                "num_warehouses": warehouses,
                "num_customers": customers,
                "plant_capacity": int_list(rng, plants, 40, 70),
                "warehouse_capacity": int_list(rng, warehouses, 25, 50),
                "demand": int_list(rng, customers, 10, 20),
                "cost_plant_warehouse": int_matrix(rng, plants, warehouses, 1, 6),
                "cost_warehouse_customer": int_matrix(rng, warehouses, customers, 1, 6),
            }

        def ok(instance: dict) -> bool:
            total = sum(instance["demand"])
            if sum(instance["plant_capacity"]) < total:
                return False
            if sum(instance["warehouse_capacity"]) < total:
                return False
            # 各顧客を最安経路だけで賄える instance は能力制約が効かないので捨てる。
            bound = sum(
                need
                * min(
                    instance["cost_plant_warehouse"][p][w]
                    + instance["cost_warehouse_customer"][w][c]
                    for p in range(plants)
                    for w in range(warehouses)
                )
                for c, need in enumerate(instance["demand"])
            )
            return solve(instance)["min_total_cost"] > bound + 1e-6

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        from scipy.optimize import linprog

        plants = range(instance["num_plants"])
        warehouses = range(instance["num_warehouses"])
        customers = range(instance["num_customers"])
        # 変数の並び: 工場→倉庫 f[p][w]、倉庫→顧客 g[w][c]。
        n_first = len(plants) * len(warehouses)
        num_vars = n_first + len(warehouses) * len(customers)

        def first(p: int, w: int) -> int:
            return p * len(warehouses) + w

        def second(w: int, c: int) -> int:
            return n_first + w * len(customers) + c

        cost = [0.0] * num_vars
        for p in plants:
            for w in warehouses:
                cost[first(p, w)] = instance["cost_plant_warehouse"][p][w]
        for w in warehouses:
            for c in customers:
                cost[second(w, c)] = instance["cost_warehouse_customer"][w][c]
        a_ub: list[list[float]] = []
        b_ub: list[float] = []
        for p in plants:
            row = [0.0] * num_vars
            for w in warehouses:
                row[first(p, w)] = 1.0
            a_ub.append(row)
            b_ub.append(float(instance["plant_capacity"][p]))
        for w in warehouses:
            row = [0.0] * num_vars
            for p in plants:
                row[first(p, w)] = 1.0
            a_ub.append(row)
            b_ub.append(float(instance["warehouse_capacity"][w]))
        a_eq: list[list[float]] = []
        b_eq: list[float] = []
        for w in warehouses:
            # 倉庫は保管しないので流入と流出が等しい。
            row = [0.0] * num_vars
            for p in plants:
                row[first(p, w)] = 1.0
            for c in customers:
                row[second(w, c)] = -1.0
            a_eq.append(row)
            b_eq.append(0.0)
        for c in customers:
            row = [0.0] * num_vars
            for w in warehouses:
                row[second(w, c)] = 1.0
            a_eq.append(row)
            b_eq.append(float(instance["demand"][c]))
        result = linprog(
            c=cost,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=[(0, None)] * num_vars,
            method="highs",
        )
        if not result.success:
            raise SolveError(f"LP failed: {result.message}")
        # 同梱参照解と同じく目的値だけを返す。費用は LP の解から計算する。
        total = sum(c * float(v) for c, v in zip(cost, result.x, strict=True))
        return {"min_total_cost": total, "note": "LP（HiGHS、厳密最適解）"}

    return generate, solve


@register(96, "min_cost")
def capacitated_covering_assignment():
    """prob_096: 容量制約付き被覆割当。ゾーン数・センター数・半径は問題文どおり固定。

    半径判定は四捨五入した整数距離で行う。同梱参照解（[2, 4]、費用 57）はゾーン 5 と
    センター 4 の距離 25.495 を半径内としており、実距離で判定すると最適値は 61 になって
    参照値を再現できない。prob_092・098 の距離行列も整数へ丸めた値なので、この問題群の
    距離は整数丸めが規約とみなす。チェッカーも半径に 1 の余裕を持たせて同じ解を通す。
    """

    def generate(rng: random.Random, base: dict) -> dict:
        num_zones = len(base["zones"])
        num_centers = len(base["centers"])
        radius = base["radius"]

        def make() -> dict:
            return {
                "zones": _random_points(rng, num_zones, 25),
                "centers": _random_points(rng, num_centers, 22),
                "radius": radius,
                "demand": int_list(rng, num_zones, 2, 7),
                "capacity": int_list(rng, num_centers, 15, 25),
                "fixed_cost": int_list(rng, num_centers, 20, 40),
            }

        def ok(instance: dict) -> bool:
            try:
                opened = solve(instance)["opened_centers"]
            except SolveError:
                return False
            # 1 センターで全ゾーンを賄える instance では開設の組合せを考える余地が無い。
            return len(opened) >= 2

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        cp_model, solver = cp_sat_solver()
        zones = instance["zones"]
        centers = instance["centers"]
        radius = instance["radius"]
        model = cp_model.CpModel()
        opened = [model.NewBoolVar(f"y{k}") for k in range(len(centers))]
        assign: dict[tuple[int, int], object] = {}
        for z, zone in enumerate(zones):
            # Why not 実距離で判定: 同梱参照解は丸めた距離で半径内を判定している（docstring 参照）。
            reachable = [
                k for k, center in enumerate(centers) if _rounded_distance(zone, center) <= radius
            ]
            if not reachable:
                raise SolveError(f"zone {z} is outside every center's radius")
            for k in reachable:
                assign[z, k] = model.NewBoolVar(f"x{z}_{k}")
                model.AddImplication(assign[z, k], opened[k])
            model.AddExactlyOne(assign[z, k] for k in reachable)
        for k in range(len(centers)):
            model.Add(
                sum(
                    instance["demand"][z] * assign[z, k]
                    for z in range(len(zones))
                    if (z, k) in assign
                )
                <= instance["capacity"][k]
            )
        # 総開設費用を優先し、同点なら開設ビット列が小さい組合せを選ぶ（決定的なタイブレーク）。
        scale = 1 << len(centers)
        model.Minimize(
            scale * sum(instance["fixed_cost"][k] * opened[k] for k in range(len(centers)))
            + sum((1 << k) * opened[k] for k in range(len(centers)))
        )
        require_optimal(cp_model, solver.Solve(model))
        chosen = [k for k in range(len(centers)) if solver.Value(opened[k])]
        return {
            "min_cost": sum(instance["fixed_cost"][k] for k in chosen),
            "opened_centers": chosen,
            "note": "CP-SAT（厳密最適解）",
        }

    return generate, solve


def _tsptw_tour(instance: dict) -> tuple[int | None, list[int] | None, int]:
    """全順列を調べ、時間枠を守る最短巡回路と、時間枠を無視した最短距離を返す。

    Returns:
        (時間枠つき最短距離, その訪問順（デポ 0 を先頭）, 時間枠なしの最短距離)。
        時間枠を守る順路が無ければ先頭 2 つは None。
    """
    n = instance["num_nodes"]
    dist = instance["distance"]
    windows = instance["time_windows"]
    service = instance["service_time"]
    best: tuple[int, tuple[int, ...]] | None = None
    unconstrained = math.inf
    for order in itertools.permutations(range(1, n)):
        route = (0, *order, 0)
        length = sum(dist[u][v] for u, v in itertools.pairwise(route))
        unconstrained = min(unconstrained, length)
        clock = 0
        feasible = True
        for u, v in itertools.pairwise(route):
            # 到着が最早時刻より前なら待ち、最遅時刻を過ぎたら不可。デポへの帰着にも枠を課す。
            clock = max(clock + dist[u][v], windows[v][0])
            if clock > windows[v][1]:
                feasible = False
                break
            clock += service[v]
        # 同点は順列の辞書順で決める（permutations は辞書順に列挙する）。
        if feasible and (best is None or length < best[0]):
            best = (length, order)
    if best is None:
        return None, None, int(unconstrained)
    return best[0], [0, *best[1]], int(unconstrained)


@register(98, "min_distance")
def tsp_time_windows():
    """prob_098: 時間枠付き巡回路。ノード数 6（デポ + 顧客 5）は問題文どおり固定。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n = base["num_nodes"]
        depot_window = list(base["time_windows"][0])

        def make() -> dict:
            points = _random_points(rng, n, 25)
            windows = [depot_window]
            for _ in range(1, n):
                early = rng.randint(0, 40)
                windows.append([early, early + rng.randint(40, 150)])
            return {
                "num_nodes": n,
                "coordinates": points,
                "distance": [[_rounded_distance(a, b) for b in points] for a in points],
                "time_windows": windows,
                "service_time": [0, *int_list(rng, n - 1, 2, 5)],
            }

        def ok(instance: dict) -> bool:
            length, _, unconstrained = _tsptw_tour(instance)
            # 時間枠が無い TSP の最適路がそのまま通る instance では時間枠が飾りなので捨てる。
            return length is not None and length > unconstrained

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        length, tour, _ = _tsptw_tour(instance)
        if tour is None:
            raise SolveError("no tour satisfies the time windows")
        return {
            "min_distance": length,
            "tour": tour,
            "feasible": True,
            "note": "順列全列挙（厳密最適解）。距離はデポへの帰路を含む",
        }

    return generate, solve


@register(99, "max_clique_size", minimize=False)
def maximum_clique():
    """prob_099: 最大クリーク。10 ノードは問題文にあり、辺数 16 は形状として保つ。"""

    def generate(rng: random.Random, base: dict) -> dict:
        num_nodes = base["num_nodes"]
        num_edges = len(base["edges"])

        def make() -> dict:
            return {"num_nodes": num_nodes, "edges": _random_edges(rng, num_nodes, num_edges)}

        def ok(instance: dict) -> bool:
            # 三角形が無いと任意の辺が最大クリークになり、問題として薄いので捨てる。
            return solve(instance)["max_clique_size"] >= 3

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        num_nodes = instance["num_nodes"]
        adjacency = {frozenset(edge) for edge in instance["edges"]}
        # 大きいサイズから辞書順に部分集合を調べ、最初に見つかったクリークを返す。
        for size in range(num_nodes, 0, -1):
            for nodes in itertools.combinations(range(num_nodes), size):
                if all(frozenset(pair) in adjacency for pair in itertools.combinations(nodes, 2)):
                    return {
                        "max_clique_size": size,
                        "clique": list(nodes),
                        "note": "部分集合全列挙（厳密最適解）",
                    }
        raise SolveError("graph has no nodes")

    return generate, solve
