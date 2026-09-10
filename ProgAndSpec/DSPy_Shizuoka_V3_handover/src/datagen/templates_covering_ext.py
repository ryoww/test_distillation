"""施設配置・被覆 系の雛形（第 2 群: prob_044〜054）。 templates.py と同じ規約で (generate, solve) を登録する。

同梱 instance の距離行列・被覆関係は、整数座標のユークリッド距離を四捨五入した値で
できている（prob_045/046/048/051/053 の全要素で一致を確認）。被覆判定（prob_044/052/054）は
丸めなしのユークリッド距離が半径以下かで行い、同梱参照解の被覆人口を再現する。
"""

from __future__ import annotations

import itertools
import math
import random

from .base import cp_sat_solver, int_list, register, require_optimal, retry

_COORD = 30
_NOTE_CP_SAT = "CP-SAT（厳密最適解）"


def _points(rng: random.Random, n: int) -> list[dict]:
    return [
        {"id": i, "x": rng.randint(-_COORD, _COORD), "y": rng.randint(-_COORD, _COORD)}
        for i in range(n)
    ]


def _dist(a: dict, b: dict) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def _rounded_distances(rows: list[dict], cols: list[dict]) -> list[list[int]]:
    return [[round(_dist(a, b)) for b in cols] for a in rows]


def _coverage(demand_points: list[dict], site_points: list[dict], radius: float) -> list[list[int]]:
    """site ごとに、半径内にある需要点のインデックス列。"""
    return [
        [j for j, p in enumerate(demand_points) if _dist(p, site) <= radius] for site in site_points
    ]


def _distinct_points(points: list[dict]) -> bool:
    return len({(p["x"], p["y"]) for p in points}) == len(points)


def _unambiguous_coverage(instance: dict, chosen: list[int]) -> bool:
    """選んだ施設の半径ぎりぎり外（radius, radius+1]に需要地がない。

    Why not 半径どおりに判定して終わり: check_covering_ip は半径に 1 の余裕（_RADIUS_SLACK）を
    持たせて被覆人口を再計算するので、この帯に需要地があると正しい参照解が
    「covered population != actual coverage」で弾かれる。帯が空の instance だけ採用する。
    """
    radius = instance["radius"]
    for i in chosen:
        site = instance["site_points"][i]
        if any(radius < _dist(p, site) <= radius + 1 for p in instance["demand_points"]):
            return False
    return True


def _max_coverage_model(instance: dict):
    """被覆人口最大化の共通部分。施設数・予算の制約は呼び出し側が足す。"""
    cp_model, solver = cp_sat_solver()
    sites = instance["site_points"]
    demand_points = instance["demand_points"]
    covers = _coverage(demand_points, sites, instance["radius"])
    model = cp_model.CpModel()
    y = [model.NewBoolVar(f"y{i}") for i in range(len(sites))]
    z = [model.NewBoolVar(f"z{j}") for j in range(len(demand_points))]
    for j in range(len(demand_points)):
        # 被覆フラグは半径内の施設が 1 つ以上開くときだけ立てられる。
        model.Add(z[j] <= sum(y[i] for i in range(len(sites)) if j in covers[i]))
    model.Maximize(sum(p * z[j] for j, p in enumerate(instance["population"])))
    return cp_model, solver, model, y


def _solve_max_coverage(instance: dict, add_limits) -> dict:
    cp_model, solver, model, y = _max_coverage_model(instance)
    add_limits(model, y)
    require_optimal(cp_model, solver.Solve(model))
    return {
        "max_covered_population": round(solver.ObjectiveValue()),
        "chosen_sites": [i for i in range(len(y)) if solver.Value(y[i])],
        "note": _NOTE_CP_SAT,
    }


@register(44, "max_covered_population", minimize=False)
def max_coverage():
    """prob_044: 最大被覆。半径 18 と施設数上限 3 は問題文にあるので雛形の値を保つ。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n_demand, n_sites = len(base["demand_points"]), len(base["site_points"])
        radius, limit = base["radius"], base["max_facilities"]

        def make() -> dict:
            return {
                "demand_points": _points(rng, n_demand),
                "site_points": _points(rng, n_sites),
                "radius": radius,
                "max_facilities": limit,
                "population": int_list(rng, n_demand, 2, 8),
            }

        def ok(inst: dict) -> bool:
            result = solve(inst)
            # 3 施設で全需要地を覆えてしまう instance は上限が効かないので捨てる。
            return result["max_covered_population"] < sum(inst["population"]) and (
                _unambiguous_coverage(inst, result["chosen_sites"])
            )

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        def add_limits(model, y):
            model.Add(sum(y) <= instance["max_facilities"])

        return _solve_max_coverage(instance, add_limits)

    return generate, solve


@register(54, "max_covered_population", minimize=False)
def budgeted_max_coverage():
    """prob_054: 予算制約付き最大被覆。半径 18 と予算 100 は問題文にあるので雛形の値を保つ。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n_demand, n_sites = len(base["demand_points"]), len(base["site_points"])
        radius, budget = base["radius"], base["budget"]

        def make() -> dict:
            return {
                "demand_points": _points(rng, n_demand),
                "site_points": _points(rng, n_sites),
                "radius": radius,
                "site_cost": int_list(rng, n_sites, 27, 42),
                "population": int_list(rng, n_demand, 2, 12),
                "budget": budget,
            }

        def ok(inst: dict) -> bool:
            result = solve(inst)
            # 予算内で全需要地を覆える、または 1 施設しか買えない instance は薄いので捨てる。
            return (
                2 <= len(result["chosen_sites"])
                and result["max_covered_population"] < sum(inst["population"])
                and _unambiguous_coverage(inst, result["chosen_sites"])
            )

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        def add_limits(model, y):
            model.Add(
                sum(c * y[i] for i, c in enumerate(instance["site_cost"])) <= instance["budget"]
            )

        return _solve_max_coverage(instance, add_limits)

    return generate, solve


@register(52, "min_facilities")
def set_covering_location():
    """prob_052: 被覆立地（最小施設数）。半径 26 は問題文にあるので雛形の値を保つ。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n_demand, n_sites = len(base["demand_points"]), len(base["site_points"])
        radius = base["radius"]

        def make() -> dict:
            return {
                "demand_points": _points(rng, n_demand),
                "site_points": _points(rng, n_sites),
                "radius": radius,
            }

        def ok(inst: dict) -> bool:
            covers = _coverage(inst["demand_points"], inst["site_points"], inst["radius"])
            if set().union(*covers) != set(range(n_demand)):
                return False  # 覆えない需要地があると実行不可能
            # 1 施設で全部覆える instance は自明なので捨てる。
            return solve(inst)["min_facilities"] >= 2

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        cp_model, solver = cp_sat_solver()
        sites = instance["site_points"]
        covers = _coverage(instance["demand_points"], sites, instance["radius"])
        model = cp_model.CpModel()
        y = [model.NewBoolVar(f"y{i}") for i in range(len(sites))]
        for j in range(len(instance["demand_points"])):
            model.AddBoolOr([y[i] for i in range(len(sites)) if j in covers[i]])
        model.Minimize(sum(y))
        require_optimal(cp_model, solver.Solve(model))
        return {
            "min_facilities": round(solver.ObjectiveValue()),
            "chosen_sites": [i for i in range(len(sites)) if solver.Value(y[i])],
            "note": _NOTE_CP_SAT,
        }

    return generate, solve


@register(45, "min_total_cost")
def uncapacitated_facility_location():
    """prob_045: 無容量施設配置。開設集合を全列挙し、各顧客を最も安い開設施設に付ける。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n_fac, n_cus = len(base["facilities"]), len(base["customers"])

        def make() -> dict:
            facilities, customers = _points(rng, n_fac), _points(rng, n_cus)
            return {
                "facilities": facilities,
                "customers": customers,
                "fixed_cost": int_list(rng, n_fac, 30, 80),
                "transport_cost": _rounded_distances(facilities, customers),
            }

        # 全施設を開くのが最適だと、固定費のトレードオフが消えるので捨てる。
        return retry(rng, make, lambda inst: len(solve(inst)["opened_facilities"]) < n_fac)

    def solve(instance: dict) -> dict:
        fixed, cost = instance["fixed_cost"], instance["transport_cost"]
        n_cus = len(instance["customers"])
        best: tuple[int, tuple[int, ...]] | None = None
        # Why not MILP: 候補地 5 箇所なら 31 通りの開設集合を調べれば厳密で、同点は辞書順で決まる。
        for k in range(1, len(fixed) + 1):
            for opened in itertools.combinations(range(len(fixed)), k):
                total = sum(fixed[i] for i in opened)
                total += sum(min(cost[i][j] for i in opened) for j in range(n_cus))
                if best is None or total < best[0]:
                    best = (total, opened)
        assert best is not None
        return {
            "min_total_cost": int(best[0]),
            "opened_facilities": list(best[1]),
            "note": "開設集合の全列挙（厳密最適解）",
        }

    return generate, solve


@register(46, "min_total_cost")
def capacitated_facility_location():
    """prob_046: 容量制約施設配置（単一供給元）。輸送費用は距離×需要量。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n_fac, n_cus = len(base["facilities"]), len(base["customers"])

        def make() -> dict:
            facilities, customers = _points(rng, n_fac), _points(rng, n_cus)
            return {
                "facilities": facilities,
                "customers": customers,
                "fixed_cost": int_list(rng, n_fac, 50, 80),
                "capacity": int_list(rng, n_fac, 20, 35),
                "demand": int_list(rng, n_cus, 3, 10),
                "distance": _rounded_distances(facilities, customers),
            }

        def ok(inst: dict) -> bool:
            total_demand = sum(inst["demand"])
            # 全開設でも足りない instance は実行不可能、1 施設で足りる instance は容量が効かない。
            if sum(inst["capacity"]) < total_demand or max(inst["capacity"]) >= total_demand:
                return False
            return len(solve(inst)["opened_facilities"]) < n_fac

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        cp_model, solver = cp_sat_solver()
        fixed, capacity = instance["fixed_cost"], instance["capacity"]
        demand, distance = instance["demand"], instance["distance"]
        n_fac, n_cus = len(fixed), len(demand)
        model = cp_model.CpModel()
        y = [model.NewBoolVar(f"y{i}") for i in range(n_fac)]
        x = [[model.NewBoolVar(f"x{i}_{j}") for j in range(n_cus)] for i in range(n_fac)]
        for j in range(n_cus):
            model.AddExactlyOne(x[i][j] for i in range(n_fac))
        for i in range(n_fac):
            model.Add(sum(demand[j] * x[i][j] for j in range(n_cus)) <= capacity[i] * y[i])
        model.Minimize(
            sum(fixed[i] * y[i] for i in range(n_fac))
            + sum(distance[i][j] * demand[j] * x[i][j] for i in range(n_fac) for j in range(n_cus))
        )
        require_optimal(cp_model, solver.Solve(model))
        return {
            "min_total_cost": round(solver.ObjectiveValue()),
            "opened_facilities": [i for i in range(n_fac) if solver.Value(y[i])],
            "note": _NOTE_CP_SAT,
        }

    return generate, solve


@register(51, "min_total_cost")
def warehouse_location_split_supply():
    """prob_051: 倉庫配置（分割供給可）。開設集合ごとに輸送問題を LP で解いて最小を取る。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n_wh, n_st = len(base["warehouses"]), len(base["stores"])

        def make() -> dict:
            warehouses, stores = _points(rng, n_wh), _points(rng, n_st)
            return {
                "warehouses": warehouses,
                "stores": stores,
                "fixed_cost": int_list(rng, n_wh, 50, 100),
                "capacity": int_list(rng, n_wh, 40, 65),
                "demand": int_list(rng, n_st, 6, 15),
                "distance": _rounded_distances(warehouses, stores),
            }

        def ok(inst: dict) -> bool:
            total_demand = sum(inst["demand"])
            if sum(inst["capacity"]) < total_demand or max(inst["capacity"]) >= total_demand:
                return False
            return len(solve(inst)["opened_warehouses"]) < n_wh

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        fixed, capacity = instance["fixed_cost"], instance["capacity"]
        demand, distance = instance["demand"], instance["distance"]
        total_demand = sum(demand)
        best: tuple[int, tuple[int, ...]] | None = None
        # Why not MILP: 開設集合を固定すれば残りは輸送問題（LP）で、4 倉庫なら 15 通りで済む。
        for k in range(1, len(fixed) + 1):
            for opened in itertools.combinations(range(len(fixed)), k):
                if sum(capacity[i] for i in opened) < total_demand:
                    continue
                transport = _transportation_cost(
                    [distance[i] for i in opened], [capacity[i] for i in opened], demand
                )
                total = sum(fixed[i] for i in opened) + transport
                if best is None or total < best[0]:
                    best = (total, opened)
        if best is None:
            raise ValueError("total capacity is below total demand")
        return {
            "min_total_cost": int(best[0]),
            "opened_warehouses": list(best[1]),
            "note": "開設集合の全列挙 + 輸送問題 LP（scipy HiGHS、厳密最適解）",
        }

    return generate, solve


def _transportation_cost(cost: list[list[int]], supply: list[int], demand: list[int]) -> int:
    """供給上限つき輸送問題の最小費用。

    供給・需要が整数なら制約行列は完全単模なので最適値も整数になり、LP の値を丸めてよい。
    """
    from scipy.optimize import linprog

    m, n = len(supply), len(demand)
    c = [cost[i][j] for i in range(m) for j in range(n)]
    a_eq = [[1 if k % n == j else 0 for k in range(m * n)] for j in range(n)]
    a_ub = [[1 if k // n == i else 0 for k in range(m * n)] for i in range(m)]
    result = linprog(c, A_ub=a_ub, b_ub=supply, A_eq=a_eq, b_eq=demand, method="highs")
    if result.status != 0:
        raise RuntimeError(f"transportation LP failed: {result.message}")
    value = round(result.fun)
    if abs(result.fun - value) > 1e-6:
        raise RuntimeError(f"transportation LP optimum is not integral: {result.fun}")
    return value


@register(48, "min_max_distance")
def p_center():
    """prob_048: p-センター。9 地点から 2 箇所なので施設の組を全列挙する。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n, p = len(base["nodes"]), base["p"]

        def make() -> dict:
            nodes = _points(rng, n)
            return {"nodes": nodes, "p": p, "distance": _rounded_distances(nodes, nodes)}

        # 座標が重なると距離 0 の地点対ができ、施設選択が事実上 p-1 箇所になる。
        return retry(rng, make, lambda inst: _distinct_points(inst["nodes"]))

    def solve(instance: dict) -> dict:
        distance, n = instance["distance"], len(instance["nodes"])
        best: tuple[int, tuple[int, ...]] | None = None
        for selected in itertools.combinations(range(n), instance["p"]):
            worst = max(min(distance[i][j] for i in selected) for j in range(n))
            if best is None or worst < best[0]:
                best = (worst, selected)
        assert best is not None
        return {
            "min_max_distance": int(best[0]),
            "selected_facilities": list(best[1]),
            "note": "施設組合せの全列挙（厳密最適解）",
        }

    return generate, solve


@register(53, "min_cost")
def single_allocation_hub_location():
    """prob_053: 単一割当ハブ配置。

    同梱参照解の 3835 は「収集配送コストモデル」で、各都市の発着流量の合計 × 割当ハブまでの
    距離を足したもの（ハブ間輸送費は含まず、hub_discount は目的値に現れない）。問題文の
    「収集・配送コスト（流量×ハブまでの距離）の合計」もこの定義なので、それに合わせる。
    ハブ集合を固定すると各都市は最寄りハブに付けるのが最適なので、ハブの組を全列挙する。
    """

    def generate(rng: random.Random, base: dict) -> dict:
        n, num_hubs, discount = len(base["nodes"]), base["num_hubs"], base["hub_discount"]

        def make() -> dict:
            nodes = _points(rng, n)
            flow = [[0 if i == j else rng.randint(1, 10) for j in range(n)] for i in range(n)]
            return {
                "nodes": nodes,
                "flow": flow,
                "distance": _rounded_distances(nodes, nodes),
                "num_hubs": num_hubs,
                "hub_discount": discount,
            }

        return retry(rng, make, lambda inst: _distinct_points(inst["nodes"]))

    def solve(instance: dict) -> dict:
        flow, distance, n = instance["flow"], instance["distance"], len(instance["nodes"])
        volume = [sum(flow[i]) + sum(flow[j][i] for j in range(n)) for i in range(n)]
        best: tuple[int, tuple[int, ...], list[int]] | None = None
        for hubs in itertools.combinations(range(n), instance["num_hubs"]):
            # 同点の最寄りハブは番号の小さいハブに付ける（min はキー順で最初の要素を返す）。
            assignment = [min(hubs, key=lambda h: (distance[i][h], h)) for i in range(n)]
            total = sum(volume[i] * distance[i][assignment[i]] for i in range(n))
            if best is None or total < best[0]:
                best = (total, hubs, assignment)
        assert best is not None
        return {
            "min_cost": int(best[0]),
            "hubs": list(best[1]),
            "assignment": {str(i): h for i, h in enumerate(best[2])},
            "note": "ハブ組合せの全列挙（収集配送コストモデル、厳密最適解）",
        }

    return generate, solve


@register(49, "min_cost")
def weighted_vertex_cover():
    """prob_049: 重み付き頂点被覆。交差点数は問題文にあるので雛形のまま、辺数も雛形に合わせる。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n, m = base["num_nodes"], len(base["edges"])
        candidates = [(u, v) for u in range(n) for v in range(u + 1, n)]

        def make() -> dict:
            edges = [list(e) for e in sorted(rng.sample(candidates, m))]
            return {"num_nodes": n, "edges": edges, "cost": int_list(rng, n, 1, 7)}

        # 辺を持たない交差点は問題に関与しないので、全交差点が道路網に含まれる instance だけ使う。
        def ok(inst: dict) -> bool:
            return len({v for e in inst["edges"] for v in e}) == n

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        cp_model, solver = cp_sat_solver()
        n, cost = instance["num_nodes"], instance["cost"]
        model = cp_model.CpModel()
        y = [model.NewBoolVar(f"y{i}") for i in range(n)]
        for u, v in instance["edges"]:
            model.AddBoolOr([y[u], y[v]])
        model.Minimize(sum(cost[i] * y[i] for i in range(n)))
        require_optimal(cp_model, solver.Solve(model))
        return {
            "min_cost": round(solver.ObjectiveValue()),
            "selected_nodes": [i for i in range(n) if solver.Value(y[i])],
            "note": _NOTE_CP_SAT,
        }

    return generate, solve


@register(50, "min_cost")
def set_partitioning():
    """prob_050: 集合分割。雛形どおり、複数タスクのチーム群の後ろに各タスクの単独チームを置く。

    単独チームを全タスク分入れておくと、どんな乱数でも実行可能解が存在する。
    """

    def generate(rng: random.Random, base: dict) -> dict:
        num_items = base["num_items"]
        num_multi = len(base["groups"]) - num_items

        def make() -> dict:
            groups = []
            for i in range(num_multi):
                members = sorted(rng.sample(range(num_items), rng.randint(2, 4)))
                groups.append({"id": i, "members": members, "cost": rng.randint(5, 15)})
            for item in range(num_items):
                groups.append(
                    {"id": num_multi + item, "members": [item], "cost": rng.randint(8, 20)}
                )
            return {"num_items": num_items, "groups": groups}

        def ok(inst: dict) -> bool:
            # 単独チームだけで組むのが最適な instance は分割の要素がないので捨てる。
            return any(i < num_multi for i in solve(inst)["chosen_groups"])

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        cp_model, solver = cp_sat_solver()
        groups = instance["groups"]
        model = cp_model.CpModel()
        y = [model.NewBoolVar(f"y{i}") for i in range(len(groups))]
        for item in range(instance["num_items"]):
            model.AddExactlyOne(y[i] for i, g in enumerate(groups) if item in g["members"])
        model.Minimize(sum(g["cost"] * y[i] for i, g in enumerate(groups)))
        require_optimal(cp_model, solver.Solve(model))
        return {
            "min_cost": round(solver.ObjectiveValue()),
            "chosen_groups": [i for i in range(len(groups)) if solver.Value(y[i])],
            "note": _NOTE_CP_SAT,
        }

    return generate, solve
