"""既存問題を雛形にした instance 生成器と厳密ソルバー。

各テンプレートは雛形 instance の件数と、問題文に書かれた容量・予算などの
スカラーを保ち、それ以外の数値だけを乱数で置き換える。参照解は雛形と同じ
トップレベルのキーで返し、note にはソルバーを記す。
"""

from __future__ import annotations

import itertools
import math
import random

from .base import (
    cp_sat_solver,
    int_list,
    int_matrix,
    partition,
    register,
    require_optimal,
    retry,
)

# ============================================================
# スケジューリング
# ============================================================


@register(1, "objective_value")
def single_machine_tardiness():
    """prob_001: 単一機械の総遅延最小化。順列を全列挙する。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n = len(base["jobs"])
        times = int_list(rng, n, 1, 9)
        horizon = sum(times)
        jobs = [
            {"id": i + 1, "processing_time": p, "due_date": rng.randint(2, max(2, horizon - 2))}
            for i, p in enumerate(times)
        ]
        return {"jobs": jobs}

    def solve(instance: dict) -> dict:
        jobs = instance["jobs"]
        if len(jobs) > 9:
            raise ValueError("brute force is limited to 9 jobs")
        best_seq: list[int] = []
        best = math.inf
        for order in itertools.permutations(jobs):
            clock = 0
            tardiness = 0
            for job in order:
                clock += job["processing_time"]
                tardiness += max(0, clock - job["due_date"])
                if tardiness >= best:
                    break
            if tardiness < best:
                best = tardiness
                best_seq = [job["id"] for job in order]
        return {
            "optimal_sequence": best_seq,
            "objective_value": int(best),
            "note": "総遅延時間 = sum(max(0, 完工時刻 - 納期))。順列全列挙（厳密最適解）",
        }

    return generate, solve


# ============================================================
# ネットワークフロー
# ============================================================


@register(31, "max_flow")
def max_flow():
    """prob_031: 最大流。源へ入る辺と集から出る辺は作らない。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n = base["num_nodes"]
        source, sink = base["source"], base["sink"]
        m = len(base["edges"])
        candidates = [
            (u, v) for u in range(n) for v in range(n) if u != v and v != source and u != sink
        ]

        def make() -> dict:
            chosen = rng.sample(candidates, m)
            edges = [{"from": u, "to": v, "capacity": rng.randint(3, 20)} for u, v in chosen]
            return {"num_nodes": n, "source": source, "sink": sink, "edges": edges}

        return retry(rng, make, lambda inst: _max_flow_value(inst) > 0)

    def solve(instance: dict) -> dict:
        return {"max_flow": int(_max_flow_value(instance)), "note": "networkx最大流（厳密最適解）"}

    return generate, solve


def _max_flow_value(instance: dict) -> float:
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_nodes_from(range(instance["num_nodes"]))
    for edge in instance["edges"]:
        graph.add_edge(edge["from"], edge["to"], capacity=edge["capacity"])
    return nx.maximum_flow_value(graph, instance["source"], instance["sink"])


@register(33, "min_cost")
def transportation():
    """prob_033: 供給と需要が釣り合う輸送問題。"""

    def generate(rng: random.Random, base: dict) -> dict:
        nf, nd = base["num_factories"], base["num_destinations"]
        supply = int_list(rng, nf, 15, 45)
        demand = partition(rng, sum(supply), nd, 5)
        return {
            "num_factories": nf,
            "num_destinations": nd,
            "supply": supply,
            "demand": demand,
            "cost": int_matrix(rng, nf, nd, 1, 12),
        }

    def solve(instance: dict) -> dict:
        import networkx as nx

        graph = nx.DiGraph()
        for i, s in enumerate(instance["supply"]):
            graph.add_node(("f", i), demand=-s)
        for j, d in enumerate(instance["demand"]):
            graph.add_node(("d", j), demand=d)
        for i, row in enumerate(instance["cost"]):
            for j, c in enumerate(row):
                graph.add_edge(("f", i), ("d", j), weight=c)
        cost = nx.min_cost_flow_cost(graph)
        return {"min_cost": int(cost), "note": "networkx最小費用流（厳密最適解）"}

    return generate, solve


# ============================================================
# 施設配置・被覆
# ============================================================


@register(43, "min_cost")
def set_cover():
    """prob_043: 集合被覆。全要素がどこかの集合に入るよう補正する。"""

    def generate(rng: random.Random, base: dict) -> dict:
        num_items = base["num_items"]
        num_sets = len(base["sets"])
        covers = [sorted(rng.sample(range(num_items), rng.randint(2, 4))) for _ in range(num_sets)]
        for item in range(num_items):
            if not any(item in c for c in covers):
                target = rng.randrange(num_sets)
                covers[target] = sorted(covers[target] + [item])
        sets = [{"id": i, "covers": c, "cost": rng.randint(2, 12)} for i, c in enumerate(covers)]
        return {"num_items": num_items, "sets": sets}

    def solve(instance: dict) -> dict:
        cp_model, solver = cp_sat_solver()
        sets = instance["sets"]
        model = cp_model.CpModel()
        y = [model.NewBoolVar(f"y{i}") for i in range(len(sets))]
        for item in range(instance["num_items"]):
            model.AddBoolOr([y[i] for i, s in enumerate(sets) if item in s["covers"]])
        model.Minimize(sum(s["cost"] * y[i] for i, s in enumerate(sets)))
        require_optimal(cp_model, solver.Solve(model))
        chosen = [i for i in range(len(sets)) if solver.Value(y[i])]
        return {
            "min_cost": round(solver.ObjectiveValue()),
            "chosen_sets": chosen,
            "note": "CP-SAT（厳密最適解）",
        }

    return generate, solve


# ============================================================
# ナップサック・パッキング
# ============================================================


@register(55, "max_value")
def knapsack_01():
    """prob_055: 0/1 ナップサック。容量は問題文にあるので雛形の値を保つ。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n = len(base["items"])
        capacity = base["capacity"]

        def make() -> dict:
            items = [
                {"id": i, "weight": rng.randint(3, 13), "value": rng.randint(5, 25)}
                for i in range(n)
            ]
            return {"items": items, "capacity": capacity}

        return retry(rng, make, lambda inst: sum(i["weight"] for i in inst["items"]) > capacity)

    def solve(instance: dict) -> dict:
        cp_model, solver = cp_sat_solver()
        items = instance["items"]
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x{i}") for i in range(len(items))]
        model.Add(sum(it["weight"] * x[i] for i, it in enumerate(items)) <= instance["capacity"])
        model.Maximize(sum(it["value"] * x[i] for i, it in enumerate(items)))
        require_optimal(cp_model, solver.Solve(model))
        chosen = [items[i]["id"] for i in range(len(items)) if solver.Value(x[i])]
        return {
            "max_value": round(solver.ObjectiveValue()),
            "chosen_items": chosen,
            "note": "CP-SAT（厳密最適解）",
        }

    return generate, solve


@register(59, "min_bins")
def bin_packing():
    """prob_059: ビンパッキング。箱容量は問題文にあるので雛形の値を保つ。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n = len(base["item_sizes"])
        capacity = base["bin_capacity"]
        return {"item_sizes": int_list(rng, n, 2, min(6, capacity)), "bin_capacity": capacity}

    def solve(instance: dict) -> dict:
        cp_model, solver = cp_sat_solver()
        sizes = instance["item_sizes"]
        capacity = instance["bin_capacity"]
        n = len(sizes)
        model = cp_model.CpModel()
        used = [model.NewBoolVar(f"u{b}") for b in range(n)]
        x = {(i, b): model.NewBoolVar(f"x{i}_{b}") for i in range(n) for b in range(n)}
        for i in range(n):
            model.AddExactlyOne(x[i, b] for b in range(n))
        for b in range(n):
            model.Add(sum(sizes[i] * x[i, b] for i in range(n)) <= capacity * used[b])
            if b + 1 < n:
                # Why not 任意の箱番号: 箱は互換なので、番号順に使うことにして対称解を切る。
                model.AddImplication(used[b + 1], used[b])
        model.Minimize(sum(used))
        require_optimal(cp_model, solver.Solve(model))
        bins: dict[str, list[int]] = {}
        for b in range(n):
            members = [i for i in range(n) if solver.Value(x[i, b])]
            if members:
                bins[str(len(bins))] = members
        return {"min_bins": len(bins), "bins": bins, "note": "CP-SAT（厳密最適解）"}

    return generate, solve


@register(62, "min_height")
def strip_packing():
    """prob_062: 2次元ストリップパッキング。帯幅と矩形数は問題文にあるので雛形の値を保つ。"""

    def generate(rng: random.Random, base: dict) -> dict:
        width = base["strip_width"]
        n = len(base["rectangles"])

        def make() -> dict:
            rects = [
                {"id": i, "w": rng.randint(2, min(6, width)), "h": rng.randint(2, 6)}
                for i in range(n)
            ]
            return {"strip_width": width, "rectangles": rects}

        # 1 段に全部並ぶ instance は配置の自由度がないので除く。
        return retry(rng, make, lambda inst: sum(r["w"] for r in inst["rectangles"]) > width)

    def solve(instance: dict) -> dict:
        cp_model, solver = cp_sat_solver()
        width = instance["strip_width"]
        rects = instance["rectangles"]
        upper = sum(r["h"] for r in rects)
        model = cp_model.CpModel()
        height = model.NewIntVar(0, upper, "height")
        xs, ys, x_iv, y_iv = [], [], [], []
        for r in rects:
            x = model.NewIntVar(0, width - r["w"], f"x{r['id']}")
            y = model.NewIntVar(0, upper - r["h"], f"y{r['id']}")
            xs.append(x)
            ys.append(y)
            x_iv.append(model.NewFixedSizeIntervalVar(x, r["w"], f"xi{r['id']}"))
            y_iv.append(model.NewFixedSizeIntervalVar(y, r["h"], f"yi{r['id']}"))
            model.Add(y + r["h"] <= height)
        model.AddNoOverlap2D(x_iv, y_iv)
        model.Minimize(height)
        require_optimal(cp_model, solver.Solve(model))
        placement = [
            {
                "id": r["id"],
                "x": solver.Value(xs[i]),
                "y": solver.Value(ys[i]),
                "w": r["w"],
                "h": r["h"],
            }
            for i, r in enumerate(rects)
        ]
        return {
            "min_height": solver.Value(height),
            "placement": placement,
            "note": "CP-SAT NoOverlap2D（厳密最適解）",
        }

    return generate, solve


# ============================================================
# 割当・マッチング
# ============================================================


@register(67, "min_cost")
def linear_assignment():
    """prob_067: 一対一割当。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n = base["num_agents"]
        return {"num_agents": n, "cost_matrix": int_matrix(rng, n, n, 3, 30)}

    def solve(instance: dict) -> dict:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        matrix = np.array(instance["cost_matrix"])
        rows, cols = linear_sum_assignment(matrix)
        return {
            "min_cost": int(matrix[rows, cols].sum()),
            "assignment": {str(int(r)): int(c) for r, c in zip(rows, cols)},
            "note": "ハンガリアン法（厳密最適解）",
        }

    return generate, solve


# ============================================================
# 生産・在庫計画 / 金融・投資
# ============================================================


@register(77, "max_profit")
def product_mix():
    """prob_077: 連続変数の生産ミックス LP。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n = base["num_products"]
        r = len(base["resource_usage"])
        return {
            "num_products": n,
            "profit": int_list(rng, n, 20, 50),
            "resource_usage": int_matrix(rng, r, n, 1, 5),
            "resource_available": int_list(rng, r, 80, 130),
        }

    def solve(instance: dict) -> dict:
        from scipy.optimize import linprog

        result = linprog(
            c=[-p for p in instance["profit"]],
            A_ub=instance["resource_usage"],
            b_ub=instance["resource_available"],
            bounds=[(0, None)] * instance["num_products"],
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"LP failed: {result.message}")
        # Why not 小数2桁に丸める: 丸めた生産量は資源上限を 0.01 だけ超えうる。
        # 保存する解は LP の値のまま、利益もその解から計算する。
        production = [float(v) for v in result.x]
        return {
            "max_profit": sum(p * q for p, q in zip(instance["profit"], production, strict=True)),
            "production": {str(i): q for i, q in enumerate(production)},
            "note": "LP（HiGHS、厳密最適解）",
        }

    return generate, solve


@register(81, "max_npv")
def capital_budgeting():
    """prob_081: 資本予算。予算は問題文にあるので雛形の値を保つ。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n = len(base["investment"])
        budget = base["budget"]

        def make() -> dict:
            return {
                "investment": int_list(rng, n, 25, 75),
                "npv": int_list(rng, n, 50, 120),
                "budget": budget,
            }

        return retry(rng, make, lambda inst: sum(inst["investment"]) > budget)

    def solve(instance: dict) -> dict:
        cp_model, solver = cp_sat_solver()
        investment, npv = instance["investment"], instance["npv"]
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x{i}") for i in range(len(investment))]
        model.Add(sum(investment[i] * x[i] for i in range(len(x))) <= instance["budget"])
        model.Maximize(sum(npv[i] * x[i] for i in range(len(x))))
        require_optimal(cp_model, solver.Solve(model))
        return {
            "max_npv": round(solver.ObjectiveValue()),
            "chosen_projects": [i for i in range(len(x)) if solver.Value(x[i])],
            "note": "CP-SAT（厳密最適解）",
        }

    return generate, solve


# ============================================================
# 複合・グラフ最適化
# ============================================================


@register(91, "max_weight")
def max_weight_independent_set():
    """prob_091: 最大重み独立集合。辺数と頂点数は雛形のまま、辺集合と重みを引き直す。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n = base["num_nodes"]
        m = len(base["edges"])
        pairs = [[u, v] for u in range(n) for v in range(u + 1, n)]
        edges = sorted(rng.sample(pairs, m))
        return {"num_nodes": n, "edges": edges, "weights": int_list(rng, n, 1, 10)}

    def solve(instance: dict) -> dict:
        cp_model, solver = cp_sat_solver()
        weights = instance["weights"]
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f"x{i}") for i in range(instance["num_nodes"])]
        for u, v in instance["edges"]:
            model.AddBoolOr([x[u].Not(), x[v].Not()])
        model.Maximize(sum(w * x[i] for i, w in enumerate(weights)))
        require_optimal(cp_model, solver.Solve(model))
        return {
            "max_weight": round(solver.ObjectiveValue()),
            "selected_nodes": [i for i in range(len(x)) if solver.Value(x[i])],
            "note": "CP-SAT（厳密最適解）",
        }

    return generate, solve


@register(89, "min_distance")
def tsp():
    """prob_089: 座標から丸めたユークリッド距離の TSP。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n = base["num_cities"]
        points = rng.sample([(x, y) for x in range(-25, 26) for y in range(-25, 26)], n)
        coordinates = [{"id": i, "x": x, "y": y} for i, (x, y) in enumerate(points)]
        matrix = [[round(math.dist(a, b)) if a != b else 0 for b in points] for a in points]
        return {"num_cities": n, "coordinates": coordinates, "distance_matrix": matrix}

    def solve(instance: dict) -> dict:
        cp_model, solver = cp_sat_solver()
        matrix = instance["distance_matrix"]
        n = instance["num_cities"]
        model = cp_model.CpModel()
        arcs = []
        lits = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    lits[i, j] = model.NewBoolVar(f"a{i}_{j}")
                    arcs.append((i, j, lits[i, j]))
        model.AddCircuit(arcs)
        model.Minimize(sum(matrix[i][j] * lit for (i, j), lit in lits.items()))
        require_optimal(cp_model, solver.Solve(model))
        successor = {i: j for (i, j), lit in lits.items() if solver.Value(lit)}
        tour = [0]
        while len(tour) < n:
            tour.append(successor[tour[-1]])
        return {
            "min_distance": round(solver.ObjectiveValue()),
            "tour": tour,
            "note": "CP-SAT AddCircuit（厳密最適解）",
        }

    return generate, solve


@register(97, "min_makespan")
def unrelated_parallel_machines():
    """prob_097: 非関連並列機械の makespan 最小化。"""

    def generate(rng: random.Random, base: dict) -> dict:
        jobs, machines = base["num_jobs"], base["num_machines"]
        return {
            "num_jobs": jobs,
            "num_machines": machines,
            "processing_time": int_matrix(rng, machines, jobs, 3, 12),
        }

    def solve(instance: dict) -> dict:
        cp_model, solver = cp_sat_solver()
        times = instance["processing_time"]
        jobs, machines = range(instance["num_jobs"]), range(instance["num_machines"])
        model = cp_model.CpModel()
        x = {(j, m): model.NewBoolVar(f"x{j}_{m}") for j in jobs for m in machines}
        horizon = sum(max(times[m][j] for m in machines) for j in jobs)
        makespan = model.NewIntVar(0, horizon, "makespan")
        for j in jobs:
            model.AddExactlyOne(x[j, m] for m in machines)
        for m in machines:
            model.Add(sum(times[m][j] * x[j, m] for j in jobs) <= makespan)
        model.Minimize(makespan)
        require_optimal(cp_model, solver.Solve(model))
        assignment = {str(j): next(m for m in machines if solver.Value(x[j, m])) for j in jobs}
        return {
            "min_makespan": int(solver.Value(makespan)),
            "assignment": assignment,
            "note": "CP-SAT（厳密最適解）",
        }

    return generate, solve
