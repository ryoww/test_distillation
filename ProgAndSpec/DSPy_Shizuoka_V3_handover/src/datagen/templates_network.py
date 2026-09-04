"""network 系の雛形。templates.py と同じ規約で (generate, solve) を登録する。"""

from __future__ import annotations

import math
import random

from .base import cp_sat_solver, int_list, register, require_optimal, retry

# ============================================================
# ネットワークフロー
# ============================================================


@register(36, "total_shortest_distance")
def shortest_path_tree():
    """prob_036: 無向グラフの単一始点最短路木。全ノードが到達可能な辺集合だけを採用する。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n = base["num_nodes"]
        source = base["source"]
        m = len(base["edges"])
        candidates = [(u, v) for u in range(n) for v in range(u + 1, n)]

        def make() -> dict:
            chosen = sorted(rng.sample(candidates, m))
            edges = [{"u": u, "v": v, "length": rng.randint(2, 15)} for u, v in chosen]
            return {"num_nodes": n, "source": source, "edges": edges}

        return retry(rng, make, lambda inst: len(_undirected_distances(inst)) == n)

    def solve(instance: dict) -> dict:
        dist = _undirected_distances(instance)
        if len(dist) != instance["num_nodes"]:
            raise ValueError("graph is not connected from the source")
        # 雛形と同じく、始点から近い順にノードを並べる。
        ordered = sorted(dist.items(), key=lambda item: (item[1], item[0]))
        return {
            "total_shortest_distance": int(sum(dist.values())),
            "distances": {str(node): int(d) for node, d in ordered},
            "note": "networkx Dijkstra（無向グラフ、厳密最適解）",
        }

    return generate, solve


def _undirected_distances(instance: dict) -> dict[int, int]:
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(instance["num_nodes"]))
    for edge in instance["edges"]:
        graph.add_edge(edge["u"], edge["v"], weight=edge["length"])
    return nx.single_source_dijkstra_path_length(graph, instance["source"])


@register(39, "max_matching")
def bipartite_matching():
    """prob_039: 二部グラフ最大マッチング。可能ペアの本数は雛形と同じに保つ。"""

    def generate(rng: random.Random, base: dict) -> dict:
        nw, nt = base["num_workers"], base["num_tasks"]
        m = len(base["compatible_pairs"])
        candidates = [(w, t) for w in range(nw) for t in range(nt)]
        chosen = sorted(rng.sample(candidates, m))
        return {
            "num_workers": nw,
            "num_tasks": nt,
            "compatible_pairs": [{"worker": w, "task": t} for w, t in chosen],
        }

    def solve(instance: dict) -> dict:
        import networkx as nx

        graph = nx.Graph()
        workers = [("w", w) for w in range(instance["num_workers"])]
        graph.add_nodes_from(workers)
        graph.add_nodes_from(("t", t) for t in range(instance["num_tasks"]))
        for pair in instance["compatible_pairs"]:
            graph.add_edge(("w", pair["worker"]), ("t", pair["task"]))
        matching = nx.bipartite.hopcroft_karp_matching(graph, top_nodes=workers)
        # hopcroft_karp_matching は両方向を含む辞書を返すので、ペア数はその半分。
        return {
            "max_matching": len(matching) // 2,
            "note": "networkx Hopcroft-Karp 二部マッチング（厳密最適解）",
        }

    return generate, solve


# ============================================================
# 施設配置・被覆
# ============================================================


@register(47, "min_weighted_distance")
def p_median():
    """prob_047: p-メディアン。距離は座標のユークリッド距離を四捨五入した値。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n = len(base["nodes"])
        p = base["p"]

        def make() -> dict:
            nodes = [
                {"id": i, "x": rng.randint(-30, 30), "y": rng.randint(-30, 30)} for i in range(n)
            ]
            distance = [
                [round(math.hypot(a["x"] - b["x"], a["y"] - b["y"])) for b in nodes] for a in nodes
            ]
            return {"nodes": nodes, "demand": int_list(rng, n, 1, 9), "p": p, "distance": distance}

        # 座標が重なると距離 0 の地点対ができ、施設選択が事実上 p-1 箇所になる。
        def ok(inst: dict) -> bool:
            points = {(node["x"], node["y"]) for node in inst["nodes"]}
            return len(points) == n

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        cp_model, solver = cp_sat_solver()
        n = len(instance["nodes"])
        demand, distance = instance["demand"], instance["distance"]
        model = cp_model.CpModel()
        open_ = [model.NewBoolVar(f"y{i}") for i in range(n)]
        assign = [[model.NewBoolVar(f"x{i}_{j}") for j in range(n)] for i in range(n)]
        model.Add(sum(open_) == instance["p"])
        for j in range(n):
            model.AddExactlyOne(assign[i][j] for i in range(n))
            for i in range(n):
                model.AddImplication(assign[i][j], open_[i])
        model.Minimize(
            sum(demand[j] * distance[i][j] * assign[i][j] for i in range(n) for j in range(n))
        )
        require_optimal(cp_model, solver.Solve(model))
        return {
            "min_weighted_distance": round(solver.ObjectiveValue()),
            "selected_facilities": [i for i in range(n) if solver.Value(open_[i])],
            "note": "CP-SAT（厳密最適解）",
        }

    return generate, solve
