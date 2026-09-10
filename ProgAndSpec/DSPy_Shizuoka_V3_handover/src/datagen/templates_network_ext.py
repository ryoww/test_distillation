"""network 系の雛形（第 2 群: prob_032〜042）。 templates.py と同じ規約で (generate, solve) を登録する。

辺は同梱 instance と同じ dict 表現（有向は from/to、無向は u/v）で持ち、参照解は同梱の
参照解と同じキー集合で返す。整数データの流問題は networkx の厳密アルゴリズムで、
LP でしか表せない多品種流・容量拡張は scipy HiGHS で解く。
"""

from __future__ import annotations

import random
from itertools import pairwise

from .base import SolveError, register, retry

# ============================================================
# 共通ヘルパー
# ============================================================


def _dag_edge_pairs(rng: random.Random, n: int, m: int) -> list[tuple[int, int]]:
    """n ノード上に from < to の有向辺を m 本重複なく選び、(from, to) 順に並べる。

    Why not 閉路を許す: 同梱の流問題 instance はいずれも番号昇順の DAG で、逆向き辺を
    混ぜると費用 0 の閉路や逆平行辺の扱いで解釈が割れる。
    """
    candidates = [(u, v) for u in range(n) for v in range(u + 1, n)]
    return sorted(rng.sample(candidates, m))


def _flow_digraph(instance: dict, *, capacity: bool, weight_key: str | None):
    """from/to 表現の辺から networkx の DiGraph を作る。ノードは 0..n-1 を全て登録する。"""
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_nodes_from(range(instance["num_nodes"]))
    for edge in instance["edges"]:
        attrs = {}
        if capacity:
            attrs["capacity"] = edge["capacity"]
        if weight_key is not None:
            attrs["weight"] = edge[weight_key]
        graph.add_edge(edge["from"], edge["to"], **attrs)
    return graph


def _max_flow_value(instance: dict) -> int:
    import networkx as nx

    graph = _flow_digraph(instance, capacity=True, weight_key=None)
    return int(nx.maximum_flow_value(graph, instance["source"], instance["sink"]))


def _linprog(c, *, a_ub, b_ub, a_eq, b_eq):
    """HiGHS で LP を解き、最適な解ベクトルを返す。最適以外は SolveError。"""
    from scipy.optimize import linprog

    result = linprog(
        c,
        A_ub=a_ub or None,
        b_ub=b_ub or None,
        A_eq=a_eq or None,
        b_eq=b_eq or None,
        bounds=(0, None),
        method="highs",
    )
    if result.status != 0:
        raise SolveError(f"HiGHS did not reach optimality (status={result.status})")
    return result.x


# ============================================================
# 最小費用流系
# ============================================================


@register(32, "min_cost")
def min_cost_flow():
    """prob_032: 単一供給・単一需要の最小費用流。ノード数・輸送量は問題文にあるので雛形のまま。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n, m = base["num_nodes"], len(base["edges"])
        source, sink, amount = base["supply_node"], base["demand_node"], base["amount"]

        def make() -> dict:
            edges = [
                {"from": u, "to": v, "capacity": rng.randint(5, 14), "cost": rng.randint(1, 10)}
                for u, v in _dag_edge_pairs(rng, n, m)
            ]
            return {
                "num_nodes": n,
                "supply_node": source,
                "demand_node": sink,
                "amount": amount,
                "edges": edges,
            }

        def ok(inst: dict) -> bool:
            # 容量が効かず「最短路に全量流す」だけで最適になる instance は問題として薄い。
            flow_inst = {**inst, "source": source, "sink": sink}
            if _max_flow_value(flow_inst) < amount:
                return False
            return solve(inst)["min_cost"] > amount * _cheapest_path_cost(inst)

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        import networkx as nx

        graph = _flow_digraph(instance, capacity=True, weight_key="cost")
        graph.nodes[instance["supply_node"]]["demand"] = -instance["amount"]
        graph.nodes[instance["demand_node"]]["demand"] = instance["amount"]
        try:
            cost = nx.min_cost_flow_cost(graph)
        except nx.NetworkXUnfeasible as exc:
            raise SolveError("min cost flow is infeasible") from exc
        return {
            "min_cost": int(cost),
            "feasible": True,
            "note": "networkx最小費用流（厳密最適解）",
        }

    return generate, solve


def _cheapest_path_cost(instance: dict) -> int:
    import networkx as nx

    graph = _flow_digraph(instance, capacity=False, weight_key="cost")
    return int(
        nx.shortest_path_length(
            graph, instance["supply_node"], instance["demand_node"], weight="weight"
        )
    )


@register(34, "min_cost")
def transshipment():
    """prob_034: 2 供給地・2 中継・3 需要地の積替問題。拠点名と辺の組は雛形のまま、費用と量を乱数にする。"""

    def generate(rng: random.Random, base: dict) -> dict:
        supply_names = list(base["supply"])
        demand_names = list(base["demand"])
        total = 5 * rng.randint(9, 13)
        # 供給合計と需要合計を一致させる（要件が「出荷量は供給量に等しい」なので余剰は許さない）。
        supply = _split_in_fives(rng, total, len(supply_names))
        demand = _split_in_fives(rng, total, len(demand_names))
        edges = [
            {"from": e["from"], "to": e["to"], "cost": rng.randint(1, 8)} for e in base["edges"]
        ]
        return {
            "supply": dict(zip(supply_names, supply)),
            "demand": dict(zip(demand_names, demand)),
            "transshipment": list(base["transshipment"]),
            "edges": edges,
        }

    def solve(instance: dict) -> dict:
        import networkx as nx

        graph = nx.DiGraph()
        for node, amount in instance["supply"].items():
            graph.add_node(node, demand=-amount)
        for node, amount in instance["demand"].items():
            graph.add_node(node, demand=amount)
        graph.add_nodes_from(instance["transshipment"])
        for edge in instance["edges"]:
            graph.add_edge(edge["from"], edge["to"], weight=edge["cost"])
        try:
            cost = nx.min_cost_flow_cost(graph)
        except nx.NetworkXUnfeasible as exc:
            raise SolveError("transshipment is infeasible") from exc
        return {"min_cost": int(cost), "note": "networkx最小費用流（厳密最適解）"}

    return generate, solve


def _split_in_fives(rng: random.Random, total: int, parts: int) -> list[int]:
    """total（5 の倍数）を parts 個の 5 以上の 5 の倍数へ分ける。同梱 instance の刻みに合わせる。"""
    units = total // 5
    cuts = sorted(rng.sample(range(1, units), parts - 1))
    bounds = [0, *cuts, units]
    return [5 * (b - a) for a, b in pairwise(bounds)]


@register(40, "min_cost")
def min_cost_flow_with_lower_bounds():
    """prob_040: 下限付き最小費用流。下限を先に流して残りを通常の最小費用流に帰着する。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n, m = base["num_nodes"], len(base["edges"])
        source, sink, amount = base["supply_node"], base["demand_node"], base["amount"]

        def make() -> dict:
            edges = []
            for u, v in _dag_edge_pairs(rng, n, m):
                lower = rng.randint(0, 3)
                edges.append(
                    {
                        "from": u,
                        "to": v,
                        "lower": lower,
                        "capacity": lower + rng.randint(4, 12),
                        "cost": rng.randint(1, 8),
                    }
                )
            return {
                "num_nodes": n,
                "supply_node": source,
                "demand_node": sink,
                "amount": amount,
                "edges": edges,
            }

        def ok(inst: dict) -> bool:
            try:
                with_lower = solve(inst)["min_cost"]
            except SolveError:
                return False
            # 下限を外しても同じ費用なら下限制約が効いておらず、prob_032 と同じ問題になる。
            relaxed = {**inst, "edges": [{**e, "lower": 0} for e in inst["edges"]]}
            return with_lower > solve(relaxed)["min_cost"]

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        import networkx as nx

        graph = nx.DiGraph()
        graph.add_nodes_from(range(instance["num_nodes"]), demand=0)
        graph.nodes[instance["supply_node"]]["demand"] -= instance["amount"]
        graph.nodes[instance["demand_node"]]["demand"] += instance["amount"]
        fixed_cost = 0
        for edge in instance["edges"]:
            lower = edge["lower"]
            # 下限ぶんを先に流したとみなし、両端の需要を調整して残余容量の問題にする。
            fixed_cost += lower * edge["cost"]
            graph.nodes[edge["from"]]["demand"] += lower
            graph.nodes[edge["to"]]["demand"] -= lower
            graph.add_edge(
                edge["from"], edge["to"], capacity=edge["capacity"] - lower, weight=edge["cost"]
            )
        try:
            cost = fixed_cost + nx.min_cost_flow_cost(graph)
        except nx.NetworkXUnfeasible as exc:
            raise SolveError("lower bounds make the flow infeasible") from exc
        return {
            "min_cost": int(cost),
            "feasible": True,
            "note": "networkx最小費用流（下限を需要へ振り替え、厳密最適解）",
        }

    return generate, solve


# ============================================================
# LP でしか表せない流問題
# ============================================================


@register(37, "min_total_flow")
def multicommodity_flow():
    """prob_037: 容量共有の多品種流。品目の始点・終点・需要は問題文にあるので雛形のまま。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n, m = base["num_nodes"], len(base["edges"])
        commodities = [dict(c) for c in base["commodities"]]

        def make() -> dict:
            edges = [
                {"from": u, "to": v, "capacity": rng.randint(3, 12)}
                for u, v in _dag_edge_pairs(rng, n, m)
            ]
            return {"num_nodes": n, "edges": edges, "commodities": commodities}

        def ok(inst: dict) -> bool:
            try:
                total = solve(inst)["min_total_flow"]
            except SolveError:
                return False
            # 各品目が最少ホップ路だけで流せるなら容量共有が効いておらず、単なる最短路になる。
            return total > _hop_lower_bound(inst) + 1e-6

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        edges, n = instance["edges"], instance["num_nodes"]
        commodities = instance["commodities"]
        m = len(edges)
        nv = m * len(commodities)
        a_eq, b_eq = [], []
        for k, item in enumerate(commodities):
            for node in range(n):
                row = [0.0] * nv
                for j, edge in enumerate(edges):
                    if edge["from"] == node:
                        row[k * m + j] += 1.0
                    if edge["to"] == node:
                        row[k * m + j] -= 1.0
                a_eq.append(row)
                if node == item["source"]:
                    b_eq.append(float(item["demand"]))
                elif node == item["sink"]:
                    b_eq.append(-float(item["demand"]))
                else:
                    b_eq.append(0.0)
        a_ub, b_ub = [], []
        for j, edge in enumerate(edges):
            row = [0.0] * nv
            for k in range(len(commodities)):
                row[k * m + j] = 1.0
            a_ub.append(row)
            b_ub.append(float(edge["capacity"]))
        x = _linprog([1.0] * nv, a_ub=a_ub, b_ub=b_ub, a_eq=a_eq, b_eq=b_eq)
        # 目的値は丸めずに解ベクトルから計算する（保存する値と整合させる）。
        return {
            "min_total_flow": float(sum(x)),
            "feasible": True,
            "note": "LP（scipy HiGHS、厳密最適解）",
        }

    return generate, solve


def _hop_lower_bound(instance: dict) -> float:
    import networkx as nx

    graph = _flow_digraph(instance, capacity=False, weight_key=None)
    total = 0.0
    for item in instance["commodities"]:
        total += item["demand"] * nx.shortest_path_length(graph, item["source"], item["sink"])
    return total


@register(41, "max_flow", minimize=False)
def max_flow_with_capacity_expansion():
    """prob_041: 予算内で容量を拡張してから流す最大流。予算は問題文にあるので雛形のまま。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n, m = base["num_nodes"], len(base["edges"])
        source, sink, budget = base["source"], base["sink"], base["budget"]

        def make() -> dict:
            edges = [
                {
                    "from": u,
                    "to": v,
                    "capacity": rng.randint(3, 12),
                    "expand_unit_cost": rng.randint(2, 6),
                }
                for u, v in _dag_edge_pairs(rng, n, m)
            ]
            return {
                "num_nodes": n,
                "source": source,
                "sink": sink,
                "budget": budget,
                "edges": edges,
            }

        def ok(inst: dict) -> bool:
            # 拡張しても流量が増えない instance は予算制約が無意味なので捨てる。
            plain = _max_flow_value(inst)
            return plain > 0 and solve(inst)["max_flow"] > plain + 1e-6

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        edges, n = instance["edges"], instance["num_nodes"]
        source, sink = instance["source"], instance["sink"]
        m = len(edges)
        # 変数は [flow_0..flow_{m-1}, expand_0..expand_{m-1}]。
        nv = 2 * m
        a_eq, b_eq = [], []
        for node in range(n):
            if node in (source, sink):
                continue
            row = [0.0] * nv
            for j, edge in enumerate(edges):
                if edge["from"] == node:
                    row[j] += 1.0
                if edge["to"] == node:
                    row[j] -= 1.0
            a_eq.append(row)
            b_eq.append(0.0)
        a_ub, b_ub = [], []
        for j, edge in enumerate(edges):
            row = [0.0] * nv
            row[j] = 1.0
            row[m + j] = -1.0
            a_ub.append(row)
            b_ub.append(float(edge["capacity"]))
        budget_row = [0.0] * nv
        for j, edge in enumerate(edges):
            budget_row[m + j] = float(edge["expand_unit_cost"])
        a_ub.append(budget_row)
        b_ub.append(float(instance["budget"]))
        objective = [0.0] * nv
        for j, edge in enumerate(edges):
            if edge["from"] == source:
                objective[j] -= 1.0
            if edge["to"] == source:
                objective[j] += 1.0
        x = _linprog(objective, a_ub=a_ub, b_ub=b_ub, a_eq=a_eq, b_eq=b_eq)
        net_out = -sum(c * v for c, v in zip(objective, x))
        return {
            "max_flow": float(net_out),
            "feasible": True,
            "note": "LP（scipy HiGHS、厳密最適解）",
        }

    return generate, solve


# ============================================================
# グラフ最適化・組合せ
# ============================================================


@register(35, "shortest_distance")
def shortest_path():
    """prob_035: 有向グラフの 2 点間最短路。同点は辞書順最小の経路を返す。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n, m = base["num_nodes"], len(base["edges"])
        start, goal = base["start"], base["goal"]

        def make() -> dict:
            # Why not 有向辺を自由に選ぶ: チェッカーは辺を無向として引くので、逆向きの辺が
            # 別の長さで並ぶと正しい経路長でも不一致になる。無向対を選んで向きだけ乱数にする。
            pairs = rng.sample([(u, v) for u in range(n) for v in range(u + 1, n)], m)
            oriented = sorted((u, v) if rng.random() < 0.5 else (v, u) for u, v in pairs)
            edges = [{"from": u, "to": v, "length": rng.randint(4, 40)} for u, v in oriented]
            return {"num_nodes": n, "start": start, "goal": goal, "edges": edges}

        def ok(inst: dict) -> bool:
            try:
                path = solve(inst)["path"]
            except SolveError:
                return False
            # 1 辺で着く最短路は問題として薄い。
            return len(path) >= 3

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        import networkx as nx

        graph = _flow_digraph(instance, capacity=False, weight_key="length")
        start, goal = instance["start"], instance["goal"]
        forward = nx.single_source_dijkstra_path_length(graph, start)
        if goal not in forward:
            raise SolveError("goal is unreachable from start")
        backward = nx.single_source_dijkstra_path_length(graph.reverse(copy=False), goal)
        best = forward[goal]
        # Why not nx の経路をそのまま使う: 同点の最短路が辺の挿入順で変わる。前後の距離が
        # 合計と一致する隣接ノードのうち番号最小を選ぶと、経路が一意に決まる。
        path = [start]
        while path[-1] != goal:
            node = path[-1]
            nxt = min(
                v
                for v in graph.successors(node)
                if v in backward and forward[node] + graph[node][v]["weight"] + backward[v] == best
            )
            path.append(nxt)
        return {
            "shortest_distance": int(best),
            "path": path,
            "note": "Dijkstra（厳密最適解）",
        }

    return generate, solve


@register(38, "min_cut_value")
def minimum_cut():
    """prob_038: 有向ネットワークの最小 s-t カット。S 側は残余グラフで源から到達できる集合。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n, m = base["num_nodes"], len(base["edges"])
        source, sink = base["source"], base["sink"]

        def make() -> dict:
            edges = [
                {"from": u, "to": v, "capacity": rng.randint(3, 14)}
                for u, v in _dag_edge_pairs(rng, n, m)
            ]
            return {"num_nodes": n, "source": source, "sink": sink, "edges": edges}

        def ok(inst: dict) -> bool:
            # 源の出容量か集の入容量がそのまま答えになる instance は問題として薄い。
            out_cap = sum(e["capacity"] for e in inst["edges"] if e["from"] == source)
            in_cap = sum(e["capacity"] for e in inst["edges"] if e["to"] == sink)
            value = solve(inst)["min_cut_value"]
            return 0 < value < min(out_cap, in_cap)

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        import networkx as nx

        graph = _flow_digraph(instance, capacity=True, weight_key=None)
        value, (s_side, t_side) = nx.minimum_cut(graph, instance["source"], instance["sink"])
        return {
            "min_cut_value": int(value),
            "S_side": sorted(s_side),
            "T_side": sorted(t_side),
            "note": "最大流最小カット定理（networkx、厳密最適解）",
        }

    return generate, solve


@register(42, "min_total_cost")
def minimum_spanning_tree():
    """prob_042: 無向グラフの最小全域木。候補辺の本数は雛形のまま、連結な候補だけを採用する。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n, m = base["num_nodes"], len(base["edges"])
        candidates = [(u, v) for u in range(n) for v in range(u + 1, n)]

        def make() -> dict:
            chosen = sorted(rng.sample(candidates, m))
            edges = [{"u": u, "v": v, "cost": rng.randint(1, 20)} for u, v in chosen]
            return {"num_nodes": n, "edges": edges}

        def ok(inst: dict) -> bool:
            import networkx as nx

            return nx.is_connected(_undirected_graph(inst))

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        import networkx as nx

        graph = _undirected_graph(instance)
        if not nx.is_connected(graph):
            raise SolveError("graph is not connected")
        # Why not Kruskal の辺集合を返す: 参照解は総費用と本数だけなので、同点の木が
        # 複数あっても目的値は一意に決まる。
        tree = nx.minimum_spanning_tree(graph, weight="weight", algorithm="kruskal")
        return {
            "min_total_cost": int(tree.size(weight="weight")),
            "num_edges_used": tree.number_of_edges(),
            "note": "Kruskal（networkx、厳密最適解）",
        }

    return generate, solve


def _undirected_graph(instance: dict):
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(instance["num_nodes"]))
    for edge in instance["edges"]:
        graph.add_edge(edge["u"], edge["v"], weight=edge["cost"])
    return graph
