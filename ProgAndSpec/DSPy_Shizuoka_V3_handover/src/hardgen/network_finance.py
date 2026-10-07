"""大規模問題集の instance 生成器（network_finance 群）。register(kind) で登録する。

網設計（mcnd / mcnd_surv）は元 instance の候補アークの作り方を推定し、同じ作り方で引き直す。

- 全アークに逆向きがある（prob_308）: 各拠点を近い順に結んだ無向対を両方向のアークに展開する。
- 隣接番号のリングが両方向にある（prob_325）: リングに乱択アークを足す。リングがあるので
  どの OD にも 2 本の辺素経路があり、max_delay はその最良対の長い方から決める。
- それ以外（prob_318）: ランダムな巡回路で強連結を保証し、乱択アークを足す。

費用は元 instance で距離との相関が強ければ距離比例、そうでなければ元の範囲の一様乱数にする。
ポートフォリオ（portfolio / portfolio_cvar）は銘柄属性を元の範囲の一様乱数で引き、シナリオ収益は
beta × 市場因子 + 固有擾乱（正規乱数。市場因子の平均・標準偏差と擾乱の標準偏差は元 instance から推定）で作る。
"""

from __future__ import annotations

import copy
import math
import random
import re
import statistics
from collections import Counter, defaultdict, deque

from . import register

# 単一ノードのカット（起点の出容量・終点の入容量）に対する需要の上限。元 instance は最大 0.21。
_NODE_CUT_RATIO = 0.35
# max_delay = 最良の辺素経路対の長い方 × この範囲。元 instance の比は 1.24〜1.33（外れ値 1.6）。
_DELAY_FACTOR = (1.24, 1.34)
# 売却不可（portfolio_cvar）の現保有合計。元 instance は 1.0 で解が現保有に固定されてしまうので、
# 新規組入れの余地を残す。
_CVAR_CURRENT_TOTAL = (0.6, 0.8)
# 現保有ウェイトの元になる一様乱数の範囲。正規化後に元 instance と同程度（0.008〜0.08）に散らばる。
_CURRENT_RAW = (0.1, 1.0)


# ---------------------------------------------------------------------------
# 共通
# ---------------------------------------------------------------------------


def _name_prefix(name: str) -> str:
    return re.sub(r"\d+$", "", str(name))


def _decimals(values: list) -> int:
    """数値列の小数桁数の最大。丸め桁を元 instance に合わせる。"""
    out = 0
    for value in values:
        if isinstance(value, int):
            continue
        text = repr(float(value))
        if "e" in text or text.endswith(".0"):
            continue
        out = max(out, len(text.split(".")[1]))
    return out


def _round(value: float, decimals: int) -> int | float:
    return round(value) if decimals == 0 else round(value, decimals)


def _corr(xs: list[float], ys: list[float]) -> float:
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else 0.0


# ---------------------------------------------------------------------------
# 網設計（mcnd / mcnd_surv）
# ---------------------------------------------------------------------------


class _CostModel:
    """元 instance のアーク費用の作り方。距離と相関が強ければ 切片 + 距離 × 一様係数、無ければ一様乱数。"""

    def __init__(self, base_arcs: list[dict], key: str) -> None:
        dist = [float(a["distance"]) for a in base_arcs]
        vals = [float(a[key]) for a in base_arcs]
        self.decimals = _decimals([a[key] for a in base_arcs])
        self.floor = min(vals)
        self.proportional = _corr(dist, vals) > 0.5
        if self.proportional:
            # 最小二乗の切片を基本料金とみなし、距離 1 あたりの係数の範囲を長めのアークから推定する。
            # Why not 全アークで推定: 短いアークは切片の誤差で係数が大きく振れ、範囲が過大になる。
            slope = _corr(dist, vals) * statistics.pstdev(vals) / statistics.pstdev(dist)
            self.intercept = max(0.0, statistics.fmean(vals) - slope * statistics.fmean(dist))
            median = statistics.median(dist)
            rates = [(v - self.intercept) / d for d, v in zip(dist, vals) if d >= median]
            self.lo, self.hi = min(rates), max(rates)
        else:
            self.intercept = 0.0
            self.lo, self.hi = min(vals), max(vals)

    def draw(self, rng: random.Random, distance: float) -> int | float:
        if self.proportional:
            value = self.intercept + distance * rng.uniform(self.lo, self.hi)
        else:
            value = rng.uniform(self.lo, self.hi)
        return _round(max(value, self.floor), self.decimals)


def _topology(base_arcs: list[dict], n: int) -> str:
    pairs = {(int(a["from"]), int(a["to"])) for a in base_arcs}
    if all((b, a) in pairs for a, b in pairs):
        return "knn"
    ring = all((i, i % n + 1) in pairs and (i % n + 1, i) in pairs for i in range(1, n + 1))
    return "ring" if ring else "cycle"


def _nodes(rng: random.Random, base_nodes: list[dict]) -> list[dict]:
    prefix = _name_prefix(base_nodes[0]["name"])
    top = max(max(v["x"] for v in base_nodes), max(v["y"] for v in base_nodes))
    bound = math.ceil(top / 100) * 100
    return [
        {
            "id": i,
            "name": f"{prefix}{i}",
            "x": round(rng.uniform(0, bound), 1),
            "y": round(rng.uniform(0, bound), 1),
        }
        for i in range(1, len(base_nodes) + 1)
    ]


def _knn_pairs(rng: random.Random, pos: dict[int, tuple], target: int) -> list[tuple[int, int]]:
    """各拠点を近い順に結んだ無向対。target 本になるまで乱択した拠点の次に近い相手を足す。"""
    ids = sorted(pos)
    nearest = {
        u: sorted((v for v in ids if v != u), key=lambda v: (math.dist(pos[u], pos[v]), v))
        for u in ids
    }
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    rank = dict.fromkeys(ids, 0)

    def add(u: int, v: int) -> None:
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            pairs.append((u, v))

    # Why not k 固定: 相互に近い対の数は配置で変わるので、target を超えない限り 1 段ずつ全拠点に足す。
    while True:
        layer = {(min(u, nearest[u][rank[u]]), max(u, nearest[u][rank[u]])) for u in ids}
        if len(pairs) + len(layer - seen) > target:
            break
        for u in ids:
            add(u, nearest[u][rank[u]])
            rank[u] += 1
    order = ids[:]
    rng.shuffle(order)
    cursor = 0
    while len(pairs) < target:
        u = order[cursor % len(order)]
        cursor += 1
        add(u, nearest[u][rank[u]])
        rank[u] += 1
    return pairs


def _connected(pairs: list[tuple[int, int]], n: int) -> bool:
    adj = defaultdict(list)
    for u, v in pairs:
        adj[u].append(v)
        adj[v].append(u)
    seen = {1}
    stack = [1]
    while stack:
        for w in adj[stack.pop()]:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return len(seen) == n


def _directed_pairs(
    rng: random.Random, n: int, count: int, topology: str
) -> list[tuple[int, int]]:
    """リング（両方向）または乱択巡回路で強連結にし、count 本まで乱択アークを足す。並行アークは作らない。"""
    pairs: list[tuple[int, int]] = []
    if topology == "ring":
        for i in range(1, n + 1):
            pairs.extend([(i, i % n + 1), (i % n + 1, i)])
    else:
        perm = list(range(1, n + 1))
        rng.shuffle(perm)
        pairs.extend((perm[i], perm[(i + 1) % n]) for i in range(n))
    seen = set(pairs)
    while len(pairs) < count:
        u, v = rng.sample(range(1, n + 1), 2)
        if (u, v) not in seen:
            seen.add((u, v))
            pairs.append((u, v))
    return sorted(pairs)


def _arcs(
    rng: random.Random, base: dict, nodes: list[dict], pairs: list[tuple[int, int]]
) -> list[dict]:
    pos = {v["id"]: (v["x"], v["y"]) for v in nodes}
    base_arcs = base["arcs"]
    dist_decimals = _decimals([a["distance"] for a in base_arcs])
    # Why not 容量の種類から等確率: 元 instance の種類ごとの本数の偏りも引き継ぐ。
    capacities = [a["capacity"] for a in base_arcs]
    fixed, flow = _CostModel(base_arcs, "fixed_cost"), _CostModel(base_arcs, "flow_cost")
    arcs = []
    for arc_id, (u, v) in enumerate(pairs, start=1):
        distance = round(math.dist(pos[u], pos[v]), dist_decimals)
        arcs.append(
            {
                "id": arc_id,
                "from": u,
                "to": v,
                "distance": distance,
                "capacity": rng.choice(capacities),
                "fixed_cost": fixed.draw(rng, distance),
                "flow_cost": flow.draw(rng, distance),
            }
        )
    return arcs


def _disjoint_pair_lengths(arcs: list[dict], source: int, target: int) -> list[float] | None:
    """距離合計が最小の 2 本の辺素経路の長さ（昇順）。無ければ None。

    単位容量の残余グラフに最短路を 2 回流す（Suurballe と同じ最適性）。負の残余費用が出るので
    Bellman-Ford（SPFA）で最短路を取る。
    """
    graph: dict[int, list[list]] = defaultdict(list)
    for arc in arcs:
        forward = [arc["to"], float(arc["distance"]), 1, None]
        backward = [arc["from"], -float(arc["distance"]), 0, forward]
        forward[3] = backward
        graph[arc["from"]].append(forward)
        graph[arc["to"]].append(backward)
    for _ in range(2):
        dist = {source: 0.0}
        prev: dict[int, tuple[int, list]] = {}
        queue = deque([source])
        queued = {source}
        while queue:
            u = queue.popleft()
            queued.discard(u)
            for edge in graph[u]:
                v, cost, residual, _ = edge
                if residual > 0 and dist[u] + cost < dist.get(v, math.inf) - 1e-9:
                    dist[v] = dist[u] + cost
                    prev[v] = (u, edge)
                    if v not in queued:
                        queue.append(v)
                        queued.add(v)
        if target not in dist:
            return None
        v = target
        while v != source:
            u, edge = prev[v]
            edge[2] -= 1
            edge[3][2] += 1
            v = u
    # 流れた前向きアーク（残余 0）を起点から辿り直して 2 本の長さに分ける。
    used: dict[int, list[list]] = defaultdict(list)
    for u, edges in graph.items():
        for edge in edges:
            if edge[1] >= 0 and edge[2] == 0:
                used[u].append(edge)
    lengths = []
    for _ in range(2):
        v, length = source, 0.0
        while v != target:
            edge = used[v].pop()
            length += edge[1]
            v = edge[0]
        lengths.append(length)
    return sorted(lengths)


def _od_demands(rng: random.Random, base: dict, arcs: list[dict], n: int) -> list[dict]:
    base_ods = base["od_demands"]
    volumes = [int(o["volume"]) for o in base_ods]
    distinct = len({(o["origin"], o["destination"]) for o in base_ods}) == len(base_ods)
    survivable = "max_delay" in base_ods[0]
    out_cap: dict[int, float] = defaultdict(float)
    in_cap: dict[int, float] = defaultdict(float)
    for arc in arcs:
        out_cap[arc["from"]] += arc["capacity"]
        in_cap[arc["to"]] += arc["capacity"]
    out_load: dict[int, float] = defaultdict(float)
    in_load: dict[int, float] = defaultdict(float)
    used: set[tuple[int, int]] = set()
    ods: list[dict] = []
    while len(ods) < len(base_ods):
        origin, destination = rng.sample(range(1, n + 1), 2)
        # Why not 最小〜最大の一様乱数: prob_308 の需要は一様でなく、経験分布から引く方が合計が合う。
        volume = rng.choice(volumes)
        if distinct and (origin, destination) in used:
            continue
        # Why not 最大流で可行性検査: 起点の出容量と終点の入容量だけ見て需要を抑え、多品種としての
        # 可行性は教師コードの実行で確認する。候補アークの総容量は需要合計の 30 倍以上ある。
        if (
            out_load[origin] + volume > _NODE_CUT_RATIO * out_cap[origin]
            or in_load[destination] + volume > _NODE_CUT_RATIO * in_cap[destination]
        ):
            continue
        used.add((origin, destination))
        out_load[origin] += volume
        in_load[destination] += volume
        od = {"origin": origin, "destination": destination, "volume": volume}
        if survivable:
            lengths = _disjoint_pair_lengths(arcs, origin, destination)
            if lengths is None:
                raise ValueError(f"no two disjoint paths for {origin}->{destination}")
            od["max_delay"] = math.ceil(lengths[1] * rng.uniform(*_DELAY_FACTOR))
        ods.append(od)
    return ods


def _network_instance(rng: random.Random, base: dict) -> dict:
    n = len(base["nodes"])
    topology = _topology(base["arcs"], n)
    while True:
        nodes = _nodes(rng, base["nodes"])
        if topology == "knn":
            pos = {v["id"]: (v["x"], v["y"]) for v in nodes}
            undirected = _knn_pairs(rng, pos, len(base["arcs"]) // 2)
            # Why not 連結化のアーク追加: 近傍グラフが非連結になるのは稀なので配置ごと引き直す。
            if not _connected(undirected, n):
                continue
            pairs = [p for u, v in undirected for p in ((u, v), (v, u))]
        else:
            pairs = _directed_pairs(rng, n, len(base["arcs"]), topology)
        break
    arcs = _arcs(rng, base, nodes, pairs)
    generated = {"nodes": nodes, "arcs": arcs, "od_demands": _od_demands(rng, base, arcs, n)}
    return {key: generated.get(key, copy.deepcopy(value)) for key, value in base.items()}


@register("mcnd")
def generate_mcnd(rng: random.Random, base: dict) -> dict:
    return _network_instance(rng, base)


@register("mcnd_surv")
def generate_mcnd_surv(rng: random.Random, base: dict) -> dict:
    return _network_instance(rng, base)


# ---------------------------------------------------------------------------
# ポートフォリオ（portfolio / portfolio_cvar）
# ---------------------------------------------------------------------------


def _factor_model(base: dict) -> tuple[float, float, float]:
    """scenario_returns を beta × 市場因子 + 固有擾乱 とみなし、市場因子の平均・標準偏差と擾乱の標準偏差を推定する。"""
    betas = [float(a["beta"]) for a in base["assets"]]
    beta_norm = sum(b * b for b in betas)
    factors = []
    residual_sq = 0.0
    count = 0
    for row in base["scenario_returns"]:
        factor = sum(b * r for b, r in zip(betas, row)) / beta_norm
        factors.append(factor)
        residual_sq += sum((r - b * factor) ** 2 for b, r in zip(betas, row))
        count += len(row)
    return statistics.fmean(factors), statistics.pstdev(factors), math.sqrt(residual_sq / count)


def _assets(rng: random.Random, base: dict) -> list[dict]:
    base_assets = base["assets"]
    prefix = _name_prefix(base_assets[0]["name"])
    returns = [a["expected_return"] for a in base_assets]
    betas = [a["beta"] for a in base_assets]
    ret_decimals, beta_decimals = _decimals(returns), _decimals(betas)
    assets = []
    for i in range(1, len(base_assets) + 1):
        expected = round(rng.uniform(min(returns), max(returns)), ret_decimals)
        assets.append(
            {
                "id": i,
                "name": f"{prefix}{i}",
                "sector": rng.choice(base["sectors"]),
                "expected_return": expected,
                "beta": round(rng.uniform(min(betas), max(betas)), beta_decimals),
                "mean_return": expected,
            }
        )
    return assets


def _scenario_returns(rng: random.Random, base: dict, assets: list[dict]) -> list[list[float]]:
    factor_mean, factor_sd, noise_sd = _factor_model(base)
    decimals = _decimals(base["scenario_returns"][0])
    betas = [a["beta"] for a in assets]
    rows = []
    for _ in range(len(base["scenario_returns"])):
        factor = rng.gauss(factor_mean, factor_sd)
        rows.append([round(b * factor + rng.gauss(0.0, noise_sd), decimals) for b in betas])
    return rows


def _weights(
    rng: random.Random, count: int, total: float, decimals: int, cap: float
) -> list[float]:
    """合計が total、各要素が cap 以下の count 個の正のウェイト。丸めの端数は最大の要素で吸収する。"""
    while True:
        raw = [rng.uniform(*_CURRENT_RAW) for _ in range(count)]
        scale = total / sum(raw)
        weights = [round(r * scale, decimals) for r in raw]
        top = max(range(count), key=weights.__getitem__)
        weights[top] = round(weights[top] + total - sum(weights), decimals)
        # Why not cap で切り詰め: 切った分を他へ配り直すより、稀な超過は引き直す方が単純。
        if weights[top] <= cap:
            return weights


def _current_portfolio(
    rng: random.Random, base: dict, assets: list[dict], *, cvar: bool
) -> dict[str, float]:
    base_current = base["current_portfolio"]
    count = len(base_current)
    decimals = _decimals(list(base_current.values()))
    sorted_keys = list(base_current) == sorted(base_current, key=int)
    limits = base["limits"]
    max_return = max(a["expected_return"] for a in assets)
    while True:
        total = rng.uniform(*_CVAR_CURRENT_TOTAL) if cvar else sum(base_current.values())
        ids = rng.sample(range(1, len(assets) + 1), count)
        values = _weights(rng, count, total, decimals, limits["max_weight"])
        weights = dict(zip(sorted(ids) if sorted_keys else ids, values))
        if not cvar:
            break
        # 売却不可なので、現保有だけで業種上限・業種数・目標収益を破っていると可行解が無い。
        sector_weight: dict[int, float] = defaultdict(float)
        for asset_id, w in weights.items():
            sector_weight[assets[asset_id - 1]["sector"]] += w
        expected = sum(w * assets[i - 1]["expected_return"] for i, w in weights.items())
        if (
            max(sector_weight.values()) <= 0.85 * limits["max_sector_weight"]
            and len(sector_weight) >= max(limits["min_sectors"], len(base["sector_cardinality"]))
            and expected + (1.0 - total) * max_return >= base["target_return"] + 0.01
        ):
            break
    return {str(asset_id): w for asset_id, w in weights.items()}


def _sector_cardinality(
    rng: random.Random, base: dict, assets: list[dict], current: dict[str, float]
) -> dict[str, list[int]]:
    """現保有のある業種から元と同じ数を選び、現保有数を含む [下限, 上限] を付ける。"""
    held = Counter(assets[int(k) - 1]["sector"] for k in current)
    chosen = rng.sample(sorted(held), len(base["sector_cardinality"]))
    return {
        str(sector): [rng.randint(1, min(2, held[sector])), held[sector] + rng.randint(0, 2)]
        for sector in chosen
    }


def _portfolio_instance(rng: random.Random, base: dict, *, cvar: bool) -> dict:
    assets = _assets(rng, base)
    generated = {
        "assets": assets,
        "scenario_returns": _scenario_returns(rng, base, assets),
        "current_portfolio": _current_portfolio(rng, base, assets, cvar=cvar),
    }
    if cvar:
        generated["sector_cardinality"] = _sector_cardinality(
            rng, base, assets, generated["current_portfolio"]
        )
    return {key: generated.get(key, copy.deepcopy(value)) for key, value in base.items()}


@register("portfolio")
def generate_portfolio(rng: random.Random, base: dict) -> dict:
    return _portfolio_instance(rng, base, cvar=False)


@register("portfolio_cvar")
def generate_portfolio_cvar(rng: random.Random, base: dict) -> dict:
    return _portfolio_instance(rng, base, cvar=True)
