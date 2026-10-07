"""ナップサック・パッキング 系の雛形（第 2 群: prob_056〜066）。 templates.py と同じ規約で (generate, solve) を登録する。

品物の id は配列位置と一致させる（チェッカーが位置でアクセスするため）。
問題文に書かれた容量・予算・母材長・目標値・件数は雛形の値を保ち、それ以外の数値だけを乱数で置き換える。
"""

from __future__ import annotations

import functools
import itertools
import random

from .base import cp_sat_solver, register, require_optimal, retry

NOTE = "CP-SAT（厳密最適解）"


def _items(rng: random.Random, n: int, **ranges: tuple[int, int]) -> list[dict]:
    """id を位置に合わせ、各フィールドを指定範囲の整数で埋めた品物リスト。"""
    return [
        {"id": i, **{key: rng.randint(lo, hi) for key, (lo, hi) in ranges.items()}}
        for i in range(n)
    ]


def _total(items: list[dict], key: str) -> int:
    return sum(item[key] for item in items)


def _maximize_selection(items: list[dict], gain: str, constrain) -> tuple[list[int], int]:
    """0/1 選択の CP-SAT 骨格。constrain(model, x) で制約を足し、選ばれた位置と目的値を返す。"""
    cp_model, solver = cp_sat_solver()
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x{i}") for i in range(len(items))]
    constrain(model, x)
    model.Maximize(sum(item[gain] * x[i] for i, item in enumerate(items)))
    require_optimal(cp_model, solver.Solve(model))
    return [i for i in range(len(items)) if solver.Value(x[i])], round(solver.ObjectiveValue())


def _enumerate_patterns(stock_length: int, lengths: list[int]) -> list[dict]:
    """母材に収まる空でない切断パターンを、ベクトルの辞書式順に列挙する。

    Why not 需要で個数を打ち切る: 同梱 instance では打ち切っても件数は変わらず、
    問題文にも書かれていない。列挙規則を単純に保つため母材長だけで打ち切る。
    """
    bounds = [stock_length // length for length in lengths]
    patterns = []
    for vec in itertools.product(*[range(b + 1) for b in bounds]):
        used = sum(v * length for v, length in zip(vec, lengths))
        if any(vec) and used <= stock_length:
            patterns.append({"pattern": list(vec), "waste": stock_length - used})
    return patterns


def _count_patterns(stock_length: int, lengths: tuple[int, ...]) -> int:
    """_enumerate_patterns の件数だけを DP で数える（候補長さの絞り込み用）。"""
    ways = [0] * (stock_length + 1)
    ways[0] = 1
    for length in lengths:
        for used in range(length, stock_length + 1):
            ways[used] += ways[used - length]
    return sum(ways) - 1


@functools.cache
def _length_tuples(stock_length: int, n: int, count: int, lo: int, hi: int) -> tuple:
    """パターン件数がちょうど count になる部材長の組（昇順）を全部集める。

    Why not 乱数で引いて件数を確かめる retry: 件数 44 に当たる確率は 1% 程度で、
    200 回の retry では取り逃す。件数の一致は生成の前提なので候補を先に絞る。
    """
    return tuple(
        combo
        for combo in itertools.combinations_with_replacement(range(lo, hi + 1), n)
        if _count_patterns(stock_length, combo) == count
    )


def _min_rolls(stock_length: int, lengths: list[int], demands: list[int]) -> int:
    """パターン列挙 + CP-SAT で母材本数を最小化する。"""
    cp_model, solver = cp_sat_solver()
    patterns = _enumerate_patterns(stock_length, lengths)
    model = cp_model.CpModel()
    upper = sum(demands)
    use = [model.NewIntVar(0, upper, f"u{p}") for p in range(len(patterns))]
    for j, demand in enumerate(demands):
        model.Add(sum(pat["pattern"][j] * use[p] for p, pat in enumerate(patterns)) >= demand)
    model.Minimize(sum(use))
    require_optimal(cp_model, solver.Solve(model))
    return round(solver.ObjectiveValue())


@register(56, "max_value")
def bounded_knapsack():
    """prob_056: 有界ナップサック。容量 35 と品物数は問題文にあるので雛形のまま。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n = len(base["items"])
        capacity = base["capacity"]

        def make() -> dict:
            items = _items(rng, n, weight=(3, 9), value=(3, 18), max_count=(1, 5))
            return {"items": items, "capacity": capacity}

        # 在庫を全部入れても容量に収まる instance は制約が効かないので捨てる。
        def ok(inst: dict) -> bool:
            return sum(i["weight"] * i["max_count"] for i in inst["items"]) > capacity

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        cp_model, solver = cp_sat_solver()
        items = instance["items"]
        model = cp_model.CpModel()
        counts = [model.NewIntVar(0, item["max_count"], f"c{i}") for i, item in enumerate(items)]
        model.Add(
            sum(item["weight"] * counts[i] for i, item in enumerate(items)) <= instance["capacity"]
        )
        model.Maximize(sum(item["value"] * counts[i] for i, item in enumerate(items)))
        require_optimal(cp_model, solver.Solve(model))
        item_counts = {
            str(i): solver.Value(counts[i]) for i in range(len(items)) if solver.Value(counts[i])
        }
        return {
            "max_value": round(solver.ObjectiveValue()),
            "item_counts": item_counts,
            "note": NOTE,
        }

    return generate, solve


@register(57, "max_value")
def multidimensional_knapsack():
    """prob_057: 重量・体積の 2 制約ナップサック。両制限 40 は問題文にあるので雛形のまま。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n = len(base["items"])
        caps = {k: base[k] for k in ("weight_capacity", "volume_capacity")}

        def make() -> dict:
            return {"items": _items(rng, n, weight=(3, 12), volume=(3, 12), value=(5, 25)), **caps}

        def ok(inst: dict) -> bool:
            return (
                _total(inst["items"], "weight") > caps["weight_capacity"]
                and _total(inst["items"], "volume") > caps["volume_capacity"]
            )

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        items = instance["items"]

        def constrain(model, x):
            for field, cap in (("weight", "weight_capacity"), ("volume", "volume_capacity")):
                model.Add(sum(item[field] * x[i] for i, item in enumerate(items)) <= instance[cap])

        chosen, value = _maximize_selection(items, "value", constrain)
        return {"max_value": value, "chosen_items": chosen, "note": NOTE}

    return generate, solve


@register(58, "max_value")
def multiple_knapsack():
    """prob_058: 複数ナップサック。容量列 [25, 25, 20] は問題文にあるので雛形のまま。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n = len(base["items"])
        capacities = list(base["capacities"])

        def make() -> dict:
            return {
                "items": _items(rng, n, weight=(4, 12), value=(5, 25)),
                "capacities": capacities,
            }

        return retry(rng, make, lambda inst: _total(inst["items"], "weight") > sum(capacities))

    def solve(instance: dict) -> dict:
        cp_model, solver = cp_sat_solver()
        items = instance["items"]
        capacities = instance["capacities"]
        model = cp_model.CpModel()
        x = {
            (i, k): model.NewBoolVar(f"x{i}_{k}")
            for i in range(len(items))
            for k in range(len(capacities))
        }
        for i in range(len(items)):
            model.AddAtMostOne(x[i, k] for k in range(len(capacities)))
        for k, cap in enumerate(capacities):
            model.Add(sum(item["weight"] * x[i, k] for i, item in enumerate(items)) <= cap)
        model.Maximize(sum(items[i]["value"] * var for (i, _), var in x.items()))
        require_optimal(cp_model, solver.Solve(model))
        assignment = {
            str(k): [i for i in range(len(items)) if solver.Value(x[i, k])]
            for k in range(len(capacities))
        }
        return {"max_value": round(solver.ObjectiveValue()), "assignment": assignment, "note": NOTE}

    return generate, solve


@register(60, "min_stock_rolls")
def cutting_stock_with_patterns():
    """prob_060: 1 次元カッティングストック。母材長 20 は問題文にあるので雛形のまま。

    instance には全切断パターンが同梱されているので、生成時も同じ規則で列挙し直す。
    形状検査でパターン件数（44）も一致させる必要があるため、件数が一致する部材長の組だけを使う。
    """

    def generate(rng: random.Random, base: dict) -> dict:
        stock = base["stock_length"]
        n = len(base["orders"])
        # 件数 44 に合う長さの組は 2〜18 の範囲でも 24 通りしかないので、
        # instance の多様性は長さの並びと需要数で出す。
        candidates = _length_tuples(stock, n, len(base["patterns"]), 2, stock - 2)

        def make() -> dict:
            lengths = list(rng.choice(candidates))
            rng.shuffle(lengths)
            orders = [{"length": length, "demand": rng.randint(2, 8)} for length in lengths]
            patterns = _enumerate_patterns(stock, lengths)
            return {"stock_length": stock, "orders": orders, "patterns": patterns}

        # 母材 2 本で足りる instance は切断計画と呼びにくいので捨てる。
        def ok(inst: dict) -> bool:
            return sum(o["length"] * o["demand"] for o in inst["orders"]) > 2 * stock

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        orders = instance["orders"]
        # Why not 列挙し直す: 同じ規則で列挙すれば同一になるが、参照解のパターン番号は
        # instance の patterns を指すので、instance 側を正として使う。
        cp_model, solver = cp_sat_solver()
        patterns = instance["patterns"]
        model = cp_model.CpModel()
        upper = sum(o["demand"] for o in orders)
        use = [model.NewIntVar(0, upper, f"u{p}") for p in range(len(patterns))]
        for j, order in enumerate(orders):
            model.Add(
                sum(pat["pattern"][j] * use[p] for p, pat in enumerate(patterns)) >= order["demand"]
            )
        model.Minimize(sum(use))
        require_optimal(cp_model, solver.Solve(model))
        usage = {str(p): solver.Value(use[p]) for p in range(len(patterns)) if solver.Value(use[p])}
        return {
            "min_stock_rolls": round(solver.ObjectiveValue()),
            "pattern_usage": usage,
            "note": "CP-SAT（列挙パターン、厳密最適解）",
        }

    return generate, solve


@register(61, "min_difference")
def subset_sum():
    """prob_061: 部分和。目標値 49 と個数は問題文にあるので雛形のまま。到達可能和の DP で解く。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n = len(base["numbers"])
        target = base["target"]

        def make() -> dict:
            return {"numbers": [rng.randint(3, 25) for _ in range(n)], "target": target}

        # 全部選んでも届かない、または 1 個で目標に当たる instance は自明なので捨てる。
        def ok(inst: dict) -> bool:
            return sum(inst["numbers"]) > target and target not in inst["numbers"]

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        numbers = instance["numbers"]
        target = instance["target"]
        # reach[i] = 位置 i 以降の数だけで作れる和の集合。後ろから積み上げる。
        reach: list[set[int]] = [set() for _ in range(len(numbers) + 1)]
        reach[-1] = {0}
        for i in range(len(numbers) - 1, -1, -1):
            reach[i] = reach[i + 1] | {s + numbers[i] for s in reach[i + 1]}
        # 同点は小さい和を採り、添字は前から貪欲に取って辞書式最小の組にする。
        best = min(reach[0], key=lambda s: (abs(s - target), s))
        chosen, remaining = [], best
        for i, value in enumerate(numbers):
            if remaining - value in reach[i + 1]:
                chosen.append(i)
                remaining -= value
        return {
            "min_difference": abs(best - target),
            "achieved_sum": best,
            "chosen_indices": chosen,
            "note": "到達可能和の DP（厳密最適解）",
        }

    return generate, solve


@register(63, "max_value")
def group_knapsack():
    """prob_063: グループ制約付きナップサック。容量 30、グループ分け（4×3）は雛形のまま。"""

    def generate(rng: random.Random, base: dict) -> dict:
        groups = [list(g) for g in base["groups"]]
        capacity = base["capacity"]
        group_of = {i: g for g, members in enumerate(groups) for i in members}

        def make() -> dict:
            items = [
                {
                    "id": i,
                    "group": group_of[i],
                    "weight": rng.randint(2, 10),
                    "value": rng.randint(5, 20),
                }
                for i in range(len(base["items"]))
            ]
            return {"items": items, "groups": groups, "capacity": capacity}

        # 各グループの最高価値品を全部入れても容量に収まるなら重量制約が効かないので捨てる。
        def ok(inst: dict) -> bool:
            heaviest_best = sum(
                max((inst["items"][i] for i in members), key=lambda it: it["value"])["weight"]
                for members in groups
            )
            return heaviest_best > capacity

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        items = instance["items"]

        def constrain(model, x):
            model.Add(
                sum(item["weight"] * x[i] for i, item in enumerate(items)) <= instance["capacity"]
            )
            for members in instance["groups"]:
                model.AddAtMostOne(x[i] for i in members)

        chosen, value = _maximize_selection(items, "value", constrain)
        return {"max_value": value, "chosen_items": chosen, "note": NOTE}

    return generate, solve


@register(64, "max_profit")
def project_selection_with_dependencies():
    """prob_064: 依存関係付きプロジェクト選択。予算 100、候補数、依存関係の本数は雛形のまま。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n = len(base["projects"])
        budget = base["budget"]
        num_deps = len(base["dependencies"])
        # 先行側を若い番号に限れば閉路はできない。
        pairs = [[a, b] for a in range(n) for b in range(a)]

        def make() -> dict:
            projects = _items(rng, n, cost=(10, 40), profit=(15, 55))
            deps = sorted(rng.sample(pairs, num_deps))
            return {"projects": projects, "budget": budget, "dependencies": deps}

        return retry(rng, make, lambda inst: _total(inst["projects"], "cost") > budget)

    def solve(instance: dict) -> dict:
        projects = instance["projects"]

        def constrain(model, x):
            model.Add(sum(p["cost"] * x[i] for i, p in enumerate(projects)) <= instance["budget"])
            for dependent, prerequisite in instance["dependencies"]:
                model.AddImplication(x[dependent], x[prerequisite])

        chosen, profit = _maximize_selection(projects, "profit", constrain)
        return {"max_profit": profit, "chosen_projects": chosen, "note": NOTE}

    return generate, solve


@register(65, "min_rolls")
def steel_cutting():
    """prob_065: 資材切出。鋼材長 15 と部材種類数は問題文にあるので雛形のまま。

    参照解は本数と端材だけなので、切断パターンは列挙して CP-SAT で本数を求め、
    端材は 鋼材長 × 本数 − 必要長の合計 とする（同梱値 6 = 15×6 − 84 と同じ定義）。
    """

    def generate(rng: random.Random, base: dict) -> dict:
        stock = base["stock_length"]
        n = len(base["pieces"])

        def make() -> dict:
            pieces = [{"length": rng.randint(3, 9), "demand": rng.randint(2, 6)} for _ in range(n)]
            return {"stock_length": stock, "pieces": pieces}

        def ok(inst: dict) -> bool:
            return sum(p["length"] * p["demand"] for p in inst["pieces"]) > 2 * stock

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        stock = instance["stock_length"]
        pieces = instance["pieces"]
        rolls = _min_rolls(stock, [p["length"] for p in pieces], [p["demand"] for p in pieces])
        needed = sum(p["length"] * p["demand"] for p in pieces)
        return {"min_rolls": rolls, "total_waste": rolls * stock - needed, "note": NOTE}

    return generate, solve


@register(66, "max_priority")
def container_loading():
    """prob_066: コンテナ積載。積載重量 120・容積 80・貨物数は問題文にあるので雛形のまま。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n = len(base["items"])
        caps = {k: base[k] for k in ("max_weight", "max_volume")}

        def make() -> dict:
            items = _items(rng, n, weight=(5, 25), volume=(3, 14), priority=(1, 10))
            return {"items": items, **caps}

        def ok(inst: dict) -> bool:
            return (
                _total(inst["items"], "weight") > caps["max_weight"]
                and _total(inst["items"], "volume") > caps["max_volume"]
            )

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        items = instance["items"]

        def constrain(model, x):
            for field, cap in (("weight", "max_weight"), ("volume", "max_volume")):
                model.Add(sum(item[field] * x[i] for i, item in enumerate(items)) <= instance[cap])

        loaded, priority = _maximize_selection(items, "priority", constrain)
        return {"max_priority": priority, "loaded_items": loaded, "note": NOTE}

    return generate, solve
