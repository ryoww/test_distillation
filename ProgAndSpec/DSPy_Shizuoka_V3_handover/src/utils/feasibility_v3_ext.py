"""feasibility.py に登録されていなかった core_type 向けの制約チェッカー。

feasibility.check_feasibility_detailed は未登録の core_type に対して
``verified: False`` を返し、metrics_v3 はそれを status="unverified" として
減点する。本モジュールは残り20 core_type / 70問ぶんのチェッカーを提供する。

設計方針:

- 同じ core_type でも instance の形状が複数あるため、各チェッカーは
  instance のキーを見てサブ形状へ振り分ける。
- 形状を判定できなかった場合は ``verified: False`` を返し、従来どおり
  unverified のままにする。検証していないものへ満点を与えない。
- 目的値だけを返す問題（解の変数が返らない LP など）は、instance から
  計算できる妥当な下界・上界との整合だけを検証する。

構造から目的値が一意に定まる問題では、申告値と再計算値の一致も検証する。
最小化問題の過少申告は不当に高いスコアへ直結するため、ここを見ない
チェッカーはラバースタンプになる。

検証しきれないと分かっている3点:

- 最大流の過少申告は制約違反ではないため検出しない（過大申告はカット上界で弾く）。
- 単一割当ハブ配置は、参照解の費用モデルを再現できなかったため構造のみ検証する。
- 発注ロット最適化は、解に発注回数が含まれず供給量を復元できない。

EXTRA_CHECKERS を feasibility.py 側が読み込んで登録する。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from itertools import pairwise
from typing import Any

# 解の値が参照値と一致しているかではなく、instance から導ける制約だけを見る。
# 数値比較には解が浮動小数で返ることを見込んだ相対許容差を使う。
_REL_TOL = 1e-6
_ABS_TOL = 1e-6
_SOFT_REL_TOL = 2e-3
_SOFT_ABS_TOL = 2e-2
# 座標が整数へ丸められた状態で被覆判定された参照解があるため、半径には
# 1単位ぶんの余裕を持たせる。
_RADIUS_SLACK = 1.0


def _result(
    violations: list[str],
    total_constraints: int,
    *,
    cost: float | None = None,
) -> dict:
    """検証を実施した結果を返す。"""
    total = max(total_constraints, 1)
    satisfied = max(total - len(violations), 0)
    return {
        "feasible": not violations,
        "verified": True,
        "partial_score": satisfied / total,
        "violation_count": len(violations),
        "total_constraints": total,
        "violations": violations,
        "cost": cost,
    }


def _unverified(reason: str) -> dict:
    """形状を判定できず検証しなかったことを示す。"""
    # Why not feasible=False: 検証できないことと制約違反は別物なので、
    # 違反として扱うと解けている解まで infeasible に落ちてしまう。
    return {
        "feasible": True,
        "verified": False,
        "partial_score": 1.0,
        "violation_count": 0,
        "total_constraints": 0,
        "violations": [f"unrecognized shape: {reason}"],
        "cost": None,
    }


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)


def _close_soft(a: float, b: float) -> bool:
    """連続量向けの比較。

    LP系の参照解は小数第2位で丸めて保存されているため、厳密一致では
    再計算値とわずかにずれる。丸め由来の差だけを許容する。
    """
    return math.isclose(a, b, rel_tol=_SOFT_REL_TOL, abs_tol=_SOFT_ABS_TOL)


def _le_soft(value: float, limit: float) -> bool:
    """連続量の上限比較。丸め由来の超過だけを許容する。"""
    return value <= limit + max(_SOFT_ABS_TOL, _SOFT_REL_TOL * abs(limit))


def _num(value: Any) -> float | None:
    """数値として読めるなら float、読めなければ None。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def _pick(solution: Any, *names: str) -> Any:
    """solution から最初に見つかったキーの値を返す。"""
    if not isinstance(solution, dict):
        return None
    for name in names:
        if name in solution:
            return solution[name]
    lowered = {str(k).lower(): v for k, v in solution.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _int_list(value: Any) -> list[int] | None:
    """整数インデックスのリストとして読めるなら返す。"""
    if isinstance(value, dict):
        value = list(value.values())
    if not isinstance(value, (list, tuple)):
        return None
    out: list[int] = []
    for item in value:
        if isinstance(item, bool):
            return None
        if isinstance(item, int):
            out.append(item)
            continue
        if isinstance(item, float) and float(item).is_integer():
            out.append(int(item))
            continue
        if isinstance(item, str):
            try:
                out.append(int(item.strip()))
                continue
            except ValueError:
                return None
        return None
    return out


def _num_list(value: Any) -> list[float] | None:
    """数値のリストとして読めるなら返す。dict は key 順に並べ直す。"""
    if isinstance(value, dict):
        keys = _int_list(list(value.keys()))
        items = sorted(zip(keys, value.values())) if keys else list(value.items())
        value = [v for _, v in items]
    if not isinstance(value, (list, tuple)):
        return None
    out: list[float] = []
    for item in value:
        parsed = _num(item)
        if parsed is None:
            return None
        out.append(parsed)
    return out


def _pair_list(
    value: Any, first: Sequence[str], second: Sequence[str]
) -> list[tuple[int, int]] | None:
    """マッチング等の組を (a, b) のリストへ正規化する。

    モデルは ``[{"left": 0, "right": 1}]`` とも ``[[0, 1]]`` とも返すため、
    どちらも受ける。1件でも解釈できない要素があれば None を返し、
    呼び出し側は unverified として扱う。
    """
    if not isinstance(value, (list, tuple)):
        return None
    out: list[tuple[int, int]] = []
    for entry in value:
        if isinstance(entry, dict):
            a = next((_num(entry[k]) for k in first if k in entry), None)
            b = next((_num(entry[k]) for k in second if k in entry), None)
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            a, b = _num(entry[0]), _num(entry[1])
        else:
            return None
        if a is None or b is None:
            return None
        out.append((int(a), int(b)))
    return out


def _period_series(value: Any, field_names: Sequence[str]) -> list[float] | None:
    """期別の系列を取り出す。数値の配列と dict の配列の両方を受ける。"""
    if isinstance(value, dict):
        value = [value[k] for k in sorted(value, key=lambda k: _num(k) or 0)]
    if not isinstance(value, (list, tuple)) or not value:
        return None
    if all(isinstance(v, dict) for v in value):
        out: list[float] = []
        for entry in value:
            found = next((_num(entry[k]) for k in field_names if k in entry), None)
            if found is None:
                return None
            out.append(found)
        return out
    return _num_list(value)


def _index_map(value: Any) -> dict[int, Any] | None:
    """{"0": x, "1": y} 形式や [x, y] 形式を dict[int, Any] へ正規化する。"""
    if isinstance(value, dict):
        out: dict[int, Any] = {}
        for key, item in value.items():
            idx = _num(key)
            if idx is None or not float(idx).is_integer():
                return None
            out[int(idx)] = item
        return out
    if isinstance(value, (list, tuple)):
        return dict(enumerate(value))
    return None


def _edge_ends(edge: Any) -> tuple[Any, Any] | None:
    """辺を (from, to) へ正規化する。dict も 2要素リストも受ける。"""
    if isinstance(edge, dict):
        head = edge.get("from", edge.get("u", edge.get("source", edge.get("left"))))
        tail = edge.get("to", edge.get("v", edge.get("target", edge.get("right"))))
        if head is None or tail is None:
            return None
        return head, tail
    if isinstance(edge, (list, tuple)) and len(edge) >= 2:
        return edge[0], edge[1]
    return None


def _undirected_adjacency(edges: Iterable[Any]) -> set[frozenset]:
    """無向辺の集合を作る。"""
    out: set[frozenset] = set()
    for edge in edges:
        ends = _edge_ends(edge)
        if ends is None:
            continue
        out.add(frozenset(ends))
    return out


def _distinct_indices(
    values: Sequence[int],
    limit: int,
    label: str,
    violations: list[str],
) -> bool:
    """インデックス列が範囲内かつ重複なしかを検査する。"""
    ok = True
    out_of_range = [v for v in values if not 0 <= v < limit]
    if out_of_range:
        violations.append(f"{label} out of range 0..{limit - 1}: {sorted(set(out_of_range))}")
        ok = False
    if len(set(values)) != len(values):
        violations.append(f"{label} contains duplicates")
        ok = False
    return ok


def _euclid(a: dict, b: dict) -> float:
    return math.dist((a.get("x", 0), a.get("y", 0)), (b.get("x", 0), b.get("y", 0)))


def _cumulative_supply_ok(
    supply: Sequence[float],
    demand: Sequence[float],
    label: str,
    violations: list[str],
) -> None:
    """各期までの累積供給が累積需要を満たすかを検査する（在庫は繰越可）。"""
    running_supply = 0.0
    running_demand = 0.0
    for period, (produced, needed) in enumerate(zip(supply, demand)):
        running_supply += produced
        running_demand += needed
        if running_supply + _ABS_TOL < running_demand:
            violations.append(
                f"{label}: cumulative supply {running_supply:g} < demand {running_demand:g} "
                f"at period {period}"
            )
            return


EXTRA_CHECKERS: dict[str, Callable[[dict, Any], dict]] = {}


def _register(
    core_type: str,
) -> Callable[[Callable[[dict, Any], dict]], Callable[[dict, Any], dict]]:
    def wrap(fn: Callable[[dict, Any], dict]) -> Callable[[dict, Any], dict]:
        EXTRA_CHECKERS[core_type] = fn
        return fn

    return wrap


# ============================================================
# ナップサック・パッキング
# ============================================================


def _items_field(items: Sequence[dict], key: str) -> list[float]:
    return [float(item.get(key, 0) or 0) for item in items]


@_register("ナップサック・パッキング_整数計画")
def check_knapsack_packing_ip(instance: dict, solution: Any) -> dict:
    """0/1・有界・多次元・複数・ビンパッキング・切出・部分和・依存付き選択。"""
    violations: list[str] = []
    items = instance.get("items")

    # ビンパッキング: 全アイテムがどこかの箱にちょうど1回入り、各箱が容量以内。
    if "item_sizes" in instance and "bin_capacity" in instance:
        sizes = _num_list(instance["item_sizes"]) or []
        capacity = _num(instance.get("bin_capacity")) or 0.0
        bins = _index_map(_pick(solution, "bins", "assignment", "packing"))
        if bins is None:
            return _unverified("bin packing without 'bins' mapping")
        placed: list[int] = []
        for bin_id, contents in bins.items():
            members = _int_list(contents)
            if members is None:
                violations.append(f"bin {bin_id} is not a list of item indices")
                continue
            placed.extend(members)
            load = sum(sizes[i] for i in members if 0 <= i < len(sizes))
            if load > capacity + _ABS_TOL:
                violations.append(f"bin {bin_id} load {load:g} exceeds capacity {capacity:g}")
        _distinct_indices(placed, len(sizes), "packed items", violations)
        if set(placed) != set(range(len(sizes))):
            missing = sorted(set(range(len(sizes))) - set(placed))
            violations.append(f"items not packed: {missing}")
        used_bins = sum(1 for contents in bins.values() if _int_list(contents))
        claimed = _num(_pick(solution, "min_bins", "num_bins", "objective_value"))
        if claimed is not None and not _close(claimed, used_bins):
            violations.append(f"min_bins {claimed:g} != {used_bins} non-empty bins")
        return _result(violations, len(bins) + 3)

    # カッティングストック: パターン使用回数が各注文の需要を満たす。
    if "patterns" in instance and "orders" in instance:
        orders = instance.get("orders") or []
        patterns = instance.get("patterns") or []
        usage = _index_map(_pick(solution, "pattern_usage", "usage", "pattern_counts"))
        if usage is None:
            return _unverified("cutting stock without 'pattern_usage'")
        produced = [0.0] * len(orders)
        for pattern_id, count in usage.items():
            times = _num(count) or 0.0
            if not 0 <= pattern_id < len(patterns):
                violations.append(f"pattern index {pattern_id} out of range")
                continue
            cuts = _num_list(patterns[pattern_id].get("pattern")) or []
            for idx, qty in enumerate(cuts):
                if idx < len(produced):
                    produced[idx] += qty * times
        for idx, order in enumerate(orders):
            need = _num(order.get("demand")) or 0.0
            if produced[idx] + _ABS_TOL < need:
                violations.append(f"order {idx} produced {produced[idx]:g} < demand {need:g}")
        rolls = sum(_num(v) or 0.0 for v in usage.values())
        claimed = _num(_pick(solution, "min_stock_rolls", "min_rolls", "objective_value"))
        if claimed is not None and not _close(claimed, rolls):
            violations.append(f"min_stock_rolls {claimed:g} != total pattern usage {rolls:g}")
        return _result(violations, len(orders) + 2)

    # 資材切出（パターン列挙なし）: 母材本数が長さ合計の下界を下回らない。
    if "pieces" in instance and "stock_length" in instance:
        pieces = instance.get("pieces") or []
        stock = _num(instance.get("stock_length")) or 0.0
        rolls = _num(_pick(solution, "min_rolls", "min_stock_rolls", "rolls", "objective_value"))
        if rolls is None or stock <= 0:
            return _unverified("cutting problem without a roll count")
        total_length = sum(
            (_num(p.get("length")) or 0.0) * (_num(p.get("demand")) or 0.0) for p in pieces
        )
        longest = max((_num(p.get("length")) or 0.0 for p in pieces), default=0.0)
        if longest > stock + _ABS_TOL:
            violations.append(f"piece length {longest:g} exceeds stock length {stock:g}")
        lower_bound = math.ceil(total_length / stock - _ABS_TOL)
        if rolls + _ABS_TOL < lower_bound:
            violations.append(f"rolls {rolls:g} below length lower bound {lower_bound}")
        return _result(violations, 2, cost=rolls)

    # 部分和: 選んだインデックスの和が achieved_sum と一致する。
    if "numbers" in instance and "target" in instance:
        numbers = _num_list(instance["numbers"]) or []
        chosen = _int_list(_pick(solution, "chosen_indices", "chosen_items", "indices"))
        if chosen is None:
            return _unverified("subset sum without chosen indices")
        _distinct_indices(chosen, len(numbers), "chosen_indices", violations)
        achieved = sum(numbers[i] for i in chosen if 0 <= i < len(numbers))
        claimed = _num(_pick(solution, "achieved_sum", "sum"))
        if claimed is not None and not _close(claimed, achieved):
            violations.append(f"achieved_sum {claimed:g} != sum of chosen {achieved:g}")
        target = _num(instance["target"]) or 0.0
        diff = _num(_pick(solution, "min_difference", "difference", "objective_value"))
        if diff is not None and not _close(diff, abs(achieved - target)):
            violations.append(f"min_difference {diff:g} != |{achieved:g} - {target:g}|")
        return _result(violations, 3)

    # 依存関係付きプロジェクト選択: 予算と先行関係。
    if "projects" in instance and "budget" in instance:
        projects = instance.get("projects") or []
        budget = _num(instance["budget"]) or 0.0
        chosen = _int_list(_pick(solution, "chosen_projects", "chosen_items", "selected"))
        if chosen is None:
            return _unverified("project selection without chosen list")
        _distinct_indices(chosen, len(projects), "chosen_projects", violations)
        spend = sum(_num(projects[i].get("cost")) or 0.0 for i in chosen if 0 <= i < len(projects))
        if spend > budget + _ABS_TOL:
            violations.append(f"total cost {spend:g} exceeds budget {budget:g}")
        picked = set(chosen)
        for pair in instance.get("dependencies") or []:
            ends = _edge_ends(pair)
            if ends is None:
                continue
            dependent, prerequisite = ends
            if dependent in picked and prerequisite not in picked:
                violations.append(
                    f"project {dependent} selected without prerequisite {prerequisite}"
                )
        profit = sum(
            _num(projects[i].get("profit")) or 0.0 for i in chosen if 0 <= i < len(projects)
        )
        claimed = _num(_pick(solution, "max_profit", "total_profit", "objective_value"))
        if claimed is not None and not _close(claimed, profit):
            violations.append(f"max_profit {claimed:g} != profit of chosen projects {profit:g}")
        return _result(violations, 4, cost=spend)

    if not isinstance(items, list) or not items:
        return _unverified("no recognizable item list")

    # 複数ナップサック: 各ナップサックの容量と、アイテム重複なし。
    capacities = _num_list(instance.get("capacities"))
    if capacities:
        assignment = _index_map(_pick(solution, "assignment", "bins", "allocation"))
        if assignment is None:
            return _unverified("multiple knapsack without assignment")
        weights = _items_field(items, "weight")
        used: list[int] = []
        for knap_id, contents in assignment.items():
            members = _int_list(contents)
            if members is None:
                violations.append(f"knapsack {knap_id} is not a list of item indices")
                continue
            used.extend(members)
            if not 0 <= knap_id < len(capacities):
                violations.append(f"knapsack index {knap_id} out of range")
                continue
            load = sum(weights[i] for i in members if 0 <= i < len(weights))
            if load > capacities[knap_id] + _ABS_TOL:
                violations.append(
                    f"knapsack {knap_id} load {load:g} exceeds {capacities[knap_id]:g}"
                )
        _distinct_indices(used, len(items), "assigned items", violations)
        values = _items_field(items, "value")
        total = sum(values[i] for i in used if 0 <= i < len(values))
        claimed = _num(_pick(solution, "max_value", "total_value", "objective_value"))
        if claimed is not None and not _close(claimed, total):
            violations.append(f"max_value {claimed:g} != value of assigned items {total:g}")
        return _result(violations, len(capacities) + 2, cost=total)

    # 多次元 / コンテナ積載: 2種類の容量を同時に満たす。
    dims = [
        ("weight", instance.get("weight_capacity", instance.get("max_weight"))),
        ("volume", instance.get("volume_capacity", instance.get("max_volume"))),
    ]
    if all(cap is not None for _, cap in dims):
        chosen = _int_list(
            _pick(solution, "chosen_items", "loaded_items", "selected_items", "items")
        )
        if chosen is None:
            return _unverified("multidimensional knapsack without chosen items")
        _distinct_indices(chosen, len(items), "chosen_items", violations)
        for field, cap in dims:
            limit = _num(cap) or 0.0
            values = _items_field(items, field)
            load = sum(values[i] for i in chosen if 0 <= i < len(values))
            if load > limit + _ABS_TOL:
                violations.append(f"total {field} {load:g} exceeds capacity {limit:g}")
        objective_field = "priority" if any("priority" in item for item in items) else "value"
        gains = _items_field(items, objective_field)
        total = sum(gains[i] for i in chosen if 0 <= i < len(gains))
        claimed = _num(
            _pick(solution, "max_value", "max_priority", "total_value", "objective_value")
        )
        if claimed is not None and not _close(claimed, total):
            violations.append(f"objective {claimed:g} != {objective_field} of chosen {total:g}")
        return _result(violations, 4, cost=total)

    capacity = _num(instance.get("capacity"))
    if capacity is None:
        return _unverified("knapsack without capacity")
    weights = _items_field(items, "weight")

    # 有界ナップサック: 個数が max_count 以内。
    counts = _index_map(_pick(solution, "item_counts", "counts", "quantities"))
    if counts is not None and any("max_count" in item for item in items):
        load = 0.0
        for idx, raw in counts.items():
            qty = _num(raw)
            if qty is None or qty < 0:
                violations.append(f"item {idx} count is not a non-negative number")
                continue
            if not 0 <= idx < len(items):
                violations.append(f"item index {idx} out of range")
                continue
            bound = _num(items[idx].get("max_count"))
            if bound is not None and qty > bound + _ABS_TOL:
                violations.append(f"item {idx} count {qty:g} exceeds max_count {bound:g}")
            load += qty * weights[idx]
        if load > capacity + _ABS_TOL:
            violations.append(f"total weight {load:g} exceeds capacity {capacity:g}")
        values = _items_field(items, "value")
        total = sum(
            (_num(raw) or 0.0) * values[idx]
            for idx, raw in counts.items()
            if 0 <= idx < len(values)
        )
        claimed = _num(_pick(solution, "max_value", "total_value", "objective_value"))
        if claimed is not None and not _close(claimed, total):
            violations.append(f"max_value {claimed:g} != value of chosen counts {total:g}")
        return _result(violations, len(counts) + 2, cost=load)

    # 0/1 ナップサック（グループ制約付きを含む）。
    chosen = _int_list(_pick(solution, "chosen_items", "selected_items", "items", "chosen"))
    if chosen is None:
        return _unverified("knapsack without chosen items")
    _distinct_indices(chosen, len(items), "chosen_items", violations)
    load = sum(weights[i] for i in chosen if 0 <= i < len(weights))
    if load > capacity + _ABS_TOL:
        violations.append(f"total weight {load:g} exceeds capacity {capacity:g}")
    values = _items_field(items, "value")
    total = sum(values[i] for i in chosen if 0 <= i < len(values))
    claimed = _num(_pick(solution, "max_value", "total_value", "objective_value"))
    if claimed is not None and not _close(claimed, total):
        violations.append(f"max_value {claimed:g} != value of chosen items {total:g}")
    return _result(violations, 3, cost=load)


@_register("ナップサック・パッキング_制約計画")
def check_strip_packing(instance: dict, solution: Any) -> dict:
    """2次元ストリップパッキング: 全矩形が幅内に重なりなく配置される。"""
    rectangles = instance.get("rectangles")
    width = _num(instance.get("strip_width"))
    if not isinstance(rectangles, list) or width is None:
        return _unverified("strip packing without rectangles or width")
    placement = _pick(solution, "placement", "placements", "positions")
    if not isinstance(placement, list) or not placement:
        return _unverified("strip packing without placement list")

    violations: list[str] = []
    boxes: list[tuple[float, float, float, float]] = []
    seen: list[int] = []
    for entry in placement:
        if not isinstance(entry, dict):
            violations.append("placement entry is not an object")
            continue
        rect_id = _num(entry.get("id"))
        x = _num(entry.get("x"))
        y = _num(entry.get("y"))
        if x is None or y is None:
            violations.append(f"placement {entry.get('id')} missing x/y")
            continue
        index = int(rect_id) if rect_id is not None and float(rect_id).is_integer() else None
        source = rectangles[index] if index is not None and 0 <= index < len(rectangles) else {}
        w = _num(entry.get("w")) or _num(source.get("w")) or 0.0
        h = _num(entry.get("h")) or _num(source.get("h")) or 0.0
        if index is not None:
            seen.append(index)
        if x < -_ABS_TOL or y < -_ABS_TOL:
            violations.append(f"rectangle {rect_id} placed at negative coordinate")
        if x + w > width + _ABS_TOL:
            violations.append(f"rectangle {rect_id} exceeds strip width {width:g}")
        boxes.append((x, y, w, h))

    _distinct_indices(seen, len(rectangles), "placed rectangles", violations)
    if len(seen) != len(rectangles):
        violations.append(f"placed {len(seen)} of {len(rectangles)} rectangles")

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ax, ay, aw, ah = boxes[i]
            bx, by, bw, bh = boxes[j]
            overlap_x = ax < bx + bw - _ABS_TOL and bx < ax + aw - _ABS_TOL
            overlap_y = ay < by + bh - _ABS_TOL and by < ay + ah - _ABS_TOL
            if overlap_x and overlap_y:
                violations.append(f"rectangles {i} and {j} overlap")

    height = max((y + h for _, y, _, h in boxes), default=0.0)
    claimed = _num(_pick(solution, "min_height", "height", "objective_value"))
    if claimed is not None and claimed + _ABS_TOL < height:
        violations.append(f"min_height {claimed:g} below actual height {height:g}")
    return _result(violations, len(rectangles) + 3, cost=height)


# ============================================================
# ネットワークフロー
# ============================================================


def _weighted_graph(edges: Iterable[Any], weight_key: str) -> dict[Any, list[tuple[Any, float]]]:
    """有向グラフの隣接リストを作る。"""
    graph: dict[Any, list[tuple[Any, float]]] = {}
    for edge in edges:
        ends = _edge_ends(edge)
        if ends is None:
            continue
        head, tail = ends
        weight = 0.0
        if isinstance(edge, dict):
            weight = _num(edge.get(weight_key)) or 0.0
        graph.setdefault(head, []).append((tail, weight))
        graph.setdefault(tail, [])
    return graph


def _dijkstra(graph: dict[Any, list[tuple[Any, float]]], start: Any) -> dict[Any, float]:
    """始点からの最短距離。到達不能ノードは含めない。"""
    import heapq

    dist: dict[Any, float] = {start: 0.0}
    queue: list[tuple[float, int, Any]] = [(0.0, 0, start)]
    order = 0
    while queue:
        current, _, node = heapq.heappop(queue)
        if current > dist.get(node, math.inf) + _ABS_TOL:
            continue
        for neighbour, weight in graph.get(node, []):
            candidate = current + weight
            if candidate < dist.get(neighbour, math.inf) - _ABS_TOL:
                dist[neighbour] = candidate
                order += 1
                heapq.heappush(queue, (candidate, order, neighbour))
    return dist


def _min_hops(edges: Iterable[Any], start: Any, goal: Any) -> float:
    """辺数の最小値。到達不能なら 0（下界として無害な値）。"""
    graph = _weighted_graph(edges, "__none__")
    unit = {node: [(nbr, 1.0) for nbr, _ in adj] for node, adj in graph.items()}
    return _dijkstra(unit, start).get(goal, 0.0)


@_register("ネットワークフロー_線形計画")
def check_network_flow_lp(instance: dict, solution: Any) -> dict:
    """最大流・最小費用流・輸送・積替・多品種流。目的値の上界／下界を検証する。"""
    violations: list[str] = []
    edges = instance.get("edges") or []

    # 輸送問題: 需要地ごとの最安値による下界。
    if "supply" in instance and "cost" in instance and "demand" in instance and not edges:
        supply = _num_list(instance["supply"]) or []
        demand = _num_list(instance["demand"]) or []
        cost = instance.get("cost") or []
        claimed = _num(_pick(solution, "min_cost", "total_cost", "objective_value"))
        if claimed is None:
            return _unverified("transportation without a cost value")
        if sum(supply) + _ABS_TOL < sum(demand):
            violations.append(f"total supply {sum(supply):g} < total demand {sum(demand):g}")
        bound = 0.0
        for j, need in enumerate(demand):
            column = [_num(row[j]) or 0.0 for row in cost if j < len(row)]
            if column:
                bound += need * min(column)
        if claimed + _ABS_TOL < bound:
            violations.append(f"min_cost {claimed:g} below lower bound {bound:g}")
        if claimed < -_ABS_TOL:
            violations.append(f"min_cost {claimed:g} is negative")
        return _result(violations, 3, cost=claimed)

    # 積替問題: 供給地から需要地への最短費用による下界。
    if isinstance(instance.get("supply"), dict) and isinstance(instance.get("demand"), dict):
        supply = instance["supply"]
        demand = instance["demand"]
        claimed = _num(_pick(solution, "min_cost", "total_cost", "objective_value"))
        if claimed is None:
            return _unverified("transshipment without a cost value")
        graph = _weighted_graph(edges, "cost")
        reach = {node: _dijkstra(graph, node) for node in supply}
        bound = 0.0
        for node, need in demand.items():
            options = [d[node] for d in reach.values() if node in d]
            if options:
                bound += (_num(need) or 0.0) * min(options)
        if claimed + _ABS_TOL < bound:
            violations.append(f"min_cost {claimed:g} below lower bound {bound:g}")
        if sum(_num(v) or 0.0 for v in supply.values()) + _ABS_TOL < sum(
            _num(v) or 0.0 for v in demand.values()
        ):
            violations.append("total supply below total demand")
        return _result(violations, 2, cost=claimed)

    # 多品種流: 各品目の需要 × 最小ホップ数による下界。
    commodities = instance.get("commodities")
    if commodities:
        claimed = _num(_pick(solution, "min_total_flow", "total_flow", "objective_value"))
        if claimed is None:
            return _unverified("multicommodity flow without a flow value")
        bound = 0.0
        for item in commodities:
            hops = _min_hops(edges, item.get("source"), item.get("sink"))
            bound += (_num(item.get("demand")) or 0.0) * hops
        if claimed + _ABS_TOL < bound:
            violations.append(f"min_total_flow {claimed:g} below lower bound {bound:g}")
        return _result(violations, 2, cost=claimed)

    # 最小費用流: 単位あたり最短費用 × 輸送量による下界。
    if "supply_node" in instance and "demand_node" in instance:
        amount = _num(instance.get("amount")) or 0.0
        claimed = _num(_pick(solution, "min_cost", "total_cost", "objective_value"))
        if claimed is None:
            return _unverified("min cost flow without a cost value")
        graph = _weighted_graph(edges, "cost")
        shortest = _dijkstra(graph, instance["supply_node"]).get(instance["demand_node"])
        if shortest is not None and claimed + _ABS_TOL < amount * shortest:
            violations.append(f"min_cost {claimed:g} below lower bound {amount * shortest:g}")
        if claimed < -_ABS_TOL:
            violations.append(f"min_cost {claimed:g} is negative")
        return _result(violations, 2, cost=claimed)

    # 最大流（容量拡張つきを含む）: 源の流出容量と集の流入容量による上界。
    if "source" in instance and "sink" in instance:
        claimed = _num(_pick(solution, "max_flow", "flow", "objective_value"))
        if claimed is None:
            return _unverified("max flow without a flow value")
        source, sink = instance["source"], instance["sink"]
        budget = _num(instance.get("budget"))
        out_cap = expand_out = 0.0
        in_cap = expand_in = 0.0
        out_units: list[float] = []
        in_units: list[float] = []
        for edge in edges:
            ends = _edge_ends(edge)
            if ends is None or not isinstance(edge, dict):
                continue
            head, tail = ends
            cap = _num(edge.get("capacity")) or 0.0
            unit = _num(edge.get("expand_unit_cost"))
            if head == source:
                out_cap += cap
                if unit:
                    out_units.append(unit)
            if tail == sink:
                in_cap += cap
                if unit:
                    in_units.append(unit)
        if budget and out_units:
            expand_out = budget / min(out_units)
        if budget and in_units:
            expand_in = budget / min(in_units)
        upper = min(out_cap + expand_out, in_cap + expand_in)
        if claimed > upper + _ABS_TOL:
            violations.append(f"max_flow {claimed:g} exceeds cut upper bound {upper:g}")
        if claimed < -_ABS_TOL:
            violations.append(f"max_flow {claimed:g} is negative")
        return _result(violations, 2, cost=claimed)

    return _unverified("unknown network flow LP shape")


@_register("ネットワークフロー_グラフ最適化")
def check_network_flow_graph(instance: dict, solution: Any) -> dict:
    """最短路・単一始点最短路木・最小全域木。"""
    violations: list[str] = []
    edges = instance.get("edges") or []

    # 最短路: 経路が実在する辺で繋がっているか。
    if "start" in instance and "goal" in instance:
        path = _int_list(_pick(solution, "path", "route", "sequence"))
        if path is None or len(path) < 1:
            return _unverified("shortest path without a path")
        if path[0] != instance["start"]:
            violations.append(f"path starts at {path[0]}, expected {instance['start']}")
        if path[-1] != instance["goal"]:
            violations.append(f"path ends at {path[-1]}, expected {instance['goal']}")
        lookup: dict[frozenset, float] = {}
        for edge in edges:
            ends = _edge_ends(edge)
            if ends is None:
                continue
            length = _num(edge.get("length")) if isinstance(edge, dict) else None
            lookup[frozenset(ends)] = length or 0.0
        total = 0.0
        for head, tail in pairwise(path):
            key = frozenset((head, tail))
            if key not in lookup:
                violations.append(f"edge {head}->{tail} does not exist")
                continue
            total += lookup[key]
        claimed = _num(_pick(solution, "shortest_distance", "distance", "objective_value"))
        if claimed is not None and not _close(claimed, total):
            violations.append(f"shortest_distance {claimed:g} != path length {total:g}")
        return _result(violations, len(path) + 2, cost=total)

    # 単一始点最短路木: 各距離が真の最短距離を下回らないこと。
    if "source" in instance:
        distances = _index_map(_pick(solution, "distances", "distance", "dist"))
        if distances is None:
            return _unverified("shortest path tree without distances")
        graph = _weighted_graph(edges, "length")
        undirected: dict[Any, list[tuple[Any, float]]] = {}
        for edge in edges:
            ends = _edge_ends(edge)
            if ends is None or not isinstance(edge, dict):
                continue
            head, tail = ends
            length = _num(edge.get("length")) or 0.0
            undirected.setdefault(head, []).append((tail, length))
            undirected.setdefault(tail, []).append((head, length))
        true_dist = _dijkstra(undirected or graph, instance["source"])
        for node, claimed_raw in distances.items():
            claimed = _num(claimed_raw)
            if claimed is None:
                violations.append(f"distance for node {node} is not numeric")
                continue
            optimal = true_dist.get(node)
            if optimal is not None and claimed + _ABS_TOL < optimal:
                violations.append(
                    f"distance {claimed:g} for node {node} below shortest {optimal:g}"
                )
        total = sum(_num(v) or 0.0 for v in distances.values())
        claimed_total = _num(
            _pick(solution, "total_shortest_distance", "total_distance", "objective_value")
        )
        if claimed_total is not None and not _close(claimed_total, total):
            violations.append(f"total {claimed_total:g} != sum of distances {total:g}")
        return _result(violations, len(distances) + 1, cost=total)

    # 最小全域木: 辺数が n-1 で、費用が最小 n-1 本の合計を下回らない。
    num_nodes = int(_num(instance.get("num_nodes")) or 0)
    claimed = _num(_pick(solution, "min_total_cost", "total_cost", "objective_value"))
    if claimed is None or num_nodes <= 0:
        return _unverified("minimum spanning tree without a cost value")
    used = _num(_pick(solution, "num_edges_used", "edge_count"))
    if used is not None and int(used) != num_nodes - 1:
        violations.append(f"num_edges_used {int(used)} != {num_nodes - 1}")
    costs = sorted(_num(e.get("cost")) or 0.0 for e in edges if isinstance(e, dict))
    bound = sum(costs[: max(num_nodes - 1, 0)])
    if claimed + _ABS_TOL < bound:
        violations.append(f"min_total_cost {claimed:g} below lower bound {bound:g}")
    return _result(violations, 2, cost=claimed)


@_register("ネットワークフロー_組合せ最適化")
def check_network_flow_combinatorial(instance: dict, solution: Any) -> dict:
    """最小カットと二部グラフ最大マッチング。"""
    violations: list[str] = []
    edges = instance.get("edges") or []

    # 最小カット: S/T が全ノードを分割し、カット容量が申告値と一致する。
    s_side = _int_list(_pick(solution, "S_side", "s_side", "source_side"))
    t_side = _int_list(_pick(solution, "T_side", "t_side", "sink_side"))
    if s_side is not None and t_side is not None:
        num_nodes = int(_num(instance.get("num_nodes")) or 0)
        source, sink = instance.get("source"), instance.get("sink")
        overlap = set(s_side) & set(t_side)
        if overlap:
            violations.append(f"nodes on both sides of the cut: {sorted(overlap)}")
        if set(s_side) | set(t_side) != set(range(num_nodes)):
            violations.append("S_side and T_side do not partition all nodes")
        if source not in s_side:
            violations.append(f"source {source} not in S_side")
        if sink not in t_side:
            violations.append(f"sink {sink} not in T_side")
        cut = 0.0
        for edge in edges:
            ends = _edge_ends(edge)
            if ends is None or not isinstance(edge, dict):
                continue
            head, tail = ends
            if head in set(s_side) and tail in set(t_side):
                cut += _num(edge.get("capacity")) or 0.0
        claimed = _num(_pick(solution, "min_cut_value", "cut_value", "objective_value"))
        if claimed is not None and not _close(claimed, cut):
            violations.append(f"min_cut_value {claimed:g} != cut capacity {cut:g}")
        return _result(violations, 5, cost=cut)

    # 二部グラフ最大マッチング: 片側の人数と辺の本数を超えない。
    pairs = instance.get("compatible_pairs")
    if pairs is not None:
        claimed = _num(_pick(solution, "max_matching", "matching_size", "objective_value"))
        if claimed is None:
            return _unverified("bipartite matching without a size")
        workers = int(_num(instance.get("num_workers")) or 0)
        tasks = int(_num(instance.get("num_tasks")) or 0)
        upper = min(workers, tasks, len(pairs))
        if claimed > upper + _ABS_TOL:
            violations.append(f"max_matching {claimed:g} exceeds upper bound {upper}")
        if claimed < -_ABS_TOL:
            violations.append(f"max_matching {claimed:g} is negative")
        return _result(violations, 2, cost=claimed)

    return _unverified("unknown combinatorial network shape")


# ============================================================
# 施設配置・被覆
# ============================================================


def _covered_points(
    demand_points: Sequence[dict],
    site_points: Sequence[dict],
    radius: float,
    chosen: Iterable[int],
) -> set[int]:
    """選んだサイトの半径内に入る需要点の集合。"""
    picked = [site_points[i] for i in chosen if 0 <= i < len(site_points)]
    covered: set[int] = set()
    for idx, point in enumerate(demand_points):
        if any(_euclid(point, site) <= radius + _RADIUS_SLACK for site in picked):
            covered.add(idx)
    return covered


@_register("施設配置・被覆_整数計画")
def check_covering_ip(instance: dict, solution: Any) -> dict:
    """集合被覆・集合分割・頂点被覆・最大被覆・被覆立地。"""
    violations: list[str] = []

    # 集合被覆: 選んだ集合の和集合が全要素を覆う。
    sets = instance.get("sets")
    if sets is not None:
        num_items = int(_num(instance.get("num_items")) or 0)
        chosen = _int_list(_pick(solution, "chosen_sets", "selected_sets", "chosen"))
        if chosen is None:
            return _unverified("set cover without chosen sets")
        _distinct_indices(chosen, len(sets), "chosen_sets", violations)
        covered: set[int] = set()
        spend = 0.0
        for idx in chosen:
            if not 0 <= idx < len(sets):
                continue
            covered |= set(_int_list(sets[idx].get("covers")) or [])
            spend += _num(sets[idx].get("cost")) or 0.0
        missing = sorted(set(range(num_items)) - covered)
        if missing:
            violations.append(f"items not covered: {missing}")
        claimed = _num(_pick(solution, "min_cost", "total_cost", "objective_value"))
        if claimed is not None and not _close(claimed, spend):
            violations.append(f"min_cost {claimed:g} != sum of chosen costs {spend:g}")
        return _result(violations, 3, cost=spend)

    # 集合分割: 各要素をちょうど1回覆う。
    groups = instance.get("groups")
    if groups is not None and isinstance(groups, list) and groups and isinstance(groups[0], dict):
        num_items = int(_num(instance.get("num_items")) or 0)
        chosen = _int_list(_pick(solution, "chosen_groups", "selected_groups", "chosen"))
        if chosen is None:
            return _unverified("set partition without chosen groups")
        _distinct_indices(chosen, len(groups), "chosen_groups", violations)
        seen: list[int] = []
        spend = 0.0
        for idx in chosen:
            if not 0 <= idx < len(groups):
                continue
            seen.extend(_int_list(groups[idx].get("members")) or [])
            spend += _num(groups[idx].get("cost")) or 0.0
        duplicated = sorted({i for i in seen if seen.count(i) > 1})
        if duplicated:
            violations.append(f"items covered more than once: {duplicated}")
        missing = sorted(set(range(num_items)) - set(seen))
        if missing:
            violations.append(f"items not covered: {missing}")
        claimed = _num(_pick(solution, "min_cost", "total_cost", "objective_value"))
        if claimed is not None and not _close(claimed, spend):
            violations.append(f"min_cost {claimed:g} != sum of chosen costs {spend:g}")
        return _result(violations, 4, cost=spend)

    # 重み付き頂点被覆: すべての辺が端点を持つ。
    if "edges" in instance and "cost" in instance and "num_nodes" in instance:
        costs = _num_list(instance["cost"]) or []
        selected = _int_list(_pick(solution, "selected_nodes", "chosen_nodes", "cover"))
        if selected is None:
            return _unverified("vertex cover without selected nodes")
        _distinct_indices(
            selected, int(_num(instance["num_nodes"]) or 0), "selected_nodes", violations
        )
        picked = set(selected)
        for edge in instance.get("edges") or []:
            ends = _edge_ends(edge)
            if ends is None:
                continue
            if ends[0] not in picked and ends[1] not in picked:
                violations.append(f"edge {list(ends)} is not covered")
        spend = sum(costs[i] for i in selected if 0 <= i < len(costs))
        claimed = _num(_pick(solution, "min_cost", "total_cost", "objective_value"))
        if claimed is not None and not _close(claimed, spend):
            violations.append(f"min_cost {claimed:g} != sum of node costs {spend:g}")
        return _result(violations, 3, cost=spend)

    demand_points = instance.get("demand_points")
    site_points = instance.get("site_points")
    if demand_points is None or site_points is None:
        return _unverified("unknown covering IP shape")

    radius = _num(instance.get("radius")) or 0.0
    chosen = _int_list(_pick(solution, "chosen_sites", "selected_sites", "sites", "facilities"))
    if chosen is None:
        return _unverified("covering location without chosen sites")
    _distinct_indices(chosen, len(site_points), "chosen_sites", violations)
    covered = _covered_points(demand_points, site_points, radius, chosen)

    max_facilities = _num(instance.get("max_facilities"))
    if max_facilities is not None and len(chosen) > max_facilities + _ABS_TOL:
        violations.append(f"{len(chosen)} sites chosen, limit is {int(max_facilities)}")

    site_cost = _num_list(instance.get("site_cost"))
    budget = _num(instance.get("budget"))
    if site_cost and budget is not None:
        spend = sum(site_cost[i] for i in chosen if 0 <= i < len(site_cost))
        if spend > budget + _ABS_TOL:
            violations.append(f"site cost {spend:g} exceeds budget {budget:g}")

    population = _num_list(instance.get("population"))
    if population:
        total = sum(population[i] for i in covered if i < len(population))
        claimed = _num(
            _pick(solution, "max_covered_population", "covered_population", "objective_value")
        )
        if claimed is not None and not _close(claimed, total):
            violations.append(f"covered population {claimed:g} != actual coverage {total:g}")
        return _result(violations, 3, cost=total)

    # 被覆立地（最小施設数）: すべての需要点が覆われる。
    missing = sorted(set(range(len(demand_points))) - covered)
    if missing:
        violations.append(f"demand points not covered: {missing}")
    claimed = _num(_pick(solution, "min_facilities", "num_facilities", "objective_value"))
    if claimed is not None and not _close(claimed, len(chosen)):
        violations.append(f"min_facilities {claimed:g} != {len(chosen)} chosen sites")
    return _result(violations, 3, cost=float(len(chosen)))


@_register("施設配置・被覆_混合整数計画")
def check_facility_location_milp(instance: dict, solution: Any) -> dict:
    """無容量／容量制約施設配置・p-メディアン・p-センター・倉庫配置・ハブ配置。"""
    violations: list[str] = []
    distance = instance.get("distance") or instance.get("transport_cost") or []
    fixed_cost = _num_list(instance.get("fixed_cost")) or []
    demand = _num_list(instance.get("demand")) or []

    # ハブ配置: ハブ数と、全ノードのハブ割当。
    num_hubs = _num(instance.get("num_hubs"))
    if num_hubs is not None:
        nodes = instance.get("nodes") or []
        hubs = _int_list(_pick(solution, "hubs", "selected_hubs", "opened_hubs"))
        if hubs is None:
            return _unverified("hub location without hubs")
        _distinct_indices(hubs, len(nodes), "hubs", violations)
        if len(hubs) != int(num_hubs):
            violations.append(f"{len(hubs)} hubs chosen, expected {int(num_hubs)}")
        assignment = _index_map(_pick(solution, "assignment", "allocation"))
        if assignment is not None:
            for node, hub in assignment.items():
                hub_idx = _num(hub)
                if hub_idx is None or int(hub_idx) not in set(hubs):
                    violations.append(f"node {node} assigned to non-hub {hub}")
            missing = sorted(set(range(len(nodes))) - set(assignment))
            if missing:
                violations.append(f"nodes without a hub assignment: {missing}")
        return _result(violations, 3)

    # p-メディアン / p-センター: 施設数が p、目的値が選択集合から一意に定まる。
    p = _num(instance.get("p"))
    if p is not None:
        nodes = instance.get("nodes") or []
        selected = _int_list(
            _pick(solution, "selected_facilities", "facilities", "selected", "opened_facilities")
        )
        if selected is None:
            return _unverified("p-median/p-center without selected facilities")
        _distinct_indices(selected, len(nodes), "selected_facilities", violations)
        if len(selected) != int(p):
            violations.append(f"{len(selected)} facilities chosen, expected p={int(p)}")
        nearest = []
        for j in range(len(nodes)):
            options = [
                _num(distance[i][j]) or 0.0
                for i in selected
                if 0 <= i < len(distance) and j < len(distance[i])
            ]
            nearest.append(min(options) if options else 0.0)
        if demand:
            total = sum(d * n for d, n in zip(demand, nearest))
            claimed = _num(
                _pick(solution, "min_weighted_distance", "weighted_distance", "objective_value")
            )
            if claimed is not None and not _close(claimed, total):
                violations.append(
                    f"weighted distance {claimed:g} != value for this facility set {total:g}"
                )
            return _result(violations, 3, cost=total)
        worst = max(nearest) if nearest else 0.0
        claimed = _num(_pick(solution, "min_max_distance", "max_distance", "objective_value"))
        if claimed is not None and not _close(claimed, worst):
            violations.append(
                f"min_max_distance {claimed:g} != value for this facility set {worst:g}"
            )
        return _result(violations, 3, cost=worst)

    opened = _int_list(
        _pick(solution, "opened_facilities", "opened_warehouses", "facilities", "opened")
    )
    if opened is None:
        return _unverified("facility location without opened facilities")
    sites = instance.get("facilities") or instance.get("warehouses") or []
    _distinct_indices(opened, len(sites), "opened facilities", violations)
    if not opened:
        violations.append("no facility opened")

    fixed_total = sum(fixed_cost[i] for i in opened if 0 <= i < len(fixed_cost))
    num_customers = len(demand) if demand else (len(distance[0]) if distance else 0)
    nearest_total = 0.0
    for j in range(num_customers):
        options = [
            _num(distance[i][j]) or 0.0
            for i in opened
            if 0 <= i < len(distance) and j < len(distance[i])
        ]
        if not options:
            continue
        weight = demand[j] if demand and j < len(demand) else 1.0
        nearest_total += weight * min(options)

    capacity = _num_list(instance.get("capacity"))
    claimed = _num(_pick(solution, "min_total_cost", "total_cost", "min_cost", "objective_value"))
    if capacity:
        available = sum(capacity[i] for i in opened if 0 <= i < len(capacity))
        if available + _ABS_TOL < sum(demand):
            violations.append(f"opened capacity {available:g} below total demand {sum(demand):g}")
        if claimed is not None and claimed + _ABS_TOL < fixed_total + nearest_total:
            violations.append(
                f"total cost {claimed:g} below lower bound {fixed_total + nearest_total:g}"
            )
        return _result(violations, 3, cost=claimed)

    # 無容量施設配置は、開設集合が決まれば最適割当費用が一意に決まる。
    exact = fixed_total + nearest_total
    if claimed is not None and claimed + _ABS_TOL < exact:
        violations.append(f"total cost {claimed:g} below achievable minimum {exact:g}")
    return _result(violations, 3, cost=claimed)


# ============================================================
# 割当・マッチング
# ============================================================


def _assignment_map(solution: Any, *names: str) -> dict[int, Any] | None:
    return _index_map(_pick(solution, *names, "assignment", "allocation", "schedule"))


def _check_full_assignment(
    assignment: dict[int, Any],
    num_tasks: int,
    num_agents: int,
    violations: list[str],
) -> list[int | None]:
    """全タスクが有効なエージェントへ割り当てられているかを検査する。"""
    chosen: list[int | None] = [None] * num_tasks
    for task, agent in assignment.items():
        agent_idx = _num(agent)
        if agent_idx is None or not float(agent_idx).is_integer():
            violations.append(f"task {task} assigned to non-integer agent {agent}")
            continue
        if not 0 <= task < num_tasks:
            violations.append(f"task index {task} out of range")
            continue
        if not 0 <= int(agent_idx) < num_agents:
            violations.append(f"task {task} assigned to unknown agent {int(agent_idx)}")
            continue
        chosen[task] = int(agent_idx)
    missing = [i for i, a in enumerate(chosen) if a is None]
    if missing:
        violations.append(f"tasks without assignment: {missing}")
    return chosen


@_register("割当・マッチング_整数計画")
def check_assignment_ip(instance: dict, solution: Any) -> dict:
    """一般化割当・スキル制約・ボトルネック・当直・教室・選択的割当。"""
    violations: list[str] = []

    # 当直割当: 各日の必要人数と、勤務不可日。
    daily_need = _num_list(instance.get("daily_need"))
    if daily_need is not None:
        num_staff = int(_num(instance.get("num_staff")) or 0)
        schedule = _index_map(_pick(solution, "schedule", "assignment", "roster"))
        if schedule is None:
            return _unverified("duty roster without a schedule")
        unavailable = {
            int(_num(k) or -1): set(_int_list(v) or [])
            for k, v in (instance.get("unavailable") or {}).items()
        }
        # schedule は {日: 職員リスト} と {職員: 勤務日リスト} の両方があるため、
        # キー数で向きを判定してから (日, 職員) の組に正規化する。
        num_days = len(daily_need)
        staff_keyed = len(schedule) == num_staff and num_staff != num_days
        pairs: list[tuple[int, int]] = []
        malformed = False
        for key, raw in schedule.items():
            members = _int_list(raw)
            if members is None:
                violations.append(f"schedule entry {key} is not a list")
                malformed = True
                continue
            for member in members:
                pairs.append((member, key) if staff_keyed else (key, member))
        counts = [0] * num_staff
        per_day: dict[int, int] = {}
        for day, person in pairs:
            per_day[day] = per_day.get(day, 0) + 1
            if not 0 <= person < num_staff:
                violations.append(f"unknown staff {person} on day {day}")
                continue
            counts[person] += 1
            if day in unavailable.get(person, set()):
                violations.append(f"staff {person} assigned on unavailable day {day}")
        if not malformed:
            for day, needed in enumerate(daily_need):
                if per_day.get(day, 0) != int(needed):
                    violations.append(
                        f"day {day} has {per_day.get(day, 0)} staff, needs {int(needed)}"
                    )
        gap = (max(counts) - min(counts)) if counts else 0
        claimed = _num(_pick(solution, "min_load_gap", "load_gap", "objective_value"))
        if claimed is not None and not _close(claimed, gap):
            violations.append(f"min_load_gap {claimed:g} != actual gap {gap}")
        return _result(violations, len(daily_need) + 2, cost=float(gap))

    # 教室割当: 定員と、(教室, 時限) の重複禁止。
    course_size = _num_list(instance.get("course_size"))
    room_capacity = _num_list(instance.get("room_capacity"))
    if course_size is not None and room_capacity is not None:
        preference = instance.get("slot_preference") or []
        assignment = _index_map(_pick(solution, "assignment", "allocation"))
        if assignment is None:
            return _unverified("classroom assignment without assignment")
        used: set[tuple[int, int]] = set()
        total_pref = 0.0
        for course, slot_info in assignment.items():
            if isinstance(slot_info, dict):
                room = _num(slot_info.get("room"))
                slot = _num(slot_info.get("slot"))
            elif isinstance(slot_info, (list, tuple)) and len(slot_info) >= 2:
                room, slot = _num(slot_info[0]), _num(slot_info[1])
            else:
                return _unverified("classroom assignment is not (room, slot)")
            if room is None or slot is None:
                return _unverified("classroom assignment is missing room or slot")
            room, slot = int(room), int(slot)
            if (room, slot) in used:
                violations.append(f"room {room} double-booked at slot {slot}")
            used.add((room, slot))
            in_range = 0 <= course < len(course_size) and 0 <= room < len(room_capacity)
            if in_range and room_capacity[room] + _ABS_TOL < course_size[course]:
                violations.append(
                    f"course {course} size {course_size[course]:g} exceeds room {room} "
                    f"capacity {room_capacity[room]:g}"
                )
            if 0 <= course < len(preference) and 0 <= slot < len(preference[course]):
                total_pref += _num(preference[course][slot]) or 0.0
        missing = sorted(set(range(len(course_size))) - set(assignment))
        if missing:
            violations.append(f"courses without assignment: {missing}")
        claimed = _num(_pick(solution, "max_preference", "total_preference", "objective_value"))
        if claimed is not None and not _close(claimed, total_pref):
            violations.append(f"max_preference {claimed:g} != actual {total_pref:g}")
        return _result(violations, len(course_size) + 2, cost=total_pref)

    # 選択的割当: エージェントと仕事が重複しないマッチング。
    profit_matrix = instance.get("profit_matrix")
    if profit_matrix is not None:
        pairs = _pair_list(
            _pick(solution, "pairs", "assignment", "matching"),
            ("agent", "a", "worker", "left"),
            ("job", "j", "task", "right"),
        )
        if pairs is None:
            return _unverified("selective assignment entries are not (agent, job) pairs")
        agents: list[int] = []
        jobs: list[int] = []
        total = 0.0
        for agent, job in pairs:
            agents.append(agent)
            jobs.append(job)
            if 0 <= agent < len(profit_matrix) and 0 <= job < len(profit_matrix[agent]):
                total += _num(profit_matrix[agent][job]) or 0.0
        _distinct_indices(agents, len(profit_matrix), "paired agents", violations)
        _distinct_indices(
            jobs, len(profit_matrix[0]) if profit_matrix else 0, "paired jobs", violations
        )
        claimed = _num(_pick(solution, "max_profit", "total_profit", "objective_value"))
        if claimed is not None and not _close(claimed, total):
            violations.append(f"max_profit {claimed:g} != sum over pairs {total:g}")
        return _result(violations, 3, cost=total)

    # スキル制約付きタスク割当: 担当者がスキルを持つこと。
    required = _int_list(instance.get("task_required_skill"))
    worker_skills = instance.get("worker_skills")
    if required is not None and worker_skills is not None:
        times = instance.get("processing_time") or []
        assignment = _assignment_map(solution)
        if assignment is None:
            return _unverified("skill assignment without assignment")
        chosen = _check_full_assignment(assignment, len(required), len(worker_skills), violations)
        loads = [0.0] * len(worker_skills)
        for task, worker in enumerate(chosen):
            if worker is None:
                continue
            skills = set(_int_list(worker_skills[worker]) or [])
            if required[task] not in skills:
                violations.append(f"worker {worker} lacks skill {required[task]} for task {task}")
            if worker < len(times) and task < len(times[worker]):
                loads[worker] += _num(times[worker][task]) or 0.0
        makespan = max(loads) if loads else 0.0
        claimed = _num(_pick(solution, "min_makespan", "makespan", "objective_value"))
        if claimed is not None and not _close(claimed, makespan):
            violations.append(f"min_makespan {claimed:g} != actual max load {makespan:g}")
        return _result(violations, len(required) + 2, cost=makespan)

    # ボトルネック割当: 順列であること。
    time_matrix = instance.get("time_matrix")
    if time_matrix is not None:
        size = int(_num(instance.get("num_agents")) or len(time_matrix))
        assignment = _assignment_map(solution)
        if assignment is None:
            return _unverified("bottleneck assignment without assignment")
        chosen = _check_full_assignment(assignment, size, size, violations)
        assigned = [a for a in chosen if a is not None]
        _distinct_indices(assigned, size, "assigned jobs", violations)
        # assignment は {エージェント: 仕事} なので、行がキー・列が値になる。
        worst = max(
            (
                _num(time_matrix[agent][job]) or 0.0
                for agent, job in enumerate(chosen)
                if job is not None and agent < len(time_matrix) and job < len(time_matrix[agent])
            ),
            default=0.0,
        )
        claimed = _num(_pick(solution, "min_bottleneck", "bottleneck", "objective_value"))
        if claimed is not None and not _close(claimed, worst):
            violations.append(f"min_bottleneck {claimed:g} != actual maximum {worst:g}")
        return _result(violations, 3, cost=worst)

    # 一般化割当: エージェントの資源容量。
    capacity = _num_list(instance.get("capacity"))
    resource = instance.get("resource")
    cost_matrix = instance.get("cost")
    if capacity is not None and resource is not None:
        num_tasks = int(_num(instance.get("num_tasks")) or 0)
        num_agents = int(_num(instance.get("num_agents")) or len(capacity))
        assignment = _assignment_map(solution)
        if assignment is None:
            return _unverified("generalized assignment without assignment")
        chosen = _check_full_assignment(assignment, num_tasks, num_agents, violations)
        used = [0.0] * num_agents
        total = 0.0
        for task, agent in enumerate(chosen):
            if agent is None:
                continue
            if agent < len(resource) and task < len(resource[agent]):
                used[agent] += _num(resource[agent][task]) or 0.0
            if cost_matrix and agent < len(cost_matrix) and task < len(cost_matrix[agent]):
                total += _num(cost_matrix[agent][task]) or 0.0
        for agent, load in enumerate(used):
            if agent < len(capacity) and load > capacity[agent] + _ABS_TOL:
                violations.append(
                    f"agent {agent} uses {load:g} resource, capacity is {capacity[agent]:g}"
                )
        claimed = _num(_pick(solution, "min_cost", "total_cost", "objective_value"))
        if claimed is not None and not _close(claimed, total):
            violations.append(f"min_cost {claimed:g} != sum of assignment costs {total:g}")
        return _result(violations, num_agents + 2, cost=total)

    return _unverified("unknown assignment IP shape")


@_register("割当・マッチング_線形計画")
def check_assignment_lp(instance: dict, solution: Any) -> dict:
    """割当問題: 一対一の完全割当で、費用が行列と一致する。"""
    violations: list[str] = []
    cost_matrix = instance.get("cost_matrix")
    if cost_matrix is None:
        return _unverified("assignment LP without a cost matrix")
    size = int(_num(instance.get("num_agents")) or len(cost_matrix))
    assignment = _assignment_map(solution)
    if assignment is None:
        return _unverified("assignment LP without assignment")
    chosen = _check_full_assignment(assignment, size, size, violations)
    assigned = [a for a in chosen if a is not None]
    _distinct_indices(assigned, size, "assigned jobs", violations)
    total = sum(
        _num(cost_matrix[i][j]) or 0.0
        for i, j in enumerate(chosen)
        if j is not None and i < len(cost_matrix) and j < len(cost_matrix[i])
    )
    claimed = _num(_pick(solution, "min_cost", "total_cost", "objective_value"))
    if claimed is not None and not _close(claimed, total):
        violations.append(f"min_cost {claimed:g} != sum of assignment costs {total:g}")
    return _result(violations, 3, cost=total)


@_register("割当・マッチング_組合せ最適化")
def check_matching_combinatorial(instance: dict, solution: Any) -> dict:
    """安定結婚・最大重み二部マッチング・二次割当。"""
    violations: list[str] = []

    # 安定結婚: 全単射かつブロッキングペアが存在しない。
    men = instance.get("men_preferences")
    women = instance.get("women_preferences")
    if men is not None and women is not None:
        matching = _index_map(_pick(solution, "stable_matching", "matching", "assignment"))
        if matching is None:
            return _unverified("stable marriage without a matching")
        pairs: dict[int, int] = {}
        for man, woman in matching.items():
            idx = _num(woman)
            if idx is None:
                violations.append(f"man {man} matched to non-index {woman}")
                continue
            pairs[man] = int(idx)
        _distinct_indices(list(pairs.values()), len(women), "matched women", violations)
        missing = sorted(set(range(len(men))) - set(pairs))
        if missing:
            violations.append(f"men without a partner: {missing}")
        wife_of = pairs
        husband_of = {w: m for m, w in pairs.items()}
        man_rank = [{w: r for r, w in enumerate(prefs)} for prefs in men]
        woman_rank = [{m: r for r, m in enumerate(prefs)} for prefs in women]
        for man, wife in wife_of.items():
            if man >= len(man_rank):
                continue
            for woman, rank in man_rank[man].items():
                if rank >= man_rank[man].get(wife, len(man_rank[man])):
                    continue
                husband = husband_of.get(woman)
                if husband is None or woman >= len(woman_rank):
                    continue
                if woman_rank[woman].get(man, len(woman_rank[woman])) < woman_rank[woman].get(
                    husband, len(woman_rank[woman])
                ):
                    violations.append(f"blocking pair (man {man}, woman {woman})")
        return _result(violations, len(men) + 2)

    # 二次割当: 配置が順列で、費用が flow x distance と一致する。
    flow = instance.get("flow")
    distance = instance.get("distance")
    if flow is not None and distance is not None:
        size = int(_num(instance.get("num_facilities")) or len(flow))
        placement = _int_list(_pick(solution, "placement", "assignment", "permutation"))
        if placement is None:
            return _unverified("QAP without a placement")
        _distinct_indices(placement, size, "placement", violations)
        if len(placement) != size:
            violations.append(f"placement has {len(placement)} entries, expected {size}")
        total = 0.0
        for i, loc_i in enumerate(placement):
            for j, loc_j in enumerate(placement):
                if (
                    i < len(flow)
                    and j < len(flow[i])
                    and loc_i < len(distance)
                    and loc_j < len(distance[loc_i])
                ):
                    total += (_num(flow[i][j]) or 0.0) * (_num(distance[loc_i][loc_j]) or 0.0)
        claimed = _num(_pick(solution, "min_cost", "total_cost", "objective_value"))
        if claimed is not None and not _close(claimed, total):
            violations.append(f"min_cost {claimed:g} != flow x distance total {total:g}")
        return _result(violations, 3, cost=total)

    # 最大重み二部マッチング: 実在する辺で、左右とも重複しない。
    edges = instance.get("edges")
    if edges is not None:
        matching = _pick(solution, "matching", "pairs", "assignment")
        if not isinstance(matching, list):
            return _unverified("bipartite matching without a matching list")
        lookup = {}
        for edge in edges:
            if isinstance(edge, dict):
                lookup[(edge.get("left"), edge.get("right"))] = _num(edge.get("weight")) or 0.0
        pairs = _pair_list(matching, ("left", "l", "u", "worker"), ("right", "r", "v", "task"))
        if pairs is None:
            return _unverified("bipartite matching entries are not (left, right) pairs")
        lefts: list[int] = []
        rights: list[int] = []
        total = 0.0
        for left, right in pairs:
            if (left, right) not in lookup:
                violations.append(f"edge ({left}, {right}) does not exist")
                continue
            lefts.append(left)
            rights.append(right)
            total += lookup[(left, right)]
        _distinct_indices(
            lefts, int(_num(instance.get("num_left")) or 0), "matched left", violations
        )
        _distinct_indices(
            rights, int(_num(instance.get("num_right")) or 0), "matched right", violations
        )
        claimed = _num(_pick(solution, "max_weight", "total_weight", "objective_value"))
        if claimed is not None and not _close(claimed, total):
            violations.append(f"max_weight {claimed:g} != sum of matched weights {total:g}")
        return _result(violations, 3, cost=total)

    return _unverified("unknown matching shape")


# ============================================================
# 複合・グラフ最適化
# ============================================================


@_register("複合・グラフ最適化_整数計画")
def check_composite_ip(instance: dict, solution: Any) -> dict:
    """グラフ彩色・最大重み独立集合・容量制約被覆・並列機械・最大クリーク。"""
    violations: list[str] = []
    edges = instance.get("edges") or []
    num_nodes = int(_num(instance.get("num_nodes")) or 0)

    # 容量制約付き被覆割当: 半径内被覆と、センター容量。
    zones = instance.get("zones")
    centers = instance.get("centers")
    if zones is not None and centers is not None:
        radius = _num(instance.get("radius")) or 0.0
        demand = _num_list(instance.get("demand")) or []
        capacity = _num_list(instance.get("capacity")) or []
        fixed_cost = _num_list(instance.get("fixed_cost")) or []
        opened = _int_list(_pick(solution, "opened_centers", "centers", "opened", "selected"))
        if opened is None:
            return _unverified("capacitated covering without opened centers")
        _distinct_indices(opened, len(centers), "opened_centers", violations)
        covered = _covered_points(zones, centers, radius, opened)
        missing = sorted(set(range(len(zones))) - covered)
        if missing:
            violations.append(f"zones not covered within radius: {missing}")
        available = sum(capacity[i] for i in opened if 0 <= i < len(capacity))
        if available + _ABS_TOL < sum(demand):
            violations.append(f"opened capacity {available:g} below total demand {sum(demand):g}")
        spend = sum(fixed_cost[i] for i in opened if 0 <= i < len(fixed_cost))
        claimed = _num(_pick(solution, "min_cost", "total_cost", "objective_value"))
        if claimed is not None and not _close(claimed, spend):
            violations.append(f"min_cost {claimed:g} != opened fixed cost {spend:g}")
        return _result(violations, 4, cost=spend)

    # 非関連並列機械: 全ジョブを機械へ割当て、メイクスパンが最大負荷と一致。
    times = instance.get("processing_time")
    if times is not None:
        num_jobs = int(_num(instance.get("num_jobs")) or 0)
        num_machines = int(_num(instance.get("num_machines")) or len(times))
        assignment = _assignment_map(solution)
        if assignment is None:
            return _unverified("parallel machine scheduling without assignment")
        chosen = _check_full_assignment(assignment, num_jobs, num_machines, violations)
        loads = [0.0] * num_machines
        for job, machine in enumerate(chosen):
            if machine is None:
                continue
            if machine < len(times) and job < len(times[machine]):
                loads[machine] += _num(times[machine][job]) or 0.0
        makespan = max(loads) if loads else 0.0
        claimed = _num(_pick(solution, "min_makespan", "makespan", "objective_value"))
        if claimed is not None and not _close(claimed, makespan):
            violations.append(f"min_makespan {claimed:g} != actual max load {makespan:g}")
        return _result(violations, 3, cost=makespan)

    adjacency = _undirected_adjacency(edges)

    # 最大重み独立集合: 選択頂点間に辺がない。
    weights = _num_list(instance.get("weights"))
    if weights is not None:
        selected = _int_list(_pick(solution, "selected_nodes", "independent_set", "selected"))
        if selected is None:
            return _unverified("independent set without selected nodes")
        _distinct_indices(selected, num_nodes, "selected_nodes", violations)
        picked = set(selected)
        for pair in adjacency:
            if pair <= picked:
                violations.append(f"edge {sorted(pair)} inside the independent set")
        total = sum(weights[i] for i in selected if 0 <= i < len(weights))
        claimed = _num(_pick(solution, "max_weight", "total_weight", "objective_value"))
        if claimed is not None and not _close(claimed, total):
            violations.append(f"max_weight {claimed:g} != sum of selected weights {total:g}")
        return _result(violations, 3, cost=total)

    # 最大クリーク: 選択頂点が互いに隣接している。
    clique = _int_list(_pick(solution, "clique", "max_clique", "nodes"))
    if clique is not None:
        _distinct_indices(clique, num_nodes, "clique", violations)
        for i, a in enumerate(clique):
            for b in clique[i + 1 :]:
                if frozenset((a, b)) not in adjacency:
                    violations.append(f"clique members {a} and {b} are not adjacent")
        claimed = _num(_pick(solution, "max_clique_size", "clique_size", "objective_value"))
        if claimed is not None and not _close(claimed, len(clique)):
            violations.append(f"max_clique_size {claimed:g} != {len(clique)} members")
        return _result(violations, 3, cost=float(len(clique)))

    # グラフ彩色: 隣接頂点が異なる色を持つ。
    coloring = _index_map(_pick(solution, "coloring", "colors", "assignment"))
    if coloring is not None:
        missing = sorted(set(range(num_nodes)) - set(coloring))
        if missing:
            violations.append(f"nodes without a color: {missing}")
        for pair in adjacency:
            head, tail = tuple(pair) if len(pair) == 2 else (None, None)
            if head is None:
                continue
            if head in coloring and tail in coloring and coloring[head] == coloring[tail]:
                violations.append(f"adjacent nodes {head} and {tail} share color {coloring[head]}")
        used = len({str(c) for c in coloring.values()})
        claimed = _num(_pick(solution, "min_colors", "num_colors", "objective_value"))
        if claimed is not None and not _close(claimed, used):
            violations.append(f"min_colors {claimed:g} != {used} distinct colors used")
        return _result(violations, 3, cost=float(used))

    return _unverified("unknown composite IP shape")


@_register("複合・グラフ最適化_混合整数計画")
def check_composite_milp(instance: dict, solution: Any) -> dict:
    """立地配送複合・時間枠付き巡回路・発電機起動停止計画。"""
    violations: list[str] = []

    # 発電機起動停止: 各期の供給が需要以上で、出力が min/max の範囲内。
    units = instance.get("units")
    if units is not None:
        demand = _num_list(instance.get("demand")) or []
        schedule = _index_map(_pick(solution, "schedule", "plan", "commitment"))
        if schedule is None:
            return _unverified("unit commitment without a schedule")
        # 各号機の系列は {"on": [...], "output": [...]}、出力だけの配列、
        # [{"period": t, "output": o}] のいずれでも返ってくる。
        normalised: dict[int, tuple[list[float], list[float]]] = {}
        for unit_id, entry in schedule.items():
            if isinstance(entry, dict) and ("output" in entry or "on" in entry):
                output = _num_list(entry.get("output")) or []
                on = _num_list(entry.get("on")) or []
            else:
                output = _period_series(entry, ("output", "production", "power")) or []
                on = _period_series(entry, ("on", "running", "committed")) or []
            if not output:
                return _unverified("unit commitment schedule has no output series")
            normalised[unit_id] = (output, on)

        supply = [0.0] * len(demand)
        for unit_id, (output, on) in normalised.items():
            if not 0 <= unit_id < len(units):
                violations.append(f"unknown unit {unit_id}")
                continue
            low = _num(units[unit_id].get("min")) or 0.0
            high = _num(units[unit_id].get("max")) or 0.0
            for period, produced in enumerate(output):
                # on が無い出力だけの系列では、正の出力を稼働とみなす。
                running = bool(on[period]) if period < len(on) else produced > _ABS_TOL
                if running and not (low - _ABS_TOL <= produced <= high + _ABS_TOL):
                    violations.append(
                        f"unit {unit_id} period {period} output {produced:g} outside "
                        f"[{low:g}, {high:g}]"
                    )
                if not running and abs(produced) > _ABS_TOL:
                    violations.append(
                        f"unit {unit_id} period {period} produces {produced:g} while off"
                    )
                if period < len(supply):
                    supply[period] += produced
        for period, needed in enumerate(demand):
            if supply[period] + _ABS_TOL < needed:
                violations.append(
                    f"period {period} supply {supply[period]:g} below demand {needed:g}"
                )
        spend = 0.0
        for unit_id, (output, on) in normalised.items():
            if not 0 <= unit_id < len(units):
                continue
            unit = units[unit_id]
            spend += (_num(unit.get("cost")) or 0.0) * sum(output)
            previous = False
            for period, produced in enumerate(output):
                running = bool(on[period]) if period < len(on) else produced > _ABS_TOL
                if running and not previous:
                    spend += _num(unit.get("startup")) or 0.0
                previous = running
        claimed = _num(_pick(solution, "min_cost", "total_cost", "objective_value"))
        if claimed is not None and not _close_soft(claimed, spend):
            violations.append(f"min_cost {claimed:g} != cost of this schedule {spend:g}")
        return _result(violations, len(demand) + len(units) + 1, cost=spend)

    # 時間枠付き巡回路: 全ノードを1回ずつ回り、各ノードの時間枠を守る。
    time_windows = instance.get("time_windows")
    if time_windows is not None:
        distance = instance.get("distance") or []
        service = _num_list(instance.get("service_time")) or []
        num_nodes = int(_num(instance.get("num_nodes")) or len(time_windows))
        tour = _int_list(_pick(solution, "tour", "route", "sequence", "path"))
        if tour is None:
            return _unverified("TSPTW without a tour")
        visited = tour[:-1] if len(tour) > 1 and tour[0] == tour[-1] else tour
        _distinct_indices(visited, num_nodes, "tour", violations)
        if set(visited) != set(range(num_nodes)):
            violations.append(f"tour visits {len(set(visited))} of {num_nodes} nodes")
        clock = 0.0
        total = 0.0
        for head, tail in pairwise(visited):
            leg = (
                _num(distance[head][tail]) or 0.0
                if head < len(distance) and tail < len(distance[head])
                else 0.0
            )
            total += leg
            clock += leg
            if tail < len(time_windows):
                early, late = time_windows[tail][0], time_windows[tail][1]
                clock = max(clock, _num(early) or 0.0)
                if clock > (_num(late) or 0.0) + _ABS_TOL:
                    violations.append(f"node {tail} served at {clock:g}, window closes at {late}")
            if tail < len(service):
                clock += service[tail]
        claimed = _num(_pick(solution, "min_distance", "total_distance", "objective_value"))
        if claimed is not None and claimed + _ABS_TOL < total:
            violations.append(f"min_distance {claimed:g} below tour length {total:g}")
        return _result(violations, num_nodes + 2, cost=total)

    # 立地配送複合: 開設施設への割当と、車両容量。
    facilities = instance.get("facilities")
    if facilities is not None:
        demand = _num_list(instance.get("demand")) or []
        opened = _int_list(_pick(solution, "opened_facilities", "facilities", "opened"))
        assignment = _index_map(_pick(solution, "assignment", "allocation"))
        if opened is None or assignment is None:
            return _unverified("location routing without opened facilities or assignment")
        _distinct_indices(opened, len(facilities), "opened_facilities", violations)
        picked = set(opened)
        for customer, facility in assignment.items():
            idx = _num(facility)
            if idx is None or int(idx) not in picked:
                violations.append(f"customer {customer} assigned to closed facility {facility}")
        missing = sorted(set(range(len(demand))) - set(assignment))
        if missing:
            violations.append(f"customers without assignment: {missing}")
        vehicle_capacity = _num(instance.get("vehicle_capacity"))
        per_facility: dict[int, float] = {}
        for customer, facility in assignment.items():
            idx = _num(facility)
            if idx is None or customer >= len(demand):
                continue
            per_facility[int(idx)] = per_facility.get(int(idx), 0.0) + demand[customer]
        vehicles = 0
        if vehicle_capacity:
            for facility, load in per_facility.items():
                needed = math.ceil(load / vehicle_capacity - _ABS_TOL)
                if needed < 1:
                    violations.append(f"facility {facility} serves {load:g} with no vehicle")
                vehicles += max(needed, 0)
        distance = instance.get("distance") or []
        fixed_cost = _num_list(instance.get("fixed_cost")) or []
        legs = 0.0
        for customer, facility in assignment.items():
            idx = _num(facility)
            if idx is None:
                continue
            row = int(idx)
            if row < len(distance) and customer < len(distance[row]):
                legs += _num(distance[row][customer]) or 0.0
        # 配送は施設と顧客の往復で数える（参照解の費用と一致する定義）。
        spend = (
            sum(fixed_cost[i] for i in opened if 0 <= i < len(fixed_cost))
            + 2 * legs
            + vehicles * (_num(instance.get("vehicle_cost")) or 0.0)
        )
        claimed = _num(_pick(solution, "min_total_cost", "total_cost", "objective_value"))
        if claimed is not None and not _close_soft(claimed, spend):
            violations.append(f"min_total_cost {claimed:g} != cost of this plan {spend:g}")
        return _result(violations, 4, cost=spend)

    return _unverified("unknown composite MILP shape")


@_register("複合・グラフ最適化_線形計画")
def check_composite_lp(instance: dict, solution: Any) -> dict:
    """在庫配送複合・生産流通複合・多段供給連鎖。"""
    violations: list[str] = []
    demand = instance.get("demand")

    # 多段供給連鎖: 目的値のみが返るため、最安経路による下界だけ検証する。
    if "num_warehouses" in instance:
        needs = _num_list(demand) or []
        plant_warehouse = instance.get("cost_plant_warehouse") or []
        warehouse_customer = instance.get("cost_warehouse_customer") or []
        claimed = _num(_pick(solution, "min_total_cost", "total_cost", "objective_value"))
        if claimed is None:
            return _unverified("multi-echelon without a cost value")
        bound = 0.0
        for j, need in enumerate(needs):
            options = [
                (_num(plant_warehouse[p][w]) or 0.0) + (_num(warehouse_customer[w][j]) or 0.0)
                for p in range(len(plant_warehouse))
                for w in range(len(warehouse_customer))
                if w < len(plant_warehouse[p]) and j < len(warehouse_customer[w])
            ]
            if options:
                bound += need * min(options)
        if claimed + _ABS_TOL < bound:
            violations.append(f"min_total_cost {claimed:g} below lower bound {bound:g}")
        capacity = _num_list(instance.get("plant_capacity")) or []
        if sum(capacity) + _ABS_TOL < sum(needs):
            violations.append("plant capacity below total demand")
        return _result(violations, 2, cost=claimed)

    # 生産流通複合: 出荷が需要を満たし、工場能力を超えない。
    if "num_plants" in instance and "transport_cost" in instance:
        needs = _num_list(demand) or []
        capacity = _num_list(instance.get("capacity")) or []
        raw = _pick(solution, "shipments", "shipment", "plan")
        # 工場×顧客の行列、入れ子 dict、[{plant, customer, amount}] のいずれもある。
        flows: dict[int, dict[int, float]] = {}
        if isinstance(raw, list) and raw and all(isinstance(e, dict) for e in raw):
            if all("plant" in e or "from" in e for e in raw):
                for entry in raw:
                    plant = _num(entry.get("plant", entry.get("from")))
                    customer = _num(entry.get("customer", entry.get("to")))
                    amount = next(
                        (
                            _num(entry[k])
                            for k in ("amount", "quantity", "qty", "flow")
                            if k in entry
                        ),
                        None,
                    )
                    if plant is None or customer is None or amount is None:
                        return _unverified("shipment entries lack plant, customer or amount")
                    flows.setdefault(int(plant), {})[int(customer)] = amount
            else:
                return _unverified("shipment entries are not plant/customer records")
        else:
            outer = _index_map(raw)
            if outer is None:
                return _unverified("production distribution without shipments")
            for plant, row in outer.items():
                per_customer = _index_map(row)
                if per_customer is None:
                    return _unverified("shipment rows are not customer mappings")
                flows[plant] = {c: _num(q) or 0.0 for c, q in per_customer.items()}

        received = [0.0] * len(needs)
        for plant, row in flows.items():
            shipped = 0.0
            for customer, qty in row.items():
                if qty < -_ABS_TOL:
                    violations.append(f"negative shipment {qty:g} from plant {plant}")
                shipped += qty
                if 0 <= customer < len(received):
                    received[customer] += qty
            if 0 <= plant < len(capacity) and shipped > capacity[plant] + _ABS_TOL:
                violations.append(
                    f"plant {plant} ships {shipped:g}, capacity is {capacity[plant]:g}"
                )
        for customer, need in enumerate(needs):
            if received[customer] + _ABS_TOL < need:
                violations.append(
                    f"customer {customer} receives {received[customer]:g} < demand {need:g}"
                )
        production_cost = _num_list(instance.get("production_cost")) or []
        transport_cost = instance.get("transport_cost") or []
        spend = 0.0
        for plant, row in flows.items():
            for customer, qty in row.items():
                unit = production_cost[plant] if 0 <= plant < len(production_cost) else 0.0
                if 0 <= plant < len(transport_cost) and customer < len(transport_cost[plant]):
                    unit += _num(transport_cost[plant][customer]) or 0.0
                spend += qty * unit
        claimed = _num(_pick(solution, "min_total_cost", "total_cost", "objective_value"))
        if claimed is not None and not _close_soft(claimed, spend):
            violations.append(f"min_total_cost {claimed:g} != cost of these shipments {spend:g}")
        return _result(violations, len(needs) + len(capacity) + 1, cost=spend)

    # 在庫配送複合: 生産量が能力以内で、累積生産が累積需要を満たす。
    if "num_retailers" in instance:
        rows = demand or []
        totals = [sum(_num(v) or 0.0 for v in col) for col in zip(*rows)] if rows else []
        plan = _pick(solution, "plan", "production_plan", "schedule")
        production = (
            _num_list(plan.get("production")) if isinstance(plan, dict) else _num_list(plan)
        )
        if production is None:
            return _unverified("inventory distribution without a production plan")
        capacity = _num(instance.get("production_capacity"))
        for period, produced in enumerate(production):
            if produced < -_ABS_TOL:
                violations.append(f"period {period} production {produced:g} is negative")
            if capacity is not None and produced > capacity + _ABS_TOL:
                violations.append(
                    f"period {period} production {produced:g} exceeds capacity {capacity:g}"
                )
        _cumulative_supply_ok(production, totals, "production", violations)
        shipments = _index_map(plan.get("shipments")) if isinstance(plan, dict) else None
        if shipments is None:
            return _result(violations, len(production) + 1, cost=sum(production))
        ship_cost = _num_list(instance.get("ship_cost")) or []
        periods = int(_num(instance.get("periods")) or len(production))
        spend = (_num(instance.get("prod_cost")) or 0.0) * sum(production)
        for retailer, series in shipments.items():
            quantities = _num_list(series) or []
            if retailer < len(ship_cost):
                spend += ship_cost[retailer] * sum(quantities)
            inventory = 0.0
            row = rows[retailer] if retailer < len(rows) else []
            for period in range(periods):
                shipped = quantities[period] if period < len(quantities) else 0.0
                needed = _num(row[period]) or 0.0 if period < len(row) else 0.0
                inventory += shipped - needed
                if inventory < -_ABS_TOL:
                    violations.append(
                        f"retailer {retailer} short by {-inventory:g} at period {period}"
                    )
                spend += (_num(instance.get("holding_retailer")) or 0.0) * inventory
        warehouse_inventory = 0.0
        for period in range(periods):
            shipped_out = sum(
                (_num_list(series) or [0.0] * periods)[period]
                for series in shipments.values()
                if period < len(_num_list(series) or [])
            )
            warehouse_inventory += (
                production[period] if period < len(production) else 0.0
            ) - shipped_out
            spend += (_num(instance.get("holding_warehouse")) or 0.0) * warehouse_inventory
        claimed = _num(_pick(solution, "min_total_cost", "total_cost", "objective_value"))
        if claimed is not None and not _close_soft(claimed, spend):
            violations.append(f"min_total_cost {claimed:g} != cost of this plan {spend:g}")
        return _result(violations, len(production) + 2, cost=spend)

    return _unverified("unknown composite LP shape")


@_register("複合・グラフ最適化_組合せ最適化")
def check_tsp(instance: dict, solution: Any) -> dict:
    """巡回セールスマン: 全都市を1回ずつ訪れ、距離が行列と一致する。"""
    violations: list[str] = []
    matrix = instance.get("distance_matrix") or instance.get("distance")
    if matrix is None:
        return _unverified("TSP without a distance matrix")
    num_cities = int(_num(instance.get("num_cities")) or len(matrix))
    tour = _int_list(_pick(solution, "tour", "route", "sequence", "path"))
    if tour is None:
        return _unverified("TSP without a tour")
    visited = tour[:-1] if len(tour) > 1 and tour[0] == tour[-1] else tour
    _distinct_indices(visited, num_cities, "tour", violations)
    if set(visited) != set(range(num_cities)):
        violations.append(f"tour visits {len(set(visited))} of {num_cities} cities")
    total = 0.0
    closed = [*visited, visited[0]] if visited else []
    for head, tail in pairwise(closed):
        if head < len(matrix) and tail < len(matrix[head]):
            total += _num(matrix[head][tail]) or 0.0
    claimed = _num(_pick(solution, "min_distance", "total_distance", "objective_value"))
    if claimed is not None and not _close(claimed, total):
        violations.append(f"min_distance {claimed:g} != tour length {total:g}")
    return _result(violations, 3, cost=total)


# ============================================================
# 生産・在庫計画
# ============================================================


@_register("生産・在庫計画_線形計画")
def check_production_lp(instance: dict, solution: Any) -> dict:
    """生産ミックス・配合・多期間生産・残業外注つき生産計画。"""
    violations: list[str] = []

    # 配合問題: 合計量と純度下限。
    materials = instance.get("materials")
    if materials is not None:
        amount = _num(instance.get("amount")) or 0.0
        min_purity = _num(instance.get("min_purity")) or 0.0
        purity = instance.get("purity") or {}
        cost = instance.get("cost") or {}
        blend = _pick(solution, "blend", "mix", "amounts")
        if not isinstance(blend, dict):
            return _unverified("blending without a blend mapping")
        total = 0.0
        weighted_purity = 0.0
        spend = 0.0
        for name, qty_raw in blend.items():
            qty = _num(qty_raw) or 0.0
            if qty < -_ABS_TOL:
                violations.append(f"material {name} quantity {qty:g} is negative")
            total += qty
            weighted_purity += qty * (_num(purity.get(name)) or 0.0)
            spend += qty * (_num(cost.get(name)) or 0.0)
        if not _close_soft(total, amount):
            violations.append(f"blend total {total:g} != required amount {amount:g}")
        if total > 0 and not _le_soft(min_purity, weighted_purity / total):
            violations.append(
                f"blend purity {weighted_purity / total:g} below minimum {min_purity:g}"
            )
        claimed = _num(_pick(solution, "min_cost", "total_cost", "objective_value"))
        if claimed is not None and not _close_soft(claimed, spend):
            violations.append(f"min_cost {claimed:g} != blend cost {spend:g}")
        return _result(violations, 4, cost=spend)

    # 生産ミックス: 資源使用量が在庫を超えない。
    resource_usage = instance.get("resource_usage")
    if resource_usage is not None:
        profit = _num_list(instance.get("profit")) or []
        available = _num_list(instance.get("resource_available")) or []
        production = _num_list(_pick(solution, "production", "plan", "quantities"))
        if production is None:
            return _unverified("product mix without production quantities")
        for product, qty in enumerate(production):
            if qty < -_ABS_TOL:
                violations.append(f"product {product} quantity {qty:g} is negative")
        for resource, limit in enumerate(available):
            used = sum(
                (_num(resource_usage[resource][p]) or 0.0) * qty
                for p, qty in enumerate(production)
                if resource < len(resource_usage) and p < len(resource_usage[resource])
            )
            if not _le_soft(used, limit):
                violations.append(f"resource {resource} usage {used:g} exceeds available {limit:g}")
        total = sum(p * q for p, q in zip(profit, production))
        claimed = _num(_pick(solution, "max_profit", "total_profit", "objective_value"))
        if claimed is not None and not _close_soft(claimed, total):
            violations.append(f"max_profit {claimed:g} != profit of this plan {total:g}")
        return _result(violations, len(available) + 2, cost=total)

    demand = _num_list(instance.get("demand")) or []

    # 残業・外注つき生産計画: 各期の区分ごとの上限。
    regular_cap = _num(instance.get("regular_cap"))
    if regular_cap is not None:
        overtime_cap = _num(instance.get("overtime_cap")) or 0.0
        plan = _pick(solution, "plan", "production_plan", "schedule")
        if not isinstance(plan, list) or not plan:
            return _unverified("production plan without per-period entries")
        supply: list[float] = []
        for entry in plan:
            if not isinstance(entry, dict):
                return _unverified("production plan entries are not objects")
            regular = _num(entry.get("regular")) or 0.0
            overtime = _num(entry.get("overtime")) or 0.0
            subcontract = _num(entry.get("subcontract")) or 0.0
            period = int(_num(entry.get("period")) or len(supply))
            if regular > regular_cap + _ABS_TOL:
                violations.append(
                    f"period {period} regular {regular:g} exceeds cap {regular_cap:g}"
                )
            if overtime > overtime_cap + _ABS_TOL:
                violations.append(
                    f"period {period} overtime {overtime:g} exceeds cap {overtime_cap:g}"
                )
            if min(regular, overtime, subcontract) < -_ABS_TOL:
                violations.append(f"period {period} has a negative quantity")
            supply.append(regular + overtime + subcontract)
        _cumulative_supply_ok(supply, demand, "production", violations)
        holding = _num(instance.get("holding_cost")) or 0.0
        rates = (
            _num(instance.get("regular_cost")) or 0.0,
            _num(instance.get("overtime_cost")) or 0.0,
            _num(instance.get("subcontract_cost")) or 0.0,
        )
        inventory = 0.0
        spend = 0.0
        for period, entry in enumerate(plan):
            quantities = (
                _num(entry.get("regular")) or 0.0,
                _num(entry.get("overtime")) or 0.0,
                _num(entry.get("subcontract")) or 0.0,
            )
            spend += sum(rate * qty for rate, qty in zip(rates, quantities))
            inventory += sum(quantities) - (demand[period] if period < len(demand) else 0.0)
            spend += holding * inventory
        claimed = _num(_pick(solution, "min_cost", "total_cost", "objective_value"))
        if claimed is not None and not _close_soft(claimed, spend):
            violations.append(f"min_cost {claimed:g} != cost of this plan {spend:g}")
        return _result(violations, len(plan) + 2, cost=spend)

    # 多期間生産計画: 期別能力と累積需要充足。
    capacity = _num(instance.get("production_capacity"))
    if capacity is not None:
        production = _num_list(_pick(solution, "production_plan", "plan", "production"))
        if production is None:
            return _unverified("multi-period production without a plan")
        periods = int(_num(instance.get("periods")) or len(demand))
        if len(production) != periods:
            violations.append(f"plan has {len(production)} periods, expected {periods}")
        for period, produced in enumerate(production):
            if produced < -_ABS_TOL:
                violations.append(f"period {period} production {produced:g} is negative")
            if produced > capacity + _ABS_TOL:
                violations.append(
                    f"period {period} production {produced:g} exceeds capacity {capacity:g}"
                )
        _cumulative_supply_ok(production, demand, "production", violations)
        unit_cost = _num(instance.get("unit_cost")) or 0.0
        holding = _num(instance.get("holding_cost")) or 0.0
        inventory = 0.0
        spend = 0.0
        for period, produced in enumerate(production):
            spend += unit_cost * produced
            inventory += produced - (demand[period] if period < len(demand) else 0.0)
            spend += holding * inventory
        claimed = _num(_pick(solution, "min_cost", "total_cost", "objective_value"))
        if claimed is not None and not _close_soft(claimed, spend):
            violations.append(f"min_cost {claimed:g} != cost of this plan {spend:g}")
        return _result(violations, periods + 3, cost=spend)

    return _unverified("unknown production LP shape")


@_register("生産・在庫計画_整数計画")
def check_production_ip(instance: dict, solution: Any) -> dict:
    """労働力計画と発注ロット最適化。"""
    violations: list[str] = []

    # 労働力計画: 各期の人員が必要数以上で、増減が前期と整合する。
    requirement = _num_list(instance.get("requirement"))
    if requirement is not None:
        initial = _num(instance.get("initial_workforce")) or 0.0
        plan = _pick(solution, "plan", "schedule", "workforce_plan")
        # 期別の人員だけを配列で返す実装と、採用・解雇まで含む dict 配列がある。
        levels = _period_series(plan, ("workforce", "staff", "headcount", "employees"))
        if levels is None:
            return _unverified("workforce plan without a per-period workforce series")
        entries = plan if isinstance(plan, list) and all(isinstance(e, dict) for e in plan) else []
        previous = initial
        spend = 0.0
        for period, workforce in enumerate(levels):
            if period < len(requirement) and workforce + _ABS_TOL < requirement[period]:
                violations.append(
                    f"period {period} workforce {workforce:g} below requirement "
                    f"{requirement[period]:g}"
                )
            entry = entries[period] if period < len(entries) else {}
            hire = next((_num(entry[k]) for k in ("hire", "hires", "hired") if k in entry), None)
            fire = next((_num(entry[k]) for k in ("fire", "fires", "fired") if k in entry), None)
            # 採用・解雇が省略された解では、人員の増減から復元する。
            derived = workforce - previous
            if hire is None:
                hire = max(derived, 0.0)
            if fire is None:
                fire = max(-derived, 0.0)
            if hire < -_ABS_TOL or fire < -_ABS_TOL:
                violations.append(f"period {period} has a negative hire/fire count")
            if not _close_soft(workforce, previous + hire - fire):
                violations.append(
                    f"period {period} workforce {workforce:g} != previous {previous:g} "
                    f"+ hire {hire:g} - fire {fire:g}"
                )
            spend += (
                (_num(instance.get("hire_cost")) or 0.0) * hire
                + (_num(instance.get("fire_cost")) or 0.0) * fire
                + (_num(instance.get("wage")) or 0.0) * workforce
            )
            previous = workforce
        claimed = _num(_pick(solution, "min_cost", "total_cost", "objective_value"))
        if claimed is not None and not _close_soft(claimed, spend):
            violations.append(f"min_cost {claimed:g} != cost of this plan {spend:g}")
        return _result(violations, len(levels) + 2, cost=spend)

    # 発注ロット最適化: 選択ロットが候補にあり、累積供給が需要を満たす。
    lot_choices = _num_list(instance.get("lot_choices"))
    if lot_choices is not None:
        demand = _num_list(instance.get("demand")) or []
        chosen = _index_map(_pick(solution, "chosen_lot_sizes", "lot_sizes", "plan"))
        if chosen is None:
            return _unverified("lot sizing without chosen lot sizes")
        # Why not check cumulative supply: 解には期ごとのロットサイズしか入らず、
        # 発注回数は別変数なので供給量を復元できない。
        allowed = set(lot_choices)
        for period, size_raw in chosen.items():
            size = _num(size_raw)
            if size is None:
                violations.append(f"period {period} lot size is not numeric")
                continue
            if size not in allowed:
                violations.append(f"period {period} lot size {size:g} not in lot_choices")
        missing = sorted({p for p, d in enumerate(demand) if d > 0} - set(chosen))
        if missing:
            violations.append(f"periods with demand but no lot size: {missing}")
        return _result(violations, len(demand) + 1)

    return _unverified("unknown production IP shape")


@_register("生産・在庫計画_混合整数計画")
def check_lot_sizing_milp(instance: dict, solution: Any) -> dict:
    """多期間ロットサイジング: 累積生産が累積需要を満たす。"""
    violations: list[str] = []
    demand = _num_list(instance.get("demand")) or []
    periods = int(_num(instance.get("periods")) or len(demand))
    production = _num_list(_pick(solution, "production_plan", "plan", "production"))
    if production is None:
        return _unverified("lot sizing without a production plan")
    if len(production) != periods:
        violations.append(f"plan has {len(production)} periods, expected {periods}")
    for period, produced in enumerate(production):
        if produced < -_ABS_TOL:
            violations.append(f"period {period} production {produced:g} is negative")
    _cumulative_supply_ok(production, demand, "production", violations)
    if not _close(sum(production), sum(demand)) and sum(production) + _ABS_TOL < sum(demand):
        violations.append("total production below total demand")
    setup = _num(instance.get("setup_cost")) or 0.0
    unit_cost = _num(instance.get("unit_cost")) or 0.0
    holding = _num(instance.get("holding_cost")) or 0.0
    inventory = 0.0
    spend = 0.0
    for period, produced in enumerate(production):
        if produced > _ABS_TOL:
            spend += setup
        spend += unit_cost * produced
        inventory += produced - (demand[period] if period < len(demand) else 0.0)
        spend += holding * inventory
    claimed = _num(_pick(solution, "min_cost", "total_cost", "objective_value"))
    if claimed is not None and not _close_soft(claimed, spend):
        violations.append(f"min_cost {claimed:g} != cost of this plan {spend:g}")
    return _result(violations, periods + 3, cost=spend)


@_register("生産・在庫計画_確率最適化")
def check_newsvendor(instance: dict, solution: Any) -> dict:
    """新聞売り子問題: 申告した発注量に対する期待利益が計算値と一致する。"""
    violations: list[str] = []
    price = _num(instance.get("price"))
    cost = _num(instance.get("cost"))
    salvage = _num(instance.get("salvage"))
    values = _num_list(instance.get("demand_values"))
    probs = _num_list(instance.get("demand_probs"))
    if None in (price, cost, salvage) or values is None or probs is None:
        return _unverified("newsvendor without full parameters")
    quantity = _num(_pick(solution, "optimal_order_quantity", "order_quantity", "quantity"))
    if quantity is None:
        return _unverified("newsvendor without an order quantity")
    if quantity < -_ABS_TOL:
        violations.append(f"order quantity {quantity:g} is negative")
    expected = sum(
        prob
        * (price * min(quantity, value) + salvage * max(quantity - value, 0.0) - cost * quantity)
        for value, prob in zip(values, probs)
    )
    claimed = _num(_pick(solution, "expected_profit", "profit", "objective_value"))
    if claimed is not None and not _close(claimed, expected):
        violations.append(
            f"expected_profit {claimed:g} != expectation for q={quantity:g} ({expected:g})"
        )
    return _result(violations, 2, cost=expected)


# ============================================================
# 金融・投資
# ============================================================


@_register("金融・投資_整数計画")
def check_finance_ip(instance: dict, solution: Any) -> dict:
    """資本予算問題と収益管理（座席配分）。"""
    violations: list[str] = []

    # 収益管理: 各クラスの配分が需要以内で、合計が座席数以内。
    classes = instance.get("classes")
    if classes is not None:
        capacity = _num(instance.get("capacity")) or 0.0
        allocation = _pick(solution, "allocation", "seats", "assignment")
        if not isinstance(allocation, dict):
            return _unverified("revenue management without an allocation")
        by_name = {str(entry.get("name")): entry for entry in classes}
        total_seats = 0.0
        revenue = 0.0
        for name, seats_raw in allocation.items():
            seats = _num(seats_raw) or 0.0
            entry = by_name.get(str(name))
            if entry is None:
                violations.append(f"unknown fare class {name}")
                continue
            limit = _num(entry.get("demand")) or 0.0
            if seats > limit + _ABS_TOL:
                violations.append(f"class {name} allocates {seats:g} above demand {limit:g}")
            if seats < -_ABS_TOL:
                violations.append(f"class {name} allocation {seats:g} is negative")
            total_seats += seats
            revenue += seats * (_num(entry.get("price")) or 0.0)
        if total_seats > capacity + _ABS_TOL:
            violations.append(f"total seats {total_seats:g} exceed capacity {capacity:g}")
        claimed = _num(_pick(solution, "max_revenue", "total_revenue", "objective_value"))
        if claimed is not None and not _close(claimed, revenue):
            violations.append(f"max_revenue {claimed:g} != revenue of this allocation {revenue:g}")
        return _result(violations, len(classes) + 2, cost=revenue)

    # 資本予算: 投資額合計が予算以内。
    investment = _num_list(instance.get("investment"))
    npv = _num_list(instance.get("npv"))
    if investment is not None and npv is not None:
        budget = _num(instance.get("budget")) or 0.0
        chosen = _int_list(_pick(solution, "chosen_projects", "selected_projects", "chosen"))
        if chosen is None:
            return _unverified("capital budgeting without chosen projects")
        _distinct_indices(chosen, len(investment), "chosen_projects", violations)
        spend = sum(investment[i] for i in chosen if 0 <= i < len(investment))
        if spend > budget + _ABS_TOL:
            violations.append(f"investment {spend:g} exceeds budget {budget:g}")
        total = sum(npv[i] for i in chosen if 0 <= i < len(npv))
        claimed = _num(_pick(solution, "max_npv", "total_npv", "objective_value"))
        if claimed is not None and not _close(claimed, total):
            violations.append(f"max_npv {claimed:g} != NPV of chosen projects {total:g}")
        return _result(violations, 3, cost=total)

    return _unverified("unknown finance IP shape")


@_register("金融・投資_線形計画")
def check_finance_lp(instance: dict, solution: Any) -> dict:
    """キャッシュフローマッチングとMADポートフォリオ。"""
    violations: list[str] = []

    # MADポートフォリオ: 重みが単体上にあり、期待収益が目標以上。
    scenarios = instance.get("scenarios")
    if scenarios is not None:
        mean_return = _num_list(instance.get("mean_return")) or []
        target = _num(instance.get("target_return")) or 0.0
        weights = _num_list(_pick(solution, "weights", "allocation", "portfolio"))
        if weights is None:
            return _unverified("portfolio without weights")
        if any(w < -_ABS_TOL for w in weights):
            violations.append("portfolio contains a negative weight")
        if not math.isclose(sum(weights), 1.0, rel_tol=1e-4, abs_tol=1e-4):
            violations.append(f"weights sum to {sum(weights):g}, expected 1")
        achieved = sum(w * r for w, r in zip(weights, mean_return))
        if achieved + 1e-6 < target:
            violations.append(f"expected return {achieved:g} below target {target:g}")
        if scenarios:
            mad = sum(
                abs(sum(w * (row[a] - mean_return[a]) for a, w in enumerate(weights)))
                for row in scenarios
            ) / len(scenarios)
            claimed = _num(_pick(solution, "min_MAD", "mad", "objective_value"))
            if claimed is not None and not _close_soft(claimed, mad):
                violations.append(f"min_MAD {claimed:g} != MAD of these weights {mad:g}")
        return _result(violations, 4, cost=achieved)

    # キャッシュフローマッチング: 保有量が非負で、費用が価格と一致する。
    bonds = instance.get("bonds")
    if bonds is not None:
        raw_holdings = _pick(solution, "bond_holdings", "holdings", "portfolio")
        if isinstance(raw_holdings, list) and all(isinstance(e, dict) for e in raw_holdings):
            rebuilt: dict[int, Any] = {}
            for entry in raw_holdings:
                bond_id = _num(entry.get("id", entry.get("bond")))
                amount = next(
                    (
                        _num(entry[k])
                        for k in ("quantity", "amount", "units", "holding")
                        if k in entry
                    ),
                    None,
                )
                if bond_id is None or amount is None:
                    return _unverified("bond holdings entries lack id or quantity")
                rebuilt[int(bond_id)] = amount
            holdings = rebuilt
        else:
            holdings = _index_map(raw_holdings)
        if holdings is None:
            return _unverified("cash flow matching without bond holdings")
        spend = 0.0
        for bond_id, qty_raw in holdings.items():
            qty = _num(qty_raw) or 0.0
            if qty < -_ABS_TOL:
                violations.append(f"bond {bond_id} holding {qty:g} is negative")
            if 0 <= bond_id < len(bonds):
                spend += qty * (_num(bonds[bond_id].get("price")) or 0.0)
        claimed = _num(_pick(solution, "min_cost", "total_cost", "objective_value"))
        if claimed is not None and not _close_soft(claimed, spend):
            violations.append(f"min_cost {claimed:g} != purchase cost {spend:g}")
        return _result(violations, 2, cost=spend)

    return _unverified("unknown finance LP shape")
