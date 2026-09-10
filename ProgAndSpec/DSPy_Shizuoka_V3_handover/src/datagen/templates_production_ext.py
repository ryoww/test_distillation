"""生産・在庫計画 系の雛形（第 2 群: prob_078〜087）。templates.py と同じ規約で (generate, solve) を登録する。

LP は scipy の HiGHS で解き、templates_production.py と同じく解は丸めずに保存し、
目的値は保存した解から再計算する（丸めた解が実行不可能に落ちるのを避ける）。
問題文にある原料数・生産量・能力・単価・ロット候補・発注回数上限は雛形の値を保ち、
需要や単価など文章に出ない数値だけを乱数で置き換える。
"""

from __future__ import annotations

import random
from itertools import product

from .base import int_list, register, retry


def _linprog(**kwargs):
    from scipy.optimize import linprog

    result = linprog(method="highs", **kwargs)
    if not result.success:
        raise RuntimeError(f"LP failed: {result.message}")
    return [float(v) for v in result.x]


def _as_int_if_integral(value: float) -> int | float:
    """整数データの LP は頂点解が整数になるので、丸め誤差だけの差なら int で返す。"""
    return round(value) if abs(value - round(value)) < 1e-6 else value


@register(78, "min_cost")
def blending():
    """prob_078: 配合問題。原料数・製品量 100・純度下限 80% は問題文にあるので雛形の値を保つ。"""

    def generate(rng: random.Random, base: dict) -> dict:
        materials = list(base["materials"])

        def make() -> dict:
            return {
                "materials": materials,
                "cost": {m: rng.randint(4, 12) for m in materials},
                "purity": {m: rng.randint(55, 98) for m in materials},
                "amount": base["amount"],
                "min_purity": base["min_purity"],
            }

        def ok(instance: dict) -> bool:
            purity = instance["purity"]
            if max(purity.values()) < instance["min_purity"]:
                return False
            # 純度下限を満たす原料が最安だと単一原料で決まるので、2 種以上を混ぜる解に限る。
            blend = solve(instance)["blend"]
            return sum(1 for q in blend.values() if q > 1e-6) >= 2

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        materials = instance["materials"]
        cost = instance["cost"]
        purity = instance["purity"]
        amount = instance["amount"]
        x = _linprog(
            c=[cost[m] for m in materials],
            A_ub=[[-purity[m] for m in materials]],
            b_ub=[-instance["min_purity"] * amount],
            A_eq=[[1.0] * len(materials)],
            b_eq=[amount],
            bounds=[(0, None)] * len(materials),
        )
        blend = dict(zip(materials, x))
        return {
            "min_cost": float(sum(cost[m] * q for m, q in blend.items())),
            "blend": blend,
            "note": "LP（HiGHS、厳密最適解）",
        }

    return generate, solve


@register(84, "min_cost")
def multi_period_production():
    """prob_084: 多期間生産計画。期間数と生産能力 70 は問題文にあるので雛形の値を保つ。"""

    def generate(rng: random.Random, base: dict) -> dict:
        periods = base["periods"]
        capacity = base["production_capacity"]

        def make() -> dict:
            return {
                "periods": periods,
                "demand": int_list(rng, periods, 30, 95),
                "production_capacity": capacity,
                "unit_cost": rng.randint(5, 15),
                "holding_cost": rng.randint(1, 5),
            }

        def ok(instance: dict) -> bool:
            demand = instance["demand"]
            cumulative = 0
            for t, d in enumerate(demand):
                cumulative += d
                if cumulative > capacity * (t + 1):
                    return False
            # 全期の需要が能力以下だと各期ちょうど生産で決まるので、在庫を使う instance に限る。
            return max(demand) > capacity

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        periods = instance["periods"]
        demand = instance["demand"]
        capacity = instance["production_capacity"]
        unit = instance["unit_cost"]
        holding = instance["holding_cost"]
        # 変数は [生産量 x_0..x_{T-1}] + [期末在庫 I_0..I_{T-1}]。I_{t-1} + x_t - I_t = d_t。
        a_eq = []
        for t in range(periods):
            row = [0.0] * (2 * periods)
            row[t] = 1.0
            row[periods + t] = -1.0
            if t > 0:
                row[periods + t - 1] = 1.0
            a_eq.append(row)
        x = _linprog(
            c=[unit] * periods + [holding] * periods,
            A_eq=a_eq,
            b_eq=[float(d) for d in demand],
            bounds=[(0, capacity)] * periods + [(0, None)] * periods,
        )
        plan = x[:periods]
        inventory = 0.0
        spend = 0.0
        for t, produced in enumerate(plan):
            spend += unit * produced
            inventory += produced - demand[t]
            spend += holding * max(inventory, 0.0)
        return {
            "min_cost": _as_int_if_integral(spend),
            "production_plan": plan,
            "feasible": True,
            "note": "LP（HiGHS、厳密最適解）",
        }

    return generate, solve


@register(86, "min_cost")
def production_with_overtime():
    """prob_086: 残業・外注つき生産計画。期間数・各手段の能力と単価は問題文にあるので保つ。"""

    def generate(rng: random.Random, base: dict) -> dict:
        periods = base["periods"]
        fixed_keys = (
            "regular_cap",
            "regular_cost",
            "overtime_cap",
            "overtime_cost",
            "subcontract_cost",
        )
        fixed = {k: base[k] for k in fixed_keys}

        def make() -> dict:
            return {
                "periods": periods,
                "demand": int_list(rng, periods, 30, 95),
                **fixed,
                "holding_cost": rng.randint(1, 3),
            }

        def ok(instance: dict) -> bool:
            plan = solve(instance)["plan"]
            demand = instance["demand"]
            # 3 手段と在庫の繰越が全部登場する instance に限る（どれかが無いと自明に近い）。
            uses_overtime = any(e["overtime"] > 1e-6 for e in plan)
            uses_subcontract = any(e["subcontract"] > 1e-6 for e in plan)
            carries = any(
                e["regular"] + e["overtime"] + e["subcontract"] > demand[t] + 1e-6
                for t, e in enumerate(plan)
            )
            return uses_overtime and uses_subcontract and carries

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        periods = instance["periods"]
        demand = instance["demand"]
        rates = (
            instance["regular_cost"],
            instance["overtime_cost"],
            instance["subcontract_cost"],
        )
        holding = instance["holding_cost"]
        # 変数は期ごとに [通常 r_t, 残業 o_t, 外注 s_t, 期末在庫 I_t] の 4 つを並べる。
        width = 4 * periods
        a_eq = []
        for t in range(periods):
            row = [0.0] * width
            row[4 * t : 4 * t + 3] = [1.0, 1.0, 1.0]
            row[4 * t + 3] = -1.0
            if t > 0:
                row[4 * (t - 1) + 3] = 1.0
            a_eq.append(row)
        x = _linprog(
            c=[*rates, holding] * periods,
            A_eq=a_eq,
            b_eq=[float(d) for d in demand],
            bounds=[
                (0, instance["regular_cap"]),
                (0, instance["overtime_cap"]),
                (0, None),
                (0, None),
            ]
            * periods,
        )
        plan = [
            {
                "period": t,
                "regular": x[4 * t],
                "overtime": x[4 * t + 1],
                "subcontract": x[4 * t + 2],
            }
            for t in range(periods)
        ]
        inventory = 0.0
        spend = 0.0
        for t, entry in enumerate(plan):
            quantities = (entry["regular"], entry["overtime"], entry["subcontract"])
            spend += sum(rate * q for rate, q in zip(rates, quantities))
            inventory += sum(quantities) - demand[t]
            spend += holding * max(inventory, 0.0)
        return {
            "min_cost": _as_int_if_integral(spend),
            "plan": plan,
            "note": "LP（HiGHS、厳密最適解）",
        }

    return generate, solve


@register(87, "min_cost")
def order_lot_optimization():
    """prob_087: 発注ロット最適化。商品数・ロット候補・発注回数上限 40 は問題文にあるので保つ。

    費用は「発注費 × ceil(年間需要 / ロット) + 保持費 × ロット / 2」。同梱参照値 919 は
    この式で再現でき、発注回数を需要/ロットの実数にすると 868 になって一致しない。
    """

    def generate(rng: random.Random, base: dict) -> dict:
        n = len(base["demand"])
        lot_choices = list(base["lot_choices"])
        max_orders = base["max_total_orders"]

        def make() -> dict:
            return {
                # 需要 360 以下なら全商品ロット 60 で発注 36 回に収まり、必ず実行可能。
                "demand": int_list(rng, n, 120, 360),
                "order_cost": int_list(rng, n, 5, 25),
                "holding_cost": int_list(rng, n, 2, 10),
                "lot_choices": lot_choices,
                "max_total_orders": max_orders,
            }

        def ok(instance: dict) -> bool:
            # 商品ごとの単独最適の発注回数が上限に収まると制約が効かないので、超える instance に限る。
            free_orders = sum(
                min(_lot_options(instance, i), key=lambda c: (c[0], c[1]))[1] for i in range(n)
            )
            return free_orders > max_orders

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        n = len(instance["demand"])
        options = [_lot_options(instance, i) for i in range(n)]
        max_orders = instance["max_total_orders"]
        # 6 商品 × 6 候補 = 46656 通りの全列挙。同点は (費用, ロット列) の辞書順で決める。
        best = None
        for combo in product(*options):
            if sum(c[1] for c in combo) > max_orders:
                continue
            key = (sum(c[0] for c in combo), tuple(c[2] for c in combo))
            if best is None or key < best:
                best = key
        if best is None:
            raise RuntimeError("no lot assignment satisfies the order limit")
        return {
            "min_cost": int(best[0]),
            "chosen_lot_sizes": {str(i): lot for i, lot in enumerate(best[1])},
            "note": "全列挙（厳密最適解）",
        }

    return generate, solve


def _lot_options(instance: dict, i: int) -> list[tuple[int, int, int]]:
    """商品 i の各ロット候補について (年間費用, 発注回数, ロット) を返す。"""
    demand = instance["demand"][i]
    order_cost = instance["order_cost"][i]
    holding = instance["holding_cost"][i]
    out = []
    for lot in instance["lot_choices"]:
        orders = -(-demand // lot)
        out.append((order_cost * orders + holding * (lot // 2), orders, lot))
    return out


# Why not prob_080（新聞売り子）: 仕入値・売値・処分価格・需要の値と確率がすべて問題文に
# 書かれており、文章を保ったまま乱数で変えられる数値が無い（templates_production.py の
# 冒頭コメントと同じ判断）。解自体は発注量の全列挙で厳密に決まるが、雛形にはならない。
