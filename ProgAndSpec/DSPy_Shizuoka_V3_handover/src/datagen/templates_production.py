"""production 系の雛形。templates.py と同じ規約で (generate, solve) を登録する。

生産・在庫計画と金融・投資の雛形を置く。問題文にある期間数・初期人員・単価は
雛形の値を保ち、需要や負債など文章に出ない数値だけを乱数で置き換える。
"""

from __future__ import annotations

import math
import random
from itertools import pairwise

from .base import cp_sat_solver, int_list, register, require_optimal, retry

# Why not prob_080（新聞売り子）: 問題文が仕入値・売値・処分価格・需要の値と確率を
# すべて明記しているので、文章を保ったまま乱数で変えられる数値が一つも無い。
# 同じ core_type（生産・在庫計画_確率最適化）に他の雛形も無いため、この型は空ける。


# ============================================================
# 生産・在庫計画
# ============================================================


@register(79, "min_cost")
def lot_sizing():
    """prob_079: 多期間ロットサイジング。期間数は問題文にあるので雛形の値を保つ。"""

    def generate(rng: random.Random, base: dict) -> dict:
        periods = base["periods"]

        def make() -> dict:
            return {
                "periods": periods,
                "demand": int_list(rng, periods, 5, 40),
                "setup_cost": rng.randint(60, 150),
                "holding_cost": rng.randint(1, 4),
                "unit_cost": rng.randint(3, 8),
            }

        def ok(instance: dict) -> bool:
            # 毎期生産（段取費が効かない）と一括生産（在庫費が効かない）は除く。
            setups = sum(1 for q in solve(instance)["production_plan"] if q > 0)
            return 2 <= setups <= periods - 1

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        demand = instance["demand"]
        periods = instance["periods"]
        setup = instance["setup_cost"]
        holding = instance["holding_cost"]
        unit = instance["unit_cost"]

        def batch_cost(start: int, end: int) -> int:
            """期 start で期 start..end の需要をまとめて生産する費用。

            在庫費は期末在庫に掛かるので、期 k の需要は (k - start) 期分の在庫費を払う。
            """
            quantity = sum(demand[start : end + 1])
            carry = sum(demand[k] * (k - start) for k in range(start + 1, end + 1))
            return setup + unit * quantity + holding * carry

        # Wagner–Whitin: best[j] は期 0..j-1 の需要を満たす最小費用。
        best = [0] + [math.inf] * periods
        last_start = [0] * (periods + 1)
        for end in range(periods):
            for start in range(end + 1):
                candidate = best[start] + batch_cost(start, end)
                if candidate < best[end + 1]:
                    best[end + 1] = candidate
                    last_start[end + 1] = start
        plan = [0.0] * periods
        end = periods
        while end > 0:
            start = last_start[end]
            plan[start] = float(sum(demand[start:end]))
            end = start
        return {
            "min_cost": int(best[periods]),
            "production_plan": plan,
            "note": "Wagner–Whitin 動的計画法（厳密最適解）",
        }

    return generate, solve


@register(85, "min_cost")
def workforce_planning():
    """prob_085: 労働力計画。期間数・初期人員・単価は問題文にあるので雛形の値を保つ。"""

    def generate(rng: random.Random, base: dict) -> dict:
        periods = base["periods"]
        initial = base["initial_workforce"]

        def make() -> dict:
            return {
                "periods": periods,
                "requirement": int_list(rng, periods, max(1, initial - 4), initial + 10),
                "hire_cost": base["hire_cost"],
                "fire_cost": base["fire_cost"],
                "wage": base["wage"],
                "initial_workforce": initial,
            }

        def ok(instance: dict) -> bool:
            req = instance["requirement"]
            # 単調増加の必要人数は「毎期ちょうど採用」で決まるので、減る期を必ず入れる。
            has_drop = any(b < a for a, b in pairwise(req))
            return max(req) > initial and has_drop

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        cp_model, solver = cp_sat_solver()
        requirement = instance["requirement"]
        periods = instance["periods"]
        upper = max(requirement + [instance["initial_workforce"]])
        model = cp_model.CpModel()
        workforce = [model.NewIntVar(0, upper, f"w{t}") for t in range(periods)]
        hire = [model.NewIntVar(0, upper, f"h{t}") for t in range(periods)]
        fire = [model.NewIntVar(0, upper, f"f{t}") for t in range(periods)]
        for t in range(periods):
            previous = instance["initial_workforce"] if t == 0 else workforce[t - 1]
            model.Add(workforce[t] == previous + hire[t] - fire[t])
            model.Add(workforce[t] >= requirement[t])
        # 賃金は採用・解雇を反映した後のその期の人員に掛かる（雛形の参照解と同じ定義）。
        model.Minimize(
            sum(
                instance["hire_cost"] * hire[t]
                + instance["fire_cost"] * fire[t]
                + instance["wage"] * workforce[t]
                for t in range(periods)
            )
        )
        require_optimal(cp_model, solver.Solve(model))
        plan = [
            {
                "period": t,
                "workforce": solver.Value(workforce[t]),
                "hire": solver.Value(hire[t]),
                "fire": solver.Value(fire[t]),
            }
            for t in range(periods)
        ]
        return {
            "min_cost": round(solver.ObjectiveValue()),
            "plan": plan,
            "note": "CP-SAT（厳密最適解）",
        }

    return generate, solve


# ============================================================
# 金融・投資
# ============================================================

# Why 100: 額面は instance に無い。雛形の参照解（債券2を 116/6 単位）は額面が 14 以上なら
# 再現できるので値は特定できず、市場慣行の 100 を採る。
BOND_FACE_VALUE = 100


@register(82, "min_cost")
def cash_flow_matching():
    """prob_082: キャッシュフローマッチング。年数と債券数は問題文にあるので雛形の値を保つ。"""

    def generate(rng: random.Random, base: dict) -> dict:
        periods = base["periods"]
        num_bonds = len(base["bonds"])

        def make() -> dict:
            maturities = rng.sample(range(1, periods + 1), num_bonds)
            bonds = [
                {
                    "id": i,
                    "price": rng.randint(90, 105),
                    "coupon": rng.randint(3, 8),
                    "maturity": maturities[i],
                }
                for i in range(num_bonds)
            ]
            return {
                "periods": periods,
                "liabilities": int_list(rng, periods, 60, 150),
                "bonds": bonds,
            }

        def ok(instance: dict) -> bool:
            # 1銘柄だけで賄える instance は選択の余地が無いので、2銘柄以上使う解を求める。
            holdings = solve(instance)["bond_holdings"]
            return sum(1 for q in holdings.values() if q > 0) >= 2

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        from scipy.optimize import linprog

        periods = instance["periods"]
        bonds = instance["bonds"]
        liabilities = instance["liabilities"]
        num_bonds = len(bonds)

        def cash_flow(bond: dict, year: int) -> float:
            """year 年目（1 始まり）に 1 単位が払うクーポンと償還金。"""
            if year > bond["maturity"]:
                return 0.0
            return bond["coupon"] + (BOND_FACE_VALUE if year == bond["maturity"] else 0)

        # 変数は [保有量 x_b] + [各年末の繰越 s_t]。年 t の収支:
        #   sum_b cf(b, t) x_b + s_{t-1} - s_t = L_t  （s_0 = 0）
        a_eq = []
        for t in range(periods):
            row = [cash_flow(b, t + 1) for b in bonds] + [0.0] * periods
            row[num_bonds + t] = -1.0
            if t > 0:
                row[num_bonds + t - 1] = 1.0
            a_eq.append(row)
        result = linprog(
            c=[b["price"] for b in bonds] + [0.0] * periods,
            A_eq=a_eq,
            b_eq=liabilities,
            bounds=[(0, None)] * (num_bonds + periods),
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"LP failed: {result.message}")
        holdings = {str(b["id"]): round(float(result.x[i]), 3) for i, b in enumerate(bonds)}
        return {
            "min_cost": round(float(result.fun), 2),
            "feasible": True,
            "bond_holdings": holdings,
            "note": "LP（HiGHS、厳密最適解）",
        }

    return generate, solve
