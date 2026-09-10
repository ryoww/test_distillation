"""割当・マッチング と 金融・投資 系の雛形（第 2 群）。 templates.py と同じ規約で (generate, solve) を登録する。

問題文にある人数・件数・教室数・座席数は雛形の値を保ち、重み・所要時間・負債など
文章に出ない数値だけを乱数で置き換える。
"""

from __future__ import annotations

import random

from .base import cp_sat_solver, int_list, int_matrix, register, require_optimal, retry

# ============================================================
# 割当・マッチング
# ============================================================


@register(70, "max_weight", minimize=False)
def max_weight_bipartite_matching():
    """prob_070: 最大重み二部マッチング。辺数は雛形のまま、端点と重みを引き直す。"""

    def generate(rng: random.Random, base: dict) -> dict:
        nl, nr = base["num_left"], base["num_right"]
        num_edges = len(base["edges"])

        def make() -> dict:
            pairs = sorted(rng.sample([(u, v) for u in range(nl) for v in range(nr)], num_edges))
            edges = [{"left": u, "right": v, "weight": rng.randint(5, 30)} for u, v in pairs]
            return {"num_left": nl, "num_right": nr, "edges": edges}

        def ok(inst: dict) -> bool:
            # 各求職者が自分の最良求人を取るだけで衝突しないなら競合が無く薄いので捨てる。
            best: dict[int, tuple[int, int]] = {}
            for e in inst["edges"]:
                if e["left"] not in best or e["weight"] > best[e["left"]][0]:
                    best[e["left"]] = (e["weight"], e["right"])
            chosen = [r for _, r in best.values()]
            return len(best) == nl and len(set(chosen)) < len(chosen)

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        from scipy.optimize import linear_sum_assignment

        nl, nr = instance["num_left"], instance["num_right"]
        weight = {(e["left"], e["right"]): e["weight"] for e in instance["edges"]}
        # 辺の無い組は重み 0 にして完全割当に埋め込む。重みは正なので 0 の組は「未マッチ」と
        # 同じ。
        matrix = [[weight.get((u, v), 0) for v in range(nr)] for u in range(nl)]
        rows, cols = linear_sum_assignment(matrix, maximize=True)
        matching = [
            {"left": int(u), "right": int(v), "weight": weight[(int(u), int(v))]}
            for u, v in sorted(zip(rows, cols))
            if (int(u), int(v)) in weight
        ]
        return {
            "max_weight": sum(m["weight"] for m in matching),
            "matching": matching,
            "note": "Hungarian 法（linear_sum_assignment、厳密最適解）",
        }

    return generate, solve


@register(72, "min_makespan")
def skill_constrained_assignment():
    """prob_072: スキル制約付きタスク割当。スキル制約が実際に効く instance だけを採用する。"""

    def generate(rng: random.Random, base: dict) -> dict:
        nt, nw = len(base["task_required_skill"]), len(base["worker_skills"])
        num_skills = max(max(s) for s in base["worker_skills"]) + 1
        sizes = [len(s) for s in base["worker_skills"]]

        def make() -> dict:
            return {
                "task_required_skill": int_list(rng, nt, 0, num_skills - 1),
                "worker_skills": [sorted(rng.sample(range(num_skills), k)) for k in sizes],
                "processing_time": int_matrix(rng, nw, nt, 2, 9),
            }

        def ok(inst: dict) -> bool:
            covered = set().union(*(set(s) for s in inst["worker_skills"]))
            if not set(inst["task_required_skill"]) <= covered:
                return False
            # スキルを無視しても同じメイクスパンなら制約が無意味なので捨てる。
            relaxed = dict(inst, worker_skills=[list(range(num_skills))] * nw)
            return solve(relaxed)["min_makespan"] < solve(inst)["min_makespan"]

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        cp_model, solver = cp_sat_solver()
        required = instance["task_required_skill"]
        skills = [set(s) for s in instance["worker_skills"]]
        times = instance["processing_time"]
        nt, nw = len(required), len(skills)
        model = cp_model.CpModel()
        x = [[model.NewBoolVar(f"x{w}_{t}") for t in range(nt)] for w in range(nw)]
        for t in range(nt):
            allowed = [x[w][t] for w in range(nw) if required[t] in skills[w]]
            model.AddExactlyOne(allowed)
        makespan = model.NewIntVar(0, sum(max(col) for col in zip(*times)), "makespan")
        for w in range(nw):
            model.Add(sum(times[w][t] * x[w][t] for t in range(nt)) <= makespan)
        model.Minimize(makespan)
        require_optimal(cp_model, solver.Solve(model))
        assignment = {
            str(t): next(w for w in range(nw) if solver.Value(x[w][t])) for t in range(nt)
        }
        return {
            "min_makespan": round(solver.ObjectiveValue()),
            "assignment": assignment,
            "note": "CP-SAT（厳密最適解）",
        }

    return generate, solve


@register(74, "min_load_gap")
def fair_duty_roster():
    """prob_074: 当直割当（公平負荷）。日数・人数は問題文にあるので雛形のまま。"""

    def generate(rng: random.Random, base: dict) -> dict:
        nd, ns = base["num_days"], base["num_staff"]
        # shape_signature は数値キー dict の先頭要素の長さを見るので、職員 0 の不可日数だけは
        # 雛形と揃える。
        first_size = len(base["unavailable"]["0"])

        def make() -> dict:
            unavailable = {}
            for s in range(ns):
                k = first_size if s == 0 else rng.choice([0, 1, 1, 2])
                unavailable[str(s)] = sorted(rng.sample(range(nd), k))
            return {
                "num_days": nd,
                "num_staff": ns,
                "daily_need": int_list(rng, nd, 1, 2),
                "unavailable": unavailable,
            }

        def ok(inst: dict) -> bool:
            need = inst["daily_need"]
            blocked = [0] * nd
            for days in inst["unavailable"].values():
                for d in days:
                    blocked[d] += 1
            # 各日の必要人数を出勤可能な人数で賄えない日があれば実行不能。
            if any(need[d] > ns - blocked[d] for d in range(nd)):
                return False
            # 不可日が無ければ順番に回すだけで公平になるので、不可日が 3 つ以上ある
            # instance に限る。
            return sum(blocked) >= 3 and 8 <= sum(need) <= 12

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        cp_model, solver = cp_sat_solver()
        nd, ns = instance["num_days"], instance["num_staff"]
        need = instance["daily_need"]
        unavailable = {int(k): set(v) for k, v in instance["unavailable"].items()}
        model = cp_model.CpModel()
        x = [[model.NewBoolVar(f"x{s}_{d}") for d in range(nd)] for s in range(ns)]
        for s in range(ns):
            for d in unavailable.get(s, set()):
                model.Add(x[s][d] == 0)
        for d in range(nd):
            model.Add(sum(x[s][d] for s in range(ns)) == need[d])
        counts = [sum(x[s]) for s in range(ns)]
        high = model.NewIntVar(0, nd, "high")
        low = model.NewIntVar(0, nd, "low")
        for c in counts:
            model.Add(c <= high)
            model.Add(c >= low)
        model.Minimize(high - low)
        require_optimal(cp_model, solver.Solve(model))
        schedule = {str(s): [d for d in range(nd) if solver.Value(x[s][d])] for s in range(ns)}
        return {
            "min_load_gap": round(solver.ObjectiveValue()),
            "schedule": schedule,
            "note": "CP-SAT（厳密最適解）",
        }

    return generate, solve


@register(75, "max_preference", minimize=False)
def classroom_assignment():
    """prob_075: 教室割当。講義数・教室数・時限数は問題文にあるので雛形のまま。"""

    def generate(rng: random.Random, base: dict) -> dict:
        nc = len(base["course_size"])
        nr, ns = base["num_rooms"], base["num_slots"]

        def make() -> dict:
            return {
                "course_size": int_list(rng, nc, 15, 40),
                "room_capacity": int_list(rng, nr, 30, 65),
                "slot_preference": int_matrix(rng, nc, ns, 1, 10),
                "num_rooms": nr,
                "num_slots": ns,
            }

        def ok(inst: dict) -> bool:
            if max(inst["course_size"]) > max(inst["room_capacity"]):
                return False
            # 定員を無視しても同じ選好度なら定員制約が無意味なので捨てる。
            relaxed = dict(inst, room_capacity=[max(inst["course_size"])] * nr)
            solved = _classroom_solve(inst)
            return solved is not None and _classroom_solve(relaxed)[0] > solved[0]

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        solved = _classroom_solve(instance)
        if solved is None:
            raise ValueError("instance has no feasible classroom assignment")
        total, placement = solved
        return {
            "max_preference": total,
            "assignment": {str(c): {"room": r, "slot": s} for c, (r, s) in enumerate(placement)},
            "note": "CP-SAT（厳密最適解）",
        }

    return generate, solve


def _classroom_solve(instance: dict) -> tuple[int, list[tuple[int, int]]] | None:
    """最大選好度と各講義の (教室, 時限)。実行不能なら None。"""
    cp_model, solver = cp_sat_solver()
    size, capacity = instance["course_size"], instance["room_capacity"]
    pref = instance["slot_preference"]
    nc, nr, ns = len(size), instance["num_rooms"], instance["num_slots"]
    model = cp_model.CpModel()
    x = {
        (c, r, s): model.NewBoolVar(f"x{c}_{r}_{s}")
        for c in range(nc)
        for r in range(nr)
        for s in range(ns)
        if size[c] <= capacity[r]
    }
    for c in range(nc):
        model.AddExactlyOne(x[k] for k in x if k[0] == c)
    for r in range(nr):
        for s in range(ns):
            model.AddAtMostOne(x[k] for k in x if k[1] == r and k[2] == s)
    model.Maximize(sum(pref[c][s] * v for (c, _, s), v in x.items()))
    status = solver.Solve(model)
    if status == cp_model.INFEASIBLE:
        return None
    require_optimal(cp_model, status)
    placement = [
        next((r, s) for (c2, r, s), v in x.items() if c2 == c and solver.Value(v))
        for c in range(nc)
    ]
    return round(solver.ObjectiveValue()), placement


@register(76, "max_profit", minimize=False)
def selective_assignment():
    """prob_076: 選択的割当。負の利益を含め、割り当てない選択が実際に選ばれる instance に限る。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n = base["num_agents"]

        def make() -> dict:
            # 約 1/3 の組を負の利益にして「割り当てない」判断が要る行列にする。
            matrix = [
                [
                    rng.randint(-8, -1) if rng.random() < 1 / 3 else rng.randint(0, 25)
                    for _ in range(n)
                ]
                for _ in range(n)
            ]
            return {"num_agents": n, "profit_matrix": matrix}

        def ok(inst: dict) -> bool:
            negatives = sum(v < 0 for row in inst["profit_matrix"] for v in row)
            # 「割り当てない」選択が最適解に現れないと選択的割当としての意味が薄い。
            return negatives >= 3 and len(solve(inst)["pairs"]) < n

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        from scipy.optimize import linear_sum_assignment

        profit = instance["profit_matrix"]
        # 負の組は利益 0 に置き換えて完全割当を解く。0 の組は割り当てないのと同じ利益なので、
        # 利益が正の組だけを残せば選択的割当の厳密最適解になる。
        clipped = [[max(v, 0) for v in row] for row in profit]
        rows, cols = linear_sum_assignment(clipped, maximize=True)
        pairs = [
            {"agent": int(a), "job": int(j)}
            for a, j in sorted(zip(rows, cols))
            if profit[int(a)][int(j)] > 0
        ]
        return {
            "max_profit": sum(profit[p["agent"]][p["job"]] for p in pairs),
            "pairs": pairs,
            "note": "Hungarian 法（linear_sum_assignment、厳密最適解）",
        }

    return generate, solve


# ============================================================
# 金融・投資
# ============================================================

# 債券の額面は instance に無いが、同梱参照解（債券 2 を 19.333 単位、費用 1933.33）は
# 1 年目のクーポン 6 × 19.333 = 116 で負債を賄い、2 年目に額面 100 で全残債を賄う解であり、
# 価格 95〜100・クーポン 4〜6 の水準とも整合するので、額面 100（パー）で一意に再現できる。
_FACE_VALUE = 100


@register(82, "min_cost")
def cash_flow_matching():
    """prob_082: キャッシュフローマッチング。年数・債券数は問題文にあるので雛形のまま。"""

    def generate(rng: random.Random, base: dict) -> dict:
        periods = base["periods"]
        nb = len(base["bonds"])

        def make() -> dict:
            maturities = sorted(rng.sample(range(2, periods + 1), nb))
            rng.shuffle(maturities)
            bonds = [
                {
                    "id": i,
                    "price": rng.randint(90, 105),
                    "coupon": rng.randint(3, 8),
                    "maturity": m,
                }
                for i, m in enumerate(maturities)
            ]
            # Why not 同梱と同じ 60〜130: 額面 100 に対して負債が小さいと 1 年目のクーポンを
            # 賄う最安の債券 1 種類が全年を賄ってしまい、複数債券の組合せが最適になる instance が
            # ほとんど出ない。
            return {
                "periods": periods,
                "liabilities": int_list(rng, periods, 200, 600),
                "bonds": bonds,
            }

        def ok(inst: dict) -> bool:
            # 1 種類の債券だけで最適になる instance（同梱 instance と同じ形）は組合せが無く薄い。
            holdings = sorted(solve(inst)["bond_holdings"].values(), reverse=True)
            return holdings[1] >= 1.0

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        from scipy.optimize import linprog

        periods = instance["periods"]
        bonds = instance["bonds"]
        liabilities = instance["liabilities"]
        nb = len(bonds)
        # 変数は [債券の購入量 x_0..x_{nb-1}] + [各年末の繰越金 s_0..s_{T-1}]。
        # 年 t: Σ_b cf_b(t) x_b + s_{t-1} - s_t = L_t。
        a_eq, b_eq = [], []
        for t in range(periods):
            row = [0.0] * (nb + periods)
            for b, bond in enumerate(bonds):
                year = t + 1
                if year <= bond["maturity"]:
                    row[b] += bond["coupon"]
                if year == bond["maturity"]:
                    row[b] += _FACE_VALUE
            if t > 0:
                row[nb + t - 1] = 1.0
            row[nb + t] = -1.0
            a_eq.append(row)
            b_eq.append(float(liabilities[t]))
        result = linprog(
            c=[float(b["price"]) for b in bonds] + [0.0] * periods,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=[(0, None)] * (nb + periods),
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"LP failed: {result.message}")
        holdings = [max(float(v), 0.0) for v in result.x[:nb]]
        return {
            "min_cost": sum(h * b["price"] for h, b in zip(holdings, bonds)),
            "feasible": True,
            "bond_holdings": {str(b["id"]): h for h, b in zip(holdings, bonds)},
            "note": "LP（HiGHS、厳密最適解）",
        }

    return generate, solve


@register(88, "max_revenue", minimize=False)
def seat_allocation():
    """prob_088: 収益管理（座席配分）。座席数とクラス名は問題文にあるので雛形のまま。"""

    def generate(rng: random.Random, base: dict) -> dict:
        names = [c["name"] for c in base["classes"]]
        capacity = base["capacity"]

        def make() -> dict:
            # First > Business > Economy の価格順を保つ。
            prices = sorted(rng.sample(range(100, 601, 10), len(names)), reverse=True)
            demands = [rng.randint(5, 12), rng.randint(12, 30), rng.randint(40, 70)]
            classes = [
                {"name": n, "price": p, "demand": d} for n, p, d in zip(names, prices, demands)
            ]
            return {"classes": classes, "capacity": capacity}

        def ok(inst: dict) -> bool:
            demands = [c["demand"] for c in inst["classes"]]
            # 座席が需要合計を賄えるなら配分の判断が無く、上位 2 クラスだけで満席なら
            # 最下位クラスが消える。どちらも捨てる。
            return sum(demands) > capacity and sum(demands[:-1]) < capacity

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        cp_model, solver = cp_sat_solver()
        classes = instance["classes"]
        model = cp_model.CpModel()
        seats = [model.NewIntVar(0, c["demand"], c["name"]) for c in classes]
        model.Add(sum(seats) <= instance["capacity"])
        model.Maximize(sum(c["price"] * v for c, v in zip(classes, seats)))
        require_optimal(cp_model, solver.Solve(model))
        return {
            "max_revenue": round(solver.ObjectiveValue()),
            "allocation": {c["name"]: solver.Value(v) for c, v in zip(classes, seats)},
            "note": "整数計画（CP-SAT、厳密最適解）",
        }

    return generate, solve
