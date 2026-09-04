"""scheduling 系の雛形。templates.py と同じ規約で (generate, solve) を登録する。"""

from __future__ import annotations

import itertools
import math
import random

from .base import cp_sat_solver, int_list, register, require_optimal, retry


@register(2, "makespan")
def identical_parallel_machines():
    """prob_002: 同一並列機械 P||Cmax。台数とジョブ数は問題文にあるので雛形のまま。"""

    def generate(rng: random.Random, base: dict) -> dict:
        m = base["num_machines"]
        n = len(base["jobs"])

        def make() -> dict:
            times = int_list(rng, n, 2, 12)
            jobs = [{"id": i + 1, "processing_time": p} for i, p in enumerate(times)]
            return {"num_machines": m, "jobs": jobs}

        def ok(inst: dict) -> bool:
            # 最長ジョブを単独で置くだけで最適になる instance は問題として薄いので捨てる。
            times = [j["processing_time"] for j in inst["jobs"]]
            return max(times) < solve(inst)["makespan"]

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        cp_model, solver = cp_sat_solver()
        jobs = instance["jobs"]
        machines = range(instance["num_machines"])
        model = cp_model.CpModel()
        x = {(j, m): model.NewBoolVar(f"x{j}_{m}") for j in range(len(jobs)) for m in machines}
        makespan = model.NewIntVar(0, sum(j["processing_time"] for j in jobs), "makespan")
        for j in range(len(jobs)):
            model.AddExactlyOne(x[j, m] for m in machines)
        loads = [
            sum(jobs[j]["processing_time"] * x[j, m] for j in range(len(jobs))) for m in machines
        ]
        for load in loads:
            model.Add(load <= makespan)
        model.Minimize(makespan)
        require_optimal(cp_model, solver.Solve(model))
        assignment = {
            str(jobs[j]["id"]): next(m + 1 for m in machines if solver.Value(x[j, m]))
            for j in range(len(jobs))
        }
        return {
            "machine_assignment": assignment,
            "makespan": int(solver.Value(makespan)),
            "machine_loads": {str(m + 1): int(solver.Value(loads[m])) for m in machines},
        }

    return generate, solve


@register(3, "makespan")
def permutation_flow_shop():
    """prob_003: 3 段の順列フローショップ。6 ジョブなので順列を全列挙する。"""

    def generate(rng: random.Random, base: dict) -> dict:
        stages = base["num_stages"]
        n = len(base["jobs"])
        jobs = [
            {
                "id": i + 1,
                "processing_times": {f"stage_{s + 1}": rng.randint(2, 10) for s in range(stages)},
            }
            for i in range(n)
        ]
        return {"num_stages": stages, "jobs": jobs}

    def solve(instance: dict) -> dict:
        jobs = instance["jobs"]
        if len(jobs) > 8:
            raise ValueError("brute force is limited to 8 jobs")
        stage_keys = [f"stage_{s + 1}" for s in range(instance["num_stages"])]
        best_seq: list[int] = []
        best = math.inf
        for order in itertools.permutations(jobs):
            # 各段の機械の空き時刻を前のジョブから引き継ぎ、前段完了との遅い方から始める。
            free = [0] * len(stage_keys)
            for job in order:
                done = 0
                for s, key in enumerate(stage_keys):
                    done = max(done, free[s]) + job["processing_times"][key]
                    free[s] = done
            if free[-1] < best:
                best = free[-1]
                best_seq = [job["id"] for job in order]
        return {
            "optimal_sequence": best_seq,
            "makespan": int(best),
            "note": "全順列探索による最適解（厳密最適解）",
        }

    return generate, solve


@register(10, "project_duration")
def critical_path():
    """prob_010: クリティカルパス法。活動数は問題文にあるので雛形のまま、DAG を引き直す。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n = len(base["activities"])

        def make() -> dict:
            activities = []
            for i in range(1, n + 1):
                # 先行活動は番号の小さい活動から選ぶので閉路はできない。
                pool = list(range(1, i))
                k = min(len(pool), rng.choice([0, 1, 1, 2]))
                activities.append(
                    {
                        "id": i,
                        "name": f"活動{i}",
                        "duration": rng.randint(1, 8),
                        "predecessors": sorted(rng.sample(pool, k)),
                    }
                )
            return {"activities": activities}

        def ok(inst: dict) -> bool:
            # 先行関係のない活動だけ、または 2 活動で終わる工程表は工程表と呼びにくい。
            has_edges = sum(len(a["predecessors"]) for a in inst["activities"]) >= n // 2
            return has_edges and len(solve(inst)["critical_path"]) >= 3

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        activities = instance["activities"]
        duration = {a["id"]: a["duration"] for a in activities}
        preds = {a["id"]: list(a["predecessors"]) for a in activities}
        succs: dict[int, list[int]] = {a["id"]: [] for a in activities}
        for a in activities:
            for p in a["predecessors"]:
                succs[p].append(a["id"])
        # 前進計算: 先行活動は必ず若い番号なので id 順に回せばよい。
        earliest: dict[int, int] = {}
        for a in activities:
            earliest[a["id"]] = max((earliest[p] + duration[p] for p in preds[a["id"]]), default=0)
        project = max(earliest[i] + duration[i] for i in earliest)
        # 後退計算: 最遅開始 = min(後続の最遅開始) - 所要日数。後続がなければ完了日から。
        latest: dict[int, int] = {}
        for a in reversed(activities):
            finish = min((latest[s] for s in succs[a["id"]]), default=project)
            latest[a["id"]] = finish - duration[a["id"]]
        critical = sorted(
            (i for i in earliest if earliest[i] == latest[i]), key=lambda i: (earliest[i], i)
        )
        return {
            "project_duration": int(project),
            "critical_path": critical,
            "earliest_start": {str(i): earliest[i] for i in earliest},
            "latest_start": {str(i): latest[i] for i in latest},
        }

    return generate, solve


# Why not prob_018（予防保全）: 参照解は `schedule` だけで数値の目的値キーがなく、
# validate_problem の目的値チェックを通せない。加えて期待コスト率 7.82 などを再現する
# 陽な式が見つからず（最も近い (Cp + Cf(1 - e^{-T/μ}))/T でも 7.85）、指数分布は
# 無記憶なので予防保全間隔に有限の厳密最適が存在しない。
