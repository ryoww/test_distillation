"""scheduling 系の雛形（第 3 群: prob_012〜020）。templates.py と同じ規約で (generate, solve) を登録する。"""

from __future__ import annotations

import itertools
import random

from .base import SolveError, cp_sat_solver, register, require_optimal, retry

# (ジョブ側の要求キー, ノード側の容量キー)。feasibility_shapes.check_cluster_node_assignment と同じ組。
_CLUSTER_RESOURCES = (
    ("cpu_required", "cpu_cores"),
    ("gpu_required", "gpu_count"),
    ("memory_required_gb", "memory_gb"),
)


@register(12, "total_priority", shipped_reference_optimal=False, minimize=False)
def cluster_batch_jobs():
    """prob_012: ノード容量つきのジョブ割当（一般化割当問題）。ノード数・ジョブ数は問題文のまま。

    同梱参照解は total_priority=6 と申告するが、載せている 6 ジョブの優先度合計は 14 で誤申告。
    さらにノード 1 にメモリ 64+32 GB のジョブを同居させ、容量 64 GB を超えている。
    容量はノードごとに割り当てジョブの要求を合算して比較する。
    Why not ジョブ単体とノード容量の比較（チェッカーの読み方）: その読みでは同梱 instance の
    全ジョブがどこかに単独で収まり、目的値が全優先度の和 14 に自明に決まって出題にならない。
    合算で扱った厳密解は 13 で、合算解はジョブ単体の条件も満たすのでチェッカーは通る。
    """

    def generate(rng: random.Random, base: dict) -> dict:
        n_nodes = len(base["nodes"])
        n_jobs = len(base["jobs"])

        def make() -> dict:
            nodes = [
                {
                    "id": i + 1,
                    "cpu_cores": rng.choice([8, 16, 32]),
                    "gpu_count": rng.choice([0, 2, 4]),
                    "memory_gb": rng.choice([32, 64, 128]),
                }
                for i in range(n_nodes)
            ]
            jobs = [
                {
                    "id": j + 1,
                    "cpu_required": rng.choice([2, 4, 8]),
                    "gpu_required": rng.choice([0, 1, 2]),
                    "memory_required_gb": rng.choice([16, 32, 64]),
                    "priority": rng.randint(1, 5),
                    "estimated_duration": rng.randint(1, 8),
                }
                for j in range(n_jobs)
            ]
            return {"nodes": nodes, "jobs": jobs}

        def ok(inst: dict) -> bool:
            # 全ジョブが載る instance は容量制約が効いておらず、載るジョブが少なすぎても薄い。
            solved = solve(inst)
            everything = sum(job["priority"] for job in inst["jobs"])
            return solved["assigned_count"] >= 3 and solved["total_priority"] < everything

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        cp_model, solver = cp_sat_solver()
        nodes = instance["nodes"]
        jobs = instance["jobs"]
        model = cp_model.CpModel()
        x = {
            (j, n): model.NewBoolVar(f"x{j}_{n}")
            for j in range(len(jobs))
            for n in range(len(nodes))
        }
        for j in range(len(jobs)):
            model.AddAtMostOne(x[j, n] for n in range(len(nodes)))
        for n, node in enumerate(nodes):
            for need_key, cap_key in _CLUSTER_RESOURCES:
                model.Add(
                    sum(jobs[j][need_key] * x[j, n] for j in range(len(jobs))) <= node[cap_key]
                )
        priority = sum(jobs[j]["priority"] * x[j, n] for (j, n) in x)
        # 同点の割当を若い番号のノードへ寄せて解を一つに近づける。重みは副目的の上限より大きい。
        spread = sum(n * x[j, n] for (j, n) in x)
        model.Maximize((len(jobs) * len(nodes) + 1) * priority - spread)
        require_optimal(cp_model, solver.Solve(model))
        assignment = {
            str(job["id"]): nodes[n]["id"]
            for j, job in enumerate(jobs)
            for n in range(len(nodes))
            if solver.Value(x[j, n])
        }
        total = sum(job["priority"] for job in jobs if str(job["id"]) in assignment)
        return {
            "node_assignment": assignment,
            "total_priority": int(total),
            "assigned_count": len(assignment),
            "note": "CP-SAT（厳密最適解）",
        }

    return generate, solve


def _clock(minutes: int) -> str:
    return f"{minutes // 60}:{minutes % 60:02d}"


@register(13, "total_expected_rating")
def broadcast_lineup():
    """prob_013: ジャンル最低数つきの番組–スロット割当。スロット数・番組数は問題文のまま。

    番組がスロットに入る条件は「継続時間 <= 最大継続時間」の閾値型なので、番組の部分集合を
    固定すれば、継続時間の降順とスロットの降順を突き合わせるだけで配置可能かが決まる。
    12 番組から高々 8 つを選ぶ部分集合は 3797 通りなので全列挙する。
    Why not CP-SAT: 視聴率が小数で、整数化の丸めが同点判定に影響する。全列挙なら丸めが要らない。
    """

    def generate(rng: random.Random, base: dict) -> dict:
        n_slots = len(base["time_slots"])
        n_programs = len(base["programs"])
        genres = list(base["genre_minimums"])

        def make() -> dict:
            slots = []
            clock = 8 * 60
            for i in range(n_slots):
                length = rng.choice([30, 60, 60, 90])
                slots.append(
                    {
                        "id": i,
                        "time_slot": f"{_clock(clock)}-{_clock(clock + length)}",
                        "max_duration": length,
                    }
                )
                clock += length
            programs = [
                {
                    "id": i + 1,
                    "name": f"番組{i + 1}",
                    "genre": rng.choice(genres),
                    "duration": rng.choice([30, 60, 90]),
                    "expected_rating": rng.uniform(5.0, 25.0),
                }
                for i in range(n_programs)
            ]
            minimums = {genre: 0 for genre in genres}
            for genre in rng.sample(genres, 2):
                minimums[genre] = rng.randint(1, 2)
            return {"time_slots": slots, "programs": programs, "genre_minimums": minimums}

        def ok(inst: dict) -> bool:
            try:
                solved = solve(inst)
            except SolveError:
                return False
            # 視聴率上位をそのまま並べた値に届くなら、ジャンルも継続時間も効いていない。
            ratings = sorted((p["expected_rating"] for p in inst["programs"]), reverse=True)
            unconstrained = sum(ratings[:n_slots])
            filled = len(solved["lineup"]) >= n_slots - 1
            return filled and solved["total_expected_rating"] < unconstrained - 1e-9

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        slots = sorted(instance["time_slots"], key=lambda s: (-s["max_duration"], s["id"]))
        programs = sorted(instance["programs"], key=lambda p: p["id"])
        minimums = instance["genre_minimums"]
        best_total = None
        best_order: list[dict] = []
        for size in range(min(len(slots), len(programs)) + 1):
            for chosen in itertools.combinations(programs, size):
                counts: dict[str, int] = {}
                for program in chosen:
                    counts[program["genre"]] = counts.get(program["genre"], 0) + 1
                if any(counts.get(genre, 0) < need for genre, need in minimums.items()):
                    continue
                ordered = sorted(chosen, key=lambda p: (-p["duration"], p["id"]))
                if any(p["duration"] > s["max_duration"] for p, s in zip(ordered, slots)):
                    continue
                total = sum(p["expected_rating"] for p in chosen)
                # 同点は先に列挙した（番組 id の辞書順で小さい）部分集合を採る。
                if best_total is None or total > best_total:
                    best_total, best_order = total, ordered
        if best_total is None:
            raise SolveError("no lineup satisfies the genre minimums")
        lineup = {str(p["id"]): s["time_slot"] for p, s in zip(best_order, slots)}
        return {
            "lineup": dict(sorted(lineup.items(), key=lambda kv: int(kv[0]))),
            "total_expected_rating": float(best_total),
        }

    return generate, solve


@register(19, "total_assignments")
def event_staff_roster():
    """prob_019: スキルと利用可能時間帯つきの要員配置。人数・役割・時間帯は問題文のまま。

    最低人数を超える配置は 1 件外しても制約を保つので、最適値は実行可能なら最低人数の総和に一致する。
    それでも配置表そのものは求めなければならないので CP-SAT で解く。
    """

    def generate(rng: random.Random, base: dict) -> dict:
        roles = base["roles"]
        slots = base["time_slots"]
        n_staff = len(base["staff"])
        skills = sorted({role["required_skill"] for role in roles})

        def make() -> dict:
            staff = []
            for i in range(n_staff):
                own = rng.sample(skills, rng.randint(2, 3))
                available = rng.sample([s["id"] for s in slots], rng.randint(4, len(slots)))
                staff.append(
                    {
                        "id": i + 1,
                        "name": f"スタッフ{i + 1}",
                        "skills": own,
                        "available_slots": available,
                    }
                )
            requirements = {
                str(role["id"]): {str(slot["id"]): rng.choice([0, 0, 0, 1, 1, 2]) for slot in slots}
                for role in roles
            }
            return {
                "roles": roles,
                "time_slots": slots,
                "staff": staff,
                "role_requirements": requirements,
            }

        def ok(inst: dict) -> bool:
            try:
                solved = solve(inst)
            except SolveError:
                return False
            return solved["total_assignments"] >= 8

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        roles = instance["roles"]
        slots = instance["time_slots"]
        staff = instance["staff"]
        requirements = instance["role_requirements"]
        roster: dict[str, dict[str, str]] = {str(person["id"]): {} for person in staff}
        total = 0
        # 制約は時間帯をまたがないので、時間帯ごとの小さな割当問題に分けて解く。
        # Why not 全時間帯を 1 つのモデルに: 同点整理の副目的を付けると最適性証明に数十秒かかる。
        for slot in slots:
            cp_model, solver = cp_sat_solver()
            model = cp_model.CpModel()
            x = {
                (s, r): model.NewBoolVar(f"x{s}_{r}")
                for s, person in enumerate(staff)
                if slot["id"] in person["available_slots"]
                for r, role in enumerate(roles)
                if role["required_skill"] in person["skills"]
            }
            for s in range(len(staff)):
                model.AddAtMostOne(x[s, r] for r in range(len(roles)) if (s, r) in x)
            for r, role in enumerate(roles):
                need = requirements[str(role["id"])][str(slot["id"])]
                model.Add(sum(x[s, r] for s in range(len(staff)) if (s, r) in x) >= need)
            count = sum(x.values())
            # 同点の配置を若い番号のスタッフへ寄せて解を一つに近づける。重みは副目的の上限より大きい。
            spread = sum(s * var for (s, _), var in x.items())
            model.Minimize((len(staff) * len(x) + 1) * count + spread)
            require_optimal(cp_model, solver.Solve(model))
            for (s, r), var in x.items():
                if solver.Value(var):
                    roster[str(staff[s]["id"])][slot["time_slot"]] = roles[r]["name"]
                    total += 1
        return {"roster": roster, "total_assignments": int(total), "feasible": True}

    return generate, solve


def _project_activities(instance: dict) -> list[dict]:
    """projects を id 一意・先行活動は同一プロジェクト内の先行行のみ、に正規化して並べる。

    Why not そのまま読む: 同梱 instance は各プロジェクトの全活動が同じ id（1, 6, 11）を持ち、
    先行活動 [1] が自分自身を指す。名前 P2-6 / P3-11 が先頭 id を示すので、
    id が重複する instance だけ「先頭 id + 位置」で振り直す。
    """
    rows = []
    for key in sorted(instance["projects"], key=int):
        rows.extend(instance["projects"][key])
    unique = len({row["id"] for row in rows}) == len(rows)
    activities: list[dict] = []
    for key in sorted(instance["projects"], key=int):
        members = instance["projects"][key]
        ids = [row["id"] if unique else members[0]["id"] + k for k, row in enumerate(members)]
        for k, row in enumerate(members):
            earlier = set(ids[:k])
            activities.append(
                {
                    "id": ids[k],
                    "project": int(key),
                    "duration": row["duration"],
                    "resource": row["resource_usage"],
                    "predecessors": sorted(set(row["predecessors"]) & earlier),
                }
            )
    return activities


@register(20, "peak_resource_usage", shipped_reference_optimal=False)
def multi_project_leveling():
    """prob_020: 共有リソースの上限 8 の下で使用量ピークを最小化する RCPSP 型の平準化。

    同梱参照解は 3 活動しか置かず、ピーク 19 で容量 8 を超える greedy 近似解。
    CP-SAT の cumulative 制約でピークを変数にして厳密に解く。
    プロジェクト数と容量 8 は問題文にあるので固定し、活動数も雛形と同じにする。
    """

    def generate(rng: random.Random, base: dict) -> dict:
        capacity = base["resource_capacity"]
        sizes = {key: len(rows) for key, rows in base["projects"].items()}

        def make() -> dict:
            projects: dict[str, list[dict]] = {}
            next_id = 1
            for key in sorted(sizes, key=int):
                rows = []
                ids: list[int] = []
                for _ in range(sizes[key]):
                    # 先頭以外は必ず先行活動を持たせる（雛形と同じく空と非空の両方が現れる）。
                    preds = sorted(rng.sample(ids, min(len(ids), rng.randint(1, 2)))) if ids else []
                    rows.append(
                        {
                            "id": next_id,
                            "project_id": int(key),
                            "name": f"P{key}-{next_id}",
                            "duration": rng.randint(2, 5),
                            "predecessors": preds,
                            "resource_usage": rng.randint(1, 4),
                        }
                    )
                    ids.append(next_id)
                    next_id += 1
                projects[key] = rows
            return {
                "projects": projects,
                "resource_capacity": capacity,
                "horizon": rng.randint(14, 20),
            }

        def ok(inst: dict) -> bool:
            try:
                solved = solve(inst)
            except SolveError:
                return False
            # ピークが単独活動の使用量に留まるなら重ね合わせを考える必要がなく薄い。
            largest = max(a["resource"] for a in _project_activities(inst))
            return solved["peak_resource_usage"] > largest

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        cp_model, solver = cp_sat_solver()
        activities = _project_activities(instance)
        horizon = instance["horizon"]
        capacity = instance["resource_capacity"]
        model = cp_model.CpModel()
        starts = {}
        intervals = []
        for a in activities:
            start = model.NewIntVar(0, horizon - a["duration"], f"s{a['id']}")
            starts[a["id"]] = start
            intervals.append(model.NewFixedSizeIntervalVar(start, a["duration"], f"i{a['id']}"))
        for a in activities:
            for p in a["predecessors"]:
                pred = next(b for b in activities if b["id"] == p)
                model.Add(starts[a["id"]] >= starts[p] + pred["duration"])
        peak = model.NewIntVar(max(a["resource"] for a in activities), capacity, "peak")
        model.AddCumulative(intervals, [a["resource"] for a in activities], peak)
        # 同点のスケジュールを前詰めに寄せて解を一つに近づける。重みは開始時刻の総和の上限より大きい。
        model.Minimize((len(activities) * horizon + 1) * peak + sum(starts.values()))
        require_optimal(cp_model, solver.Solve(model))
        schedule = {}
        for a in activities:
            begin = solver.Value(starts[a["id"]])
            schedule[str(a["id"])] = {
                "start": int(begin),
                "end": int(begin + a["duration"]),
                "project": a["project"],
            }
        return {
            "schedule": schedule,
            "peak_resource_usage": int(solver.Value(peak)),
            "makespan": max(entry["end"] for entry in schedule.values()),
            "note": "CP-SAT（cumulative 制約、厳密最適解）",
        }

    return generate, solve


# Why not prob_014（スポーツリーグ戦日程）: 各試合の移動距離は instance に固定値で与えられ、
# 30 試合すべてを行う以上、総移動距離はラウンド割りに関係なく一定（合計 8,487 km）になる。
# 「ホーム/アウェイ均衡」も数値化の定義が問題文にない。同梱参照解は 30 試合中 4 試合しか
# 日程に載せず、その 4 試合の距離の和 901 を目的値としているので、「最適」を問題文から一意に
# 定義できない。
# Why not prob_015（教員授業表作成）: 目的は「すべての科目の配置（可行性）」で、参照解の
# total_period_sum=0 は問題文に定義がなく、モデルが再現すべき数値目的にならない。同梱参照解は
# 週 3 時限の科目にも 1 時限しか割り当てておらず、num_periods を無視した形になっている。
