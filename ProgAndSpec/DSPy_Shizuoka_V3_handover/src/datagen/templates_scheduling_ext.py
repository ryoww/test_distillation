"""scheduling 系の雛形（第 2 群: prob_004〜011）。 templates.py と同じ規約で (generate, solve) を登録する。

区間を持つ問題（ジョブショップ・手術室・RCPSP）は CP-SAT の区間変数で解く。目的は
makespan などの主目的に開始時刻の和を下位の目的として重ねた辞書式にし、同点最適解の
中から左詰めのスケジュールを参照解に選ぶ。
"""

from __future__ import annotations

import random
from collections import defaultdict

from .base import SolveError, cp_sat_solver, register, require_optimal, retry


def _overlaps(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    """半開区間 [start, end) が重なるか。同梱参照解は終了時刻と開始時刻の一致を許している。"""
    return start_a < end_b and start_b < end_a


@register(4, "makespan")
def job_shop():
    """prob_004: ジョブショップ J||Cmax。ジョブ数・機械数・各ジョブの工程数は雛形のまま。"""

    def generate(rng: random.Random, base: dict) -> dict:
        m = base["num_machines"]
        counts = [len(job["operations"]) for job in base["jobs"]]

        def make() -> dict:
            jobs = []
            for i, k in enumerate(counts):
                # 同じジョブの工程は別々の機械で加工する（雛形と同じ）。
                machines = rng.sample(range(1, m + 1), k)
                jobs.append(
                    {
                        "id": i + 1,
                        "operations": [
                            {
                                "operation": o + 1,
                                "machine": mc,
                                "processing_time": rng.randint(1, 9),
                            }
                            for o, mc in enumerate(machines)
                        ],
                    }
                )
            return {"num_jobs": len(counts), "num_machines": m, "jobs": jobs}

        def ok(inst: dict) -> bool:
            # 最長ジョブを通すだけで最適になる（機械の競合が効かない）instance は捨てる。
            longest = max(
                sum(op["processing_time"] for op in job["operations"]) for job in inst["jobs"]
            )
            return solve(inst)["makespan"] > longest

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        cp_model, solver = cp_sat_solver()
        model = cp_model.CpModel()
        jobs = instance["jobs"]
        horizon = sum(op["processing_time"] for job in jobs for op in job["operations"])
        by_machine: dict[int, list] = defaultdict(list)
        starts: dict[tuple[int, int], object] = {}
        last_ends = []
        for job in jobs:
            previous_end = None
            for op in job["operations"]:
                key = (job["id"], op["operation"])
                start = model.NewIntVar(0, horizon, f"s{key}")
                end = model.NewIntVar(0, horizon, f"e{key}")
                by_machine[op["machine"]].append(
                    model.NewIntervalVar(start, op["processing_time"], end, f"i{key}")
                )
                if previous_end is not None:
                    model.Add(start >= previous_end)
                previous_end = end
                starts[key] = start
            last_ends.append(previous_end)
        for intervals in by_machine.values():
            model.AddNoOverlap(intervals)
        makespan = model.NewIntVar(0, horizon, "makespan")
        model.AddMaxEquality(makespan, last_ends)
        # 下位目的の開始時刻和は最大でも len(starts) * horizon なので、この重みで辞書式になる。
        model.Minimize(makespan * (len(starts) * horizon + 1) + sum(starts.values()))
        require_optimal(cp_model, solver.Solve(model))
        schedule = {}
        for job in jobs:
            rows = []
            for op in job["operations"]:
                begin = int(solver.Value(starts[job["id"], op["operation"]]))
                rows.append(
                    {
                        "operation": op["operation"],
                        "machine": op["machine"],
                        "start_time": begin,
                        "end_time": begin + op["processing_time"],
                    }
                )
            schedule[str(job["id"])] = rows
        return {"makespan": int(solver.Value(makespan)), "schedule": schedule}

    return generate, solve


@register(5, "preferred_shift_matches", minimize=False)
def nurse_rostering():
    """prob_005: ナース勤務表。人数・日数・連続夜勤上限 2 は問題文にあるので雛形のまま。"""

    def generate(rng: random.Random, base: dict) -> dict:
        nurses = base["num_nurses"]
        days = base["num_days"]
        shifts = list(base["shifts"])
        work_shifts = [s for s in shifts if s != "off"]
        max_night = base["nurses"][0]["max_consecutive_night"]

        def make() -> dict:
            # 各日の必要人数は最大 2+2+2=6 で総人数を超えないので必ず実行可能。
            requirements = {
                str(d + 1): {s: (rng.randint(1, 2) if s != "off" else 0) for s in shifts}
                for d in range(days)
            }
            staff = [
                {
                    "id": i + 1,
                    "name": f"ナース{i + 1}",
                    "preferred_shift": rng.choice(work_shifts),
                    "max_consecutive_night": max_night,
                }
                for i in range(nurses)
            ]
            return {
                "num_nurses": nurses,
                "num_days": days,
                "shifts": shifts,
                "shift_requirements": requirements,
                "nurses": staff,
            }

        def ok(inst: dict) -> bool:
            # 全員が毎日希望どおりに働ける instance は制約が効いていないので捨てる。
            solved = solve(inst)
            return solved["preferred_shift_matches"] < solved["total_assignments"]

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        cp_model, solver = cp_sat_solver()
        model = cp_model.CpModel()
        shifts = list(instance["shifts"])
        nurses = instance["nurses"]
        days = instance["num_days"]
        requirements = instance["shift_requirements"]
        x = {
            (n["id"], d, s): model.NewBoolVar(f"x{n['id']}_{d}_{s}")
            for n in nurses
            for d in range(days)
            for s in shifts
        }
        for n in nurses:
            for d in range(days):
                model.AddExactlyOne(x[n["id"], d, s] for s in shifts)
            # 連続 (上限+1) 日の窓ごとに夜勤を上限以下にすると、連続夜勤が上限を超えない。
            window = n["max_consecutive_night"] + 1
            for d in range(days - window + 1):
                model.Add(sum(x[n["id"], d + k, "night"] for k in range(window)) <= window - 1)
        for d in range(days):
            for s in shifts:
                model.Add(sum(x[n["id"], d, s] for n in nurses) >= requirements[str(d + 1)][s])
        matches = sum(x[n["id"], d, n["preferred_shift"]] for n in nurses for d in range(days))
        model.Maximize(matches)
        require_optimal(cp_model, solver.Solve(model))
        roster = {
            f"nurse_{n['id']}": [
                next(s for s in shifts if solver.Value(x[n["id"], d, s])) for d in range(days)
            ]
            for n in nurses
        }
        total = len(nurses) * days
        matched = int(solver.Value(matches))
        return {
            "roster": roster,
            "preferred_shift_matches": matched,
            "total_assignments": total,
            "match_rate": round(matched / total, 3),
        }

    return generate, solve


def _solve_interval_packing(
    items: list[dict], slots: list[int], fits: dict[tuple[int, int], bool]
) -> tuple[int, dict[str, int]]:
    """時間の重なる item を同じ slot に置かずに、置けない item 数を最小化する。

    items は id/start/end を持つ。fits[(item id, slot id)] が True の組だけ割り当てられる。
    同点は slot 番号の和が小さい割当（若い番号を使う）を選ぶ。
    Returns: (置けなかった件数, {item id: slot id})
    """
    cp_model, solver = cp_sat_solver()
    model = cp_model.CpModel()
    x = {
        (it["id"], slot): model.NewBoolVar(f"x{it['id']}_{slot}")
        for it in items
        for slot in slots
        if fits[it["id"], slot]
    }
    assigned = []
    for it in items:
        own = [x[it["id"], slot] for slot in slots if (it["id"], slot) in x]
        if own:
            model.AddAtMostOne(own)
        assigned.extend(own)
    for i, a in enumerate(items):
        for b in items[i + 1 :]:
            if not _overlaps(a["start"], a["end"], b["start"], b["end"]):
                continue
            for slot in slots:
                if (a["id"], slot) in x and (b["id"], slot) in x:
                    model.AddBoolOr([x[a["id"], slot].Not(), x[b["id"], slot].Not()])
    unassigned = len(items) - sum(assigned)
    weight = sum(slots) * len(items) + 1
    model.Minimize(unassigned * weight + sum(slot * var for (_, slot), var in x.items()))
    require_optimal(cp_model, solver.Solve(model))
    assignment = {str(item_id): slot for (item_id, slot), var in x.items() if solver.Value(var)}
    return int(solver.Value(unassigned)), assignment


def _contention_matters(items: list[dict], slots: list[int], fits: dict, unassigned: int) -> bool:
    """置けなかった件数が「どの slot にも条件で入らない item」の数を超えているか。

    超えていなければ時間の重なりが効いておらず、条件だけで答えが決まる薄い instance。
    """
    hopeless = sum(1 for it in items if not any(fits[it["id"], slot] for slot in slots))
    return unassigned > hopeless


@register(6, "unassigned_count")
def meeting_rooms():
    """prob_006: 会議室割当。部屋数と会議数は問題文にあるので雛形のまま。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n_rooms = len(base["rooms"])
        n_meetings = len(base["meetings"])

        def make() -> dict:
            rooms = [
                {
                    "id": i + 1,
                    "name": f"会議室{i + 1}",
                    "capacity": rng.choice([10, 15, 20, 25, 30, 40]),
                    "has_projector": rng.random() < 0.6,
                }
                for i in range(n_rooms)
            ]
            meetings = []
            for i in range(n_meetings):
                start = rng.randint(9, 16)
                meetings.append(
                    {
                        "id": i + 1,
                        "name": f"会議{i + 1}",
                        "start_time": start,
                        "duration": rng.randint(1, min(3, 18 - start)),
                        "attendees": rng.randint(5, 45),
                        "needs_projector": rng.random() < 0.4,
                    }
                )
            return {"rooms": rooms, "meetings": meetings}

        def ok(inst: dict) -> bool:
            items, slots, fits = _meeting_model(inst)
            return _contention_matters(items, slots, fits, solve(inst)["unassigned_count"])

        return retry(rng, make, ok)

    def _meeting_model(instance: dict):
        items = [
            {"id": m["id"], "start": m["start_time"], "end": m["start_time"] + m["duration"]}
            for m in instance["meetings"]
        ]
        slots = [r["id"] for r in instance["rooms"]]
        fits = {
            (m["id"], r["id"]): r["capacity"] >= m["attendees"]
            and (r["has_projector"] or not m["needs_projector"])
            for m in instance["meetings"]
            for r in instance["rooms"]
        }
        return items, slots, fits

    def solve(instance: dict) -> dict:
        unassigned, assignment = _solve_interval_packing(*_meeting_model(instance))
        return {"room_assignment": assignment, "unassigned_count": unassigned}

    return generate, solve


_SIZE_RANK = {"small": 0, "medium": 1, "large": 2}


@register(7, "unassigned_count")
def gate_assignment():
    """prob_007: ゲート割当。ゲート数とフライト数は問題文にあるので雛形のまま。"""

    def generate(rng: random.Random, base: dict) -> dict:
        n_gates = len(base["gates"])
        n_flights = len(base["flights"])
        sizes = list(_SIZE_RANK)

        def make() -> dict:
            gates = [
                {"id": i + 1, "name": f"ゲート{i + 1}", "size_class": rng.choice(sizes)}
                for i in range(n_gates)
            ]
            flights = []
            for i in range(n_flights):
                arrival = rng.randint(600, 1300)
                flights.append(
                    {
                        "id": i + 1,
                        "arrival_time": arrival,
                        "departure_time": arrival + rng.randint(40, 120),
                        "aircraft_size": rng.choice(sizes),
                    }
                )
            return {"gates": gates, "flights": flights}

        def ok(inst: dict) -> bool:
            items, slots, fits = _gate_model(inst)
            return _contention_matters(items, slots, fits, solve(inst)["unassigned_count"])

        return retry(rng, make, ok)

    def _gate_model(instance: dict):
        items = [
            {"id": f["id"], "start": f["arrival_time"], "end": f["departure_time"]}
            for f in instance["flights"]
        ]
        slots = [g["id"] for g in instance["gates"]]
        fits = {
            (f["id"], g["id"]): _SIZE_RANK[g["size_class"]] >= _SIZE_RANK[f["aircraft_size"]]
            for f in instance["flights"]
            for g in instance["gates"]
        }
        return items, slots, fits

    def solve(instance: dict) -> dict:
        unassigned, assignment = _solve_interval_packing(*_gate_model(instance))
        return {"gate_assignment": assignment, "unassigned_count": unassigned}

    return generate, solve


@register(8, "total_start_time")
def operating_rooms():
    """prob_008: 手術室スケジュール。部屋 3（種別固定）・手術 6 は問題文どおり、医師は instance の 4 人。

    問題文の目的は「全手術の組込み（可行性）」だが参照解は total_start_time（開始時刻の和）を
    持つので、開始を 1 時間刻みにして開始時刻の和を最小化する解を参照解にする。同梱参照解の
    64 はこの解釈で再現できる。
    Why not 分刻みの開始: 同梱参照解は全手術が正時に始まり、分刻みでは 64 を再現できない。
    """
    day_start, day_end = 8, 18

    def generate(rng: random.Random, base: dict) -> dict:
        rooms = [dict(r) for r in base["operating_rooms"]]
        surgeons = [dict(s) for s in base["surgeons"]]
        types = [r["type"] for r in rooms]
        n = len(base["surgeries"])

        def make() -> dict:
            surgeries = [
                {
                    "id": i + 1,
                    "surgeon_id": rng.choice(surgeons)["id"],
                    "duration": rng.randint(60, 180),
                    "type": rng.choice(types),
                    "priority": rng.randint(1, 3),
                }
                for i in range(n)
            ]
            return {"operating_rooms": rooms, "surgeons": surgeons, "surgeries": surgeries}

        def ok(inst: dict) -> bool:
            # 医師の重複制約を外しても同じ値なら、部屋ごとの SPT 順で解ける薄い instance。
            try:
                with_surgeons = _solve(inst, respect_surgeons=True)[0]
            except SolveError:
                return False
            return with_surgeons > _solve(inst, respect_surgeons=False)[0]

        return retry(rng, make, ok)

    def _solve(instance: dict, *, respect_surgeons: bool) -> tuple[int, dict[int, int]]:
        cp_model, solver = cp_sat_solver()
        model = cp_model.CpModel()
        room_of = {}
        for room in instance["operating_rooms"]:
            if room["type"] in room_of:
                raise ValueError("expects exactly one room per type")
            room_of[room["type"]] = room["id"]
        by_room: dict[int, list] = defaultdict(list)
        by_surgeon: dict[int, list] = defaultdict(list)
        starts = {}
        for s in instance["surgeries"]:
            # 正時開始なので、次の手術は前の手術の終了を切り上げた時刻から始められる。
            hours = -(-s["duration"] // 60)
            start = model.NewIntVar(day_start, day_end - hours, f"s{s['id']}")
            interval = model.NewIntervalVar(start, hours, start + hours, f"i{s['id']}")
            by_room[room_of[s["type"]]].append(interval)
            by_surgeon[s["surgeon_id"]].append(interval)
            starts[s["id"]] = start
        for intervals in by_room.values():
            model.AddNoOverlap(intervals)
        if respect_surgeons:
            for intervals in by_surgeon.values():
                model.AddNoOverlap(intervals)
        model.Minimize(sum(starts.values()))
        require_optimal(cp_model, solver.Solve(model))
        return int(solver.ObjectiveValue()), {
            s["id"]: int(solver.Value(starts[s["id"]])) for s in instance["surgeries"]
        }

    def solve(instance: dict) -> dict:
        total, starts = _solve(instance, respect_surgeons=True)
        room_of = {r["type"]: r["id"] for r in instance["operating_rooms"]}
        schedule = {
            str(s["id"]): {
                "room": room_of[s["type"]],
                "start_time": starts[s["id"]],
                "end_time": starts[s["id"]] + s["duration"] / 60,
            }
            for s in instance["surgeries"]
        }
        return {"schedule": schedule, "total_start_time": total, "all_scheduled": True}

    return generate, solve


@register(11, "makespan", shipped_reference_optimal=False)
def rcpsp():
    """prob_011: RCPSP。活動数と資源容量（作業員 6、設備 3）は問題文にあるので雛形のまま。

    同梱参照解は note のとおり「greedy+shifting近似解」で、makespan 20 のうえに活動 5 と 7 が
    重なって作業員 7 人（peak_workers も 7 と申告）と容量 6 を超えている。CP-SAT の厳密解は 17。
    """

    def generate(rng: random.Random, base: dict) -> dict:
        n = len(base["activities"])
        capacity = dict(base["resource_capacity"])

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
                        "duration": rng.randint(1, 6),
                        "predecessors": sorted(rng.sample(pool, k)),
                        "resource_usage": {
                            "workers": rng.randint(1, 4),
                            "equipment": rng.randint(0, 2),
                        },
                    }
                )
            return {"activities": activities, "resource_capacity": capacity}

        def ok(inst: dict) -> bool:
            acts = inst["activities"]
            if sum(len(a["predecessors"]) for a in acts) < n // 2:
                return False
            # 資源制約を無視した最短（クリティカルパス長）で終わる instance は資源が効いていない。
            duration = {a["id"]: a["duration"] for a in acts}
            earliest: dict[int, int] = {}
            for a in acts:
                earliest[a["id"]] = max(
                    (earliest[p] + duration[p] for p in a["predecessors"]), default=0
                )
            critical = max(earliest[a["id"]] + a["duration"] for a in acts)
            return solve(inst)["makespan"] > critical

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        cp_model, solver = cp_sat_solver()
        model = cp_model.CpModel()
        acts = instance["activities"]
        capacity = instance["resource_capacity"]
        horizon = sum(a["duration"] for a in acts)
        starts, ends, intervals = {}, {}, {}
        for a in acts:
            starts[a["id"]] = model.NewIntVar(0, horizon, f"s{a['id']}")
            ends[a["id"]] = model.NewIntVar(0, horizon, f"e{a['id']}")
            intervals[a["id"]] = model.NewIntervalVar(
                starts[a["id"]], a["duration"], ends[a["id"]], f"i{a['id']}"
            )
        for a in acts:
            for p in a["predecessors"]:
                model.Add(starts[a["id"]] >= ends[p])
        for resource, cap in capacity.items():
            model.AddCumulative(
                [intervals[a["id"]] for a in acts],
                [a["resource_usage"][resource] for a in acts],
                cap,
            )
        makespan = model.NewIntVar(0, horizon, "makespan")
        model.AddMaxEquality(makespan, list(ends.values()))
        model.Minimize(makespan * (len(acts) * horizon + 1) + sum(starts.values()))
        require_optimal(cp_model, solver.Solve(model))
        schedule = {
            str(a["id"]): {
                "start_time": int(solver.Value(starts[a["id"]])),
                "end_time": int(solver.Value(ends[a["id"]])),
            }
            for a in acts
        }

        def peak(resource: str) -> int:
            rows = [(schedule[str(a["id"])], a["resource_usage"][resource]) for a in acts]
            usage = [
                sum(use for row, use in rows if row["start_time"] <= t < row["end_time"])
                for t in range(int(solver.Value(makespan)))
            ]
            return max(usage, default=0)

        return {
            "schedule": schedule,
            "makespan": int(solver.Value(makespan)),
            "peak_workers": peak("workers"),
            "peak_equipment": peak("equipment"),
            "note": "CP-SAT（厳密最適解）",
        }

    return generate, solve
