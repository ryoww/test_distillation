"""旧チェッカーの core_type に属する参照解固有の形を、解析してから厳密に検証する。

feasibility.py の旧チェッカーは要素数と目的値の有無しか見ない。ここでは同梱参照解が
使う4つの形（並列機械の machine_assignment、フローショップの optimal_sequence、
クラスタ割当の node_assignment、配送＋在庫の day1_routes）について、ID の範囲、重複、
容量、目的値の再計算まで行う。形を読めなければ None を返し、呼び出し側の従来処理に戻す。
"""

from __future__ import annotations

from typing import Any

from .feasibility_v3_ext import (
    _SOFT_ABS_TOL,
    _close,
    _close_soft,
    _index_map,
    _num,
    _result,
    _unverified,
)


def _job_ids(jobs: list[dict]) -> list[Any]:
    return [job.get("id", i) for i, job in enumerate(jobs)]


def check_parallel_machine_assignment(instance: dict, solution: dict) -> dict | None:
    """machine_assignment {job_id: machine(1始まり)} と makespan の整合を検証する。"""
    raw = solution.get("machine_assignment")
    jobs = instance.get("jobs")
    machines = _num(instance.get("num_machines"))
    if raw is None or not isinstance(jobs, list) or machines is None:
        return None
    machines = int(machines)
    assignment = _index_map(raw)
    if assignment is None:
        return _unverified("machine_assignment is not a job -> machine mapping")
    times = {
        job.get("id", i): _num(job.get("processing_time")) or 0.0 for i, job in enumerate(jobs)
    }
    # Why not job -> machine だけ: 機械 -> [ジョブ id] で返す解も同じ割当を表す。値が全部
    # リストなら転置して読む。機械番号は 0 始まりでも 1 始まりでもよい（下の判定に任せる）。
    if assignment and all(isinstance(v, list) for v in assignment.values()):
        transposed: dict[Any, Any] = {}
        for machine, members in assignment.items():
            for job_id in members:
                key = job_id if job_id in times else _num(job_id)
                transposed[key] = machine
        assignment = transposed
    violations: list[str] = []
    # Why not 1始まりに固定: instance は台数しか与えないので、0始まりで返す解も正当。
    # 全ジョブが 0..m-1 に収まるなら 0始まり、1..m に収まるなら 1始まりとして読む。
    values = [_num(v) for v in assignment.values()]
    numeric = [v for v in values if v is not None and float(v).is_integer()]
    zero_based = bool(numeric) and all(0 <= v < machines for v in numeric)
    one_based = bool(numeric) and all(1 <= v <= machines for v in numeric)
    low, high = (0, machines - 1) if zero_based and not one_based else (1, machines)
    loads: dict[int, float] = {}
    for job_id in _job_ids(jobs):
        machine = _num(assignment.get(job_id))
        if machine is None:
            violations.append(f"job {job_id} is not assigned")
            continue
        if not float(machine).is_integer() or not low <= machine <= high:
            violations.append(f"job {job_id} assigned to unknown machine {machine:g}")
            continue
        loads[int(machine)] = loads.get(int(machine), 0.0) + times[job_id]
    unknown = sorted(set(assignment) - set(times), key=str)
    if unknown:
        violations.append(f"assignment names unknown jobs: {unknown}")
    makespan = max(loads.values(), default=0.0)
    claimed = _num(solution.get("makespan"))
    if claimed is None:
        violations.append("no makespan field")
    elif not _close(claimed, makespan):
        violations.append(f"makespan {claimed:g} != max machine load {makespan:g}")
    reported = _index_map(solution.get("machine_loads")) if "machine_loads" in solution else None
    if reported is not None:
        # 機械番号の付け方は解ごとに違うので、負荷の多重集合で比較する。
        declared = sorted(_num(v) or 0.0 for v in reported.values())
        actual = sorted(loads.get(m, 0.0) for m in range(low, high + 1))
        if len(declared) != len(actual) or any(not _close(a, b) for a, b in zip(declared, actual)):
            violations.append("machine_loads do not match the loads implied by the assignment")
    return _result(violations, len(jobs) + 2, cost=makespan)


def _flow_shop_makespan(
    sequence: list[Any], jobs_by_id: dict[Any, dict], stages: list[str]
) -> float:
    """順列フローショップの完了時刻。各段は1台なので前ジョブの完了と前段の完了の遅いほうから始まる。"""
    finish = [0.0] * len(stages)
    for job_id in sequence:
        times = jobs_by_id[job_id].get("processing_times") or {}
        previous = 0.0
        for k, stage in enumerate(stages):
            start = max(previous, finish[k])
            finish[k] = start + (_num(times.get(stage)) or 0.0)
            previous = finish[k]
    return finish[-1] if stages else 0.0


def check_flow_shop_sequence(instance: dict, solution: dict) -> dict | None:
    """optimal_sequence が全ジョブの順列で、makespan が再計算値と一致するか検証する。"""
    sequence = solution.get("optimal_sequence")
    jobs = instance.get("jobs")
    if sequence is None or "num_stages" not in instance or not isinstance(jobs, list):
        return None
    if not isinstance(sequence, list):
        return _unverified("optimal_sequence is not a list")
    violations: list[str] = []
    jobs_by_id = {job.get("id", i): job for i, job in enumerate(jobs)}
    if len(set(sequence)) != len(sequence):
        violations.append("optimal_sequence repeats a job")
    missing = [job_id for job_id in jobs_by_id if job_id not in sequence]
    if missing:
        violations.append(f"jobs not sequenced: {missing}")
    unknown = [job_id for job_id in sequence if job_id not in jobs_by_id]
    if unknown:
        violations.append(f"unknown jobs in sequence: {unknown}")
    stages = sorted(
        {stage for job in jobs for stage in (job.get("processing_times") or {})},
        key=lambda name: _num(str(name).rsplit("_", 1)[-1]) or 0.0,
    )
    valid = [job_id for job_id in sequence if job_id in jobs_by_id]
    makespan = _flow_shop_makespan(valid, jobs_by_id, stages) if not violations else 0.0
    claimed = _num(solution.get("makespan"))
    if claimed is None:
        violations.append("no makespan field")
    elif not violations and not _close(claimed, makespan):
        violations.append(f"makespan {claimed:g} != flow shop makespan {makespan:g}")
    return _result(violations, 4, cost=makespan)


def check_cluster_node_assignment(instance: dict, solution: dict) -> dict | None:
    """node_assignment {job_id: node_id} が、ノードごとの要求合計を容量内に収めるか検証する。"""
    raw = solution.get("node_assignment")
    nodes = instance.get("nodes")
    jobs = instance.get("jobs")
    if raw is None or not isinstance(nodes, list) or not isinstance(jobs, list):
        return None
    assignment = _index_map(raw)
    if assignment is None:
        return _unverified("node_assignment is not a job -> node mapping")
    violations: list[str] = []
    nodes_by_id = {node.get("id", i + 1): node for i, node in enumerate(nodes)}
    jobs_by_id = {job.get("id", i + 1): job for i, job in enumerate(jobs)}
    requirements = (
        ("cpu_required", "cpu_cores"),
        ("gpu_required", "gpu_count"),
        ("memory_required_gb", "memory_gb"),
    )
    priority = 0.0
    assigned = 0
    loads: dict[int, dict[str, float]] = {}
    for job_id, node_raw in assignment.items():
        job = jobs_by_id.get(job_id)
        if job is None:
            violations.append(f"unknown job {job_id} in node_assignment")
            continue
        node_id = _num(node_raw)
        node = nodes_by_id.get(int(node_id)) if node_id is not None else None
        if node is None:
            violations.append(f"job {job_id} assigned to unknown node {node_raw}")
            continue
        usage = loads.setdefault(int(node_id), {})
        for need_key, _cap_key in requirements:
            usage[need_key] = usage.get(need_key, 0.0) + (_num(job.get(need_key)) or 0.0)
        priority += _num(job.get("priority")) or 0.0
        assigned += 1
    # Why not ジョブ単体と容量の比較: 単体なら同梱 instance の全ジョブがどこかに単独で収まり、
    # 目的値が全優先度の和に自明に決まる。同居するジョブの合計で読む（雛形ソルバーも合算）。
    for node_id, usage in loads.items():
        for need_key, cap_key in requirements:
            cap = _num(nodes_by_id[node_id].get(cap_key)) or 0.0
            if usage[need_key] > cap:
                violations.append(
                    f"node {node_id} {cap_key}={cap:g} < total {need_key} {usage[need_key]:g}"
                )
    claimed = _num(solution.get("total_priority"))
    if claimed is not None and not _close(claimed, priority):
        violations.append(f"total_priority {claimed:g} != sum of assigned priorities {priority:g}")
    count = _num(solution.get("assigned_count"))
    if count is not None and not _close(count, assigned):
        violations.append(f"assigned_count {count:g} != {assigned} assigned jobs")
    return _result(violations, len(jobs) + 2, cost=priority)


def check_day1_routes(instance: dict, solution: dict) -> dict | None:
    """day1_routes が倉庫発着で顧客を重複なく回り、台数と距離申告が整合するか検証する。

    dict でも list でも受け、訪問先は顧客の部分集合でよい。
    """
    routes = solution.get("day1_routes")
    customers = instance.get("customers")
    if routes is None or not isinstance(customers, list):
        return None
    entries = list(routes.items()) if isinstance(routes, dict) else list(enumerate(routes))
    if not isinstance(routes, (dict, list)) or not entries:
        return _unverified("day1_routes is not a non-empty mapping or list")
    violations: list[str] = []
    depot_id = (instance.get("warehouse") or {}).get("id", 0)
    customer_ids = {customer.get("id", i + 1) for i, customer in enumerate(customers)}
    vehicles = _num(instance.get("num_vehicles"))
    if vehicles is not None and len(entries) > vehicles:
        violations.append(f"{len(entries)} routes exceed {vehicles:g} vehicles")
    visited: list[Any] = []
    declared = 0.0
    for name, entry in entries:
        path = entry.get("route") if isinstance(entry, dict) else entry
        if not isinstance(path, list) or len(path) < 3:
            violations.append(f"route {name} is not a depot-to-depot path")
            continue
        if path[0] != depot_id or path[-1] != depot_id:
            violations.append(f"route {name} does not start and end at the warehouse")
        inner = path[1:-1]
        unknown = [node for node in inner if node not in customer_ids]
        if unknown:
            violations.append(f"route {name} visits unknown nodes {unknown}")
        visited.extend(node for node in inner if node in customer_ids)
        if isinstance(entry, dict):
            declared += _num(entry.get("distance")) or 0.0
    if len(set(visited)) != len(visited):
        violations.append("a customer is visited more than once")
    # Why not 全顧客の訪問を要求: 問題文は「配送を行うかどうかを判断」する問題で、
    # 発注点未満の顧客だけを回る解が正当。ただし誰も訪ねない経路は解ではない。
    if not visited:
        violations.append("no customer is visited")
    total = _num(solution.get("total_distance"))
    if total is None or total <= 0:
        violations.append("total_distance missing or not positive")
    elif declared and not _close_soft(total, declared):
        violations.append(f"total_distance {total:g} != sum of route distances {declared:g}")
    return _result(violations, len(customers) + 3, cost=total)


def _lookup(mapping: dict[int, Any], key: Any) -> Any:
    """_index_map の int 鍵を、instance 側の id（int または数字文字列）で引く。"""
    idx = _num(key)
    if idx is None or not float(idx).is_integer():
        return None
    return mapping.get(int(idx))


def _span(entry: Any, start_keys: tuple[str, ...], end_keys: tuple[str, ...]) -> tuple | None:
    """行から (start, end) を数値で読む。読めなければ None。"""
    if not isinstance(entry, dict):
        return None
    start = next((_num(entry[k]) for k in start_keys if k in entry), None)
    end = next((_num(entry[k]) for k in end_keys if k in entry), None)
    if start is None or end is None:
        return None
    return start, end


def _sequential_overlaps(spans: list[tuple[float, float, Any]], label: str) -> list[str]:
    """同じ資源に載った半開区間 [start, end) の重なりを報告する。"""
    violations = []
    latest_end = None
    latest_name = None
    for start, end, name in sorted(spans, key=lambda s: (s[0], s[1])):
        if latest_end is not None and start < latest_end:
            violations.append(f"{label}: {latest_name} and {name} overlap")
        if latest_end is None or end > latest_end:
            latest_end, latest_name = end, name
    return violations


def check_job_shop_schedule(instance: dict, solution: dict) -> dict | None:
    """schedule {job: [{operation, machine, start_time, end_time}]} が工程順・機械の排他・処理時間を
    守り、makespan が最終完了時刻と一致するか検証する。"""
    raw = solution.get("schedule")
    jobs = instance.get("jobs")
    if raw is None or not isinstance(jobs, list) or not jobs:
        return None
    if not all(isinstance(job, dict) and isinstance(job.get("operations"), list) for job in jobs):
        return None
    schedule = _index_map(raw)
    if schedule is None:
        return _unverified("schedule is not a job -> operations mapping")
    violations: list[str] = []
    by_machine: dict[Any, list[tuple[float, float, Any]]] = {}
    finish = 0.0
    for i, job in enumerate(jobs):
        job_id = job.get("id", i + 1)
        rows = _lookup(schedule, job_id)
        if isinstance(rows, dict):
            rows = [rows[k] for k in sorted(rows, key=lambda k: _num(k) or 0.0)]
        operations = job["operations"]
        if not isinstance(rows, list) or len(rows) != len(operations):
            violations.append(f"job {job_id} does not schedule all {len(operations)} operations")
            continue
        if all(isinstance(r, dict) and _num(r.get("operation")) is not None for r in rows):
            rows = sorted(rows, key=lambda r: _num(r.get("operation")))
        previous_end = 0.0
        for op, row in zip(operations, rows):
            span = _span(row, ("start_time", "start"), ("end_time", "end"))
            if span is None:
                violations.append(f"job {job_id} operation {op.get('operation')} has no times")
                break
            start, end = span
            machine = _num(row.get("machine"))
            if machine is None or not _close(machine, _num(op.get("machine")) or -1):
                violations.append(f"job {job_id} operation {op.get('operation')} on wrong machine")
            if not _close(end - start, _num(op.get("processing_time")) or 0.0):
                violations.append(
                    f"job {job_id} operation {op.get('operation')} does not run for its full time"
                )
            if start < previous_end:
                violations.append(f"job {job_id} starts operation {op.get('operation')} early")
            previous_end = end
            finish = max(finish, end)
            by_machine.setdefault(op.get("machine"), []).append((start, end, f"job {job_id}"))
    for machine, spans in by_machine.items():
        violations.extend(_sequential_overlaps(spans, f"machine {machine}"))
    claimed = _num(solution.get("makespan"))
    if claimed is None:
        violations.append("no makespan field")
    elif not _close(claimed, finish):
        violations.append(f"makespan {claimed:g} != last completion {finish:g}")
    return _result(violations, len(jobs) + len(by_machine) + 1, cost=finish)


def check_nurse_roster(instance: dict, solution: dict) -> dict | None:
    """roster {nurse: [shift per day]} が各日の必要人数と連続夜勤上限を守り、希望一致数が合うか。"""
    raw = solution.get("roster")
    nurses = instance.get("nurses")
    requirements = _index_map(instance.get("shift_requirements"))
    if raw is None or not isinstance(nurses, list) or not requirements:
        return None
    roster = _index_map(raw)
    if roster is None:
        return _unverified("roster is not a nurse -> shifts mapping")
    shifts = instance.get("shifts") or sorted({s for day in requirements.values() for s in day})
    days = int(_num(instance.get("num_days")) or len(requirements))
    day_offset = 0 if 0 in requirements else 1
    violations: list[str] = []
    counts: dict[tuple[int, str], int] = {}
    matches = 0
    for i, nurse in enumerate(nurses):
        nurse_id = nurse.get("id", i + 1)
        row = _lookup(roster, nurse_id)
        if isinstance(row, dict):
            row = [row[k] for k in sorted(row, key=lambda k: _num(k) or 0.0)]
        if not isinstance(row, list) or len(row) != days:
            violations.append(f"nurse {nurse_id} does not have exactly one shift on each day")
            continue
        unknown = [s for s in row if s not in shifts]
        if unknown:
            violations.append(f"nurse {nurse_id} has unknown shifts {unknown}")
            continue
        run = longest = 0
        for d, shift in enumerate(row):
            counts[d, shift] = counts.get((d, shift), 0) + 1
            matches += shift == nurse.get("preferred_shift")
            run = run + 1 if shift == "night" else 0
            longest = max(longest, run)
        limit = _num(nurse.get("max_consecutive_night"))
        if limit is not None and longest > limit:
            violations.append(f"nurse {nurse_id} works {longest} consecutive nights > {limit:g}")
    for d in range(days):
        for shift, need in (requirements.get(d + day_offset) or {}).items():
            have = counts.get((d, shift), 0)
            if have < (_num(need) or 0.0):
                violations.append(f"day {d + day_offset} {shift}: {have} nurses < required {need}")
    total = len(nurses) * days
    claimed = _num(solution.get("preferred_shift_matches"))
    if claimed is None:
        violations.append("no preferred_shift_matches field")
    elif not _close(claimed, matches):
        violations.append(f"preferred_shift_matches {claimed:g} != recomputed {matches}")
    declared_total = _num(solution.get("total_assignments"))
    if declared_total is not None and not _close(declared_total, total):
        violations.append(f"total_assignments {declared_total:g} != {total}")
    rate = _num(solution.get("match_rate"))
    if rate is not None and total and not _close_soft(rate, matches / total):
        violations.append(f"match_rate {rate:g} != {matches}/{total}")
    return _result(violations, len(nurses) + days + 1, cost=float(matches))


def _interval_packing(
    items: dict[Any, tuple[float, float]],
    slot_ids: set[Any],
    assignment: dict[int, Any],
    misfit: Any,
    names: tuple[str, str],
) -> tuple[list[str], int]:
    """区間つき item を slot に重ねずに置く割当の違反と、置かれなかった item 数を返す。

    misfit(item_id, slot_id) は条件違反の説明文か None を返す。
    """
    violations: list[str] = []
    per_slot: dict[Any, list[tuple[float, float, Any]]] = {}
    placed = 0
    for item_id, slot_raw in assignment.items():
        if item_id not in items:
            violations.append(f"unknown {names[0]} {item_id} in assignment")
            continue
        slot = _num(slot_raw)
        if slot is None or int(slot) not in slot_ids:
            violations.append(f"{names[0]} {item_id} assigned to unknown {names[1]} {slot_raw}")
            continue
        placed += 1
        reason = misfit(item_id, int(slot))
        if reason:
            violations.append(reason)
        start, end = items[item_id]
        per_slot.setdefault(int(slot), []).append((start, end, f"{names[0]} {item_id}"))
    for slot, spans in per_slot.items():
        violations.extend(_sequential_overlaps(spans, f"{names[1]} {slot}"))
    return violations, len(items) - placed


def _check_unassigned_count(
    solution: dict, violations: list[str], unassigned: int, total: int
) -> dict:
    claimed = _num(solution.get("unassigned_count"))
    if claimed is None:
        violations.append("no unassigned_count field")
    elif not _close(claimed, unassigned):
        violations.append(f"unassigned_count {claimed:g} != {unassigned} left unassigned")
    return _result(violations, total + 2, cost=float(unassigned))


def check_meeting_room_assignment(instance: dict, solution: dict) -> dict | None:
    """room_assignment {meeting: room} が定員・プロジェクター・時間の重なりを守るか検証する。"""
    raw = solution.get("room_assignment")
    rooms = instance.get("rooms")
    meetings = instance.get("meetings")
    if raw is None or not isinstance(rooms, list) or not isinstance(meetings, list):
        return None
    assignment = _index_map(raw)
    if assignment is None:
        return _unverified("room_assignment is not a meeting -> room mapping")
    rooms_by_id = {room.get("id", i + 1): room for i, room in enumerate(rooms)}
    meetings_by_id = {m.get("id", i + 1): m for i, m in enumerate(meetings)}
    items = {
        mid: (_num(m.get("start_time")) or 0.0, (_num(m.get("start_time")) or 0.0)
              + (_num(m.get("duration")) or 0.0))
        for mid, m in meetings_by_id.items()
    }

    def misfit(meeting_id: Any, room_id: Any) -> str | None:
        meeting, room = meetings_by_id[meeting_id], rooms_by_id[room_id]
        if (_num(room.get("capacity")) or 0.0) < (_num(meeting.get("attendees")) or 0.0):
            return f"meeting {meeting_id} has more attendees than room {room_id} holds"
        if meeting.get("needs_projector") and not room.get("has_projector"):
            return f"meeting {meeting_id} needs a projector but room {room_id} has none"
        return None

    violations, unassigned = _interval_packing(
        items, set(rooms_by_id), assignment, misfit, ("meeting", "room")
    )
    return _check_unassigned_count(solution, violations, unassigned, len(meetings))


_SIZE_RANK = {"small": 0, "medium": 1, "large": 2}


def check_gate_assignment(instance: dict, solution: dict) -> dict | None:
    """gate_assignment {flight: gate} がサイズ適合と時間の重なりを守るか検証する。"""
    raw = solution.get("gate_assignment")
    gates = instance.get("gates")
    flights = instance.get("flights")
    if raw is None or not isinstance(gates, list) or not isinstance(flights, list):
        return None
    assignment = _index_map(raw)
    if assignment is None:
        return _unverified("gate_assignment is not a flight -> gate mapping")
    gates_by_id = {gate.get("id", i + 1): gate for i, gate in enumerate(gates)}
    flights_by_id = {f.get("id", i + 1): f for i, f in enumerate(flights)}
    items = {
        fid: (_num(f.get("arrival_time")) or 0.0, _num(f.get("departure_time")) or 0.0)
        for fid, f in flights_by_id.items()
    }

    def misfit(flight_id: Any, gate_id: Any) -> str | None:
        gate_size = _SIZE_RANK.get(gates_by_id[gate_id].get("size_class"), -1)
        aircraft = _SIZE_RANK.get(flights_by_id[flight_id].get("aircraft_size"), 99)
        if gate_size < aircraft:
            return f"flight {flight_id} is too large for gate {gate_id}"
        return None

    violations, unassigned = _interval_packing(
        items, set(gates_by_id), assignment, misfit, ("flight", "gate")
    )
    return _check_unassigned_count(solution, violations, unassigned, len(flights))


def check_operating_room_schedule(instance: dict, solution: dict) -> dict | None:
    """schedule {surgery: {room, start_time, end_time}} が部屋種別・部屋と医師の排他・
    8:00-18:00 の枠を守り、total_start_time が開始時刻の和と一致するか検証する。"""
    raw = solution.get("schedule")
    rooms = instance.get("operating_rooms")
    surgeries = instance.get("surgeries")
    if raw is None or not isinstance(rooms, list) or not isinstance(surgeries, list):
        return None
    schedule = _index_map(raw)
    if schedule is None:
        return _unverified("schedule is not a surgery -> slot mapping")
    rooms_by_id = {room.get("id", i + 1): room for i, room in enumerate(rooms)}
    spans = {}
    for i, surgery in enumerate(surgeries):
        span = _span(_lookup(schedule, surgery.get("id", i + 1)), ("start_time", "start"),
                     ("end_time", "end"))
        if span is not None:
            spans[surgery.get("id", i + 1)] = span
    # Why not 時間単位に固定: 参照解は時間（14 → 16.6167）だが、分で返す解も同じ計画を表す。
    # 終了時刻が 24 を超えていれば分と読む。
    scale = 60.0 if any(end > 24 for _, end in spans.values()) else 1.0
    day_start, day_end = 8 * scale, 18 * scale
    violations: list[str] = []
    by_room: dict[Any, list] = {}
    by_surgeon: dict[Any, list] = {}
    total_start = 0.0
    for i, surgery in enumerate(surgeries):
        sid = surgery.get("id", i + 1)
        if sid not in spans:
            violations.append(f"surgery {sid} is not scheduled")
            continue
        start, end = spans[sid]
        entry = _lookup(schedule, sid)
        room = _lookup(rooms_by_id, entry.get("room", entry.get("room_id")))
        if room is None:
            violations.append(f"surgery {sid} is in an unknown room")
        elif room.get("type") != surgery.get("type"):
            violations.append(f"surgery {sid} is in a {room.get('type')} room")
        else:
            by_room.setdefault(room.get("id"), []).append((start, end, f"surgery {sid}"))
        duration = (_num(surgery.get("duration")) or 0.0) * scale / 60.0
        if not _close_soft(end - start, duration):
            violations.append(f"surgery {sid} does not run for its duration")
        if start < day_start or end > day_end + _SOFT_ABS_TOL:
            violations.append(f"surgery {sid} falls outside 8:00-18:00")
        by_surgeon.setdefault(surgery.get("surgeon_id"), []).append((start, end, f"surgery {sid}"))
        total_start += start
    for room_id, room_spans in by_room.items():
        violations.extend(_sequential_overlaps(room_spans, f"room {room_id}"))
    for surgeon_id, surgeon_spans in by_surgeon.items():
        violations.extend(_sequential_overlaps(surgeon_spans, f"surgeon {surgeon_id}"))
    claimed = _num(solution.get("total_start_time"))
    if claimed is not None and not _close_soft(claimed, total_start):
        violations.append(f"total_start_time {claimed:g} != sum of start times {total_start:g}")
    return _result(violations, len(surgeries) + len(rooms) + 1, cost=total_start)


def _cumulative_violations(
    rows: list[tuple[float, float, float]], capacity: float, resource: str
) -> tuple[list[str], float]:
    """(start, end, usage) の累積使用量が各開始時刻で容量以下か。ピーク値も返す。"""
    peak = 0.0
    for t, _, _ in rows:
        usage = sum(use for start, end, use in rows if start <= t < end)
        peak = max(peak, usage)
    if peak > capacity:
        return [f"{resource} usage peaks at {peak:g} > capacity {capacity:g}"], peak
    return [], peak


def _precedence_violations(
    spans: dict[Any, tuple[float, float]], activities: list[dict], label: str
) -> list[str]:
    violations = []
    for act in activities:
        if act["id"] not in spans:
            continue
        for pred in act.get("predecessors") or []:
            if pred in spans and spans[act["id"]][0] < spans[pred][1]:
                violations.append(f"{label} {act['id']} starts before predecessor {pred} ends")
    return violations


def check_rcpsp_schedule(instance: dict, solution: dict) -> dict | None:
    """schedule {activity: {start_time, end_time}} が所要日数・先行関係・資源容量を守り、
    makespan と peak_* が再計算値と一致するか検証する。"""
    raw = solution.get("schedule")
    activities = instance.get("activities")
    capacity = instance.get("resource_capacity")
    if raw is None or not isinstance(activities, list) or not isinstance(capacity, dict):
        return None
    schedule = _index_map(raw)
    if schedule is None:
        return _unverified("schedule is not an activity -> times mapping")
    violations: list[str] = []
    spans: dict[Any, tuple[float, float]] = {}
    for i, act in enumerate(activities):
        act_id = act.get("id", i + 1)
        span = _span(_lookup(schedule, act_id), ("start_time", "start"), ("end_time", "end"))
        if span is None:
            violations.append(f"activity {act_id} is not scheduled")
            continue
        if span[0] < 0 or not _close(span[1] - span[0], _num(act.get("duration")) or 0.0):
            violations.append(f"activity {act_id} does not run for its duration")
        spans[act_id] = span
    acts = [{**a, "id": a.get("id", i + 1)} for i, a in enumerate(activities)]
    violations.extend(_precedence_violations(spans, acts, "activity"))
    for resource, cap in capacity.items():
        rows = [
            (*spans[a["id"]], _num((a.get("resource_usage") or {}).get(resource)) or 0.0)
            for a in acts
            if a["id"] in spans
        ]
        found, peak = _cumulative_violations(rows, _num(cap) or 0.0, resource)
        violations.extend(found)
        claimed_peak = _num(solution.get(f"peak_{resource}"))
        if claimed_peak is not None and not _close(claimed_peak, peak):
            violations.append(f"peak_{resource} {claimed_peak:g} != recomputed {peak:g}")
    finish = max((end for _, end in spans.values()), default=0.0)
    claimed = _num(solution.get("makespan"))
    if claimed is None:
        violations.append("no makespan field")
    elif not _close(claimed, finish):
        violations.append(f"makespan {claimed:g} != last completion {finish:g}")
    return _result(violations, len(activities) + len(capacity) + 1, cost=finish)


def _resolve(value: Any, by_label: dict[str, Any], by_id: dict[Any, Any]) -> Any:
    """時間帯や役割を、表示名でも id でも引けるようにする。見つからなければ None。"""
    if isinstance(value, str) and value in by_label:
        return by_label[value]
    return _lookup(by_id, value)


def check_broadcast_lineup(instance: dict, solution: dict) -> dict | None:
    """lineup {program: time_slot} がスロットの重複なし・継続時間・ジャンル最低数を守り、
    total_expected_rating が配置番組の視聴率和と一致するか検証する。"""
    raw = solution.get("lineup")
    slots = instance.get("time_slots")
    programs = instance.get("programs")
    if raw is None or not isinstance(slots, list) or not isinstance(programs, list):
        return None
    lineup = _index_map(raw)
    if lineup is None:
        return _unverified("lineup is not a program -> slot mapping")
    slots_by_id = {int(_num(s.get("id", i)) or 0): s for i, s in enumerate(slots)}
    slots_by_label = {s.get("time_slot"): s for s in slots}
    programs_by_id = {p.get("id", i + 1): p for i, p in enumerate(programs)}
    violations: list[str] = []
    used: dict[Any, Any] = {}
    counts: dict[str, int] = {}
    total = 0.0
    for program_id, slot_raw in lineup.items():
        program = programs_by_id.get(program_id)
        if program is None:
            violations.append(f"unknown program {program_id} in lineup")
            continue
        slot = _resolve(slot_raw, slots_by_label, slots_by_id)
        if slot is None:
            violations.append(f"program {program_id} is placed in unknown slot {slot_raw}")
            continue
        key = slot.get("time_slot", slot.get("id"))
        if key in used:
            violations.append(f"slot {key} holds programs {used[key]} and {program_id}")
        used.setdefault(key, program_id)
        if (_num(program.get("duration")) or 0.0) > (_num(slot.get("max_duration")) or 0.0):
            violations.append(f"program {program_id} is longer than slot {key} allows")
        counts[program.get("genre")] = counts.get(program.get("genre"), 0) + 1
        total += _num(program.get("expected_rating")) or 0.0
    for genre, need in (instance.get("genre_minimums") or {}).items():
        if counts.get(genre, 0) < (_num(need) or 0.0):
            violations.append(f"genre {genre}: {counts.get(genre, 0)} programs < minimum {need}")
    claimed = _num(solution.get("total_expected_rating"))
    if claimed is None:
        violations.append("no total_expected_rating field")
    elif not _close_soft(claimed, total):
        violations.append(f"total_expected_rating {claimed:g} != sum of placed ratings {total:g}")
    return _result(violations, len(slots) + 3, cost=total)


def check_class_timetable(instance: dict, solution: dict) -> dict | None:
    """timetable {class: {day, period, room} | [...]} が週の時限数・教室容量・教室と教員の排他を
    守るか検証する。目的値は問題文に定義がないので見ない。"""
    raw = solution.get("timetable")
    rooms = instance.get("rooms")
    classes = instance.get("classes")
    if raw is None or not isinstance(rooms, list) or not isinstance(classes, list):
        return None
    timetable = _index_map(raw)
    if timetable is None:
        return _unverified("timetable is not a class -> slots mapping")
    rooms_by_id = {room.get("id", i + 1): room for i, room in enumerate(rooms)}
    days = instance.get("days") or []
    periods = instance.get("periods") or []
    violations: list[str] = []
    room_use: dict[tuple, Any] = {}
    teacher_use: dict[tuple, Any] = {}
    for i, cls in enumerate(classes):
        cls_id = cls.get("id", i + 1)
        entries = _lookup(timetable, cls_id)
        entries = entries if isinstance(entries, list) else [entries] if entries else []
        need = int(_num(cls.get("num_periods")) or 1)
        if len(entries) != need:
            violations.append(f"class {cls_id} has {len(entries)} periods, needs {need}")
        for entry in entries:
            if not isinstance(entry, dict):
                violations.append(f"class {cls_id} has a malformed slot")
                continue
            day, period = entry.get("day"), entry.get("period")
            if (days and day not in days) or (periods and period not in periods):
                violations.append(f"class {cls_id} is placed outside the week ({day}, {period})")
            room = _lookup(rooms_by_id, entry.get("room", entry.get("room_id")))
            if room is None:
                violations.append(f"class {cls_id} is in an unknown room")
            elif (_num(room.get("capacity")) or 0.0) < (_num(cls.get("room_required")) or 0.0):
                violations.append(f"class {cls_id} needs a larger room than {room.get('id')}")
            else:
                key = (day, period, room.get("id"))
                if key in room_use:
                    violations.append(f"room {room.get('id')} is double booked on {day} {period}")
                room_use.setdefault(key, cls_id)
            key = (day, period, cls.get("teacher_id"))
            if key in teacher_use:
                teacher = cls.get("teacher_id")
                violations.append(f"teacher {teacher} is double booked on {day} {period}")
            teacher_use.setdefault(key, cls_id)
    return _result(violations, len(classes) + 2)


def check_event_staff_roster(instance: dict, solution: dict) -> dict | None:
    """roster {staff: {time_slot: role}} がスキル・利用可能時間帯・各役割各時間帯の最低人数を守り、
    total_assignments が配置数と一致するか検証する。"""
    raw = solution.get("roster")
    roles = instance.get("roles")
    slots = instance.get("time_slots")
    staff = instance.get("staff")
    if raw is None or not all(isinstance(x, list) for x in (roles, slots, staff)):
        return None
    roster = _index_map(raw)
    if roster is None:
        return _unverified("roster is not a staff -> assignments mapping")
    roles_by_id = {r.get("id", i + 1): r for i, r in enumerate(roles)}
    roles_by_name = {r.get("name"): r for r in roles}
    slots_by_id = {s.get("id", i + 1): s for i, s in enumerate(slots)}
    slots_by_label = {s.get("time_slot"): s for s in slots}
    violations: list[str] = []
    counts: dict[tuple, int] = {}
    total = 0
    for staff_id, row in roster.items():
        person = _lookup({p.get("id", i + 1): p for i, p in enumerate(staff)}, staff_id)
        if person is None:
            violations.append(f"unknown staff {staff_id} in roster")
            continue
        if not isinstance(row, dict):
            violations.append(f"staff {staff_id} row is not a time_slot -> role mapping")
            continue
        for slot_raw, role_raw in row.items():
            slot = _resolve(slot_raw, slots_by_label, slots_by_id)
            role = _resolve(role_raw, roles_by_name, roles_by_id)
            if slot is None or role is None:
                violations.append(f"staff {staff_id}: unknown slot/role {slot_raw!r}/{role_raw!r}")
                continue
            if slot.get("id") not in (person.get("available_slots") or []):
                violations.append(f"staff {staff_id} is unavailable at {slot.get('time_slot')}")
            if role.get("required_skill") not in (person.get("skills") or []):
                violations.append(f"staff {staff_id} lacks the skill for {role.get('name')}")
            key = (role.get("id"), slot.get("id"))
            counts[key] = counts.get(key, 0) + 1
            total += 1
    requirements = _index_map(instance.get("role_requirements")) or {}
    for role_id, per_slot in requirements.items():
        for slot_id, need in (_index_map(per_slot) or {}).items():
            have = counts.get((role_id, slot_id), 0)
            if have < (_num(need) or 0.0):
                violations.append(f"role {role_id} slot {slot_id}: {have} staff < required {need}")
    claimed = _num(solution.get("total_assignments"))
    if claimed is None:
        violations.append("no total_assignments field")
    elif not _close(claimed, total):
        violations.append(f"total_assignments {claimed:g} != {total} placements")
    return _result(violations, len(roles) * len(slots) + 2, cost=float(total))


def _project_activities(projects: dict) -> list[dict]:
    """projects {pid: [rows]} を id 一意の活動一覧にする。

    Why not そのまま読む: 同梱 instance は各プロジェクトの全活動が同じ id を持つ。その場合だけ
    「先頭 id + 位置」で振り直し、先行活動は同じプロジェクト内の先行行に限る（雛形ソルバーと同じ）。
    """
    keys = sorted(projects, key=lambda k: _num(k) or 0.0)
    rows = [row for key in keys for row in projects[key]]
    unique = len({row.get("id") for row in rows}) == len(rows)
    activities = []
    for key in keys:
        members = projects[key]
        ids = [
            row.get("id") if unique else (members[0].get("id") or 0) + k
            for k, row in enumerate(members)
        ]
        for k, row in enumerate(members):
            earlier = set(ids[:k])
            activities.append(
                {
                    "id": ids[k],
                    "duration": row.get("duration"),
                    "resource_usage": row.get("resource_usage"),
                    "predecessors": sorted(set(row.get("predecessors") or []) & earlier),
                }
            )
    return activities


def check_multi_project_schedule(instance: dict, solution: dict) -> dict | None:
    """schedule {activity: {start, end}} が所要日数・先行関係・共有資源の上限を守り、
    peak_resource_usage と makespan が再計算値と一致するか検証する。"""
    raw = solution.get("schedule")
    projects = instance.get("projects")
    capacity = _num(instance.get("resource_capacity"))
    if raw is None or not isinstance(projects, dict) or capacity is None:
        return None
    schedule = _index_map(raw)
    if schedule is None:
        return _unverified("schedule is not an activity -> times mapping")
    activities = _project_activities(projects)
    horizon = _num(instance.get("horizon"))
    violations: list[str] = []
    spans: dict[Any, tuple[float, float]] = {}
    for act in activities:
        span = _span(_lookup(schedule, act["id"]), ("start", "start_time"), ("end", "end_time"))
        if span is None:
            violations.append(f"activity {act['id']} is not scheduled")
            continue
        if span[0] < 0 or not _close(span[1] - span[0], _num(act.get("duration")) or 0.0):
            violations.append(f"activity {act['id']} does not run for its duration")
        if horizon is not None and span[1] > horizon:
            violations.append(f"activity {act['id']} ends after the horizon {horizon:g}")
        spans[act["id"]] = span
    violations.extend(_precedence_violations(spans, activities, "activity"))
    rows = [
        (*spans[a["id"]], _num(a.get("resource_usage")) or 0.0)
        for a in activities
        if a["id"] in spans
    ]
    found, peak = _cumulative_violations(rows, capacity, "resource")
    violations.extend(found)
    claimed_peak = _num(solution.get("peak_resource_usage"))
    if claimed_peak is None:
        violations.append("no peak_resource_usage field")
    elif not _close(claimed_peak, peak):
        violations.append(f"peak_resource_usage {claimed_peak:g} != recomputed {peak:g}")
    finish = max((end for _, end in spans.values()), default=0.0)
    claimed = _num(solution.get("makespan"))
    if claimed is not None and not _close(claimed, finish):
        violations.append(f"makespan {claimed:g} != last completion {finish:g}")
    return _result(violations, len(activities) + 2, cost=peak)
