"""旧チェッカーの core_type に属する参照解固有の形を、解析してから厳密に検証する。

feasibility.py の旧チェッカーは要素数と目的値の有無しか見ない。ここでは同梱参照解が
使う4つの形（並列機械の machine_assignment、フローショップの optimal_sequence、
クラスタ割当の node_assignment、配送＋在庫の day1_routes）について、ID の範囲、重複、
容量、目的値の再計算まで行う。形を読めなければ None を返し、呼び出し側の従来処理に戻す。
"""

from __future__ import annotations

from typing import Any

from .feasibility_v3_ext import _close, _close_soft, _index_map, _num, _result, _unverified


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
    violations: list[str] = []
    times = {
        job.get("id", i): _num(job.get("processing_time")) or 0.0 for i, job in enumerate(jobs)
    }
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
    """node_assignment {job_id: node_id} が各ジョブの要求をノード容量内に収めるか検証する。"""
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
        # Why not ノード単位で合算: 同梱参照解はメモリ合計が容量を超える割当を含む。
        # 問題文も「ジョブの必要リソースがノードの容量を超えてはいけない」と個別に述べている。
        for need_key, cap_key in requirements:
            need = _num(job.get(need_key)) or 0.0
            cap = _num(node.get(cap_key)) or 0.0
            if need > cap:
                violations.append(
                    f"job {job_id} needs {need_key}={need:g} > node {cap_key}={cap:g}"
                )
        priority += _num(job.get("priority")) or 0.0
        assigned += 1
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
