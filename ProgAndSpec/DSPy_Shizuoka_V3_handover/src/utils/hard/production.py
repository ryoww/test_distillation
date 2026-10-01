"""大規模問題集の検証器（production 群）。register_kind で種別を登録する。

- clsp   : 多品目・多機械 容量制約付きロットサイズ決定（prob_301, prob_311）
- prp    : 生産・配送統合（prob_320）
- prp_tw : 配送時間枠とルート時間上限つきの生産・配送統合（prob_327）

目的値は申告値を読まず、instance と解の構造から再計算する。
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from ..feasibility_v3_ext import _result, _unverified
from . import register_kind

_EPS = 1e-6
# 申告目的値と再計算値のずれをどこまで許すか（brief の 0.5%）。
_DECLARED_REL_TOL = 5e-3
# ロット量は小数 6 桁程度に丸めて返されるため、加工時間の合計が能力を 1e-6 前後だけ超えることが
# ある。_EPS ではこの丸めを違反に数えてしまうので、能力照合だけ少し広い許容差を使う。
_CAPACITY_TOL = 1e-5


def _as_num(value: Any) -> float | None:
    """有限の数値なら float、そうでなければ None（bool は数値扱いしない）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _as_int(value: Any) -> int | None:
    """id や期のように整数として読める値（"4" や 4.0 も可）を int にする。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def _as_flag(value: Any) -> bool | None:
    """段取りフラグ（True/False、0/1）を bool にする。読めなければ None。"""
    if isinstance(value, bool):
        return value
    number = _as_num(value)
    return None if number is None else number != 0


def _pick(entry: dict, *names: str) -> Any:
    """同義キーのうち最初に見つかった値を返す（どれも無ければ None）。"""
    for name in names:
        if name in entry:
            return entry[name]
    return None


# モデルが返す同義キー。id は参照解の名前を先頭に置き、以降は自然な別名。
_ITEM_KEYS = ("item", "item_id", "id", "product", "product_id")
_MACHINE_KEYS = ("machine", "machine_id")
_PLANT_KEYS = ("plant", "plant_id")
_CUSTOMER_KEYS = ("customer", "customer_id")
_QTY_KEYS = ("qty", "quantity", "amount", "production")
_INVENTORY_KEYS = ("end_inventory", "inventory")


def _positional(values: Any, num_periods: int) -> list | None:
    """期順のリスト（index 0 が第 1 期）として読めるときだけ返す。長さが期数と違えば None。"""
    if isinstance(values, list) and len(values) == num_periods:
        return values
    return None


def _declared_mismatch(solution: dict, cost: float, violations: list[str]) -> None:
    """申告目的値があれば再計算値と突き合わせ、ずれが大きければ違反にする。"""
    for key in ("objective_value", "total_cost"):
        declared = _as_num(solution.get(key))
        if declared is None:
            continue
        if abs(declared - cost) > _DECLARED_REL_TOL * max(1.0, abs(cost)):
            violations.append(f"declared {key} {declared:.2f} != recomputed {cost:.2f}")
        return


# ============================================================
# clsp: 容量制約付きロットサイズ決定
# ============================================================


def _detect_clsp(instance: dict) -> bool:
    items = instance.get("items")
    machines = instance.get("machines")
    return (
        isinstance(items, list)
        and isinstance(machines, list)
        and bool(items)
        and "num_periods" in instance
        and isinstance(items[0], dict)
        and "compatible_machines" in items[0]
    )


def _parse_clsp_lots(entries: Any, num_periods: int) -> dict[tuple[int, int], float] | None:
    """1 品目分のロット列を {(machine, period): qty} にする。形が違えば None。

    読める形: record のリスト [{machine, period, qty}]、それを "production" に入れた dict、
    期順の [{machine: qty, ...}, ...]（長さが期数と一致するときだけ位置を期と読む）。
    """
    if isinstance(entries, dict):
        entries = _pick(entries, "production", "lots")
    if not isinstance(entries, list):
        return None
    lots: dict[tuple[int, int], float] = defaultdict(float)
    if entries and all(
        isinstance(e, dict) and all(_as_int(k) is not None for k in e) for e in entries
    ):
        if _positional(entries, num_periods) is None:
            return None
        for period, by_machine in enumerate(entries, start=1):
            for raw_machine, raw_qty in by_machine.items():
                qty = _as_num(raw_qty)
                if qty is None:
                    return None
                lots[(_as_int(raw_machine), period)] += qty
        return dict(lots)
    for entry in entries:
        if not isinstance(entry, dict):
            return None
        machine = _as_int(_pick(entry, *_MACHINE_KEYS))
        period = _as_int(entry.get("period"))
        qty = _as_num(_pick(entry, *_QTY_KEYS))
        if machine is None or period is None or qty is None:
            return None
        # 同じ (機械, 期) の重複エントリは 1 ロットに合算する（段取りは 1 回）。
        lots[(machine, period)] += qty
    return dict(lots)


def _parse_clsp_plan(plan: Any, num_periods: int) -> dict[int, dict[tuple[int, int], float]] | None:
    """production_plan を {item_id: {(machine, period): qty}} に正規化する。形が違えば None。

    {item: ロット列} の dict のほか、品目 id を持つ record のフラットなリストも品目ごとに
    まとめて読む。
    """
    if isinstance(plan, list):
        grouped: dict[Any, list] = defaultdict(list)
        for entry in plan:
            item_id = _as_int(_pick(entry, *_ITEM_KEYS)) if isinstance(entry, dict) else None
            if item_id is None:
                return None
            grouped[item_id].append(entry)
        plan = grouped
    if not isinstance(plan, dict):
        return None
    parsed: dict[int, dict[tuple[int, int], float]] = {}
    for raw_item, entries in plan.items():
        item_id = _as_int(raw_item)
        lots = _parse_clsp_lots(entries, num_periods)
        if item_id is None or lots is None:
            return None
        parsed[item_id] = lots
    return parsed


def _check_clsp(instance: dict, solution: Any) -> dict:
    if not isinstance(solution, dict):
        return _unverified("clsp solution is not a dict")
    num_periods = int(instance["num_periods"])
    plan = _parse_clsp_plan(solution.get("production_plan"), num_periods)
    if plan is None:
        return _unverified("clsp solution without a readable 'production_plan'")

    items = {int(it["id"]): it for it in instance["items"]}
    machines = {int(m["id"]): m for m in instance["machines"]}
    violations: list[str] = []
    load: dict[tuple[int, int], float] = defaultdict(float)
    setup_cost = holding_cost = lost_cost = 0.0
    num_setups = 0
    total_lost = 0.0

    for item_id, lots in plan.items():
        if item_id not in items:
            violations.append(f"item {item_id} is not in the instance")
    # 計画に無い品目は生産ゼロ（需要は全て欠品）として費用に入れる。
    for item_id, item in items.items():
        lots = plan.get(item_id, {})
        compatible = {int(m) for m in item["compatible_machines"]}
        bad_machine = sorted(m for (m, _t) in lots if m not in compatible or m not in machines)
        if bad_machine:
            violations.append(f"item {item_id} produced on incompatible machines {bad_machine}")
        bad_period = sorted(t for (_m, t) in lots if not 1 <= t <= num_periods)
        if bad_period:
            violations.append(f"item {item_id} has lots outside periods 1..{num_periods}: {bad_period}")
        if any(q < -_EPS for q in lots.values()):
            violations.append(f"item {item_id} has a negative lot quantity")

        produced = [0.0] * (num_periods + 1)
        for (machine, period), qty in lots.items():
            if machine in machines and 1 <= period <= num_periods and qty > _EPS:
                produced[period] += qty
                num_setups += 1
                setup_cost += float(item["setup_cost"])
                load[(machine, period)] += (
                    float(item["unit_process_time"]) * qty + float(item["setup_time"])
                )
        # 収支: 前期在庫 + 生産 + 欠品 = 需要 + 当期在庫。在庫と欠品は生産量から一意に決まる。
        inventory = 0.0
        for period in range(1, num_periods + 1):
            available = inventory + produced[period]
            demand = float(item["demand"][period - 1])
            lost = max(0.0, demand - available)
            inventory = max(0.0, available - demand)
            holding_cost += float(item["holding_cost"]) * inventory
            lost_cost += float(item["lost_sale_cost"]) * lost
            total_lost += lost

    for (machine, period), used in load.items():
        capacity = float(machines[machine]["capacity_per_period"])
        if used > capacity + _CAPACITY_TOL:
            violations.append(
                f"machine {machine} period {period} uses {used:.1f} > capacity {capacity:.1f}"
            )

    cost = setup_cost + holding_cost + lost_cost
    _declared_mismatch(solution, cost, violations)
    declared_setups = _as_int(solution.get("num_setups"))
    if declared_setups is not None and declared_setups != num_setups:
        violations.append(f"declared num_setups {declared_setups} != recomputed {num_setups}")
    declared_lost = _as_num(solution.get("total_lost_sales"))
    if declared_lost is not None and abs(declared_lost - total_lost) > _DECLARED_REL_TOL * max(
        1.0, total_lost
    ):
        violations.append(
            f"declared total_lost_sales {declared_lost:.3f} != recomputed {total_lost:.3f}"
        )
    # 品目ごとに互換機械・期範囲・非負の 3 件、機械×期ごとに能力 1 件、申告値 3 件。
    total = 3 * len(items) + len(machines) * num_periods + 3
    return _result(violations, total, cost=cost)


register_kind("clsp", _detect_clsp, _check_clsp)


# ============================================================
# prp / prp_tw: 生産・配送統合
# ============================================================


def _is_prp(instance: dict) -> bool:
    plants = instance.get("plants")
    customers = instance.get("customers")
    return (
        isinstance(plants, list)
        and isinstance(customers, list)
        and bool(plants)
        and bool(customers)
        and "vehicle_capacity" in instance
        and "max_vehicles_per_plant" in instance
        and "periods" in instance
        and isinstance(plants[0], dict)
        and "capacity_per_period" in plants[0]
    )


def _has_time_windows(instance: dict) -> bool:
    customers = instance.get("customers") or [{}]
    return isinstance(customers[0], dict) and "delivery_window" in customers[0]


def _detect_prp(instance: dict) -> bool:
    return _is_prp(instance) and not _has_time_windows(instance)


def _detect_prp_tw(instance: dict) -> bool:
    return _is_prp(instance) and _has_time_windows(instance)


def _distance(a: dict, b: dict, kind: str) -> float:
    dx, dy = float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"])
    if kind == "manhattan":
        return abs(dx) + abs(dy)
    return math.hypot(dx, dy)


def _prp_plan_records(plan: dict, num_periods: int) -> list[dict] | None:
    """{plant: {production: [期順], setup: [期順], inventory: [期順]}} を record のリストに展開する。"""
    records: list[dict] = []
    for raw_plant, fields in plan.items():
        if not isinstance(fields, dict):
            return None
        production = _positional(_pick(fields, *_QTY_KEYS), num_periods)
        setups = fields.get("setup")
        inventories = _pick(fields, *_INVENTORY_KEYS)
        if production is None:
            return None
        # 補助列は無くてもよいが、あるのに期数と長さが違えば位置が決まらない。
        if setups is not None and _positional(setups, num_periods) is None:
            return None
        if inventories is not None and _positional(inventories, num_periods) is None:
            return None
        for idx, qty in enumerate(production):
            record = {"plant": raw_plant, "period": idx + 1, "production": qty}
            if setups is not None:
                record["setup"] = setups[idx]
            if inventories is not None:
                record["end_inventory"] = inventories[idx]
            records.append(record)
    return records


def _parse_prp_plan(plan: Any, num_periods: int) -> dict[tuple[int, int], dict] | None:
    """production_plan を {(plant, period): {production, setup, end_inventory}} にする。"""
    if isinstance(plan, dict):
        plan = _prp_plan_records(plan, num_periods)
    if not isinstance(plan, list):
        return None
    parsed: dict[tuple[int, int], dict] = {}
    for entry in plan:
        if not isinstance(entry, dict):
            return None
        plant = _as_int(_pick(entry, *_PLANT_KEYS))
        period = _as_int(entry.get("period"))
        production = _as_num(_pick(entry, *_QTY_KEYS))
        if plant is None or period is None or production is None:
            return None
        setup = entry.get("setup")
        setup_flag = None if setup is None else _as_flag(setup)
        end_inventory = _as_num(_pick(entry, *_INVENTORY_KEYS))
        # 同じ (工場, 期) が複数回あれば生産量を合算する。
        prev = parsed.get((plant, period))
        if prev is not None:
            production += prev["production"]
            setup_flag = setup_flag or prev["setup"]
        parsed[(plant, period)] = {
            "production": production,
            "setup": setup_flag,
            "end_inventory": end_inventory,
        }
    return parsed


def _parse_routes(routes: Any) -> list[dict] | None:
    """routes を {plant, period, stops: [(customer, qty)]} のリストにする。"""
    if not isinstance(routes, list):
        return None
    parsed = []
    for route in routes:
        if not isinstance(route, dict):
            return None
        plant = _as_int(_pick(route, *_PLANT_KEYS))
        period = _as_int(route.get("period"))
        customers = route.get("customers")
        quantities = route.get("quantities")
        if plant is None or period is None or not isinstance(customers, list):
            return None
        if not isinstance(quantities, list) or len(quantities) != len(customers):
            return None
        stops = []
        for customer, qty in zip(customers, quantities):
            cid, q = _as_int(customer), _as_num(qty)
            if cid is None or q is None:
                return None
            stops.append((cid, q))
        parsed.append({"plant": plant, "period": period, "stops": stops})
    return parsed


def _parse_allocation(
    allocation: Any, plant_ids: set[int], customer_ids: set[int], num_periods: int
) -> dict[tuple[int, int], float] | None:
    """allocation を {(customer, period): 配送量} にする。量が取れない形なら None（照合しない）。

    読める形: record のリスト [{customer, period, qty}]、{customer: {period: {plant: qty}}}、
    {customer: {"deliveries": {period: qty}}}、{plant: {customer: [期順]}} とその逆向き
    （向きは id の集合で決め、どちらとも取れれば None）。
    """
    allocated: dict[tuple[int, int], float] = defaultdict(float)
    if isinstance(allocation, list):
        for entry in allocation:
            if not isinstance(entry, dict):
                return None
            cid = _as_int(_pick(entry, *_CUSTOMER_KEYS))
            period = _as_int(entry.get("period"))
            qty = _as_num(_pick(entry, *_QTY_KEYS))
            if cid is None or period is None or qty is None:
                return None
            allocated[(cid, period)] += qty
        return dict(allocated)
    if not isinstance(allocation, dict):
        return None
    series: list[tuple[int, int, list]] = []
    for raw_outer, inner in allocation.items():
        outer = _as_int(raw_outer)
        if outer is None or not isinstance(inner, dict):
            return None
        by_period = isinstance(inner.get("deliveries"), dict)
        if by_period:
            inner = inner["deliveries"]
        for raw_key, value in inner.items():
            key = _as_int(raw_key)
            if key is None:
                return None
            if isinstance(value, dict):
                quantities = [_as_num(q) for q in value.values()]
                if any(q is None for q in quantities):
                    return None
                allocated[(outer, key)] += sum(quantities)
            elif isinstance(value, list):
                if _positional(value, num_periods) is None:
                    return None
                series.append((outer, key, value))
            elif by_period and _as_num(value) is not None:
                allocated[(outer, key)] += _as_num(value)
            else:
                # {customer: {k: 量}} の k は期とも工場とも取れるので "deliveries" 配下だけ読む。
                return None
    if series:
        outers = {o for o, _k, _v in series}
        inners = {k for _o, k, _v in series}
        outer_is_plant = outers <= plant_ids and inners <= customer_ids
        outer_is_customer = outers <= customer_ids and inners <= plant_ids
        if outer_is_plant == outer_is_customer:
            return None
        for outer, key, values in series:
            cid = key if outer_is_plant else outer
            for idx, raw_qty in enumerate(values):
                qty = _as_num(raw_qty)
                if qty is None:
                    return None
                allocated[(cid, idx + 1)] += qty
    return dict(allocated)


def _check_prp(instance: dict, solution: Any) -> dict:
    if not isinstance(solution, dict):
        return _unverified("prp solution is not a dict")
    num_periods = int(instance["periods"])
    plan = _parse_prp_plan(solution.get("production_plan"), num_periods)
    if plan is None:
        return _unverified("prp solution without a readable 'production_plan'")
    routes = _parse_routes(solution.get("routes"))
    if routes is None:
        return _unverified("prp solution without a readable 'routes'")

    plants = {int(p["id"]): p for p in instance["plants"]}
    customers = {int(c["id"]): c for c in instance["customers"]}
    vehicle_capacity = float(instance["vehicle_capacity"])
    max_vehicles = int(instance["max_vehicles_per_plant"])
    distance_type = str(instance.get("distance_type", "euclidean"))
    distance_cost = float(instance["distance_cost_per_unit"])
    fixed_cost = float(instance.get("vehicle_fixed_cost", 0.0))
    time_windows = _has_time_windows(instance)
    speed = float(instance.get("vehicle_speed_kmh", 0.0) or 0.0)
    depart = float(instance.get("depot_open_time", 0.0) or 0.0)
    max_route_time = _as_num(instance.get("max_route_time"))

    violations: list[str] = []
    delivered: dict[tuple[int, int], float] = defaultdict(float)
    shipped: dict[tuple[int, int], float] = defaultdict(float)
    vehicles: dict[tuple[int, int], int] = defaultdict(int)
    route_cost = 0.0

    # ルート: 積載、時間枠、ルート時間。配送量はルートの quantities を正とする。
    for idx, route in enumerate(routes):
        plant_id, period, stops = route["plant"], route["period"], route["stops"]
        label = f"route {idx} (plant {plant_id}, period {period})"
        if plant_id not in plants or not 1 <= period <= num_periods:
            violations.append(f"{label} has an unknown plant or period")
            continue
        unknown = [c for c, _q in stops if c not in customers]
        if unknown:
            violations.append(f"{label} visits unknown customers {unknown}")
            continue
        if any(q < -_EPS for _c, q in stops):
            violations.append(f"{label} has a negative delivery quantity")
        load = sum(q for _c, q in stops)
        if load > vehicle_capacity + _EPS:
            violations.append(f"{label} load {load:.1f} > vehicle capacity {vehicle_capacity:.0f}")
        vehicles[(plant_id, period)] += 1
        plant = plants[plant_id]
        prev, distance, clock, late = plant, 0.0, depart, []
        for cid, q in stops:
            customer = customers[cid]
            leg = _distance(prev, customer, distance_type)
            distance += leg
            delivered[(cid, period)] += q
            shipped[(plant_id, period)] += q
            if time_windows and speed > 0:
                # 到着が早ければ窓の開始まで待ち、サービス開始が窓の終了を超えたら違反。
                window = customer["delivery_window"]
                clock = max(clock + leg / speed, float(window["start"]))
                if clock > float(window["end"]) + _EPS:
                    late.append(cid)
            prev = customer
        back = _distance(prev, plant, distance_type)
        distance += back
        route_cost += distance_cost * distance + fixed_cost
        if time_windows:
            if late:
                violations.append(f"{label} starts service after the window at {late}")
            if speed > 0 and max_route_time is not None:
                # 出発から帰還までの経過時間（待ち時間込み）が上限を超えたら違反。
                elapsed = clock + back / speed - depart
                if elapsed > max_route_time + _EPS:
                    violations.append(
                        f"{label} takes {elapsed:.2f}h > max route time {max_route_time:.1f}h"
                    )

    for (plant_id, period), count in vehicles.items():
        if count > max_vehicles:
            violations.append(
                f"plant {plant_id} period {period} uses {count} vehicles > {max_vehicles}"
            )

    # 工場: 能力・段取り・在庫上限・申告在庫。計画に無い (工場, 期) は生産ゼロ。
    plant_cost = 0.0
    for (plant_id, period) in plan:
        if plant_id not in plants or not 1 <= period <= num_periods:
            violations.append(f"production_plan has unknown plant {plant_id} or period {period}")
    for plant_id, plant in plants.items():
        inventory = float(plant.get("initial_inventory", 0.0))
        for period in range(1, num_periods + 1):
            entry = plan.get((plant_id, period), {"production": 0.0, "setup": None, "end_inventory": None})
            production = entry["production"]
            setup = bool(entry["setup"]) if entry["setup"] is not None else production > _EPS
            if production < -_EPS:
                violations.append(f"plant {plant_id} period {period} has negative production")
            if production > float(plant["capacity_per_period"]) + _EPS:
                violations.append(
                    f"plant {plant_id} period {period} production {production:.1f} exceeds capacity"
                )
            if production > _EPS and not setup:
                violations.append(f"plant {plant_id} period {period} produces without a setup")
            inventory = inventory + production - shipped.get((plant_id, period), 0.0)
            if inventory < -_EPS:
                violations.append(f"plant {plant_id} period {period} ships more than it holds")
            if inventory > float(plant["max_inventory"]) + _EPS:
                violations.append(f"plant {plant_id} period {period} exceeds max inventory")
            declared_inv = entry["end_inventory"]
            if declared_inv is not None and abs(declared_inv - inventory) > 1e-3:
                violations.append(
                    f"plant {plant_id} period {period} declared end_inventory {declared_inv:.1f}"
                    f" != recomputed {inventory:.1f}"
                )
            # 段取りは生産があるか申告されていれば費用に入れる。
            plant_cost += float(plant["setup_cost"]) * (setup or production > _EPS)
            plant_cost += float(plant["unit_production_cost"]) * max(production, 0.0)
            plant_cost += float(plant["inventory_cost"]) * max(inventory, 0.0)

    # 顧客: 欠品不可、期末在庫が上限以下。
    customer_cost = 0.0
    for cid, customer in customers.items():
        inventory = float(customer.get("initial_inventory", 0.0))
        for period in range(1, num_periods + 1):
            inventory += delivered.get((cid, period), 0.0) - float(customer["demand"][period - 1])
            if inventory < -_EPS:
                violations.append(f"customer {cid} period {period} has a stockout")
            if inventory > float(customer["max_inventory"]) + _EPS:
                violations.append(f"customer {cid} period {period} exceeds max inventory")
            customer_cost += float(customer["inventory_cost"]) * max(inventory, 0.0)

    # allocation があればルートの配送量と一致することを確かめる（配送の正はルート側）。
    allocated = _parse_allocation(
        solution.get("allocation"), set(plants), set(customers), num_periods
    )
    if allocated is not None:
        mismatched = sorted(
            key
            for key in set(allocated) | set(delivered)
            if abs(allocated.get(key, 0.0) - delivered.get(key, 0.0)) > 1e-3
        )
        if mismatched:
            violations.append(
                f"allocation differs from route deliveries for {len(mismatched)} (customer, period)"
            )

    cost = plant_cost + customer_cost + route_cost
    _declared_mismatch(solution, cost, violations)
    # ルートごとに積載 1 件（TW ならさらに時間枠・時間上限 2 件）、工場×期ごとに
    # 能力/段取り・在庫・申告在庫・台数の 4 件、顧客×期ごとに欠品・上限の 2 件、
    # allocation 整合と申告目的値の 2 件。
    per_route = 3 if time_windows else 1
    total = (
        per_route * len(routes)
        + 4 * len(plants) * num_periods
        + 2 * len(customers) * num_periods
        + 2
    )
    return _result(violations, total, cost=cost)


register_kind("prp", _detect_prp, _check_prp)
register_kind("prp_tw", _detect_prp_tw, _check_prp)
