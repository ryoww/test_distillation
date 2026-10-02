"""大規模問題集の instance 生成器（production 群）。register(kind) で登録する。

- clsp   : 多品目・多機械 容量制約付きロットサイズ決定（prob_301, prob_311）
- prp    : 生産・配送統合（prob_320）
- prp_tw : 配送時間枠とルート時間上限つきの生産・配送統合（prob_327）

件数（品目数・機械数・期数・工場数・顧客数）とスカラー（積載量、台数上限、速度、距離定義、note）は
base から写し、各要素の属性は base の経験分布（補間つき）から引く。
"""

from __future__ import annotations

import math
import random

from . import register


def _decimals(value: float) -> int:
    """数値の小数桁数。1.0 や 837.0 は 0 桁。"""
    text = repr(float(value))
    return 0 if text.endswith(".0") else len(text.split(".")[1])


def _like(rng: random.Random, values: list) -> int | float:
    """base の経験分布から引き、base と同じ小数桁数・型（int か float）に丸める。

    ソート済みの base 値を分位点とみなし、隣り合う 2 値の間を線形補間する。
    Why not [min, max] の一様乱数: 加工時間や能力は値域の中で偏っており、一様に引くと
    平均がずれて負荷率や費用の水準が base から離れる。
    """
    ordered = sorted(float(v) for v in values)
    position = rng.random() * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    drawn = ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])
    drawn = round(drawn, max(_decimals(v) for v in values))
    return int(drawn) if all(isinstance(v, int) for v in values) else float(drawn)


def _name_prefix(entries: list[dict]) -> str:
    """"品目1" → "品目" のように、先頭要素の name から id を取り除いた接頭辞。"""
    first = entries[0]
    return str(first["name"]).replace(str(first["id"]), "", 1)


# ============================================================
# clsp
# ============================================================

_CLSP_ITEM_KEYS = ("unit_process_time", "holding_cost", "setup_cost", "setup_time", "lost_sale_cost")


def clsp_load_ratio(instance: dict) -> float:
    """きつさの指標: 全需要の加工時間 / 全機械・全期の能力。"""
    items, machines = instance["items"], instance["machines"]
    work = sum(float(it["unit_process_time"]) * sum(it["demand"]) for it in items)
    capacity = sum(float(m["capacity_per_period"]) for m in machines) * int(instance["num_periods"])
    return work / capacity


@register("clsp")
def generate_clsp(rng: random.Random, base: dict) -> dict:
    items, machines = base["items"], base["machines"]
    num_periods = int(base["num_periods"])
    machine_ids = [int(m["id"]) for m in machines]
    compat_sizes = [len(it["compatible_machines"]) for it in items]
    item_prefix, machine_prefix = _name_prefix(items), _name_prefix(machines)

    new_items = []
    for idx in range(len(items)):
        # 需要は base の品目を 1 つ写し、その系列を期の順だけ入れ替える。品目ごとの水準差（prob_301）も
        # 全品目同一分布（prob_311）も、品目単位の平均・ばらつき・ゼロ需要の割合ごと引き継げる。
        template = rng.choice(items)
        item = {
            "id": idx + 1,
            "name": f"{item_prefix}{idx + 1}",
            "compatible_machines": sorted(rng.sample(machine_ids, rng.choice(compat_sizes))),
            "demand": rng.sample(template["demand"], num_periods),
        }
        for key in _CLSP_ITEM_KEYS:
            item[key] = _like(rng, [it[key] for it in items])
        new_items.append(item)

    capacities = [_like(rng, [m["capacity_per_period"] for m in machines]) for _ in machines]
    new_machines = [
        {"id": idx + 1, "name": f"{machine_prefix}{idx + 1}", "capacity_per_period": cap}
        for idx, cap in enumerate(capacities)
    ]
    instance = {"items": new_items, "machines": new_machines, "num_periods": num_periods}
    # 需要と加工時間の乱数で負荷率が base からずれるので、能力を一括で伸縮して base の負荷率に合わせる。
    factor = clsp_load_ratio(instance) / clsp_load_ratio(base)
    for machine in new_machines:
        machine["capacity_per_period"] = max(1, round(machine["capacity_per_period"] * factor))
    return instance


# ============================================================
# prp / prp_tw
# ============================================================

_PLANT_KEYS = (
    "capacity_per_period",
    "setup_cost",
    "unit_production_cost",
    "inventory_cost",
    "max_inventory",
    "initial_inventory",
)
_CUSTOMER_KEYS = ("inventory_cost", "max_inventory", "initial_inventory")


def _distance(a: dict, b: dict, kind: str) -> float:
    dx, dy = float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"])
    return abs(dx) + abs(dy) if kind == "manhattan" else math.hypot(dx, dy)


def coordinate_bound(instance: dict) -> int:
    """座標の上限。base の最大座標を 10 刻みに切り上げる（prob_320 は 250、prob_327 は 200）。"""
    points = instance["plants"] + instance["customers"]
    return math.ceil(max(max(float(p["x"]), float(p["y"])) for p in points) / 10) * 10


def direct_trip_fits(instance: dict, customer: dict) -> bool:
    """いずれかの工場から単独往復で、時間枠内にサービスを始めて時間上限内に戻れるか。"""
    speed = float(instance["vehicle_speed_kmh"])
    depart = float(instance["depot_open_time"])
    window = customer["delivery_window"]
    for plant in instance["plants"]:
        travel = _distance(plant, customer, str(instance.get("distance_type", "euclidean"))) / speed
        arrival = max(depart + travel, float(window["start"]))
        if arrival <= window["end"] and arrival + travel - depart <= float(instance["max_route_time"]):
            return True
    return False


def _generate_prp(rng: random.Random, base: dict, with_windows: bool) -> dict:
    plants, customers = base["plants"], base["customers"]
    num_periods = int(base["periods"])
    bound = coordinate_bound(base)
    plant_prefix, customer_prefix = _name_prefix(plants), _name_prefix(customers)
    demand_pool = [d for c in customers for d in c["demand"]]
    if with_windows:
        starts = [c["delivery_window"]["start"] for c in customers]
        widths = [c["delivery_window"]["end"] - c["delivery_window"]["start"] for c in customers]
        width = round(sum(widths) / len(widths), 2)

    new_plants = []
    for idx in range(len(plants)):
        plant = {
            "id": idx + 1,
            "name": f"{plant_prefix}{idx + 1}",
            "x": round(rng.uniform(0, bound), 1),
            "y": round(rng.uniform(0, bound), 1),
        }
        for key in _PLANT_KEYS:
            plant[key] = _like(rng, [p[key] for p in plants])
        new_plants.append(plant)
    # スカラー（積載量・台数上限・速度・距離定義・note）は base のまま。工場は先に確定させ、
    # 顧客の到達可能性の判定に使う。
    instance = {**base, "plants": new_plants, "customers": []}

    for idx in range(len(customers)):
        while True:
            customer = {
                "id": idx + 1,
                "name": f"{customer_prefix}{idx + 1}",
                "x": round(rng.uniform(0, bound), 1),
                "y": round(rng.uniform(0, bound), 1),
                "demand": [rng.choice(demand_pool) for _ in range(num_periods)],
            }
            for key in _CUSTOMER_KEYS:
                customer[key] = _like(rng, [c[key] for c in customers])
            if not with_windows:
                break
            start = _like(rng, starts)
            customer["delivery_window"] = {"start": start, "end": round(start + width, 2)}
            # 単独往復すら時間枠・ルート時間に収まらない顧客は誰にも配送できないので引き直す。
            if direct_trip_fits(instance, customer):
                break
        instance["customers"].append(customer)
    return instance


@register("prp")
def generate_prp(rng: random.Random, base: dict) -> dict:
    return _generate_prp(rng, base, with_windows=False)


@register("prp_tw")
def generate_prp_tw(rng: random.Random, base: dict) -> dict:
    return _generate_prp(rng, base, with_windows=True)
