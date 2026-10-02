"""大規模問題集の instance 生成器（routing 群）。register(kind) で登録する。

- vrptw_md（prob_303 / prob_313）: 顧客の座標は元の顧客位置のまわりに散らし（デポ周りの密集など
  元の空間分布を保つ）、需要・枠幅・サービス時間は元の値から引き直し、枠の開始は元の値域に置く。
- pdptw（prob_321）: 受取・配送地点は元の座標の箱に一様、需要は元の値から引き直す。配送枠は受取枠
  から移動時間ぶんあとに置く。

デポの位置・営業時間と車種表は元のまま使い、保有台数は車種ごとに 1 台をデポ間で移す。時間枠は最も遅い車種でも
デポから枠内に着け、営業終了までに帰着できる範囲に収める（元 instance の顧客もこの性質を満たす）。
"""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Iterable

from . import register

# 到達・帰着の余裕（時間）。丸め誤差を吸収し、枠の端ぎりぎりの地点を作らない。
_MARGIN_H = 0.25
# 顧客位置を元の位置から散らす幅（座標の箱の一辺に対する比）。
_SCATTER = 0.1


def _dist(a: dict, b: dict) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def _span(values: Iterable[float]) -> tuple[float, float]:
    seq = list(values)
    return min(seq), max(seq)


def _shuffle_fleet(rng: random.Random, depots: list[dict]) -> list[dict]:
    """車種ごとに 1 台をあるデポから別のデポへ移す。位置・営業時間はそのまま。

    Why not 台数を ±1 揺らす: デポが 2 箇所しかない prob_321 では総容量が 1 割以上動き、
    需要合計/容量の比が元から外れる。総台数を保てば比は需要の揺らぎぶんしか動かない。
    """
    result = copy.deepcopy(depots)
    for vtype in result[0]["fleet"]:
        donors = [d for d in result if d["fleet"][vtype] > 1]
        if len(result) < 2 or not donors:
            continue
        source = rng.choice(donors)
        target = rng.choice([d for d in result if d is not source])
        source["fleet"][vtype] -= 1
        target["fleet"][vtype] += 1
    return result


def _clip(value: float, bounds: tuple[float, float]) -> float:
    return min(max(value, bounds[0]), bounds[1])


def _start_bounds(
    start_range: tuple[float, float], width: float, earliest: float, latest: float
) -> tuple[float, float]:
    """時間枠の開始が取れる範囲。earliest は最早到着、latest はそこで始めれば帰着できる最遅開始。

    枠の終わりが最早到着より後（着ける）、枠の始まりが最遅開始より前（帰れる）であることを課す。
    """
    lo = max(start_range[0], earliest + _MARGIN_H - width)
    hi = min(start_range[1], latest - _MARGIN_H)
    return lo, hi


# ----------------------------------------------------------------------------
# vrptw_md
# ----------------------------------------------------------------------------


@register("vrptw_md")
def generate_vrptw_md(rng: random.Random, base: dict) -> dict:
    depots = _shuffle_fleet(rng, base["depots"])
    vehicles = copy.deepcopy(base["vehicle_types"])
    speed = min(v["speed_kmh"] for v in vehicles)
    source = base["customers"]
    xr = _span(c["x"] for c in source)
    yr = _span(c["y"] for c in source)
    scatter = (_SCATTER * (xr[1] - xr[0]), _SCATTER * (yr[1] - yr[0]))
    demands = [c["demand"] for c in source]
    widths = [round(c["tw_end"] - c["tw_start"], 2) for c in source]
    services = [c["service_time"] for c in source]
    start_range = _span(c["tw_start"] for c in source)
    customers = []
    for template in source:
        while True:
            # Why not 箱に一様: prob_303 の顧客はデポの周りに固まっており、一様に撒くと最寄りデポ
            # までの距離が元より 2 割ほど伸びる。元の位置を核に散らせば一様な元（prob_313）も保てる。
            point = {
                "x": round(_clip(rng.gauss(template["x"], scatter[0]), xr), 1),
                "y": round(_clip(rng.gauss(template["y"], scatter[1]), yr), 1),
            }
            depot = min(depots, key=lambda d: _dist(d, point))
            travel = _dist(depot, point) / speed
            service = rng.choice(services)
            width = rng.choice(widths)
            lo, hi = _start_bounds(
                start_range, width, depot["open_time"] + travel,
                depot["close_time"] - travel - service,
            )
            # 最も遅い車種で往復できない遠方の点は引き直す（元の箱の大きさでは滅多に起きない）。
            if lo <= hi:
                break
        start = round(rng.uniform(lo, hi), 2)
        customers.append({
            "id": template["id"],
            "x": point["x"],
            "y": point["y"],
            "demand": rng.choice(demands),
            "tw_start": start,
            "tw_end": round(start + width, 2),
            "service_time": service,
        })
    instance = dict(base)
    instance.update(depots=depots, customers=customers, vehicle_types=vehicles)
    return instance


# ----------------------------------------------------------------------------
# pdptw
# ----------------------------------------------------------------------------


@register("pdptw")
def generate_pdptw(rng: random.Random, base: dict) -> dict:
    depots = _shuffle_fleet(rng, base["depots"])
    vehicles = copy.deepcopy(base["vehicle_types"])
    speed = min(v["speed_kmh"] for v in vehicles)
    source = base["pairs"]
    stops = [p["pickup"] for p in source] + [p["delivery"] for p in source]
    xr = _span(s["x"] for s in stops)
    yr = _span(s["y"] for s in stops)
    demands = [s["demand"] for s in stops]
    start_range = _span(p["pickup"]["tw_start"] for p in source)
    pickup_widths = [round(p["pickup"]["tw_end"] - p["pickup"]["tw_start"], 2) for p in source]
    delivery_widths = [
        round(p["delivery"]["tw_end"] - p["delivery"]["tw_start"], 2) for p in source
    ]
    pairs = []
    for template in source:
        while True:
            pickup = {"x": round(rng.uniform(*xr), 1), "y": round(rng.uniform(*yr), 1)}
            delivery = {"x": round(rng.uniform(*xr), 1), "y": round(rng.uniform(*yr), 1)}
            depot = min(depots, key=lambda d: _dist(d, pickup) + _dist(delivery, d))
            pickup_width = rng.choice(pickup_widths)
            delivery_width = rng.choice(delivery_widths)
            # 配送枠は「受取枠の開始に出て着く時刻」から受取枠幅ぶんまでの間に開く。受取枠の
            # どこで積んでも配送枠内に着ける。
            gap = _dist(pickup, delivery) / speed + rng.uniform(0.0, pickup_width)
            lo, hi = _start_bounds(
                start_range, pickup_width,
                depot["open_time"] + _dist(depot, pickup) / speed,
                depot["close_time"] - gap - _dist(delivery, depot) / speed,
            )
            # 営業時間内に受取→配送→帰着が収まらない遠いペアは引き直す。
            if lo <= hi:
                break
        pickup_start = round(rng.uniform(lo, hi), 2)
        delivery_start = round(pickup_start + gap, 2)
        pairs.append({
            "id": template["id"],
            "pickup": {
                **pickup,
                "demand": rng.choice(demands),
                "tw_start": pickup_start,
                "tw_end": round(pickup_start + pickup_width, 2),
            },
            "delivery": {
                **delivery,
                "demand": rng.choice(demands),
                "tw_start": delivery_start,
                "tw_end": round(delivery_start + delivery_width, 2),
            },
        })
    instance = dict(base)
    instance.update(depots=depots, vehicle_types=vehicles, pairs=pairs)
    return instance
