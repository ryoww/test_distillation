"""大規模問題集の instance 生成器（rosters 群）。register(kind) で登録する。

対象種別と元問題:
- crew_pairing (prob_302, prob_312) / crew_pairing_seniority (prob_330): 便と基地（+クルー）。
- nurse_roster (prob_306, prob_316) / role_roster (prob_323): 人と日ごとの必要人数。
件数・規則・費用係数・文字列定数は base から読み、乱数にするのは便の時刻と区間、人の属性、
日ごとの必要人数だけにする。
"""

from __future__ import annotations

import copy
import random
import statistics
from itertools import pairwise

from . import register


def _name_prefix(items: list[dict]) -> str:
    """"看護師12" のような名前から数字を除いた接頭辞。"""
    return items[0]["name"].rstrip("0123456789")


def _ratio(values: list[int | float]) -> float:
    return sum(values) / len(values)


# ---------------------------------------------------------------- 便
def _flights(rng: random.Random, base: dict) -> list[dict]:
    """便を base と同じ件数・時刻分布・区間長分布で作る。

    出発時刻は base の便の出発時刻を一つ選んで ±0.5h ずらす（経験分布のカーネル法）。
    Why not 一様乱数: prob_302 は 60h 期間で夜間が薄く、prob_312/330 は 20h 期間なので、
    期間と時間帯の濃淡を一つの手順で引き継ぐには base の経験分布から引くのが簡単。
    出発・到着空港は空港集合から異なる 2 つを等確率に選ぶ（base の空港ペア別件数はほぼ一様）。
    同一空港・接続時間 0.75〜8h の接続アーク数は、便数・空港数・期間・時刻分布を揃えることで
    base と同程度になる。
    """
    base_flights = base["flights"]
    airports = list(base["airports"])
    deps = [f["dep_time"] for f in base_flights]
    dep_lo, dep_hi = min(deps), max(deps)
    durations = [f["duration"] for f in base_flights]
    dur_lo, dur_hi = min(durations), max(durations)
    costs = [f["flight_cost"] for f in base_flights]
    ratios = [f["flight_cost"] / f["duration"] for f in base_flights]
    # 乗務費が区間長に比例している base（prob_302: 1500/h）はその単価で、
    # そうでない base（prob_312/330）は乗務費の範囲の一様整数で作る。
    proportional = (max(ratios) - min(ratios)) < 0.01 * _ratio(ratios)
    rate = _ratio(ratios)
    sorted_by_dep = all(a["dep_time"] <= b["dep_time"] for a, b in pairwise(base_flights))

    flights = []
    for _ in base_flights:
        dep = rng.choice(deps) + rng.uniform(-0.5, 0.5)
        while not dep_lo <= dep <= dep_hi:
            dep = rng.choice(deps) + rng.uniform(-0.5, 0.5)
        dep = round(dep, 2)
        duration = round(rng.uniform(dur_lo, dur_hi), 2)
        origin, destination = rng.sample(airports, 2)
        cost = round(duration * rate) if proportional else rng.randint(min(costs), max(costs))
        flights.append(
            {
                "origin": origin,
                "destination": destination,
                "dep_time": dep,
                "arr_time": round(dep + duration, 2),
                "duration": duration,
                "flight_cost": cost,
            }
        )
    if sorted_by_dep:
        flights.sort(key=lambda f: f["dep_time"])
    # id は base と同じ 1 始まりの連番。キー順も base（id が先頭）に合わせる。
    return [{"id": i, **f} for i, f in enumerate(flights, 1)]


def _crew_pairing_instance(rng: random.Random, base: dict) -> dict:
    instance = copy.deepcopy(base)
    instance["flights"] = _flights(rng, base)
    return instance


@register("crew_pairing")
def generate_crew_pairing(rng: random.Random, base: dict) -> dict:
    """基地・空港・接続規則・費用係数は base のまま、便だけを引き直す。"""
    return _crew_pairing_instance(rng, base)


@register("crew_pairing_seniority")
def generate_crew_pairing_seniority(rng: random.Random, base: dict) -> dict:
    """便に加えてクルー（基地・シニアリティ・担当数上限・拘束時間嗜好）を引き直す。"""
    instance = _crew_pairing_instance(rng, base)
    base_crews = base["crews"]
    prefix = _name_prefix(base_crews)
    seniorities = [c["seniority"] for c in base_crews]
    duties = sorted({c["max_duties"] for c in base_crews})
    prefs = [c["span_pref"] for c in base_crews]
    # 基地ごとのクルー数は base の構成比（prob_330 は A4 に 1/3 が偏る）で引く。
    base_weights = [sum(c["home_base"] == b for c in base_crews) for b in base["bases"]]
    instance["crews"] = [
        {
            "id": i,
            "name": f"{prefix}{i}",
            "home_base": rng.choices(base["bases"], weights=base_weights)[0],
            "seniority": rng.randint(min(seniorities), max(seniorities)),
            "max_duties": rng.choice(duties),
            "span_pref": round(rng.uniform(min(prefs), max(prefs)), 2),
        }
        for i in range(1, len(base_crews) + 1)
    ]
    return instance


# ---------------------------------------------------------------- 勤務表
def _requirements(rng: random.Random, base_reqs: list[dict]) -> list[dict]:
    """日ごとの必要人数を、base の曜日別平均 + 残差の標準偏差の正規ノイズで作る。

    1 日目を月曜として 7 日周期で平均を取り、base の最小〜最大に収める。
    Why not 全日一様: prob_306 は土日の必要人数が平日より約 1 割少ないので曜日の形を保つ。
    """
    days = len(base_reqs)
    out = [{"day": d + 1} for d in range(days)]
    for key in base_reqs[0]:
        if key == "day":
            continue
        values = [r[key] for r in base_reqs]
        lo, hi = min(values), max(values)
        means = [statistics.fmean(values[w::7] or values) for w in range(7)]
        sd = statistics.pstdev(v - means[d % 7] for d, v in enumerate(values))
        for d in range(days):
            out[d][key] = min(hi, max(lo, round(rng.gauss(means[d % 7], sd))))
    return out


def _preferred_off_days(rng: random.Random, base_people: list[dict], days: int) -> list[int]:
    counts = [len(p["preferred_off_days"]) for p in base_people]
    return sorted(rng.sample(range(1, days + 1), rng.randint(min(counts), max(counts))))


def _roster_instance(rng: random.Random, base: dict) -> dict:
    instance = copy.deepcopy(base)
    instance["daily_requirements"] = _requirements(rng, base["daily_requirements"])
    return instance


@register("nurse_roster")
def generate_nurse_roster(rng: random.Random, base: dict) -> dict:
    """規則・ペナルティ・シフト名は base のまま、看護師の属性と必要人数を引き直す。"""
    instance = _roster_instance(rng, base)
    nurses = base["nurses"]
    days = len(base["daily_requirements"])
    qualified_rate = _ratio([p["qualified"] for p in nurses])
    prefix = _name_prefix(nurses)
    instance["nurses"] = [
        {
            "id": i,
            "name": f"{prefix}{i}",
            "qualified": rng.random() < qualified_rate,
            "preferred_off_days": _preferred_off_days(rng, nurses, days),
        }
        for i in range(1, len(nurses) + 1)
    ]
    return instance


@register("role_roster")
def generate_role_roster(rng: random.Random, base: dict) -> dict:
    """役割の並び（base は看護師・医師・薬剤師の繰り返し）は保ち、地域・資格・シニアリティ・
    希望休と必要人数を引き直す。
    """
    instance = _roster_instance(rng, base)
    staff = base["staff"]
    days = len(base["daily_requirements"])
    regions = sorted({p["region"] for p in staff})
    qualified_rate = _ratio([p["qualified"] for p in staff])
    seniorities = [p["seniority"] for p in staff]
    prefix = _name_prefix(staff)
    instance["staff"] = [
        {
            "id": i,
            "name": f"{prefix}{i}",
            "role": person["role"],
            "qualified": rng.random() < qualified_rate,
            "seniority": rng.randint(min(seniorities), max(seniorities)),
            "region": rng.choice(regions),
            "preferred_off_days": _preferred_off_days(rng, staff, days),
        }
        for i, person in enumerate(staff, 1)
    ]
    return instance
