"""大規模問題集の instance 生成器（facility 群）。register(kind) で登録する。

facility_multi  : 多期間・単一供給元の施設配置・在庫統合（prob_304 / prob_314）
facility_2ech   : 工場→DC→顧客の 2 階層（prob_328）
facility_robust : シナリオ min-max（prob_329）

件数・スカラー・文字列は base から取り、数値属性は base の同じ属性の最小〜最大の一様乱数にする。
需要は base から「顧客間の水準のばらつき・期ごとの季節性・期内のノイズ」を推定して再現し、
候補施設は base と同じ件数の最近傍（距離昇順）にする。これで能力合計と需要合計の比や
候補集合の密度が base の近傍に収まる。
"""

from __future__ import annotations

import math
import random
import re
import statistics
from typing import Any

from . import register

# 座標の最大値をこの刻みで切り上げて正方形領域の一辺にする（399.7 → 400、299.7 → 300）。
_EXTENT_STEP = 10
# 需要水準・ノイズの一様分布は半幅 = sqrt(3) × 標準偏差で同じ分散になる。
_SQRT3 = math.sqrt(3.0)
# 候補施設リストのキー（種別で名前が違う）。
_CANDIDATE_KEYS = ("candidate_facilities", "candidate_dcs")


def _decimals(values: list[Any]) -> int | None:
    """base の値が全て整数なら None、小数なら最大の小数桁数。"""
    if all(float(v).is_integer() for v in values):
        return None
    return max(len(str(v).split(".")[1]) if "." in str(v) else 0 for v in values)


def _uniform_like(rng: random.Random, values: list[Any]) -> int | float:
    """base の属性値と平均・分散が同じ一様乱数。整数属性は整数、小数属性は同じ桁に丸める。

    Why not 最小〜最大: 標本の最小・最大は真の範囲より内側に寄り、base の分布が一様から
    ずれているときは平均も外れる。平均 ± sqrt(3)·標準偏差なら能力合計などの比が base に揃う。
    """
    mean = statistics.fmean(values)
    half = _SQRT3 * statistics.pstdev(values)
    value = rng.uniform(mean - half, mean + half)
    decimals = _decimals(values)
    return round(value) if decimals is None else round(value, decimals)


def _extent(customers: list[dict]) -> float:
    coords = [c["x"] for c in customers] + [c["y"] for c in customers]
    return math.ceil(max(coords) / _EXTENT_STEP) * _EXTENT_STEP


def _sites(rng: random.Random, base_rows: list[dict], extent: float) -> list[dict]:
    """拠点・工場・DC の行を base と同じキー順で作る。id は 1 始まり、name は base の接頭辞 + id。"""
    prefix = re.sub(r"\d+$", "", str(base_rows[0]["name"]))
    rows = []
    for i in range(len(base_rows)):
        row: dict[str, Any] = {}
        for key in base_rows[0]:
            if key == "id":
                row[key] = i + 1
            elif key == "name":
                row[key] = f"{prefix}{i + 1}"
            elif key in ("x", "y"):
                row[key] = round(rng.uniform(0.0, extent), 1)
            else:
                row[key] = _uniform_like(rng, [r[key] for r in base_rows])
        rows.append(row)
    return rows


def _demand_model(customers: list[dict], periods: int) -> dict[str, Any]:
    """base の需要を「水準 × 季節 × ノイズ」に分解したパラメータ。

    季節は期ごとの総需要の比。顧客間の水準の分散は、顧客平均の分散から期内ノイズ由来の
    標本分散（ノイズ分散 / 期数）を引いて推定する（i.i.d. 一様需要の base なら 0 になる）。
    """
    values = [d for c in customers for d in c["demand"]]
    totals = [sum(c["demand"][t] for c in customers) for t in range(periods)]
    mean_total = sum(totals) / periods
    raw_season = [v / mean_total for v in totals]
    # 季節性のない base でも期ごとの総需要は標本誤差でぶれるので、その分散を引いた比率だけ残す。
    sampling_var = statistics.pvariance(values) * len(customers) / mean_total**2
    shrink = max(0.0, 1.0 - sampling_var / max(statistics.pvariance(raw_season), 1e-12))
    season = [1.0 + (v - 1.0) * shrink for v in raw_season]
    means = [statistics.fmean(c["demand"]) for c in customers]
    residuals = [
        c["demand"][t] / (m * season[t]) - 1.0
        for c, m in zip(customers, means)
        for t in range(periods)
    ]
    noise_sd = statistics.pstdev(residuals)
    level_mean = statistics.fmean(means)
    within_var = statistics.fmean(statistics.pvariance(c["demand"]) for c in customers) / periods
    level_sd = math.sqrt(max(0.0, statistics.pvariance(means) - within_var))
    return {
        "season": season,
        "level_mean": level_mean,
        "level_half": _SQRT3 * level_sd,
        "noise_half": _SQRT3 * noise_sd,
        "floor": min(values),
    }


def _demand(rng: random.Random, model: dict[str, Any], periods: int) -> list[int]:
    level = model["level_mean"] + rng.uniform(-1.0, 1.0) * model["level_half"]
    half = model["noise_half"]
    return [
        max(model["floor"], round(level * model["season"][t] * (1.0 + rng.uniform(-half, half))))
        for t in range(periods)
    ]


def _customers(
    rng: random.Random, base_rows: list[dict], sites: list[dict], extent: float, periods: int
) -> list[dict]:
    """顧客を base と同じキー順で作る。候補は base と同じ件数の最近傍を距離昇順に並べる。"""
    cand_key = next(k for k in _CANDIDATE_KEYS if k in base_rows[0])
    n_cand = len(base_rows[0][cand_key])
    model = _demand_model(base_rows, periods)
    rows = []
    for i in range(len(base_rows)):
        x, y = round(rng.uniform(0.0, extent), 1), round(rng.uniform(0.0, extent), 1)
        nearest = sorted(sites, key=lambda s: (math.hypot(s["x"] - x, s["y"] - y), s["id"]))
        row: dict[str, Any] = {}
        for key in base_rows[0]:
            if key == "id":
                row[key] = i + 1
            elif key == "x":
                row[key] = x
            elif key == "y":
                row[key] = y
            elif key == "demand":
                row[key] = _demand(rng, model, periods)
            else:
                row[key] = [s["id"] for s in nearest[:n_cand]]
        rows.append(row)
    return rows


def _scenarios(rng: random.Random, base_rows: list[dict], customers: list[dict]) -> list[dict]:
    """需要乗数のシナリオ。乗数は base の最小〜最大の一様乱数（2 桁）、キーは顧客 id の文字列。"""
    factors = [v for s in base_rows for v in s["factors"].values()]
    lo, hi = min(factors), max(factors)
    return [
        {
            "id": i + 1,
            "factors": {str(c["id"]): round(rng.uniform(lo, hi), 2) for c in customers},
        }
        for i in range(len(base_rows))
    ]


def _generate(rng: random.Random, base: dict, site_keys: tuple[str, ...], cand_from: str) -> dict:
    """base のキー順を保ったまま、拠点群・顧客・シナリオだけを作り直し、他は base の値を写す。"""
    extent = _extent(base["customers"])
    periods = int(base["num_periods"])
    sites = {key: _sites(rng, base[key], extent) for key in site_keys}
    # Why not キー順に生成: シナリオは顧客 id に依存するので、base のキー順に関係なく顧客を先に作る。
    customers = _customers(rng, base["customers"], sites[cand_from], extent, periods)
    out: dict[str, Any] = {}
    for key, value in base.items():
        if key in sites:
            out[key] = sites[key]
        elif key == "customers":
            out[key] = customers
        elif key == "scenarios":
            out[key] = _scenarios(rng, value, customers)
        else:
            out[key] = value
    return out


@register("facility_multi")
def generate_multi(rng: random.Random, base: dict) -> dict:
    return _generate(rng, base, ("facilities",), "facilities")


@register("facility_robust")
def generate_robust(rng: random.Random, base: dict) -> dict:
    return _generate(rng, base, ("facilities",), "facilities")


@register("facility_2ech")
def generate_2ech(rng: random.Random, base: dict) -> dict:
    return _generate(rng, base, ("plants", "dcs"), "dcs")
