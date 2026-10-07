"""routing 系の雛形（第 3 群: prob_026〜030）。 templates.py と同じ規約で (generate, solve) を登録する。

同梱参照解はいずれも ortools routing の近似解で、区間距離を整数に切り捨てて足している。
ここでは templates_routing と同じく丸めなしのユークリッド距離で厳密解を返すので、
shipped_reference_optimal=False で登録し、テストは「厳密解が同梱値より良い」ことを検査する。
"""

from __future__ import annotations

import random

from .base import register, retry
from .templates_routing import _route_length, _solve_cvrp


def _cvrp_routes(depot: dict, customers: list[dict], capacity: int, vehicles: int) -> dict:
    """顧客リストを CVRP として厳密に解き、prob_021 と同じ形の routes 辞書を返す。

    customers の各要素には demand キーが必要。距離は丸めなしのユークリッドで、
    templates_routing の採点側規約（スコアラーが経路から再計算する）に合わせる。
    """
    _, routes = _solve_cvrp(
        {
            "depot": depot,
            "customers": customers,
            "vehicle_capacity": capacity,
            "num_vehicles": vehicles,
        }
    )
    points = [depot, *customers]
    return {
        str(i + 1): {"route": [0, *order, 0], "distance": _route_length(points, order)}
        for i, order in enumerate(routes)
    }


def _random_point(rng: random.Random) -> dict:
    return {"x": rng.randint(-15, 15), "y": rng.randint(-15, 15)}


@register(26, "total_initial_distance", shipped_reference_optimal=False)
def dynamic_vrp_initial_routes():
    """prob_026: 動的 VRP の初期経路。目的値は初期顧客だけの CVRP 最適値と定義する。

    問題文は「初期時点で5人の顧客」だが instance の initial_customers は 4 人で、
    instance を正として件数は雛形のまま保つ。参照解が返すのも初期経路のみなので、
    total_initial_distance = 初期顧客集合に対する CVRP（2 台、容量 15）の最小総距離とする。
    動的顧客は instance に残すが、再最適化戦略は目的値に含めない（問題文が方針を定めていない）。

    同梱参照解は 1 台で 4 人を回る単一経路で、区間距離を切り捨てて 88 と申告している
    （丸めなしでは 90.88）。厳密最適は 75.49。
    """

    def generate(rng: random.Random, base: dict) -> dict:
        n_initial = len(base["initial_customers"])
        n_dynamic = len(base["dynamic_customers"])
        capacity = base["vehicle_capacity"]
        vehicles = base["num_vehicles"]

        def make() -> dict:
            initial = [
                {"id": i + 1, **_random_point(rng), "demand": rng.randint(2, 8), "known_at": 0}
                for i in range(n_initial)
            ]
            dynamic = [
                {
                    "id": n_initial + i + 1,
                    **_random_point(rng),
                    "demand": rng.randint(1, 4),
                    "known_at": rng.randint(1, 8),
                }
                for i in range(n_dynamic)
            ]
            return {
                "depot": {"id": 0, "x": 0, "y": 0},
                "initial_customers": initial,
                "dynamic_customers": dynamic,
                "vehicle_capacity": capacity,
                "num_vehicles": vehicles,
            }

        def ok(inst: dict) -> bool:
            initial = sum(c["demand"] for c in inst["initial_customers"])
            dynamic = sum(c["demand"] for c in inst["dynamic_customers"])
            # 初期顧客が 1 台に載ると経路分割が要らないので 2 台必要な instance に限る。
            # 動的顧客まで含めても 2 台の容量に収まるようにし、再最適化の余地を残す。
            fleet = capacity * vehicles
            return capacity < initial <= fleet and initial + dynamic <= fleet

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        routes = _cvrp_routes(
            instance["depot"],
            instance["initial_customers"],
            instance["vehicle_capacity"],
            instance["num_vehicles"],
        )
        return {
            "initial_routes": routes,
            "total_initial_distance": sum(r["distance"] for r in routes.values()),
            "note": "初期顧客の CVRP を部分集合 Held-Karp + 分割 DP で厳密に解いた初期経路。"
            "距離は丸めなしのユークリッド。動的顧客の再最適化は目的値に含めない",
        }

    return generate, solve


@register(28, "total_distance", shipped_reference_optimal=False)
def waste_collection_vrp():
    """prob_028: 収集 VRP。路段の現在のゴミ量を需要とする CVRP（2 台、容量 50）。

    fill_rate_per_day と路段の capacity は当日の収集量に影響しないので目的値には使わない
    （ゴミ量が路段容量を超えないことだけ生成時に保つ）。
    同梱参照解は 1 台で 10 路段を回る単一経路で、区間距離を切り捨てて 201 と申告している
    （丸めなしでは 204.19）。厳密最適は 107.48。
    """

    def generate(rng: random.Random, base: dict) -> dict:
        n = len(base["segments"])
        capacity = base["truck_capacity"]
        trucks = base["num_trucks"]
        segment_capacity = base["segments"][0]["capacity"]

        def make() -> dict:
            segments = [
                {
                    "id": i + 1,
                    **_random_point(rng),
                    "waste_volume": rng.randint(2, min(12, segment_capacity)),
                    "fill_rate_per_day": rng.uniform(0.5, 2.0),
                    "capacity": segment_capacity,
                }
                for i in range(n)
            ]
            return {
                "depot": {"id": 0, "x": 0, "y": 0},
                "segments": segments,
                "truck_capacity": capacity,
                "num_trucks": trucks,
            }

        def ok(inst: dict) -> bool:
            total = sum(s["waste_volume"] for s in inst["segments"])
            # 1 台に全部載ると TSP になり容量制約が効かないので、2 台必要な instance に限る。
            return capacity < total <= capacity * trucks

        return retry(rng, make, ok)

    def solve(instance: dict) -> dict:
        segments = [{**s, "demand": s["waste_volume"]} for s in instance["segments"]]
        routes = _cvrp_routes(
            instance["depot"], segments, instance["truck_capacity"], instance["num_trucks"]
        )
        return {
            "routes": routes,
            "total_distance": sum(r["distance"] for r in routes.values()),
            "note": "部分集合 Held-Karp + 分割 DP（厳密最適解）。距離は丸めなしのユークリッド",
        }

    return generate, solve


# Why not prob_027（配送＋在庫連立）: 問題文は「顧客の在庫が発注点未満になった場合に配送」と
# いうが、instance の reorder_point は倉庫の補充パラメータ（reorder_quantity と対）で、顧客には
# 発注点がない。訪問集合を問題文から一意に定められない。さらに配送量が instance に定義されず、
# 車両容量 30 と倉庫在庫 50 の制約を課す荷物量が決まらない（満杯まで配送すると 64 > 60 で
# 実行不可能、顧客 1 は在庫 12 > 最大容量 10）。参照解の inventory_projection も配送量を
# 含まず、after_delivery は単に在庫 − 日次需要である。全 6 顧客を回る TSP は 70.02 で
# 参照値 103 より短いが、それは問題文の「配送を行うかどうかを判断」を無視した値になる。

# Why not prob_029（乗客ピックアップ＆ドロップオフ）: 目的は「待機時間の合計の最小化」だが、
# 待機時間の定義（誰の、いつからいつまで）も車両の出発地点も速度も instance にない。参照解は
# 別の量 total_distance=166 を申告しており、これは (0,0) を出発地として区間を切り捨てた値で、
# 1 台に 18 人を乗せて（容量 4）全員を先に拾ってから降ろす実行不可能な経路である。
# 文章の目的と参照解の目的値が一致せず、最適を一意に定義できない。

# Why not prob_030（貨物列車編成計画）: 目的が「列車数の最小化と目的地別集約」の二目的で、
# 重みが与えられていない。同梱 instance の 12 両は合計 593 t・153 m で機関車の上限
# （800 t・250 m・15 両）に収まるので、列車数だけなら 1 が最小だが、参照解は目的地ごとに
# 1 列車ずつ 5 本に分けている。列車を目的地単位に限れば列車数は常に目的地数に一致して自明。
# wagon_order を決める基準も、stations の max_weight / max_length の意味も問題文にない。
