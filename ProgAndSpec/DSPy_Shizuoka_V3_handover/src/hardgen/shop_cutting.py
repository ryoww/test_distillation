"""大規模問題集の instance 生成器（shop_cutting 群）。register(kind) で登録する。

対象種別:
- fjsp（prob_305, 315）/ fjsp_setup（prob_322）: ジョブ数・機械数・総工程数・release_time の
  経験分布・工程ごとの (選択肢数, 加工時間の組) を base から取り、機械の割り当てだけを引き直す。
- cutting_1d（prob_307, 317）/ cutting_2d（prob_324）: 原材（stocks / plates）と上限は base のまま、
  品目の寸法を base の値域から引き直し、需要合計は base と一致させる。
件数・文字列・スカラーは base から読み、定数は埋め込まない。
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from copy import deepcopy

from . import register

# 加工時間の揺らぎ幅（base の組をそのまま使い回さないための相対ジッタ）。
_DURATION_JITTER = 0.1


def _counts_with_total(
    rng: random.Random, pool: Sequence[int], n: int, total: int
) -> list[int]:
    """pool から n 個を復元抽出し、合計が total になるまで pool の値域内で 1 ずつ寄せる。

    description の「合計 598 工程」「合計 4023 本」を生成 instance でも正しく保つため、
    合計を base と一致させる。total は base 由来なので [n*min, n*max] に収まる。
    """
    lo, hi = min(pool), max(pool)
    counts = [rng.choice(pool) for _ in range(n)]
    diff = total - sum(counts)
    while diff != 0:
        i = rng.randrange(n)
        if diff > 0 and counts[i] < hi:
            counts[i] += 1
            diff -= 1
        elif diff < 0 and counts[i] > lo:
            counts[i] -= 1
            diff += 1
    return counts


# ---------------------------------------------------------------- FJSP
def _fjsp_jobs(rng: random.Random, base: dict) -> list[dict]:
    """ジョブ列を作る。工程数・release_time・加工時間の組は base の経験分布から再標本化する。"""
    jobs = base["jobs"]
    machine_ids = [m["id"] if isinstance(m, dict) else m for m in base["machines"]]
    ops_pool = [len(j["operations"]) for j in jobs]
    release_pool = [j.get("release_time", 0) for j in jobs]
    # 選択肢数ごとの加工時間の組。(選択肢数, 加工時間) の同時分布を保つため工程単位で持つ。
    duration_pool: dict[int, list[list[int]]] = {}
    for job in jobs:
        for op in job["operations"]:
            durations = sorted(op["machine_options"].values())
            duration_pool.setdefault(len(durations), []).append(durations)
    dur_lo = min(d[0] for pool in duration_pool.values() for d in pool)
    dur_hi = max(d[-1] for pool in duration_pool.values() for d in pool)
    n_ops = _counts_with_total(rng, ops_pool, len(jobs), sum(ops_pool))

    new_jobs = []
    for job, count in zip(jobs, n_ops):
        # Why not 選択肢数を全工程で独立に引く: shape_signature はジョブごとに「工程の選択肢数の
        # 集合」を記録するので、各ジョブは base の同じジョブに現れる選択肢数をちょうど 1 回ずつ
        # 含み、残りの工程はその集合から引く（工程数は集合の大きさより常に多い）。
        option_counts = sorted({len(op["machine_options"]) for op in job["operations"]})
        option_counts += [rng.choice(option_counts) for _ in range(count - len(option_counts))]
        rng.shuffle(option_counts)
        operations = []
        for op_index, k in enumerate(option_counts, start=1):
            durations = list(rng.choice(duration_pool[k]))
            rng.shuffle(durations)
            chosen = sorted(rng.sample(machine_ids, k))
            # Why not 選択肢ごとに独立に揺らす: 工程内の機械間の加工時間比（速い機械と遅い機械の
            # 差）が base より広がるので、組全体に共通の倍率を掛ける。
            scale = rng.uniform(1 - _DURATION_JITTER, 1 + _DURATION_JITTER)
            options = {
                str(m): min(dur_hi, max(dur_lo, round(d * scale)))
                for m, d in zip(chosen, durations)
            }
            operations.append({"op_index": op_index, "machine_options": options})
        new_job = dict(job)
        new_job["operations"] = operations
        if "release_time" in job:
            new_job["release_time"] = rng.choice(release_pool)
        new_jobs.append(new_job)
    return new_jobs


@register("fjsp")
def generate_fjsp(rng: random.Random, base: dict) -> dict:
    new = deepcopy(base)
    new["jobs"] = _fjsp_jobs(rng, base)
    return new


def _setups(rng: random.Random, base_setups: dict) -> dict:
    """setups[m][a][b] を base と同じキーで引き直す。対角は 0、他は base の値域の一様乱数。"""
    values = [
        v for table in base_setups.values()
        for a, row in table.items() for b, v in row.items() if a != b
    ]
    lo, hi = min(values), max(values)
    return {
        m: {a: {b: 0 if a == b else rng.randint(lo, hi) for b in row} for a, row in table.items()}
        for m, table in base_setups.items()
    }


def _breakdowns(rng: random.Random, base_breakdowns: dict) -> dict:
    """機械ごとの故障区間。件数は base の件数を機械間で並べ替え、開始・長さは base の値域。"""
    rows = [br for lst in base_breakdowns.values() for br in lst]
    start_lo, start_hi = min(br["start"] for br in rows), max(br["start"] for br in rows)
    dur_lo, dur_hi = min(br["duration"] for br in rows), max(br["duration"] for br in rows)
    counts = [len(lst) for lst in base_breakdowns.values()]
    # Why not 全件数をシャッフル: shape_signature は数値キー dict の先頭値の長さを見るので、
    # 先頭機械の件数だけは base と同じにし、残りの件数を機械間で入れ替える。
    rest = counts[1:]
    rng.shuffle(rest)
    return {
        m: [
            {"start": rng.randint(start_lo, start_hi), "duration": rng.randint(dur_lo, dur_hi)}
            for _ in range(count)
        ]
        for m, count in zip(base_breakdowns, [counts[0], *rest])
    }


@register("fjsp_setup")
def generate_fjsp_setup(rng: random.Random, base: dict) -> dict:
    new = deepcopy(base)
    new["jobs"] = _fjsp_jobs(rng, base)
    if "setups" in base:
        new["setups"] = _setups(rng, base["setups"])
    if "breakdowns" in base:
        new["breakdowns"] = _breakdowns(rng, base["breakdowns"])
    return new


# ---------------------------------------------------------------- カッティング
def _demands(rng: random.Random, items: list[dict]) -> list[int]:
    demands = [i["demand"] for i in items]
    return _counts_with_total(
        rng, range(min(demands), max(demands) + 1), len(items), sum(demands)
    )


@register("cutting_1d")
def generate_cutting_1d(rng: random.Random, base: dict) -> dict:
    """品目幅は base の値域から重複なしで引き、需要は base の値域の一様乱数で合計を合わせる。"""
    items = base["items"]
    widths = [i["width"] for i in items]
    new_widths = rng.sample(range(min(widths), max(widths) + 1), len(items))
    new = deepcopy(base)
    new["items"] = [
        dict(item, width=w, demand=d)
        for item, w, d in zip(items, new_widths, _demands(rng, items))
    ]
    return new


@register("cutting_2d")
def generate_cutting_2d(rng: random.Random, base: dict) -> dict:
    """品目の幅・高さは base に現れる値（100 刻みの格子）から独立に引き、需要合計を合わせる。"""
    items = base["items"]
    width_pool = sorted({i["width"] for i in items})
    height_pool = sorted({i["height"] for i in items})
    new = deepcopy(base)
    new["items"] = [
        dict(item, width=rng.choice(width_pool), height=rng.choice(height_pool), demand=d)
        for item, d in zip(items, _demands(rng, items))
    ]
    return new
