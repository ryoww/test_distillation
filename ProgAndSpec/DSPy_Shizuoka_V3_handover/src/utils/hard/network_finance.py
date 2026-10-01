"""大規模問題集の検証器（network_finance 群）。register_kind で種別を登録する。

種別と判定キー:

- mcnd: ``arcs`` + ``od_demands``（``disjoint_paths`` なし）。容量制約付き多品種網設計。
- mcnd_surv: ``arcs`` + ``od_demands`` + ``disjoint_paths``。各 OD を edge-disjoint な
  複数経路に等分し、各経路の距離合計に上限がある網設計。
- portfolio: ``assets`` + ``scenario_returns``（``risk.measure`` が cvar でない）。
  MAD 最小化のカーディナリティ制約ポートフォリオ。
- portfolio_cvar: ``assets`` + ``scenario_returns`` + ``risk.measure == "cvar"``。

目的値の定義は同梱参照解の objective_value を再現するものを採用した:

- 網設計の費用 = opened_arc_ids の固定費合計 + Σ(flow_cost × アーク流量)。流量は od_paths から
  再構成する（arc_flows は申告値なので整合だけ見る）。
- MAD = (1/S) Σ_s |r_s − μ_w|。r_s はシナリオ s のポートフォリオ収益、μ_w = Σ w_i expected_return_i。
  Why not シナリオ平均からの偏差: scenario_returns は平均がほぼ 0 の擾乱で、参照解は
  銘柄の expected_return を中心に偏差を取っている。
- CVaR_α = 損失 L_s = μ_w − r_s の上位 (1−α)S 個の平均（(1−α)S が整数でなければ端数按分）。
- 回転率 = Σ_i |w_i − w0_i|（半分にしない）。

網設計の od_paths は参照解の形（OD 番号 → {path, volume} のリスト / path1..pathK + volume_each）に加え、
モデルが実際に返した次の形も読む。意味が一意に取れない形（流量の無い複数経路、並行アークのある
ノード列、何の OD か分からないキー）は unverified のままにする。

- 要素が origin/destination を持つ record のリスト、"o-d" / "o_d" 形式のキー、キー=origin + ``to``。
  同じ起終点の OD は経路上区別できないので対にまとめて需要を合計する（mcnd）。mcnd_surv は
  相違性が OD ごとなので、対の要素が 1 つなら同じ対の全 OD に適用する。
- 経路は ``arc_ids``/``arcs``（アーク id 列）、``nodes``（ノード列）、``path``（起点で始まり終点で
  終わり全ホップがアークならノード列、そうでなければアーク id 列）。``paths`` に複数経路を入れる形。
- OD ごとの ``flow_distribution`` {arc_id: 流量}（mcnd）。経路に分解せず流量保存則で検査する。
- 申告流量 ``arc_flows``/``arc_loads`` は dict でもアーク順の list でもよい。
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from itertools import pairwise
from typing import Any

from ..feasibility_v3_ext import _num, _result, _unverified
from . import register_kind

_FLOW_TOL = 1e-6
_DECLARED_REL_TOL = 5e-3
# 参照解の weights は小数 4〜5 桁に丸められており、半連続の上下限を 1e-4 だけ超えている。
# 丸め由来の超過だけを許すため、重み系の制約は絶対許容差 5e-4 で比較する。
_PORT_TOL = 5e-4
_HOLD_EPS = 1e-9


# ---------------------------------------------------------------------------
# 共通
# ---------------------------------------------------------------------------


def _declared_mismatch(solution: dict, cost: float, *names: str) -> str | None:
    """申告目的値があり、再計算値と 0.5% 以上ずれていれば違反文を返す。"""
    for name in names:
        declared = _num(solution.get(name))
        if declared is None:
            continue
        # Why not max(1, |cost|): ポートフォリオの目的値は 0.1 程度なので、絶対 0.005 の許容では
        # 数 % のずれを見逃す。相対 0.5% に微小な絶対下限だけを付ける。
        if abs(declared - cost) > max(_DECLARED_REL_TOL * abs(cost), 1e-6):
            return f"declared {name} {declared:.6g} differs from recomputed {cost:.6g}"
        return None
    return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def _id_list(value: Any) -> list[int] | None:
    """整数 id のリストとして読めるなら返す。"""
    if not isinstance(value, (list, tuple)):
        return None
    out = []
    for item in value:
        parsed = _int_or_none(item)
        if parsed is None:
            return None
        out.append(parsed)
    return out


def _indexed_entries(value: Any) -> dict[int, Any] | None:
    """OD 番号 → 解の要素。dict（キーは番号）でも list（位置が番号）でも読む。"""
    if isinstance(value, list):
        return dict(enumerate(value))
    if not isinstance(value, dict):
        return None
    out: dict[int, Any] = {}
    for key, item in value.items():
        index = _int_or_none(key)
        if index is None:
            return None
        out[index] = item
    return out


# ---------------------------------------------------------------------------
# 網設計（mcnd / mcnd_surv）
# ---------------------------------------------------------------------------


def _detect_network(instance: dict) -> bool:
    return isinstance(instance.get("arcs"), list) and isinstance(instance.get("od_demands"), list)


def _detect_mcnd(instance: dict) -> bool:
    return _detect_network(instance) and "disjoint_paths" not in instance


def _detect_mcnd_surv(instance: dict) -> bool:
    return _detect_network(instance) and "disjoint_paths" in instance


class _Unreadable(Exception):
    """解の形から意味が一意に取れないときに投げ、検証器の入口で unverified に変換する。"""


_ARC_KEYS = ("arc_ids", "arcs", "arc_path")
_NODE_KEYS = ("nodes", "node_path", "path_nodes")
_VOLUME_KEYS = ("volume", "flow", "amount", "quantity")
_FLOW_DICT_KEYS = ("flow_distribution", "arc_flows", "flows")
_PAIR_KEY = re.compile(r"^\(?\s*(\d+)\s*(?:[-_,:;|>\s]|->)+\s*(\d+)\s*\)?$")


class _Net:
    """instance のアーク表と、同じ起終点を持つ OD をまとめた群。

    同じ (origin, destination) の OD は経路上区別できない（流量保存と費用は合計しか見ない）ので、
    対にまとめて需要を合計し、対で記述された解も番号で記述された解も同じ群に落とす。
    """

    def __init__(self, instance: dict) -> None:
        self.arcs: dict[int, dict] = {int(arc["id"]): arc for arc in instance["arcs"]}
        self.by_pair: dict[tuple[int, int], list[int]] = defaultdict(list)
        for arc_id, arc in self.arcs.items():
            self.by_pair[(int(arc["from"]), int(arc["to"]))].append(arc_id)
        self.demands: list[dict] = instance["od_demands"]
        self.groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, od in enumerate(self.demands):
            self.groups[(int(od["origin"]), int(od["destination"]))].append(index)

    def label(self, pair: tuple[int, int]) -> str:
        return "OD " + "/".join(str(i) for i in self.groups[pair])

    def demand(self, pair: tuple[int, int]) -> float:
        return sum(float(self.demands[i]["volume"]) for i in self.groups[pair])

    def nodes_to_arcs(self, nodes: list[int]) -> list[int]:
        """ノード列をアーク id 列に変換する。無いホップは落とす（_walk が不連続として報告する）。"""
        out = []
        for a, b in pairwise(nodes):
            candidates = self.by_pair.get((a, b), [])
            if len(candidates) > 1:
                raise _Unreadable(f"node path hop {a}->{b} has parallel arcs {candidates}")
            out.extend(candidates)
        return out

    def is_node_walk(self, ids: list[int], origin: int, destination: int) -> bool:
        """起点で始まり終点で終わり、全ホップがアークであればノード列とみなす。"""
        # Why not 常にアーク列: 参照解はアーク id 列だが、ノード列で返すモデルもある。
        # 両方の読みで妥当になることは事実上ないので、ノード列として歩けるかで判定する。
        return (
            len(ids) >= 2
            and ids[0] == origin
            and ids[-1] == destination
            and all((a, b) in self.by_pair for a, b in pairwise(ids))
        )

    def route(self, item: Any, origin: int, destination: int) -> tuple[list[int], float | None]:
        """1 本の経路を (アーク id 列, 申告流量 or None) に読む。list でも dict でもよい。"""
        if isinstance(item, dict):
            volume = next((_num(item[k]) for k in _VOLUME_KEYS if k in item), None)
            for key in _ARC_KEYS:
                if key in item:
                    return self._arcs(item[key], key), volume
            for key in _NODE_KEYS:
                if key in item:
                    nodes = _id_list(item[key])
                    if nodes is None:
                        raise _Unreadable(f"{key} is not a list of node ids")
                    return self.nodes_to_arcs(nodes), volume
            if "path" not in item:
                raise _Unreadable("route lacks path/arc_ids/nodes")
            item = item["path"]
        else:
            volume = None
        ids = _id_list(item)
        if ids is None:
            raise _Unreadable("path is not a list of ids")
        if self.is_node_walk(ids, origin, destination):
            return self.nodes_to_arcs(ids), volume
        return ids, volume

    @staticmethod
    def _arcs(value: Any, key: str) -> list[int]:
        ids = _id_list(value)
        if ids is None:
            raise _Unreadable(f"{key} is not a list of arc ids")
        return ids

    def resolve(self, key: Any, entry: Any) -> int | tuple[int, int] | str:
        """解の要素がどの OD 群のものかを返す。該当 OD が無ければ違反文を返す。"""
        index = None
        pair = None
        if isinstance(entry, dict):
            index = _int_or_none(entry.get("od", entry.get("od_index")))
            o, d = _int_or_none(entry.get("origin")), _int_or_none(entry.get("destination"))
            if o is not None and d is not None:
                pair = (o, d)
            elif _int_or_none(key) is not None and _int_or_none(entry.get("to")) is not None:
                pair = (_int_or_none(key), _int_or_none(entry["to"]))
        if index is None and pair is None:
            index = _int_or_none(key)
            match = _PAIR_KEY.match(key) if isinstance(key, str) and index is None else None
            if match:
                pair = (int(match.group(1)), int(match.group(2)))
        if index is not None:
            # Why not 対を優先: 明示の OD 番号は起終点より狭い指定なので、同じ対の OD を取り違えない。
            if index < 0 or index >= len(self.demands):
                return f"od_paths has unknown OD index {index}"
            return index
        if pair is None:
            raise _Unreadable(f"od_paths key {key!r} names no OD index or origin/destination")
        if pair not in self.groups:
            return f"od_paths names unknown OD pair {pair[0]}->{pair[1]}"
        return pair

    def pair_of(self, index: int) -> tuple[int, int]:
        od = self.demands[index]
        return (int(od["origin"]), int(od["destination"]))

    def grouped(self, raw: Any, *, per_od: bool) -> tuple[dict[Any, list[Any]], list[str]]:
        """od_paths を OD 群（per_od=False）または OD 番号（per_od=True）ごとの要素リストにまとめる。

        対で書かれた要素を番号に割り付けるとき、要素が 1 つなら同じ対の全 OD に適用し、
        要素数が OD 数と等しければ順に対応させる。
        """
        items = list(raw.items()) if isinstance(raw, dict) else None
        if items is None:
            if not isinstance(raw, list):
                raise _Unreadable("od_paths is neither dict nor list")
            items = list(enumerate(raw))
        by_index: dict[int, list[Any]] = defaultdict(list)
        by_pair: dict[tuple[int, int], list[Any]] = defaultdict(list)
        violations: list[str] = []
        for key, entry in items:
            where = self.resolve(key, entry)
            if isinstance(where, str):
                violations.append(where)
            elif isinstance(where, int):
                by_index[where].append(entry)
            else:
                by_pair[where].append(entry)
        if not per_od:
            for index, entries in by_index.items():
                by_pair[self.pair_of(index)].extend(entries)
            return by_pair, violations
        for pair, entries in by_pair.items():
            indices = self.groups[pair]
            if len(entries) == 1:
                entries = entries * len(indices)
            elif len(entries) != len(indices):
                raise _Unreadable(f"{self.label(pair)} has {len(entries)} entries")
            for index, entry in zip(indices, entries):
                by_index[index].append(entry)
        return by_index, violations


def _walk(
    arcs: dict[int, dict], path: list[int], origin: int, destination: int, label: str
) -> tuple[list[str], float]:
    """経路を辿り、違反文と距離合計を返す。未知アークは距離に数えない。"""
    violations: list[str] = []
    node = origin
    distance = 0.0
    for arc_id in path:
        arc = arcs.get(arc_id)
        if arc is None:
            violations.append(f"{label} uses unknown arc {arc_id}")
            continue
        if arc["from"] != node:
            violations.append(f"{label} is disconnected at arc {arc_id}")
        node = arc["to"]
        distance += float(arc.get("distance", 0.0))
    if node != destination:
        violations.append(f"{label} ends at node {node}, not destination {destination}")
    return violations, distance


def _declared_flows(solution: dict, arcs: dict[int, dict]) -> dict[int, float] | None:
    """申告アーク流量を アーク id → 流量 に読む。dict でもアーク数と同じ長さの list でもよい。"""
    raw = next((solution[k] for k in ("arc_flows", "arc_loads") if k in solution), None)
    if isinstance(raw, list) and len(raw) == len(arcs):
        raw = dict(zip(arcs, raw))
    if not isinstance(raw, dict):
        return None
    out: dict[int, float] = {}
    for key, value in raw.items():
        arc_id, flow = _int_or_none(key), _num(value)
        if arc_id is None or flow is None:
            return None
        out[arc_id] = flow
    return out


def _network_cost_checks(
    instance: dict,
    solution: dict,
    opened: set[int],
    flows: dict[int, float],
    violations: list[str],
) -> tuple[int, float]:
    """開設アークの妥当性、容量、費用、申告値の整合を検査し、検査数と費用を返す。"""
    arcs = {int(arc["id"]): arc for arc in instance["arcs"]}
    checks = 0
    unknown = sorted(a for a in opened if a not in arcs)
    checks += 1
    if unknown:
        violations.append(f"opened_arc_ids contains unknown arcs {unknown[:5]}")
    for arc_id, arc in arcs.items():
        checks += 1
        flow = flows.get(arc_id, 0.0)
        capacity = float(arc["capacity"])
        if flow > capacity + _FLOW_TOL:
            violations.append(f"arc {arc_id} carries {flow:.6g} over capacity {capacity:.6g}")
    declared_flows = _declared_flows(solution, arcs)
    if declared_flows is not None:
        checks += 1
        mismatched = sum(
            1
            for arc_id in set(flows) | set(declared_flows)
            if abs(declared_flows.get(arc_id, 0.0) - flows.get(arc_id, 0.0)) > _FLOW_TOL
        )
        if mismatched:
            violations.append(f"arc_flows disagrees with od_paths on {mismatched} arcs")
    fixed = sum(float(arcs[a]["fixed_cost"]) for a in opened if a in arcs)
    variable = sum(float(arcs[a]["flow_cost"]) * f for a, f in flows.items() if a in arcs)
    cost = fixed + variable
    checks += 1
    mismatch = _declared_mismatch(solution, cost, "objective_value", "total_cost")
    if mismatch:
        violations.append(mismatch)
    return checks, cost


def _flow_dict(entry: Any) -> dict[int, float] | None:
    """OD 要素が {arc_id: 流量} の流量表で書かれていれば読む。"""
    if not isinstance(entry, dict):
        return None
    raw = next((entry[k] for k in _FLOW_DICT_KEYS if isinstance(entry.get(k), dict)), None)
    if raw is None:
        return None
    out: dict[int, float] = {}
    for key, value in raw.items():
        arc_id, flow = _int_or_none(key), _num(value)
        if arc_id is None or flow is None:
            raise _Unreadable("flow_distribution is not arc_id -> number")
        out[arc_id] = flow
    return out


def _balance(
    net: _Net, arc_flow: dict[int, float], origin: int, destination: int, label: str
) -> tuple[list[str], float]:
    """流量表の保存則を検査し、違反文と起点からの正味送出量を返す。"""
    violations: list[str] = []
    net_out: dict[int, float] = defaultdict(float)
    for arc_id, flow in arc_flow.items():
        arc = net.arcs.get(arc_id)
        if arc is None:
            violations.append(f"{label} uses unknown arc {arc_id}")
            continue
        if flow < -_FLOW_TOL:
            violations.append(f"{label} has negative flow {flow:.6g} on arc {arc_id}")
        net_out[int(arc["from"])] += flow
        net_out[int(arc["to"])] -= flow
    carried = net_out.pop(origin, 0.0)
    for node, value in net_out.items():
        expected = -carried if node == destination else 0.0
        if abs(value - expected) > _FLOW_TOL * max(1.0, abs(carried)):
            violations.append(f"{label} flow is unbalanced at node {node}")
    return violations, carried


def _mcnd_routes(net: _Net, entry: Any, origin: int, destination: int) -> list:
    """mcnd の OD 要素を経路のリストに読む。参照解は {path, volume} のリスト。"""
    if isinstance(entry, dict) and isinstance(entry.get("paths"), list):
        entry = entry["paths"]
    if isinstance(entry, dict) or (isinstance(entry, list) and _id_list(entry) is not None):
        entry = [entry]
    if not isinstance(entry, list):
        raise _Unreadable("od_paths entry is neither a route nor a list of routes")
    return [net.route(item, origin, destination) for item in entry]


def _check_mcnd(instance: dict, solution: Any) -> dict:
    if not isinstance(solution, dict):
        return _unverified("solution is not a dict")
    opened_list = _id_list(solution.get("opened_arc_ids"))
    if opened_list is None or "od_paths" not in solution:
        return _unverified("opened_arc_ids or od_paths missing")
    net = _Net(instance)
    try:
        groups, violations = net.grouped(solution["od_paths"], per_od=False)
        parsed: dict[tuple[int, int], list] = {}
        for pair, entries in groups.items():
            routes: list = []
            for entry in entries:
                flow_dict = _flow_dict(entry)
                if flow_dict is not None:
                    routes.append(flow_dict)
                else:
                    routes.extend(_mcnd_routes(net, entry, *pair))
            parsed[pair] = routes
    except _Unreadable as exc:
        return _unverified(str(exc))

    opened = set(opened_list)
    flows: dict[int, float] = defaultdict(float)
    checks = 1
    for pair in net.groups:
        checks += 2
        label = net.label(pair)
        demand = net.demand(pair)
        carried = 0.0
        routes = parsed.get(pair, [])
        unsized = [r for r in routes if isinstance(r, tuple) and r[1] is None]
        if len(unsized) > 1:
            # Why not 等分: 分流の割合は解が申告しない限り決められない。
            return _unverified(f"{label} splits over {len(unsized)} paths without volumes")
        for route in routes:
            if isinstance(route, dict):
                balance_violations, sent = _balance(net, route, *pair, label)
                violations.extend(balance_violations)
                carried += sent
                used = [a for a, f in route.items() if f > _FLOW_TOL]
            else:
                path, volume = route
                volume = demand if volume is None else volume
                if volume < 0:
                    violations.append(f"{label} has negative volume {volume:.6g}")
                    continue
                carried += volume
                walk_violations, _ = _walk(net.arcs, path, *pair, label)
                violations.extend(walk_violations)
                route = dict.fromkeys(path, volume)
                used = path
            for arc_id in used:
                if arc_id in net.arcs and arc_id not in opened:
                    violations.append(f"{label} flows on closed arc {arc_id}")
            for arc_id, volume in route.items():
                flows[arc_id] += volume
        if abs(carried - demand) > _FLOW_TOL * max(1.0, demand):
            violations.append(f"{label} carries {carried:.6g} of demand {demand:.6g}")
    extra, cost = _network_cost_checks(instance, solution, opened, flows, violations)
    return _result(violations, checks + extra, cost=cost)


def _surv_routes(
    net: _Net, entry: Any, count: int, origin: int, destination: int
) -> tuple[list[tuple[list[int], float | None]], float | None]:
    """mcnd_surv の OD 要素を (経路リスト, volume_each) に読む。参照解は path1..pathK + volume_each。"""
    if not isinstance(entry, dict):
        raise _Unreadable("od_paths entry is not a dict")
    if "path1" in entry:
        items = [entry[k] for k in (f"path{k}" for k in range(1, count + 1)) if k in entry]
    elif isinstance(entry.get("paths"), list):
        items = entry["paths"]
    else:
        raise _Unreadable(f"od_paths entry lacks path1..path{count} or paths")
    routes = [net.route(item, origin, destination) for item in items]
    volume_each = _num(entry["volume_each"]) if "volume_each" in entry else None
    return routes, volume_each


def _check_mcnd_surv(instance: dict, solution: Any) -> dict:
    if not isinstance(solution, dict):
        return _unverified("solution is not a dict")
    count = int(instance["disjoint_paths"])
    opened_list = _id_list(solution.get("opened_arc_ids"))
    if opened_list is None or "od_paths" not in solution:
        return _unverified("opened_arc_ids or od_paths missing")
    net = _Net(instance)
    try:
        groups, violations = net.grouped(solution["od_paths"], per_od=True)
        parsed = {}
        for index, entries in groups.items():
            if len(entries) > 1:
                # Why not 連結: 1 つの OD に複数の要素があると相違性の対象が決められない。
                raise _Unreadable(f"OD {index} has {len(entries)} entries")
            parsed[index] = _surv_routes(net, entries[0], count, *net.pair_of(index))
    except _Unreadable as exc:
        return _unverified(str(exc))

    opened = set(opened_list)
    flows: dict[int, float] = defaultdict(float)
    checks = 1
    for index, od in enumerate(net.demands):
        # 経路の妥当性、相違性、遅延、分流量の 4 項目を OD ごとに検査する。
        checks += 4
        label = f"OD {index}"
        pair = net.pair_of(index)
        demand = float(od["volume"])
        if index not in parsed:
            violations.append(f"{label} has no paths")
            continue
        routes, volume_each = parsed[index]
        if len(routes) != count:
            violations.append(f"{label} has {len(routes)} paths instead of {count}")
        declared = [v for _, v in routes if v is not None]
        if volume_each is None and declared:
            volume_each = declared[0]
            if max(declared) - min(declared) > _FLOW_TOL * max(1.0, demand):
                violations.append(f"{label} splits unequally {declared}")
        if volume_each is None:
            volume_each = demand / count
        elif abs(volume_each * count - demand) > _FLOW_TOL * max(1.0, demand):
            violations.append(
                f"{label} splits {volume_each:.6g}x{count} instead of demand {demand:.6g}"
            )
        if volume_each < 0:
            violations.append(f"{label} has negative volume_each {volume_each:.6g}")
            continue
        used: set[int] = set()
        max_delay = _num(od.get("max_delay"))
        for k, (path, _) in enumerate(routes, start=1):
            path_label = f"{label} path{k}"
            walk_violations, distance = _walk(net.arcs, path, *pair, path_label)
            violations.extend(walk_violations)
            if max_delay is not None and distance > max_delay + _FLOW_TOL:
                violations.append(
                    f"{path_label} length {distance:.6g} exceeds max_delay {max_delay:.6g}"
                )
            for arc_id in path:
                if arc_id in used:
                    violations.append(f"{label} shares arc {arc_id} between paths")
                used.add(arc_id)
                if arc_id in net.arcs and arc_id not in opened:
                    violations.append(f"{path_label} flows on closed arc {arc_id}")
                flows[arc_id] += volume_each
    extra, cost = _network_cost_checks(instance, solution, opened, flows, violations)
    return _result(violations, checks + extra, cost=cost)


# ---------------------------------------------------------------------------
# ポートフォリオ（portfolio / portfolio_cvar）
# ---------------------------------------------------------------------------


def _detect_portfolio_any(instance: dict) -> bool:
    return isinstance(instance.get("assets"), list) and isinstance(
        instance.get("scenario_returns"), list
    )


def _is_cvar(instance: dict) -> bool:
    risk = instance.get("risk")
    return isinstance(risk, dict) and str(risk.get("measure", "")).lower() == "cvar"


def _detect_portfolio(instance: dict) -> bool:
    return _detect_portfolio_any(instance) and not _is_cvar(instance)


def _detect_portfolio_cvar(instance: dict) -> bool:
    return _detect_portfolio_any(instance) and _is_cvar(instance)


def _parse_weights(solution: dict, assets: list[dict]) -> dict[int, float] | None:
    """weights を 銘柄 id → 重み に読む。dict でも銘柄数と同じ長さの list でもよい。"""
    raw = solution.get("weights")
    if isinstance(raw, list):
        if len(raw) != len(assets):
            return None
        raw = {asset["id"]: value for asset, value in zip(assets, raw)}
    if not isinstance(raw, dict):
        return None
    out: dict[int, float] = {}
    for key, value in raw.items():
        asset_id = _int_or_none(key)
        weight = _num(value)
        if asset_id is None or weight is None:
            return None
        out[asset_id] = weight
    return out


def _mad(portfolio: list[float], expected: float) -> float:
    return sum(abs(r - expected) for r in portfolio) / len(portfolio)


def _cvar(portfolio: list[float], expected: float, alpha: float) -> float:
    losses = sorted((expected - r for r in portfolio), reverse=True)
    tail = (1.0 - alpha) * len(losses)
    if tail <= 0:
        return losses[0]
    whole = math.floor(tail)
    total = sum(losses[:whole])
    if whole < len(losses):
        total += (tail - whole) * losses[whole]
    return total / tail


def _check_portfolio(instance: dict, solution: Any) -> dict:
    if not isinstance(solution, dict):
        return _unverified("solution is not a dict")
    assets = instance["assets"]
    weights = _parse_weights(solution, assets)
    if weights is None:
        return _unverified("weights missing or not id -> number")
    column = {int(asset["id"]): idx for idx, asset in enumerate(assets)}
    asset_by_id = {int(asset["id"]): asset for asset in assets}
    limits = instance.get("limits") if isinstance(instance.get("limits"), dict) else {}
    violations: list[str] = []
    checks = 0

    checks += 1
    unknown = sorted(a for a in weights if a not in column)
    if unknown:
        violations.append(f"weights contain unknown asset ids {unknown[:5]}")
    held = {a: w for a, w in weights.items() if a in column and w > _HOLD_EPS}

    checks += 1
    negative = sorted(a for a, w in weights.items() if w < -_PORT_TOL)
    if negative:
        violations.append(f"negative weights on assets {negative[:5]}")

    checks += 1
    total = sum(weights.values())
    if abs(total - 1.0) > _PORT_TOL:
        violations.append(f"weights sum to {total:.6g}, not 1")

    checks += 1
    min_hold, max_hold = limits.get("min_holdings"), limits.get("max_holdings")
    if min_hold is not None and len(held) < int(min_hold):
        violations.append(f"{len(held)} holdings below minimum {min_hold}")
    if max_hold is not None and len(held) > int(max_hold):
        violations.append(f"{len(held)} holdings above maximum {max_hold}")

    min_w, max_w = _num(limits.get("min_weight")), _num(limits.get("max_weight"))
    for asset_id, w in held.items():
        checks += 1
        if min_w is not None and w < min_w - _PORT_TOL:
            violations.append(f"asset {asset_id} weight {w:.6g} below min_weight {min_w:.6g}")
        if max_w is not None and w > max_w + _PORT_TOL:
            violations.append(f"asset {asset_id} weight {w:.6g} above max_weight {max_w:.6g}")

    sector_weight: dict[Any, float] = defaultdict(float)
    sector_count: dict[Any, int] = defaultdict(int)
    for asset_id, w in held.items():
        sector = asset_by_id[asset_id].get("sector")
        sector_weight[sector] += w
        sector_count[sector] += 1
    max_sector = _num(limits.get("max_sector_weight"))
    if max_sector is not None:
        for sector, w in sector_weight.items():
            checks += 1
            if w > max_sector + _PORT_TOL:
                violations.append(f"sector {sector} weight {w:.6g} above {max_sector:.6g}")
    min_sectors = limits.get("min_sectors")
    if min_sectors is not None:
        checks += 1
        if len(sector_count) < int(min_sectors):
            violations.append(f"{len(sector_count)} sectors held, below {min_sectors}")

    expected = sum(
        w * float(asset_by_id[a].get("expected_return", asset_by_id[a].get("mean_return", 0.0)))
        for a, w in held.items()
    )
    target = _num(instance.get("target_return"))
    if target is not None:
        checks += 1
        if expected < target - _PORT_TOL:
            violations.append(f"expected return {expected:.6g} below target {target:.6g}")

    current = instance.get("current_portfolio")
    current = current if isinstance(current, dict) else {}
    w0 = {int(k): float(v) for k, v in current.items()}
    max_turnover = _num(limits.get("max_turnover"))
    if max_turnover is not None:
        checks += 1
        turnover = sum(abs(weights.get(a, 0.0) - w0.get(a, 0.0)) for a in set(weights) | set(w0))
        if turnover > max_turnover + _PORT_TOL:
            violations.append(f"turnover {turnover:.6g} above {max_turnover:.6g}")

    if _is_cvar(instance):
        cardinality = instance.get("sector_cardinality")
        if isinstance(cardinality, dict):
            for sector_key, bounds in cardinality.items():
                checks += 1
                sector = _int_or_none(sector_key)
                count = sector_count.get(sector, 0)
                lo, hi = bounds
                if count < int(lo) or count > int(hi):
                    violations.append(f"sector {sector} holds {count}, outside [{lo}, {hi}]")
        for asset_id, w in w0.items():
            checks += 1
            if weights.get(asset_id, 0.0) < w - _PORT_TOL:
                violations.append(f"asset {asset_id} sold below current weight {w:.6g}")

    scenarios = instance["scenario_returns"]
    portfolio = [sum(w * row[column[a]] for a, w in held.items()) for row in scenarios]
    if _is_cvar(instance):
        alpha = _num(instance["risk"].get("alpha"))
        cost = _cvar(portfolio, expected, 0.9 if alpha is None else alpha)
        names = ("objective_value", "cvar")
    else:
        cost = _mad(portfolio, expected)
        names = ("objective_value", "mad_risk", "mad")
    checks += 1
    mismatch = _declared_mismatch(solution, cost, *names)
    if mismatch:
        violations.append(mismatch)
    return _result(violations, checks, cost=cost)


register_kind("mcnd", _detect_mcnd, _check_mcnd)
register_kind("mcnd_surv", _detect_mcnd_surv, _check_mcnd_surv)
register_kind("portfolio", _detect_portfolio, _check_portfolio)
register_kind("portfolio_cvar", _detect_portfolio_cvar, _check_portfolio)
