"""BestKnownRegistry: 各インスタンスの既知最良コストを管理。

V1から移植。JSONL に永続化し、スレッドセーフに更新。
"""
from __future__ import annotations

import json
import threading
from pathlib import Path


class BestKnownRegistry:
    """各 instance_id に対する最良コストを保持・更新するレジストリ。"""

    def __init__(self, storage_path: str | Path | None = None) -> None:
        self._data: dict[str, float] = {}
        # Self-baseline: the first feasible cost observed per instance.
        # Used by the reference-free scoring mode to normalize improvement
        # (RL-style advantage) without relying on the external reference value.
        self._baseline: dict[str, float] = {}
        self._lock = threading.Lock()
        self._storage_path = Path(storage_path) if storage_path else None

    def get(self, instance_id: str) -> float | None:
        """instance_id の既知最良コストを返す (None = 未観測)。"""
        with self._lock:
            return self._data.get(instance_id)

    def get_baseline(self, instance_id: str) -> float | None:
        """instance_id のセルフベースライン (初回実行可能コスト) を返す。"""
        with self._lock:
            return self._baseline.get(instance_id)

    def set_baseline_if_absent(self, instance_id: str, cost: float) -> bool:
        """未登録なら初回実行可能コストをベースラインとして記録。"""
        with self._lock:
            if instance_id not in self._baseline:
                self._baseline[instance_id] = cost
                return True
            return False

    def update_if_better(self, instance_id: str, cost: float) -> bool:
        """cost が既知最良より小さければ更新し True を返す。"""
        with self._lock:
            current = self._data.get(instance_id)
            if current is None or cost < current:
                self._data[instance_id] = cost
                return True
            return False

    def register(self, instance_id: str, cost: float) -> None:
        """無条件で登録 (初回シード用)。"""
        with self._lock:
            self._data[instance_id] = cost

    def snapshot(self) -> dict[str, float]:
        """現在の状態のスナップショット (コピー) を返す。"""
        with self._lock:
            return dict(self._data)

    def load_from_disk(self) -> int:
        """JSONL から読み込み、ロードした行数を返す。"""
        if self._storage_path is None or not self._storage_path.exists():
            return 0
        count = 0
        with open(self._storage_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                iid = record.get("instance_id")
                cost = record.get("best_cost")
                if iid is not None and cost is not None:
                    with self._lock:
                        current = self._data.get(iid)
                        if current is None or cost < current:
                            self._data[iid] = cost
                    count += 1
        return count

    def save_to_disk(self) -> None:
        """現在の状態を JSONL に上書き保存。"""
        if self._storage_path is None:
            return
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._storage_path, "w", encoding="utf-8") as f:
            with self._lock:
                for iid, cost in sorted(self._data.items()):
                    f.write(json.dumps({"instance_id": iid, "best_cost": cost}) + "\n")

    def __repr__(self) -> str:
        return f"BestKnownRegistry(entries={len(self._data)})"


# グローバルデフォルトインスタンス (スクリプトから直接アクセス用)
registry: BestKnownRegistry | None = None


def init_registry(storage_path: str | Path | None = None) -> BestKnownRegistry:
    """グローバルレジストリを初期化・返す。"""
    global registry
    registry = BestKnownRegistry(storage_path=storage_path)
    return registry
