"""ローカル JSONL からの学習データ読み込みを検証する。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_train_module():
    # scripts/ はパッケージではないので、ファイルから直接読み込む。torch 等の重い import は
    # このテストの範囲外なので、無い環境では skip する。
    pytest.importorskip("datasets")
    spec = importlib.util.spec_from_file_location(
        "train_distillation", ROOT / "scripts" / "train_distillation.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:  # torch/peft/transformers が無い環境
        pytest.skip(f"training dependencies unavailable: {exc}")
    return module


def test_local_jsonl_dataset_is_loaded_with_train_and_validation(tmp_path):
    module = _load_train_module()
    row = {
        "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        "tools": None,
    }
    for split in ("train", "validation"):
        (tmp_path / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    args = argparse.Namespace(
        dataset_dir=tmp_path, dataset=None, dataset_config=None, dataset_revision=None
    )
    dataset = module.load_training_dataset(args)
    assert set(dataset) == {"train", "validation"}
    assert dataset["train"][0]["messages"][1]["content"] == "a"


def test_missing_local_split_is_reported(tmp_path):
    module = _load_train_module()
    (tmp_path / "train.jsonl").write_text("{}\n", encoding="utf-8")
    args = argparse.Namespace(
        dataset_dir=tmp_path, dataset=None, dataset_config=None, dataset_revision=None
    )
    with pytest.raises(FileNotFoundError):
        module.load_training_dataset(args)
