"""GEPA の train/val 分割が core_type で層化されることを検証する。"""

from __future__ import annotations

import collections
from pathlib import Path

from src.data_loader import (
    load_and_split_stratified,
    load_v3_data,
    split_train_val_stratified,
)

BASE_DIR = Path(__file__).resolve().parents[1]
PROBLEM_DIR = BASE_DIR / "data" / "problems"
N_VAL = 13


def _train40() -> list[dict]:
    train, _ = load_and_split_stratified(
        str(PROBLEM_DIR), n_train=40, n_test=20, seed=42, use_reference=True
    )
    return train


def _domains() -> dict[str, str]:
    return {
        f"prob_{record['id']:03d}": record["domain"] for record in load_v3_data(str(PROBLEM_DIR))
    }


def test_split_sizes_and_disjointness():
    train_only, val = split_train_val_stratified(_train40(), N_VAL, seed=42)
    assert len(val) == N_VAL
    assert len(train_only) == 40 - N_VAL
    train_ids = {e["instance_id"] for e in train_only}
    val_ids = {e["instance_id"] for e in val}
    assert train_ids & val_ids == set()
    assert len(train_ids | val_ids) == 40


def test_split_is_deterministic_for_a_seed():
    first = split_train_val_stratified(_train40(), N_VAL, seed=42)
    second = split_train_val_stratified(_train40(), N_VAL, seed=42)
    assert [e["instance_id"] for e in first[1]] == [e["instance_id"] for e in second[1]]


def test_validation_split_covers_many_domains():
    """末尾スライスでは val が3 domain に偏り、train とほとんど重ならなかった。"""
    train_only, val = split_train_val_stratified(_train40(), N_VAL, seed=42)
    domains = _domains()
    val_domains = {domains[e["instance_id"]] for e in val}
    train_domains = {domains[e["instance_id"]] for e in train_only}
    assert len(val_domains) >= 6
    # 候補選択に使う集合が、反省に使う集合とほぼ重ならない状態を防ぐ。
    assert len(val_domains & train_domains) >= 5


def test_split_beats_the_historical_tail_slice_on_diversity():
    train40 = _train40()
    domains = _domains()
    tail_val = train40[-N_VAL:]
    _, val = split_train_val_stratified(train40, N_VAL, seed=42)
    tail_domains = {domains[e["instance_id"]] for e in tail_val}
    new_domains = {domains[e["instance_id"]] for e in val}
    assert len(new_domains) > len(tail_domains)


def test_split_keeps_no_domain_overwhelmingly_dominant():
    _, val = split_train_val_stratified(_train40(), N_VAL, seed=42)
    domains = _domains()
    counts = collections.Counter(domains[e["instance_id"]] for e in val)
    assert counts.most_common(1)[0][1] <= max(3, N_VAL // 3)


def test_both_halves_are_sorted_by_instance_id():
    train_only, val = split_train_val_stratified(_train40(), N_VAL, seed=42)
    for half in (train_only, val):
        ids = [e["instance_id"] for e in half]
        assert ids == sorted(ids)


def test_degenerate_val_sizes_return_everything_as_train():
    train40 = _train40()
    for n_val in (0, -1, len(train40), len(train40) + 5):
        train_only, val = split_train_val_stratified(train40, n_val, seed=42)
        assert val == []
        assert len(train_only) == len(train40)
