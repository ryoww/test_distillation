"""SFT データ作成の候補集めが、手書きの教師コードを雛形 id ごとに読むことを確認する。"""

from __future__ import annotations

from pathlib import Path

import scripts.build_sft_dataset as build


def test_handwritten_candidates_map_files_to_template_ids(tmp_path: Path):
    (tmp_path / "prob_007.py").write_text("def solve(instance):\n    return {}\n", encoding="utf-8")
    (tmp_path / "prob_053_alt.py").write_text("def solve(instance):\n    return {1: 2}\n")
    (tmp_path / "notes.txt").write_text("ignored")
    pools = build.handwritten_candidates(tmp_path)
    assert sorted(pools) == [7, 53]
    assert pools[7][0]["source"] == "handwritten"
    assert pools[7][0]["origin"] == "prob_007.py"
    assert pools[53][0]["code"].startswith("def solve")


def test_handwritten_source_is_last_in_selection_order():
    assert build.SOURCE_ORDER[-1] == "handwritten"
    assert build.SOURCE_ORDER[0] == "fable"
