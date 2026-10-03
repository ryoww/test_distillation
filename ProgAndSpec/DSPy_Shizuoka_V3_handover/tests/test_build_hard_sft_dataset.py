"""大規模生成問題の SFT 対を作る経路が、出典の指定・対象外の除外・学習入力の形を守ることを確認する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import scripts.build_hard_sft_dataset as build


def test_labelled_specs_require_label_and_path():
    assert build._labelled(["opus=/tmp/a"]) == [("opus", Path("/tmp/a"))]
    with pytest.raises(SystemExit):
        build._labelled(["no_label_here"])


def test_answers_outside_the_known_splits_are_ignored(tmp_path: Path):
    answers = tmp_path / "answers"
    answers.mkdir()
    (answers / "prob_4001.py").write_text("def solve(instance):\n    return {}\n")
    (answers / "prob_9999.py").write_text("def solve(instance):\n    return {}\n")
    runs = tmp_path / "run" / "glm__shard01of01"
    runs.mkdir(parents=True)
    (runs / build.RESULT_FILENAME).write_text(
        json.dumps(
            {
                "test": {
                    "results": [
                        {"instance_id": "prob_4001", "code": "x"},
                        {"instance_id": "prob_4999"},
                    ]
                }
            }
        )
    )
    args = argparse.Namespace(answers=[f"opus={answers}"], runs=[f"glm={tmp_path / 'run'}"])
    found = build.collect_answers(args, {"prob_4001": {}})
    assert sorted((a["source"], a["instance_id"]) for a in found) == [
        ("glm", "prob_4001"),
        ("opus", "prob_4001"),
    ]


def test_training_input_hides_the_reference_value():
    record = json.loads(next((build.GEN_DIR / "train").glob("prob_*.json")).read_text())
    row = build.to_messages(
        record, "def solve(instance):\n    return {}", "SYSTEM", {"source": "t"}
    )
    user = row["messages"][1]["content"]
    assert "Reference Value" not in user
    assert row["messages"][2]["content"].startswith("```python\n")
    assert row["kind"] == record["provenance"]["kind"]
