"""大規模生成問題で GEPA を回すための追加オプションが、意図どおりに配線されていることを確認する。"""

from __future__ import annotations

import importlib
import json
from collections import Counter
from pathlib import Path

import train_gepa_v3 as train
from src.best_known import BestKnownRegistry
from src.data_loader import load_split_dirs
from src.metrics_v3 import evaluate_algorithm_v3

BASE_DIR = Path(__file__).resolve().parents[1]
HARD_GEN = BASE_DIR / "data" / "problems_hard_gen"
# src/__init__.py が同名の関数を公開しているので、モジュールは明示的に読み込む。
gepa_feedback_v3 = importlib.import_module("src.gepa_feedback_v3")


def _write(path: Path, rid: int, kind: str) -> None:
    path.write_text(
        json.dumps(
            {
                "id": rid,
                "domain": "d",
                "math_type": "m",
                "description": "x",
                "requirements": {"objective": "最小化", "constraints": []},
                "instance": {"n": 1},
                "reference_solution": {"objective_value": 1.0},
                "provenance": {"kind": kind},
            }
        )
    )


def test_split_dirs_take_the_first_n_per_kind_in_id_order(tmp_path: Path):
    for split in ("train", "validation", "test"):
        (tmp_path / split).mkdir()
    for rid, kind in [(3, "a"), (1, "a"), (2, "b"), (4, "b"), (5, "b")]:
        _write(tmp_path / "train" / f"prob_{rid}.json", rid, kind)
    _write(tmp_path / "validation" / "prob_9.json", 9, "a")
    _write(tmp_path / "test" / "prob_8.json", 8, "b")
    out = load_split_dirs(str(tmp_path), {"train": 1, "validation": 0, "test": 0})
    assert [e["instance_id"] for e in out["train"]] == ["prob_001", "prob_002"]
    assert len(out["validation"]) == 1 and len(out["test"]) == 1


def test_split_dirs_on_the_generated_hard_set_cover_every_kind_once():
    out = load_split_dirs(str(HARD_GEN), {"train": 1, "validation": 1, "test": 0})
    assert len(out["train"]) == 20 and len(out["validation"]) == 20 and len(out["test"]) == 120
    assert not {e["instance_id"] for e in out["train"]} & {e["instance_id"] for e in out["validation"]}


def test_exec_timeout_reaches_the_gepa_metric(monkeypatch):
    seen = {}

    def fake_eval(*args, **kwargs):
        seen.update(kwargs)
        return {"score": 0.0, "status": "exec_error", "detail": "x", "error_category": "runtime"}

    monkeypatch.setattr(gepa_feedback_v3, "evaluate_algorithm_v3", fake_eval)
    gepa_feedback_v3.set_exec_timeout(600)
    try:
        example = {"instance": {}, "core_type": "c", "instance_id": "prob_1", "reference_solution": {}}
        gepa_feedback_v3.gepa_feedback_v3(example, type("P", (), {"algorithm_code": "x"})())
    finally:
        gepa_feedback_v3.set_exec_timeout(60)
    assert seen["timeout"] == 600


def test_violation_messages_appear_in_the_scoring_detail():
    record = json.loads(next((HARD_GEN / "train").glob("prob_4321.json")).read_text())
    code = "def solve(instance):\n    return {'makespan': 1, 'schedule': []}\n"
    result = evaluate_algorithm_v3(
        code,
        record["instance"],
        f"{record['domain']}_{record['math_type']}",
        instance_id="prob_4321",
        registry=BestKnownRegistry(),
        reference_value=record["reference_solution"]["objective_value"],
        reference_solution=record["reference_solution"],
        timeout=30,
    )
    assert result["status"] in {"partial_feasible", "infeasible"}
    assert "not scheduled" in result["detail"] or "operation" in result["detail"]


def test_seed_instruction_and_reflection_template_reach_gepa(monkeypatch, tmp_path: Path):
    captured = {}

    class FakeGEPA:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def compile(self, program, trainset, valset):
            captured["instruction"] = program.generate.predict.signature.instructions
            return program

    monkeypatch.setattr(train.dspy, "GEPA", FakeGEPA)
    monkeypatch.setattr(train, "bootstrap_parse_codes_v3", lambda examples: {})
    example = load_split_dirs(str(HARD_GEN), {"train": 1, "validation": 0, "test": 0})["train"][:1]
    train.run_gepa_training(
        example,
        example,
        [],
        breadth=1,
        depth=1,
        output_dir=tmp_path,
        seed_instruction="SEED RULES",
        reflection_template="T <curr_param> <side_info>",
        log_dir=tmp_path / "gepa_logs" / "run_v3",
    )
    assert captured["instruction"] == "SEED RULES"
    assert captured["kwargs"]["instruction_proposer"].template.startswith("T ")
    assert "gepa_kwargs" not in captured["kwargs"]
    assert captured["kwargs"]["log_dir"] == str(tmp_path / "gepa_logs" / "run_v3")


def test_shipped_reflection_template_has_both_placeholders():
    text = (BASE_DIR / "prompts" / "reflection_template_general.md").read_text()
    assert Counter(p for p in ("<curr_param>", "<side_info>") if p in text) == Counter(
        {"<curr_param>": 1, "<side_info>": 1}
    )


def test_real_gepa_accepts_the_template_proposer():
    proposer = train.TemplateInstructionProposer("T <curr_param> <side_info>")
    dspy_gepa = train.dspy.GEPA(
        metric=lambda *a, **k: 0.0, reflection_lm=None, instruction_proposer=proposer, max_full_evals=1
    )
    assert dspy_gepa.custom_instruction_proposer is proposer


def test_template_proposer_fills_template_and_extracts_instruction(monkeypatch):
    prompts = []

    def fake_lm(prompt):
        prompts.append(prompt)
        return ["```\nNEW RULES\n```"]

    monkeypatch.setattr(train.dspy.settings, "lm", fake_lm, raising=False)
    proposer = train.TemplateInstructionProposer("CUR=<curr_param> INFO=<side_info>")
    out = proposer({"generate": "OLD"}, {"generate": [{"Feedback": "bad"}]}, ["generate"])
    assert out == {"generate": "NEW RULES"}
    assert prompts[0].startswith("CUR=OLD INFO=") and "bad" in prompts[0]
