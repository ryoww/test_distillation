"""指示文の変種（compact / modular）と分野別補足の選択を検証する。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.compare_prompt_models as compare
from scripts.build_prompt_variants import build_variant, phase_e_demos
from src.modules import AlgorithmGenerator, load_supplements_for, supplements_path_for

BASE_DIR = Path(__file__).resolve().parents[1]
PHASE_E = BASE_DIR / "compiled_program_v3_gepa_phaseE.json"


def test_supplement_prefers_core_type_over_domain():
    generator = AlgorithmGenerator(
        supplements={"スケジューリング": "domain note", "スケジューリング_整数計画": "core note"}
    )
    assert generator.supplement_for("スケジューリング_整数計画") == "core note"
    assert generator.supplement_for("スケジューリング_混合整数計画") == "domain note"
    assert generator.supplement_for("金融・投資_線形計画") == ""


def test_sidecar_supplements_load_next_to_the_program(tmp_path):
    program = tmp_path / "variant.json"
    program.write_text("{}", encoding="utf-8")
    assert load_supplements_for(program) == {}
    supplements_path_for(program).write_text(
        json.dumps({"配送・輸送": "routes start at the depot"}), encoding="utf-8"
    )
    assert load_supplements_for(program) == {"配送・輸送": "routes start at the depot"}


def test_sidecar_rejects_non_text_values(tmp_path):
    program = tmp_path / "variant.json"
    supplements_path_for(program).write_text(json.dumps({"x": 1}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_supplements_for(program)


def test_forward_appends_supplement_to_requirement(monkeypatch):
    generator = AlgorithmGenerator(supplements={"配送・輸送": "ROUTE NOTE"})
    seen = {}

    def fake_generate(**kwargs):
        seen.update(kwargs)

        class Out:
            algorithm_code = "def solve(instance):\n    return {}"
            rationale = ""

        return Out()

    monkeypatch.setattr(generator, "generate", fake_generate)
    generator.forward(requirement="## Problem\nbase text", core_type="配送・輸送_混合整数計画")
    assert seen["requirement"].startswith("## Problem\nbase text")
    assert "## Approach notes for this problem class\nROUTE NOTE" in seen["requirement"]
    generator.forward(requirement="plain", core_type="金融・投資_線形計画")
    assert seen["requirement"] == "plain"


def test_build_variant_keeps_phase_e_demos_and_new_instructions(tmp_path):
    demos = phase_e_demos(PHASE_E)
    output = build_variant("SHORT RULES", demos, tmp_path / "compact.json")
    loaded = AlgorithmGenerator()
    loaded.load(str(output))
    assert loaded.generate.predict.signature.instructions == "SHORT RULES"
    assert len(loaded.generate.predict.demos) == len(demos) == 2
    assert (
        loaded.improve.predict.signature.instructions
        == AlgorithmGenerator().improve.predict.signature.instructions
    )


def test_shipped_variants_are_shorter_than_phase_e_and_carry_all_domains():
    phase_e = json.loads(PHASE_E.read_text(encoding="utf-8"))["generate.predict"]["signature"][
        "instructions"
    ]
    for name in ("compact", "modular"):
        path = BASE_DIR / "prompts" / f"compiled_program_v3_{name}.json"
        loaded = AlgorithmGenerator()
        loaded.load(str(path))
        assert len(loaded.generate.predict.signature.instructions) < len(phase_e) / 5
    supplements = load_supplements_for(BASE_DIR / "prompts" / "compiled_program_v3_modular.json")
    domains = {
        json.loads(p.read_text(encoding="utf-8"))["domain"]
        for p in (BASE_DIR / "data" / "problems").glob("prob_*.json")
    }
    assert set(supplements) == domains
    assert load_supplements_for(BASE_DIR / "prompts" / "compiled_program_v3_compact.json") == {}


def test_before_demos_variant_keeps_the_original_instructions():
    loaded = AlgorithmGenerator()
    loaded.load(str(BASE_DIR / "prompts" / "compiled_program_v3_before_demos.json"))
    original = (BASE_DIR / "prompts" / "original_generate_instructions.md").read_text(
        encoding="utf-8"
    )
    assert loaded.generate.predict.signature.instructions == original.strip()
    assert len(loaded.generate.predict.demos) == 2


def test_default_generate_instruction_is_the_compact_one():
    """What: the uncompiled AlgorithmGenerator now starts from the 1.8KB compact instruction."""
    default = AlgorithmGenerator().generate.predict.signature.instructions
    compact = AlgorithmGenerator()
    compact.load(str(BASE_DIR / "prompts" / "compiled_program_v3_compact.json"))
    assert default == compact.generate.predict.signature.instructions
    assert len(default) < 2500
    assert "Required Return Schema" in default


def test_extra_program_adds_a_prompt_condition_for_dry_run(capsys, tmp_path):
    program = tmp_path / "x.json"
    program.write_text("{}", encoding="utf-8")
    code = compare.main(
        [
            "--dry-run",
            "--only-model",
            "qwen3_6_27b",
            "--only-prompt",
            "compact",
            "--extra-program",
            f"compact={program}",
            "--shards",
            "2",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    labels = [line.split("]")[0].strip("[") for line in out.splitlines() if line.startswith("[")]
    assert labels == ["compact__qwen3_6_27b__shard01of02", "compact__qwen3_6_27b__shard02of02"]
    assert str(program) in out


def test_extra_program_rejects_reserved_labels():
    with pytest.raises(ValueError):
        compare.parse_extra_programs(["after=/tmp/x.json"])
    with pytest.raises(ValueError):
        compare.parse_extra_programs(["nolabel"])


def test_ensure_parse_helpers_prepends_only_when_called_but_undefined():
    from src.modules import GENERIC_PARSE_CODE, ensure_parse_helpers

    uses = "def solve(instance):\n    return {'n': len(get_list(instance, 'jobs'))}\n"
    assert ensure_parse_helpers(uses).startswith(GENERIC_PARSE_CODE)
    defines = "def get_list(d, k, default=None):\n    return d.get(k, [])\n" + uses
    assert ensure_parse_helpers(defines) == defines
    plain = "def solve(instance):\n    return {'n': len(instance.get('jobs', []))}\n"
    assert ensure_parse_helpers(plain) == plain


def test_ensure_parse_helpers_ignores_strings_and_respects_other_bindings():
    from src.modules import GENERIC_PARSE_CODE, ensure_parse_helpers

    in_string = "def solve(instance):\n    return {'note': 'call get_list(x) here'}\n"
    assert ensure_parse_helpers(in_string) == in_string
    via_lambda = "get_list = lambda d, k, default=None: d.get(k, [])\ndef solve(instance):\n    return {'n': len(get_list(instance, 'jobs'))}\n"
    assert ensure_parse_helpers(via_lambda) == via_lambda
    via_import = "from helpers import get_list\ndef solve(instance):\n    return get_list(instance, 'jobs')\n"
    assert ensure_parse_helpers(via_import) == via_import
    broken = "def solve(instance:\n    return get_list(instance, 'jobs')\n"
    assert ensure_parse_helpers(broken) == broken
    uses = "def solve(instance):\n    return {'n': len(get_scalar(instance, 'n'))}\n"
    assert ensure_parse_helpers(uses).startswith(GENERIC_PARSE_CODE)
