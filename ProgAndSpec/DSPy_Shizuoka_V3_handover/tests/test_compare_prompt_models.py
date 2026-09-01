"""Contract tests for the four-cell prompt and model comparison wrapper."""

from __future__ import annotations

import json
import subprocess
from dataclasses import is_dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.compare_prompt_models as compare

TARGET_ROOT = Path(__file__).parents[1]
DATA_DIR = TARGET_ROOT / "data" / "problems"


def _option(command: list[str], name: str) -> str:
    """What: retrieve one option value from a child command."""
    return command[command.index(name) + 1]


def _write_eval_result(command: list[str], score: float) -> None:
    """What: create a minimal 100-row result emitted by train_gepa_v3.py."""
    output_dir = Path(_option(command, "--output-dir"))
    run_name = _option(command, "--run-name")
    result_path = output_dir / run_name / "evaluation_results_v3_gepa_phaseE.json"
    rows = [
        {
            "instance_id": f"prob_{index:03d}",
            "score": score,
            "status": "exact_match" if score > 0 else "invalid_solution",
            "beat_reference_analysis": score >= 0.8,
            "beat_reference_strict": score > 0.8,
        }
        for index in range(1, 101)
    ]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "test": {
                    "results": rows,
                    "mean_score": score,
                    "valid_count": 100 if score > 0 else 0,
                    "total_count": 100,
                    "beat_reference": 100 if score >= 0.8 else 0,
                    "beat_reference_analysis": 100 if score >= 0.8 else 0,
                    "beat_reference_strict": 100 if score > 0.8 else 0,
                }
            }
        ),
        encoding="utf-8",
    )


def _write_shard_eval_result(command: list[str], score: float) -> list[str]:
    """What: create a child result containing exactly the shard's problem IDs."""
    output_dir = Path(_option(command, "--output-dir"))
    run_name = _option(command, "--run-name")
    data_dir = Path(_option(command, "--data-dir"))
    instance_ids = sorted(path.stem for path in data_dir.glob("prob_*.json"))
    result_path = output_dir / run_name / "evaluation_results_v3_gepa_phaseE.json"
    rows = [
        {
            "instance_id": instance_id,
            "score": score,
            "status": "exact_match",
            "beat_reference_analysis": True,
            "beat_reference_strict": False,
        }
        for instance_id in instance_ids
    ]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps({"test": {"results": rows}}), encoding="utf-8")
    return instance_ids


def _run_partial_model(
    tmp_path: Path,
    run_name: str,
    model_label: str,
    scores: dict[str, float],
    monkeypatch,
) -> Path:
    """What: execute one model slice with a fake evaluator and return its artifact directory."""

    monkeypatch.setattr(
        compare,
        "create_baseline_program",
        lambda path: path.write_text("{}", encoding="utf-8"),
    )

    def fake_runner(command, *args, **kwargs):
        del args, kwargs
        prompt = "before" if "before" in _option(command, "--run-name") else "after"
        _write_shard_eval_result(command, scores[prompt])
        return subprocess.CompletedProcess(command, 0)

    exit_code = compare.main(
        [
            "--only-model",
            model_label,
            "--shards",
            "2",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            run_name,
            "--skip-connection-check",
        ],
        runner=fake_runner,
    )
    assert exit_code == 0
    return tmp_path / run_name


def test_condition_is_frozen_dataclass_with_required_public_fields():
    """What: one evaluation cell stores prompt, model, and program identity."""
    target = SimpleNamespace(label="qwen3_6", model="Qwen/Qwen3.6-27B")
    condition = compare.Condition("baseline_qwen36", "baseline", target, Path("baseline.json"))

    assert is_dataclass(condition)
    assert condition.label == "baseline_qwen36"
    assert condition.prompt_label == "baseline"
    assert condition.model_target is target
    assert condition.program_path == Path("baseline.json")


def test_split_instance_ids_reproduces_legacy_and_excludes_untouched_questions():
    """What: the 100 problems are partitioned into historical 40/20 and untouched 40 sets."""
    splits = compare.split_instance_ids(DATA_DIR)

    assert set(splits) == {"train40", "legacy_test20", "untouched40", "all100"}
    assert [
        len(splits[name]) for name in ("train40", "legacy_test20", "untouched40", "all100")
    ] == [
        40,
        20,
        40,
        100,
    ]
    assert splits["all100"] == sorted(splits["all100"])
    assert set(splits["train40"]).isdisjoint(splits["legacy_test20"])
    assert set(splits["train40"]).isdisjoint(splits["untouched40"])
    assert set(splits["legacy_test20"]).isdisjoint(splits["untouched40"])
    assert set(splits["train40"]) | set(splits["legacy_test20"]) | set(
        splits["untouched40"]
    ) == set(splits["all100"])
    assert set(splits["all100"]) == {f"prob_{index:03d}" for index in range(1, 101)}

    # These are the IDs used by the existing n_train=40/n_test=20 seed-42 run.
    assert splits["train40"] == [
        "prob_002",
        "prob_004",
        "prob_006",
        "prob_009",
        "prob_016",
        "prob_019",
        "prob_026",
        "prob_027",
        "prob_029",
        "prob_032",
        "prob_033",
        "prob_034",
        "prob_035",
        "prob_036",
        "prob_039",
        "prob_045",
        "prob_048",
        "prob_052",
        "prob_054",
        "prob_055",
        "prob_064",
        "prob_066",
        "prob_068",
        "prob_070",
        "prob_071",
        "prob_073",
        "prob_077",
        "prob_081",
        "prob_082",
        "prob_083",
        "prob_084",
        "prob_085",
        "prob_088",
        "prob_089",
        "prob_091",
        "prob_092",
        "prob_094",
        "prob_095",
        "prob_097",
        "prob_100",
    ]
    assert splits["legacy_test20"] == [
        "prob_001",
        "prob_007",
        "prob_010",
        "prob_017",
        "prob_018",
        "prob_038",
        "prob_041",
        "prob_042",
        "prob_049",
        "prob_051",
        "prob_058",
        "prob_062",
        "prob_067",
        "prob_069",
        "prob_072",
        "prob_078",
        "prob_079",
        "prob_080",
        "prob_087",
        "prob_090",
    ]


def test_create_baseline_program_writes_loadable_uncompiled_program(tmp_path):
    """What: the baseline cell is materialized as a DSPy program artifact."""
    pytest.importorskip("dspy")
    path = tmp_path / "nested" / "baseline_program_v3.json"

    compare.create_baseline_program(path)

    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert "generate.predict" in payload


def test_aggregate_results_reports_score_validity_and_status_breakdown():
    """What: raw per-instance rows become one comparable subset summary."""
    rows = [
        {
            "instance_id": "prob_001",
            "score": 1.0,
            "status": "exact_match",
            "cost": 5,
            "reference_value": 10,
            "beat_reference_analysis": True,
            "beat_reference_strict": True,
        },
        {
            "instance_id": "prob_002",
            "score": 0.5,
            "status": "beat_reference",
            "cost": 10,
            "reference_value": 10,
            "beat_reference_analysis": True,
            "beat_reference_strict": False,
        },
        {
            "instance_id": "prob_003",
            "score": 0.0,
            "status": "exec_error",
            "beat_reference_analysis": False,
            "beat_reference_strict": False,
        },
        {
            "instance_id": "prob_004",
            "score": -0.5,
            "status": "invalid_solution",
            "beat_reference_analysis": False,
            "beat_reference_strict": False,
        },
    ]

    summary = compare.aggregate_results(rows)

    assert summary["mean_score"] == pytest.approx(0.25)
    assert summary["valid_count"] == 2
    assert summary["total_count"] == 4
    assert summary["valid_rate"] == pytest.approx(0.5)
    assert summary["beat_reference"] == 2
    assert summary["beat_reference_analysis"] == 2
    assert summary["beat_reference_strict"] == 1
    assert summary["status_counts"] == {
        "exact_match": 1,
        "beat_reference": 1,
        "exec_error": 1,
        "invalid_solution": 1,
    }
    assert summary["instance_ids"] == [
        "prob_001",
        "prob_002",
        "prob_003",
        "prob_004",
    ]


def test_compute_effects_returns_prompt_model_and_interaction_effects():
    """What: a 2x2 score table yields main effects and difference-in-differences."""
    summaries = [
        {
            "label": "baseline_qwen36",
            "prompt_label": "before",
            "model_label": "qwen3_6_27b",
            "subsets": {"all100": {"mean_score": 0.4}},
        },
        {
            "label": "baseline_qwen38",
            "prompt_label": "before",
            "model_label": "qwen3_8_27b",
            "subsets": {"all100": {"mean_score": 0.6}},
        },
        {
            "label": "improved_qwen36",
            "prompt_label": "after",
            "model_label": "qwen3_6_27b",
            "subsets": {"all100": {"mean_score": 0.8}},
        },
        {
            "label": "improved_qwen38",
            "prompt_label": "after",
            "model_label": "qwen3_8_27b",
            "subsets": {"all100": {"mean_score": 1.2}},
        },
    ]

    effects = compare.compute_effects(summaries, "all100")

    assert effects["prompt_effect_by_model"] == {
        "qwen3_6_27b": pytest.approx(0.4),
        "qwen3_8_27b": pytest.approx(0.6),
    }
    assert effects["model_effect_by_prompt"] == {
        "before": pytest.approx(0.2),
        "after": pytest.approx(0.4),
    }
    assert effects["interaction"] == pytest.approx(0.2)


def test_parse_args_defaults_to_100_cases_and_supports_endpoint_overrides():
    """What: the wrapper defaults to the full benchmark and exposes both model overrides."""
    args = compare.parse_args(
        [
            "--qwen36-api-base",
            "http://localhost:8501/v1",
            "--qwen38-api-base",
            "http://localhost:8502/v1",
            "--qwen36-revision",
            "rev36",
            "--qwen38-revision",
            "rev38",
            "--max-tokens",
            "4096",
            "--lm-timeout",
            "900",
            "--parallel",
            "--dry-run",
        ]
    )

    assert args.temperature == pytest.approx(0.0)
    assert args.max_tokens == 4096
    assert args.lm_timeout == 900
    assert args.qwen36_api_base == "http://localhost:8501/v1"
    assert args.qwen38_api_base == "http://localhost:8502/v1"
    assert args.qwen36_revision == "rev36"
    assert args.qwen38_revision == "rev38"
    assert args.parallel is True
    assert args.dry_run is True


def test_main_dry_run_prints_four_commands_without_creating_output(tmp_path, capsys, monkeypatch):
    """What: dry-run shows all four cells and performs no baseline or subprocess write."""
    monkeypatch.setattr(
        compare,
        "create_baseline_program",
        lambda _path: pytest.fail("dry-run must not create baseline program"),
    )

    def fail_runner(*_args, **_kwargs):
        raise AssertionError("dry-run must not run child evaluators")

    exit_code = compare.main(
        ["--dry-run", "--output-dir", str(tmp_path), "--run-name", "dry-run"],
        runner=fail_runner,
    )

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert captured.count("--eval-only") == 4
    assert captured.count("Qwen/Qwen3.6-27B") == 4
    assert captured.count("Qwen/Qwen3.8-27B") == 4
    assert "before" in captured
    assert "after" in captured
    assert not list(tmp_path.iterdir())


def test_main_runs_all_four_conditions_and_writes_factorial_comparison(tmp_path, monkeypatch):
    """What: four child evaluations produce subset scores and factorial effects."""
    calls: list[list[str]] = []
    baseline_path: Path | None = None
    scores = {
        ("before", "Qwen/Qwen3.6-27B"): 0.4,
        ("before", "Qwen/Qwen3.8-27B"): 0.6,
        ("after", "Qwen/Qwen3.6-27B"): 0.8,
        ("after", "Qwen/Qwen3.8-27B"): 1.2,
    }

    def fake_baseline(path: Path) -> None:
        nonlocal baseline_path
        baseline_path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(compare, "create_baseline_program", fake_baseline)

    def fake_runner(command, *args, **kwargs):
        del args, kwargs
        calls.append(command)
        prompt = "before" if "before" in _option(command, "--run-name") else "after"
        model = _option(command, "--generation-model")
        _write_eval_result(command, scores[(prompt, model)])
        return subprocess.CompletedProcess(command, 0)

    exit_code = compare.main(
        ["--output-dir", str(tmp_path), "--run-name", "four-cell", "--skip-connection-check"],
        runner=fake_runner,
    )

    assert exit_code == 0
    assert len(calls) == 4
    assert baseline_path == tmp_path / "four-cell" / "baseline_program_v3.json"
    assert all("--eval-only" in command for command in calls)
    assert all(_option(command, "--n-train") == "0" for command in calls)
    assert all(_option(command, "--n-test") == "100" for command in calls)
    assert all("--skip-connection-check" in command for command in calls)
    assert {_option(command, "--generation-model") for command in calls} == {
        "Qwen/Qwen3.6-27B",
        "Qwen/Qwen3.8-27B",
    }
    assert {"before" in _option(command, "--run-name") for command in calls} == {True, False}

    comparison_path = tmp_path / "four-cell" / "factorial_comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert {condition["prompt_label"] for condition in comparison["conditions"]} == {
        "before",
        "after",
    }
    assert {condition["model_label"] for condition in comparison["conditions"]} == {
        "qwen3_6_27b",
        "qwen3_8_27b",
    }
    assert set(comparison["effects"]) >= {
        "legacy_test20",
        "train40",
        "untouched40",
        "all100",
    }
    all_effects = comparison["effects"]["all100"]
    assert all_effects["prompt_effect_by_model"] == {
        "qwen3_6_27b": pytest.approx(0.4),
        "qwen3_8_27b": pytest.approx(0.6),
    }
    assert all_effects["model_effect_by_prompt"] == {
        "before": pytest.approx(0.2),
        "after": pytest.approx(0.4),
    }
    assert all_effects["interaction"] == pytest.approx(0.2)


def test_main_writes_comparison_json_and_returns_one_when_runner_raises(tmp_path, monkeypatch):
    """What: an evaluator exception leaves an auditable failed comparison artifact."""
    monkeypatch.setattr(
        compare,
        "create_baseline_program",
        lambda path: path.write_text("{}", encoding="utf-8"),
    )

    def raising_runner(*_args, **_kwargs):
        raise RuntimeError("endpoint unavailable")

    exit_code = compare.main(
        ["--output-dir", str(tmp_path), "--run-name", "runner-failure"],
        runner=raising_runner,
    )

    comparison_path = tmp_path / "runner-failure" / "factorial_comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert comparison["benchmark"]["comparable"] is False
    assert comparison["failures"]
    assert "endpoint unavailable" in json.dumps(comparison, ensure_ascii=False)


def test_main_writes_comparison_json_and_returns_one_for_malformed_result(tmp_path, monkeypatch):
    """What: malformed child output is recorded as a failed, non-comparable run."""
    monkeypatch.setattr(
        compare,
        "create_baseline_program",
        lambda path: path.write_text("{}", encoding="utf-8"),
    )

    def malformed_runner(command, *args, **kwargs):
        del args, kwargs
        output_dir = Path(_option(command, "--output-dir"))
        run_name = _option(command, "--run-name")
        result_path = output_dir / run_name / "evaluation_results_v3_gepa_phaseE.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text("not-json", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    exit_code = compare.main(
        ["--output-dir", str(tmp_path), "--run-name", "malformed"],
        runner=malformed_runner,
    )

    comparison_path = tmp_path / "malformed" / "factorial_comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert comparison["benchmark"]["comparable"] is False
    assert comparison["failures"]
    assert "JSON" in json.dumps(comparison, ensure_ascii=False)


def test_shards_two_dry_run_emits_eight_half_corpus_commands_without_writes(
    tmp_path, capsys, monkeypatch
):
    """What: two shards expand the four-cell design into eight 50-problem jobs."""
    monkeypatch.setattr(
        compare,
        "create_baseline_program",
        lambda _path: pytest.fail("dry-run must not create baseline program"),
    )

    exit_code = compare.main(
        [
            "--shards",
            "2",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "two-shards",
        ]
    )

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert captured.count("--eval-only") == 8
    assert captured.count("--n-test 50") == 8
    assert captured.count("--data-dir") == 8
    assert not list(tmp_path.iterdir())


def test_shards_two_merges_each_real_shard_into_four_complete_conditions(tmp_path, monkeypatch):
    """What: shard-local result IDs merge to 100 rows per prompt/model condition."""
    calls: list[list[str]] = []
    shard_ids_seen: dict[str, list[str]] = {}
    scores = {
        ("before", "Qwen/Qwen3.6-27B"): 0.4,
        ("before", "Qwen/Qwen3.8-27B"): 0.6,
        ("after", "Qwen/Qwen3.6-27B"): 0.8,
        ("after", "Qwen/Qwen3.8-27B"): 1.2,
    }

    def fake_baseline(path: Path) -> None:
        path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(compare, "create_baseline_program", fake_baseline)

    def fake_runner(command, *args, **kwargs):
        del args, kwargs
        calls.append(command)
        run_name = _option(command, "--run-name")
        prompt = "before" if "before" in run_name else "after"
        model = _option(command, "--generation-model")
        shard_ids_seen[run_name] = _write_shard_eval_result(command, scores[(prompt, model)])
        return subprocess.CompletedProcess(command, 0)

    exit_code = compare.main(
        [
            "--shards",
            "2",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "two-shards",
            "--skip-connection-check",
        ],
        runner=fake_runner,
    )

    assert exit_code == 0
    assert len(calls) == 8
    assert all(_option(command, "--n-test") == "50" for command in calls)
    assert all(len(shard_ids) == 50 for shard_ids in shard_ids_seen.values())

    run_root = tmp_path / "two-shards"
    shard_dirs = sorted((run_root / "shards").glob("shard_*"))
    assert len(shard_dirs) == 2
    ids_by_shard = [
        {path.stem for path in shard_dir.glob("prob_*.json")} for shard_dir in shard_dirs
    ]
    assert ids_by_shard[0].isdisjoint(ids_by_shard[1])
    assert len(ids_by_shard[0] | ids_by_shard[1]) == 100
    assert ids_by_shard[0] | ids_by_shard[1] == {f"prob_{index:03d}" for index in range(1, 101)}

    comparison_path = run_root / "factorial_comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert comparison["benchmark"]["shards_per_condition"] == 2
    assert comparison["benchmark"]["comparable"] is True
    assert len(comparison["conditions"]) == 4
    assert all(
        condition["subsets"]["all100"]["total_count"] == 100
        for condition in comparison["conditions"]
    )
    assert set(comparison["effects"]) >= {"legacy_test20", "train40", "untouched40", "all100"}
    assert comparison["failures"] == []


def test_only_model_qwen36_with_two_shards_dry_run_emits_four_qwen36_commands(
    tmp_path, capsys, monkeypatch
):
    """What: selecting Qwen3.6 keeps both prompts and both 50-problem shards."""
    monkeypatch.setattr(
        compare,
        "create_baseline_program",
        lambda _path: pytest.fail("dry-run must not create baseline program"),
    )

    exit_code = compare.main(
        [
            "--only-model",
            "qwen3_6_27b",
            "--shards",
            "2",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "qwen36-only-dry-run",
        ]
    )

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert captured.count("--eval-only") == 4
    assert captured.count("--n-test 50") == 4
    assert captured.count("Qwen/Qwen3.6-27B") == 8
    assert "Qwen/Qwen3.8-27B" not in captured
    assert "before__qwen3_6_27b" in captured
    assert "after__qwen3_6_27b" in captured
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("mutator", "error_fragment"),
    [
        (
            lambda payload: payload["conditions"].__setitem__(
                0, {**payload["conditions"][0], "model_label": "unknown_model"}
            ),
            "condition set mismatch",
        ),
        (
            lambda payload: payload["benchmark"].__setitem__("temperature", 0.5),
            "benchmark field 'temperature' differs",
        ),
    ],
)
def test_merge_runs_rejects_condition_or_compatibility_mismatch(
    tmp_path, monkeypatch, mutator, error_fragment, capsys
):
    """What: merge refuses partial artifacts with different cells or benchmark settings."""
    qwen36_root = _run_partial_model(
        tmp_path, "qwen36-partial", "qwen3_6_27b", {"before": 0.4, "after": 0.8}, monkeypatch
    )
    qwen38_root = _run_partial_model(
        tmp_path, "qwen38-partial", "qwen3_8_27b", {"before": 0.6, "after": 1.2}, monkeypatch
    )
    source_path = qwen38_root / "factorial_comparison.json"
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    mutator(payload)
    source_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = compare.main(
        [
            "--merge-runs",
            str(qwen36_root),
            str(qwen38_root),
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "rejected-merge",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert error_fragment in captured.err
    assert not (tmp_path / "rejected-merge").exists()


def test_only_model_partial_runs_merge_to_complete_factorial_comparison(tmp_path, monkeypatch):
    """What: Qwen3.6 and Qwen3.8 partial artifacts combine into four cells with effects."""
    qwen36_root = _run_partial_model(
        tmp_path, "qwen36-partial", "qwen3_6_27b", {"before": 0.4, "after": 0.8}, monkeypatch
    )
    qwen38_root = _run_partial_model(
        tmp_path, "qwen38-partial", "qwen3_8_27b", {"before": 0.6, "after": 1.2}, monkeypatch
    )

    for run_root, model_label in (
        (qwen36_root, "qwen3_6_27b"),
        (qwen38_root, "qwen3_8_27b"),
    ):
        partial = json.loads((run_root / "factorial_comparison.json").read_text(encoding="utf-8"))
        assert len(partial["conditions"]) == 2
        assert partial["benchmark"]["selected_models"] == [model_label]
        assert partial["benchmark"]["run_complete"] is True
        assert partial["benchmark"]["partial"] is True
        assert partial["benchmark"]["comparable"] is False
        assert partial["effects"] == {}

    exit_code = compare.main(
        [
            "--merge-runs",
            str(qwen36_root),
            str(qwen38_root),
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "merged-factorial",
        ]
    )

    assert exit_code == 0
    merged_path = tmp_path / "merged-factorial" / "factorial_comparison.json"
    merged = json.loads(merged_path.read_text(encoding="utf-8"))
    assert len(merged["conditions"]) == 4
    assert merged["benchmark"]["run_complete"] is True
    assert merged["benchmark"]["partial"] is False
    assert merged["benchmark"]["comparable"] is True
    assert merged["failures"] == []
    assert set(merged["effects"]) >= {"legacy_test20", "train40", "untouched40", "all100"}
    assert merged["effects"]["all100"]["prompt_effect_by_model"]["qwen3_6_27b"] == pytest.approx(
        0.4
    )
    assert merged["effects"]["all100"]["model_effect_by_prompt"]["before"] == pytest.approx(0.2)
