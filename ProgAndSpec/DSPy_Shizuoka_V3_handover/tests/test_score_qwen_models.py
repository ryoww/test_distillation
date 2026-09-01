"""What: verify the public contract of the Qwen model score wrapper."""

import hashlib
import json
import subprocess
from dataclasses import is_dataclass
from pathlib import Path

import pytest

from scripts.score_qwen_models import (
    ModelTarget,
    build_eval_command,
    main,
    parse_args,
    rank_models,
    summarize_result,
)

TARGET_ROOT = Path(__file__).parents[1]


def _option(command, name):
    """What: retrieve the value following a command-line option."""
    return command[command.index(name) + 1]


def _write_model_result(command, scores, instance_ids=("prob_001", "prob_002", "prob_003")):
    """What: write the result file that train_gepa_v3.py --eval-only would produce."""
    output_dir = Path(_option(command, "--output-dir"))
    run_name = _option(command, "--run-name")
    phase = "phaseF_noref" if "--no-reference" in command else "phaseE"
    result_path = output_dir / run_name / f"evaluation_results_v3_gepa_{phase}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    results = [
        {
            "instance_id": instance_ids[0],
            "score": scores[0],
            "status": "exact_match",
            "beat_reference_analysis": True,
            "beat_reference_strict": False,
        },
        {
            "instance_id": instance_ids[1],
            "score": scores[1],
            "status": "beat_reference",
            "beat_reference_analysis": True,
            "beat_reference_strict": True,
        },
        {
            "instance_id": instance_ids[2],
            "score": -0.5,
            "status": "invalid_solution",
            "beat_reference_analysis": False,
            "beat_reference_strict": False,
        },
    ]
    result_path.write_text(
        json.dumps(
            {
                "test": {
                    "results": results,
                    "mean_score": sum(row["score"] for row in results) / len(results),
                    "valid_count": 2,
                    "total_count": len(results),
                    "beat_reference": 2,
                    "beat_reference_analysis": 2,
                    "beat_reference_strict": 1,
                }
            }
        ),
        encoding="utf-8",
    )


def test_model_target_is_a_dataclass_with_public_fields():
    """What: a model target stores endpoint metadata and an optional expected revision."""
    target = ModelTarget("qwen36", "Qwen/Qwen3.6-27B", "http://127.0.0.1:7501/v1")
    pinned = ModelTarget(
        "qwen38",
        "Qwen/Qwen3.8-27B",
        "http://127.0.0.1:7502/v1",
        "revision-38",
    )

    assert is_dataclass(target)
    assert target.label == "qwen36"
    assert target.model == "Qwen/Qwen3.6-27B"
    assert target.api_base == "http://127.0.0.1:7501/v1"
    assert target.revision is None
    assert pinned.revision == "revision-38"


def test_parse_args_defaults_to_phase_e_qwen_pair_and_twenty_test_cases():
    """What: default CLI settings select both local Qwen models and Phase E evaluation."""
    args = parse_args([])

    assert args.n_train == 0
    assert args.n_test == 20
    assert args.no_reference is False
    assert args.max_tokens is not None
    assert args.lm_timeout is not None
    assert args.temperature == pytest.approx(0.0)
    assert args.qwen36_revision == "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
    assert args.qwen38_revision == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"

    command_36 = build_eval_command(
        ModelTarget(
            "qwen36",
            "Qwen/Qwen3.6-27B",
            "http://127.0.0.1:7501/v1",
            args.qwen36_revision,
        ),
        args,
        TARGET_ROOT / "outputs",
    )
    command_38 = build_eval_command(
        ModelTarget(
            "qwen38",
            "Qwen/Qwen3.8-27B",
            "http://127.0.0.1:7502/v1",
            args.qwen38_revision,
        ),
        args,
        TARGET_ROOT / "outputs",
    )

    for command in (command_36, command_38):
        assert isinstance(command, list)
        assert any(Path(part).name == "train_gepa_v3.py" for part in command)
        assert "--eval-only" in command
        assert _option(command, "--n-train") == "0"
        assert _option(command, "--n-test") == "20"
        assert _option(command, "--temperature") == "0.0"
        assert "--no-reference" not in command
    assert _option(command_36, "--generation-model") == "Qwen/Qwen3.6-27B"
    assert _option(command_36, "--generation-api-base") == "http://127.0.0.1:7501/v1"
    assert _option(command_38, "--generation-model") == "Qwen/Qwen3.8-27B"
    assert _option(command_38, "--generation-api-base") == "http://127.0.0.1:7502/v1"
    assert _option(command_36, "--run-name") != _option(command_38, "--run-name")


def test_parse_args_exposes_wrapper_overrides_and_builds_reference_free_command(tmp_path):
    """What: documented CLI overrides reach the Phase F model evaluation command."""
    program_path = tmp_path / "program.json"
    args = parse_args(
        [
            "--no-reference",
            "--program-path",
            str(program_path),
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "pair",
            "--max-tokens",
            "256",
            "--lm-timeout",
            "31",
            "--temperature",
            "0.25",
            "--qwen38-revision",
            "custom-revision-38",
            "--skip-connection-check",
            "--parallel",
            "--dry-run",
        ]
    )

    assert args.no_reference is True
    assert Path(args.program_path) == program_path
    assert Path(args.output_dir) == tmp_path
    assert args.run_name == "pair"
    assert args.max_tokens == 256
    assert args.lm_timeout == 31
    assert args.temperature == pytest.approx(0.25)
    assert args.qwen38_revision == "custom-revision-38"
    assert args.skip_connection_check is True
    assert args.parallel is True
    assert args.dry_run is True

    command = build_eval_command(
        ModelTarget(
            "qwen38",
            "Qwen/Qwen3.8-27B",
            "http://127.0.0.1:7502/v1",
            args.qwen38_revision,
        ),
        args,
        tmp_path,
    )

    assert "--eval-only" in command
    assert "--no-reference" in command
    assert _option(command, "--program-path") == str(program_path)
    assert _option(command, "--max-tokens") == "256"
    assert _option(command, "--lm-timeout") == "31"
    assert _option(command, "--temperature") == "0.25"
    assert "--skip-connection-check" in command
    assert _option(command, "--output-dir") == str(tmp_path)
    assert _option(command, "--run-name") != "pair"


def test_summarize_result_aggregates_test_scores_statuses_and_reference_counts(tmp_path):
    """What: the test split becomes a complete per-model score summary."""
    result_path = tmp_path / "evaluation.json"
    result_path.write_text(
        json.dumps(
            {
                "test": {
                    "results": [
                        {
                            "instance_id": "prob_001",
                            "score": 0.8,
                            "status": "exact_match",
                            "beat_reference_analysis": True,
                            "beat_reference_strict": False,
                        },
                        {
                            "instance_id": "prob_002",
                            "score": 0.4,
                            "status": "beat_reference",
                            "beat_reference_analysis": True,
                            "beat_reference_strict": True,
                        },
                        {
                            "instance_id": "prob_003",
                            "score": -0.5,
                            "status": "invalid_solution",
                            "beat_reference_analysis": False,
                            "beat_reference_strict": False,
                        },
                    ],
                    "mean_score": (0.8 + 0.4 - 0.5) / 3,
                    "valid_count": 2,
                    "total_count": 3,
                    "beat_reference": 2,
                    "beat_reference_analysis": 2,
                    "beat_reference_strict": 1,
                }
            }
        ),
        encoding="utf-8",
    )
    target = ModelTarget(
        "qwen36",
        "Qwen/Qwen3.6-27B",
        "http://127.0.0.1:7501/v1",
        "revision-36",
    )

    summary = summarize_result(target, result_path, elapsed_seconds=12.5)

    assert summary["label"] == "qwen36"
    assert summary["model"] == "Qwen/Qwen3.6-27B"
    assert summary["api_base"] == "http://127.0.0.1:7501/v1"
    assert summary["expected_revision"] == "revision-36"
    assert summary["instance_ids"] == ["prob_001", "prob_002", "prob_003"]
    assert summary["elapsed_seconds"] == pytest.approx(12.5)
    assert summary["mean_score"] == pytest.approx((0.8 + 0.4 - 0.5) / 3)
    assert summary["valid_count"] == 2
    assert summary["total_count"] == 3
    assert summary["valid_rate"] == pytest.approx(2 / 3)
    assert summary["beat_reference"] == 2
    assert summary["beat_reference_analysis"] == 2
    assert summary["beat_reference_strict"] == 1
    assert summary["status_counts"] == {
        "exact_match": 1,
        "beat_reference": 1,
        "invalid_solution": 1,
    }


def test_rank_models_sorts_mean_score_and_assigns_equal_ranks():
    """What: higher mean scores rank first and ties share one rank."""
    summaries = [
        {"label": "qwen36", "mean_score": 0.75},
        {"label": "qwen38", "mean_score": 0.90},
        {"label": "tie", "mean_score": 0.90},
        {"label": "low", "mean_score": 0.10},
    ]

    ranked = rank_models(summaries)

    assert [item["label"] for item in ranked] == ["qwen38", "tie", "qwen36", "low"]
    assert [item["rank"] for item in ranked] == [1, 1, 3, 4]
    assert summaries[0].get("rank") is None


def test_main_dry_run_prints_two_eval_commands_without_writing_files(tmp_path, capsys):
    """What: dry-run displays both model evaluations and leaves the output directory untouched."""

    def fail_runner(*_args, **_kwargs):
        raise AssertionError("dry-run must not invoke subprocess.run")

    exit_code = main(
        ["--dry-run", "--output-dir", str(tmp_path), "--run-name", "dry-run"],
        runner=fail_runner,
    )

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert captured.count("--eval-only") == 2
    assert "Qwen/Qwen3.6-27B" in captured
    assert "Qwen/Qwen3.8-27B" in captured
    assert not list(tmp_path.iterdir())


def test_main_dry_run_prints_commands_when_program_path_does_not_exist(tmp_path, capsys):
    """What: dry-run remains useful before a compiled program has been copied into place."""
    missing_program = tmp_path / "missing-program.json"

    exit_code = main(
        [
            "--dry-run",
            "--program-path",
            str(missing_program),
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "missing-program",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.count("--eval-only") == 2
    assert captured.out.count(str(missing_program)) == 2
    assert "Compiled program not found" not in captured.err
    assert not missing_program.exists()


def test_main_runs_both_models_and_writes_comparison_json(tmp_path):
    """What: normal mode evaluates both targets and aggregates their summaries."""
    calls = []

    def fake_runner(command, *args, **kwargs):
        calls.append((command, args, kwargs))
        model = _option(command, "--generation-model")
        scores = (0.6, 0.5) if model.endswith("3.6-27B") else (0.9, 0.8)
        _write_model_result(command, scores)
        return subprocess.CompletedProcess(command, 0)

    exit_code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "pair-score",
            "--skip-connection-check",
        ],
        runner=fake_runner,
    )

    assert exit_code == 0
    assert len(calls) == 2
    commands = [call[0] for call in calls]
    assert all("--eval-only" in command for command in commands)
    assert {_option(command, "--generation-model") for command in commands} == {
        "Qwen/Qwen3.6-27B",
        "Qwen/Qwen3.8-27B",
    }
    assert {_option(command, "--generation-api-base") for command in commands} == {
        "http://127.0.0.1:7501/v1",
        "http://127.0.0.1:7502/v1",
    }
    assert len({_option(command, "--run-name") for command in commands}) == 2

    comparison_paths = list(tmp_path.rglob("comparison.json"))
    assert len(comparison_paths) == 1
    comparison = json.loads(comparison_paths[0].read_text(encoding="utf-8"))
    comparison_text = json.dumps(comparison, ensure_ascii=False)
    assert "Qwen/Qwen3.6-27B" in comparison_text
    assert "Qwen/Qwen3.8-27B" in comparison_text
    assert "pair-score" in comparison_text
    benchmark = comparison["benchmark"]
    expected_program = TARGET_ROOT / "compiled_program_v3_gepa_phaseE.json"
    data_manifest = json.loads((TARGET_ROOT / "data_manifest.json").read_text(encoding="utf-8"))
    assert benchmark["seed"] == 42
    assert benchmark["data_manifest_sha256"] == data_manifest["sha256"]
    assert benchmark["program_sha256"] == hashlib.sha256(expected_program.read_bytes()).hexdigest()
    assert benchmark["temperature"] == pytest.approx(0.0)
    assert benchmark["test_instance_ids"] == ["prob_001", "prob_002", "prob_003"]
    assert benchmark["comparable"] is True
    assert comparison["winner"] == ["qwen3_8_27b"]
    summaries = {summary["label"]: summary for summary in comparison["models"]}
    assert summaries["qwen3_6_27b"]["expected_revision"] == args_revision("qwen36")
    assert summaries["qwen3_8_27b"]["expected_revision"] == args_revision("qwen38")


def args_revision(model):
    """What: return the manifest revision expected by a default wrapper invocation."""
    manifest = json.loads((TARGET_ROOT / "model_manifest.json").read_text(encoding="utf-8"))
    key = "generation" if model == "qwen36" else "reflection"
    return manifest[key]["revision"]


def test_main_marks_mismatched_instance_sets_noncomparable(tmp_path):
    """What: differing test instance IDs suppress winners and fail the comparison run."""

    def fake_runner(command, *_args, **_kwargs):
        model = _option(command, "--generation-model")
        instance_ids = (
            ("prob_001", "prob_002", "prob_003")
            if model.endswith("3.6-27B")
            else ("prob_001", "prob_002", "prob_099")
        )
        _write_model_result(command, (0.6, 0.5), instance_ids=instance_ids)
        return subprocess.CompletedProcess(command, 0)

    exit_code = main(
        ["--output-dir", str(tmp_path), "--run-name", "id-mismatch"],
        runner=fake_runner,
    )

    comparison_path = tmp_path / "id-mismatch" / "comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert comparison["benchmark"]["comparable"] is False
    assert "test_instance_ids" in comparison["benchmark"]
    assert comparison["winner"] == []
    assert {tuple(row["instance_ids"]) for row in comparison["models"]} == {
        ("prob_001", "prob_002", "prob_003"),
        ("prob_001", "prob_002", "prob_099"),
    }


def test_main_records_runner_exception_and_still_writes_comparison(tmp_path):
    """What: a child runner exception becomes a recorded failure instead of losing the run."""

    def fake_runner(command, *_args, **_kwargs):
        model = _option(command, "--generation-model")
        if model.endswith("3.6-27B"):
            raise RuntimeError("runner exploded")
        _write_model_result(command, (0.9, 0.8))
        return subprocess.CompletedProcess(command, 0)

    exit_code = main(
        ["--output-dir", str(tmp_path), "--run-name", "runner-error"],
        runner=fake_runner,
    )

    comparison_path = tmp_path / "runner-error" / "comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert comparison["benchmark"]["comparable"] is False
    assert comparison["winner"] == []
    assert "runner exploded" in json.dumps(comparison["failures"])


def test_main_records_malformed_result_and_still_writes_comparison(tmp_path):
    """What: malformed child JSON becomes a recorded failure with a durable comparison file."""

    def fake_runner(command, *_args, **_kwargs):
        model = _option(command, "--generation-model")
        if model.endswith("3.6-27B"):
            output_dir = Path(_option(command, "--output-dir"))
            run_name = _option(command, "--run-name")
            result_path = output_dir / run_name / "evaluation_results_v3_gepa_phaseE.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text("{broken", encoding="utf-8")
        else:
            _write_model_result(command, (0.9, 0.8))
        return subprocess.CompletedProcess(command, 0)

    exit_code = main(
        ["--output-dir", str(tmp_path), "--run-name", "broken-json"],
        runner=fake_runner,
    )

    comparison_path = tmp_path / "broken-json" / "comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert comparison["benchmark"]["comparable"] is False
    assert comparison["winner"] == []
    assert comparison["failures"]
