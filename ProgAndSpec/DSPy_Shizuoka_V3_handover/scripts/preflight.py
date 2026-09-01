#!/usr/bin/env python3
"""Validate the handover package before evaluation or GEPA training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.data_loader import load_v3_data, reference_value_from_solution
from src.lm_config import (
    DEFAULT_GENERATION_API_BASE,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_REFLECTION_API_BASE,
    DEFAULT_REFLECTION_MODEL,
    qwen_configs_from_env,
)
from src.modules import AlgorithmGenerator
from src.utils.safe_exec import safe_run

REQUIRED_FIELDS = {
    "id",
    "name",
    "domain",
    "math_type",
    "description",
    "requirements",
    "instance",
    "reference_solution",
}


def check_data(data_dir: Path) -> dict:
    records = load_v3_data(str(data_dir))
    failures = []
    ids = []
    objective_count = 0
    for record in records:
        missing = sorted(REQUIRED_FIELDS - record.keys())
        if missing:
            failures.append({"id": record.get("id"), "missing": missing})
        ids.append(record.get("id"))
        if reference_value_from_solution(record) is not None:
            objective_count += 1
    unique_ids = len(set(ids))
    checksum_lines = []
    for path in sorted(data_dir.glob("prob_*.json")):
        checksum_lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
    checksum = hashlib.sha256("".join(checksum_lines).encode()).hexdigest()
    manifest = json.loads((BASE_DIR / "data_manifest.json").read_text(encoding="utf-8"))
    return {
        "ok": (
            len(records) == manifest["files"]
            and unique_ids == manifest["files"]
            and checksum == manifest["sha256"]
            and not failures
        ),
        "records": len(records),
        "unique_ids": unique_ids,
        "sha256": checksum,
        "numeric_objectives": objective_count,
        "schema_failures": failures,
    }


def check_program(path: Path) -> dict:
    try:
        program = AlgorithmGenerator()
        program.load(str(path))
    # Why not narrow: preflight must report any load failure as ok=false,
    # so an unanticipated exception type must not abort the whole check run.
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "path": str(path), "error": str(exc)}
    return {
        "ok": True,
        "path": str(path),
        "demos": len(program.generate.predict.demos),
        "instruction_chars": len(program.generate.predict.signature.instructions),
    }


def check_model_manifest() -> dict:
    manifest = json.loads((BASE_DIR / "model_manifest.json").read_text(encoding="utf-8"))
    generation = manifest["generation"]
    reflection = manifest["reflection"]
    serving = manifest["serving"]
    required_context = serving["verified_phase_e_input_tokens"] + serving["generation_max_tokens"]
    return {
        "ok": (
            generation["model_id"] == DEFAULT_GENERATION_MODEL
            and generation["api_base"] == DEFAULT_GENERATION_API_BASE
            and reflection["model_id"] == DEFAULT_REFLECTION_MODEL
            and reflection["api_base"] == DEFAULT_REFLECTION_API_BASE
            and serving["max_model_len"] >= required_context
        ),
        "generation_model": generation["model_id"],
        "reflection_model": reflection["model_id"],
        "max_model_len": serving["max_model_len"],
        "required_context": required_context,
    }


def check_endpoint(label: str, model: str, api_base: str, credential: str) -> dict:
    try:
        client = OpenAI(base_url=api_base, api_key=credential, timeout=10.0)
        served = sorted(item.id for item in client.models.list().data)
    # Why not narrow: the OpenAI client raises transport, auth and API errors from
    # separate hierarchies; preflight reports all of them as ok=false instead of raising.
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "label": label, "api_base": api_base, "error": str(exc)}
    return {
        "ok": model in served,
        "label": label,
        "api_base": api_base,
        "expected_model": model,
        "served_models": served,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "data" / "problems")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--skip-programs", action="store_true")
    args = parser.parse_args()

    checks = {
        "runtime": {
            "ok": sys.version_info[:2] == (3, 11),
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "dspy": version("dspy"),
            "gepa": version("gepa"),
        },
        "data": check_data(args.data_dir),
        "model_manifest": check_model_manifest(),
    }
    if not args.skip_programs:
        checks["phase_e_program"] = check_program(BASE_DIR / "compiled_program_v3_gepa_phaseE.json")
        checks["phase_f_program"] = check_program(
            BASE_DIR / "compiled_program_v3_gepa_phaseF_noref.json"
        )
    safe_ok, safe_result = safe_run(
        "def solve(instance):\n    return {'value': instance.get('value', 0) + 1}",
        {"value": 1},
        timeout=5.0,
    )
    checks["safe_exec"] = {"ok": safe_ok and safe_result == {"value": 2}}

    if not args.offline:
        generation, reflection = qwen_configs_from_env()
        checks["generation_endpoint"] = check_endpoint(
            "generation", generation.model, generation.api_base, generation.api_key
        )
        checks["reflection_endpoint"] = check_endpoint(
            "reflection", reflection.model, reflection.api_base, reflection.api_key
        )

    ok = all(item.get("ok", False) for item in checks.values())
    print(json.dumps({"ok": ok, "checks": checks}, indent=2, ensure_ascii=False))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
