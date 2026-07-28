#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


REQUIRE_CHOICES = (
    "nvidia-smi",
    "torch",
    "torch-gpu",
    "transformers",
    "tensorflow",
    "tensorflow-gpu",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a Python ML environment for PyTorch, Transformers, and TensorFlow."
    )
    parser.add_argument(
        "--model-dir",
        type=expand_path,
        help="Optional local model directory to validate with Transformers AutoConfig.",
    )
    parser.add_argument(
        "--expected-gpu",
        help="Expected GPU name substring. Succeeds when any visible GPU contains it.",
    )
    parser.add_argument(
        "--expected-model-type",
        help="Expected model_type in config.json or Transformers AutoConfig.",
    )
    parser.add_argument(
        "--require",
        action="append",
        choices=REQUIRE_CHOICES,
        default=[],
        help="Stack component required for --strict. May be passed more than once.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON instead of text.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when required components or expectations fail.",
    )
    return parser.parse_args()


def expand_path(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser()


def run_command(command: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False, "missing"
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()
        stdout = exc.stdout.strip()
        return False, stderr or stdout or str(exc)
    return True, result.stdout.strip()


def probe_nvidia_smi() -> dict[str, object]:
    binary = shutil.which("nvidia-smi")
    if not binary:
        return {"available": False, "error": "nvidia-smi not found"}

    ok, output = run_command([binary])
    if not ok:
        return {"available": False, "error": output}

    driver_match = re.search(r"Driver Version:\s*([0-9.]+)", output)
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", output)
    gpu_names: list[str] = []
    ok_query, query_output = run_command(
        [binary, "--query-gpu=name", "--format=csv,noheader"]
    )
    if ok_query and query_output:
        gpu_names = [line.strip() for line in query_output.splitlines() if line.strip()]

    return {
        "available": True,
        "driver_version": driver_match.group(1) if driver_match else None,
        "cuda_version": cuda_match.group(1) if cuda_match else None,
        "gpu_names": gpu_names,
    }


def probe_binary(name: str) -> dict[str, object]:
    binary = shutil.which(name)
    return {"available": bool(binary), "path": binary}


def probe_torch() -> dict[str, object]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - import failure path
        return {"import_ok": False, "error": repr(exc)}

    info: dict[str, object] = {
        "import_ok": True,
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "hip_runtime": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
    }

    mps = getattr(getattr(torch, "backends", None), "mps", None)
    info["mps_available"] = bool(
        getattr(mps, "is_available", lambda: False)()
    )

    if torch.cuda.is_available():
        devices = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": f"{properties.major}.{properties.minor}",
                    "total_memory_gb": round(properties.total_memory / (1024**3), 2),
                }
            )
        info["device_count"] = torch.cuda.device_count()
        info["devices"] = devices
        info["bf16_supported"] = bool(
            getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        )

    info["accelerator_available"] = bool(
        info.get("cuda_available") or info.get("mps_available")
    )
    return info


def read_model_config(model_dir: Path | None) -> dict[str, object]:
    if model_dir is None:
        return {}

    info: dict[str, object] = {"model_dir": str(model_dir)}
    config_path = model_dir / "config.json"
    if not config_path.exists():
        info["config_json_ok"] = False
        info["config_json_error"] = "config.json not found"
        return info

    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - malformed json path
        info["config_json_ok"] = False
        info["config_json_error"] = repr(exc)
        return info

    info["config_json_ok"] = True
    info["raw_model_type"] = raw_config.get("model_type")
    info["architectures"] = raw_config.get("architectures")
    return info


def probe_transformers(
    model_dir: Path | None,
    expected_model_type: str | None,
) -> dict[str, object]:
    model_info = read_model_config(model_dir)

    try:
        import transformers
    except Exception as exc:  # pragma: no cover - import failure path
        return {"import_ok": False, "error": repr(exc), **model_info}

    info: dict[str, object] = {
        "import_ok": True,
        "version": transformers.__version__,
        **model_info,
    }

    if model_dir is None:
        return info

    if expected_model_type and "raw_model_type" in info:
        info["raw_model_type_matches"] = info.get("raw_model_type") == expected_model_type

    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    except Exception as exc:
        info["auto_config_ok"] = False
        info["auto_config_error"] = repr(exc)
    else:
        info["auto_config_ok"] = True
        info["auto_config_model_type"] = getattr(config, "model_type", None)
        info["auto_config_class"] = config.__class__.__name__
        if expected_model_type:
            info["auto_config_model_type_matches"] = (
                getattr(config, "model_type", None) == expected_model_type
            )
    return info


def probe_tensorflow() -> dict[str, object]:
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - import failure path
        return {"import_ok": False, "error": repr(exc)}

    try:
        build_info = tf.sysconfig.get_build_info()
    except Exception as exc:  # pragma: no cover - build info failure
        build_info = {"error": repr(exc)}

    try:
        gpus = tf.config.list_physical_devices("GPU")
    except Exception as exc:  # pragma: no cover - device query failure
        return {
            "import_ok": True,
            "version": tf.__version__,
            "gpu_query_ok": False,
            "gpu_query_error": repr(exc),
            "build_info": build_info,
        }

    return {
        "import_ok": True,
        "version": tf.__version__,
        "built_with_cuda": bool(getattr(tf.test, "is_built_with_cuda", lambda: False)()),
        "gpu_query_ok": True,
        "gpu_available": bool(gpus),
        "gpus": [device.name for device in gpus],
        "build_info": build_info,
    }


def apply_expectations(report: dict[str, object], expected_gpu: str | None) -> None:
    if not expected_gpu:
        return

    nvidia_info = report["nvidia_smi"]
    gpu_names = nvidia_info.get("gpu_names") or []
    nvidia_info["expected_gpu"] = expected_gpu
    nvidia_info["expected_gpu_matches"] = any(
        expected_gpu.lower() in str(name).lower() for name in gpu_names
    )


def build_report(
    model_dir: Path | None,
    expected_gpu: str | None,
    expected_model_type: str | None,
) -> dict[str, object]:
    report = {
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "nvidia_smi": probe_nvidia_smi(),
        "ffmpeg": probe_binary("ffmpeg"),
        "ffprobe": probe_binary("ffprobe"),
        "torch": probe_torch(),
        "transformers": probe_transformers(model_dir, expected_model_type),
        "tensorflow": probe_tensorflow(),
    }
    apply_expectations(report, expected_gpu)
    return report


def strict_failures(
    report: dict[str, object],
    required: list[str],
    model_dir: Path | None,
) -> list[str]:
    failures: list[str] = []
    required_set = set(required)

    nvidia_info = report["nvidia_smi"]
    if "nvidia-smi" in required_set and not nvidia_info.get("available"):
        failures.append("nvidia-smi is unavailable")
    if nvidia_info.get("expected_gpu_matches") is False:
        failures.append("expected GPU name was not found")

    torch_info = report["torch"]
    if {"torch", "torch-gpu"} & required_set:
        if not torch_info.get("import_ok"):
            failures.append("torch import failed")
        elif "torch-gpu" in required_set and not torch_info.get("accelerator_available"):
            failures.append("torch accelerator is unavailable")

    transformers_info = report["transformers"]
    if "transformers" in required_set or model_dir is not None:
        if not transformers_info.get("import_ok"):
            failures.append("transformers import failed")
    if model_dir is not None:
        if not transformers_info.get("config_json_ok"):
            failures.append("model config.json could not be read")
        if transformers_info.get("import_ok") and not transformers_info.get("auto_config_ok"):
            failures.append("AutoConfig.from_pretrained failed")
        if transformers_info.get("raw_model_type_matches") is False:
            failures.append("config.json model_type did not match expectation")
        if transformers_info.get("auto_config_model_type_matches") is False:
            failures.append("AutoConfig model_type did not match expectation")

    tensorflow_info = report["tensorflow"]
    if {"tensorflow", "tensorflow-gpu"} & required_set:
        if not tensorflow_info.get("import_ok"):
            failures.append("tensorflow import failed")
        elif "tensorflow-gpu" in required_set and not tensorflow_info.get("gpu_available"):
            failures.append("tensorflow GPU is unavailable")

    return failures


def render_text(report: dict[str, object]) -> str:
    lines: list[str] = []

    python_info = report["python"]
    lines.append(f"python.version={python_info['version']}")
    lines.append(f"python.executable={python_info['executable']}")
    lines.append(f"python.platform={python_info['platform']}")

    nvidia_info = report["nvidia_smi"]
    lines.append(f"nvidia_smi.available={nvidia_info.get('available')}")
    if nvidia_info.get("available"):
        lines.append(f"nvidia_smi.driver_version={nvidia_info.get('driver_version')}")
        lines.append(f"nvidia_smi.cuda_version={nvidia_info.get('cuda_version')}")
        gpu_names = nvidia_info.get("gpu_names") or []
        if gpu_names:
            lines.append(f"nvidia_smi.gpus={', '.join(gpu_names)}")
        if "expected_gpu" in nvidia_info:
            lines.append(f"nvidia_smi.expected_gpu={nvidia_info.get('expected_gpu')}")
            lines.append(
                "nvidia_smi.expected_gpu_matches="
                f"{nvidia_info.get('expected_gpu_matches')}"
            )
    else:
        lines.append(f"nvidia_smi.error={nvidia_info.get('error')}")

    for name in ("ffmpeg", "ffprobe"):
        binary_info = report[name]
        lines.append(f"{name}.available={binary_info.get('available')}")
        lines.append(f"{name}.path={binary_info.get('path')}")

    torch_info = report["torch"]
    lines.append(f"torch.import_ok={torch_info.get('import_ok')}")
    if torch_info.get("import_ok"):
        lines.append(f"torch.version={torch_info.get('version')}")
        lines.append(f"torch.cuda_runtime={torch_info.get('cuda_runtime')}")
        lines.append(f"torch.hip_runtime={torch_info.get('hip_runtime')}")
        lines.append(f"torch.cuda_available={torch_info.get('cuda_available')}")
        lines.append(f"torch.mps_available={torch_info.get('mps_available')}")
        lines.append(
            f"torch.accelerator_available={torch_info.get('accelerator_available')}"
        )
        if torch_info.get("cuda_available"):
            lines.append(f"torch.device_count={torch_info.get('device_count')}")
            lines.append(f"torch.bf16_supported={torch_info.get('bf16_supported')}")
            for device in torch_info.get("devices", []):
                lines.append(
                    "torch.device="
                    f"{device['index']}:{device['name']}:{device['capability']}:"
                    f"{device['total_memory_gb']}GB"
                )
    else:
        lines.append(f"torch.error={torch_info.get('error')}")

    transformers_info = report["transformers"]
    lines.append(f"transformers.import_ok={transformers_info.get('import_ok')}")
    if transformers_info.get("import_ok"):
        lines.append(f"transformers.version={transformers_info.get('version')}")
    else:
        lines.append(f"transformers.error={transformers_info.get('error')}")
    if "model_dir" in transformers_info:
        lines.append(f"transformers.model_dir={transformers_info.get('model_dir')}")
        lines.append(
            f"transformers.config_json_ok={transformers_info.get('config_json_ok')}"
        )
        if "raw_model_type" in transformers_info:
            lines.append(
                f"transformers.raw_model_type={transformers_info.get('raw_model_type')}"
            )
        if "architectures" in transformers_info:
            lines.append(
                f"transformers.architectures={transformers_info.get('architectures')}"
            )
        if "raw_model_type_matches" in transformers_info:
            lines.append(
                "transformers.raw_model_type_matches="
                f"{transformers_info.get('raw_model_type_matches')}"
            )
        if "auto_config_ok" in transformers_info:
            lines.append(
                f"transformers.auto_config_ok={transformers_info.get('auto_config_ok')}"
            )
        if transformers_info.get("auto_config_ok"):
            lines.append(
                "transformers.auto_config="
                f"{transformers_info.get('auto_config_class')}:"
                f"{transformers_info.get('auto_config_model_type')}"
            )
        elif "auto_config_error" in transformers_info:
            lines.append(
                f"transformers.auto_config_error={transformers_info.get('auto_config_error')}"
            )

    tensorflow_info = report["tensorflow"]
    lines.append(f"tensorflow.import_ok={tensorflow_info.get('import_ok')}")
    if tensorflow_info.get("import_ok"):
        lines.append(f"tensorflow.version={tensorflow_info.get('version')}")
        lines.append(f"tensorflow.built_with_cuda={tensorflow_info.get('built_with_cuda')}")
        lines.append(f"tensorflow.gpu_available={tensorflow_info.get('gpu_available')}")
        gpus = tensorflow_info.get("gpus") or []
        if gpus:
            lines.append(f"tensorflow.gpus={', '.join(gpus)}")
    else:
        lines.append(f"tensorflow.error={tensorflow_info.get('error')}")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = build_report(args.model_dir, args.expected_gpu, args.expected_model_type)
    failures = strict_failures(report, args.require, args.model_dir) if args.strict else []
    if failures:
        report["strict_failures"] = failures

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if failures:
            raise SystemExit(1)
        return

    print(render_text(report))
    if failures:
        print("strict.failures=" + "; ".join(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
