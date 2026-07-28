#!/usr/bin/env python3
"""Serve the base model or a distilled LoRA through vLLM."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from project_paths import RUNTIME_DIR, configure_storage

DEFAULT_MODEL = "InternScience/Agents-A1-4B"
MODEL_REVISION = "945c40a4aa6f534d434a353207b8d42ecf7a5293"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=os.getenv("VLLM_MODEL_PATH", DEFAULT_MODEL))
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--served-model-name", default="agents-a1-4b")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default="7501")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", default="4096")
    parser.add_argument("--gpu-memory-utilization", default="0.90")
    parser.add_argument(
        "--gdn-prefill-backend",
        choices=["flashinfer", "triton"],
        default="triton",
        help="Triton avoids FlashInfer's nvcc-dependent first-run JIT on this server.",
    )
    parser.add_argument("--vllm-env-dir", type=Path, default=RUNTIME_DIR / "vllm")
    args, extra = parser.parse_known_args()
    configure_storage(include_vllm=True)
    os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

    vllm = args.vllm_env_dir / "bin/vllm"
    if not vllm.exists():
        raise SystemExit("vLLM environment missing; run uv run scripts/bootstrap_vllm_env.py")
    model_path = Path(args.model_path)
    model = str(model_path.resolve()) if model_path.exists() else args.model_path
    command = [
        str(vllm),
        "serve",
        model,
        "--served-model-name",
        args.served_model_name,
        "--revision",
        args.model_revision,
        "--host",
        args.host,
        "--port",
        args.port,
        "--api-key",
        args.api_key,
        "--dtype",
        args.dtype,
        "--max-model-len",
        args.max_model_len,
        "--gpu-memory-utilization",
        args.gpu_memory_utilization,
        "--gdn-prefill-backend",
        args.gdn_prefill_backend,
    ]
    if args.adapter_path:
        adapter = args.adapter_path.resolve()
        if not adapter.exists():
            raise SystemExit(f"Adapter path does not exist: {adapter}")
        command.extend(
            [
                "--enable-lora",
                "--max-lora-rank",
                "64",
                "--max-loras",
                "1",
                "--lora-modules",
                f"{args.served_model_name}={adapter}",
            ]
        )
    command.extend(extra)
    print(" ".join(command), flush=True)
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
