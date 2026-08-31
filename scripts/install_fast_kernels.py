#!/usr/bin/env python3
"""Build the optional Agents-A1 training kernels against local CUDA."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from project_paths import CUDA_ENV_DIR, RUNTIME_DIR, configure_storage


def run(command: list[str], environment: dict[str, str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, env=environment)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-dir", type=Path, default=RUNTIME_DIR / "train")
    parser.add_argument("--cuda-dir", type=Path, default=CUDA_ENV_DIR)
    parser.add_argument("--max-jobs", type=int, default=8)
    args = parser.parse_args()
    configure_storage(include_uv=True)

    python = args.env_dir / "bin" / "python"
    nvcc = args.cuda_dir / "bin" / "nvcc"
    target = args.cuda_dir / "targets" / "x86_64-linux"
    if not python.is_file():
        raise SystemExit("Training environment missing; run bootstrap_train_env.py first")
    if not nvcc.is_file():
        raise SystemExit("Local nvcc missing; run bootstrap_cuda_env.py first")

    environment = os.environ.copy()
    environment["CUDA_HOME"] = str(args.cuda_dir)
    environment["CUDACXX"] = str(nvcc)
    environment["MAX_JOBS"] = str(args.max_jobs)
    environment["PATH"] = f"{args.cuda_dir / 'bin'}:{environment['PATH']}"
    for variable, path in (
        ("CPATH", target / "include"),
        ("LIBRARY_PATH", target / "lib"),
        ("LD_LIBRARY_PATH", target / "lib"),
    ):
        previous = environment.get(variable)
        environment[variable] = f"{path}:{previous}" if previous else str(path)

    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "ninja",
            "packaging",
        ],
        environment,
    )
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--no-build-isolation",
            "causal-conv1d==1.6.2.post1",
        ],
        environment,
    )
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "flash-linear-attention==0.5.2",
        ],
        environment,
    )
    run(
        [
            str(python),
            "-c",
            (
                "import causal_conv1d, fla; "
                "print('causal-conv1d', causal_conv1d.__version__); "
                "print('flash-linear-attention', fla.__version__)"
            ),
        ],
        environment,
    )


if __name__ == "__main__":
    main()
