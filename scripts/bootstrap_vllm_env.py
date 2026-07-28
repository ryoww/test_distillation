#!/usr/bin/env python3
"""Create the CUDA 13 vLLM environment used for adapter serving."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from project_paths import RUNTIME_DIR, configure_storage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-dir", type=Path, default=RUNTIME_DIR / "vllm")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configure_storage(include_uv=True)

    venv = ["uv", "venv", "--no-project", "--python", "3.13"]
    if args.rebuild:
        venv.append("--clear")
    venv.append(str(args.env_dir))
    commands = [
        venv,
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(args.env_dir / "bin/python"),
            "--torch-backend",
            "auto",
            "vllm==0.20.0",
        ],
    ]
    for command in commands:
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
