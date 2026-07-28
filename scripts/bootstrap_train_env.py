#!/usr/bin/env python3
"""Create an isolated CUDA training environment with uv."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from project_paths import RUNTIME_DIR, configure_storage

PACKAGES = [
    "torch>=2.9",
    "transformers>=5.6.0",
    "datasets>=4.0.0",
    "accelerate>=1.10.0",
    "peft>=0.18.0",
    "bitsandbytes>=0.49.0",
    "safetensors>=0.6.0",
    "tensorboard>=2.20.0",
    "pytest>=8.0.0",
    "ruff>=0.12.0",
]


def run(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-dir", type=Path, default=RUNTIME_DIR / "train")
    parser.add_argument("--python", default="3.13")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configure_storage(include_uv=True)

    command = ["uv", "venv", "--no-project", "--python", args.python]
    if args.rebuild:
        command.append("--clear")
    command.append(str(args.env_dir))
    run(command, args.dry_run)
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(args.env_dir / "bin/python"),
            "--torch-backend",
            "auto",
            *PACKAGES,
        ],
        args.dry_run,
    )
    print(f"Training Python: {args.env_dir / 'bin/python'}")


if __name__ == "__main__":
    main()
