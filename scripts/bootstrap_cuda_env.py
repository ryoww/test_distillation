#!/usr/bin/env python3
"""Install a project-local CUDA compiler without sudo."""

from __future__ import annotations

import argparse
import os
import stat
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from project_paths import CUDA_ENV_DIR, MICROMAMBA_BIN, MICROMAMBA_ROOT

MICROMAMBA_URL = "https://micro.mamba.pm/api/micromamba/linux-64/latest"
CUDA_CHANNEL = "nvidia/label/cuda-13.0.0"
CUDA_PACKAGES = [
    "cuda-nvcc=13.0.48",
    "cuda-cudart-dev=13.0.48",
]


def install_micromamba(binary: Path) -> None:
    if binary.is_file():
        return

    binary.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="distill-micromamba-") as temp_dir:
        archive = Path(temp_dir) / "micromamba.tar.bz2"
        print(f"Downloading {MICROMAMBA_URL}", flush=True)
        urllib.request.urlretrieve(MICROMAMBA_URL, archive)
        with tarfile.open(archive, "r:bz2") as tar:
            member = tar.getmember("bin/micromamba")
            source = tar.extractfile(member)
            if source is None:
                raise RuntimeError("micromamba archive did not contain bin/micromamba")
            binary.write_bytes(source.read())
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-dir", type=Path, default=CUDA_ENV_DIR)
    args = parser.parse_args()

    install_micromamba(MICROMAMBA_BIN)
    action = "install" if (args.env_dir / "conda-meta").is_dir() else "create"
    environment = os.environ.copy()
    environment["MAMBA_ROOT_PREFIX"] = str(MICROMAMBA_ROOT)
    command = [
        str(MICROMAMBA_BIN),
        action,
        "--yes",
        "--prefix",
        str(args.env_dir),
        "--channel",
        CUDA_CHANNEL,
        "--channel",
        "conda-forge",
        "--strict-channel-priority",
        *CUDA_PACKAGES,
    ]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, env=environment)

    nvcc = args.env_dir / "bin" / "nvcc"
    subprocess.run([str(nvcc), "--version"], check=True)
    print(f"\nCUDA_HOME={args.env_dir}")
    print(f"Add to PATH: export PATH={args.env_dir / 'bin'}:$PATH")


if __name__ == "__main__":
    main()
