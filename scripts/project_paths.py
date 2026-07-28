"""Project-local storage defaults with environment-variable overrides."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("DISTILL_MODEL_DIR", REPO_ROOT / "model")).resolve()
DATA_DIR = Path(os.getenv("DISTILL_DATA_DIR", REPO_ROOT / "data")).resolve()
OUTPUT_DIR = Path(os.getenv("DISTILL_OUTPUT_DIR", REPO_ROOT / "outputs")).resolve()
CACHE_DIR = Path(os.getenv("DISTILL_CACHE_DIR", REPO_ROOT / ".cache")).resolve()
RUNTIME_DIR = Path(os.getenv("DISTILL_RUNTIME_DIR", REPO_ROOT / ".runtime")).resolve()

HF_HOME = Path(os.getenv("HF_HOME", MODEL_DIR / "hf_home")).resolve()
HF_HUB_CACHE = Path(os.getenv("HF_HUB_CACHE", HF_HOME / "hub")).resolve()
HF_DATASETS_CACHE = Path(os.getenv("HF_DATASETS_CACHE", DATA_DIR / "hf_datasets")).resolve()
TRANSFORMERS_CACHE = Path(os.getenv("TRANSFORMERS_CACHE", HF_HOME / "transformers")).resolve()
XDG_CACHE_HOME = Path(os.getenv("XDG_CACHE_HOME", CACHE_DIR / "xdg")).resolve()
UV_CACHE_DIR = Path(os.getenv("UV_CACHE_DIR", CACHE_DIR / "uv")).resolve()
VLLM_CACHE_ROOT = Path(os.getenv("VLLM_CACHE_ROOT", CACHE_DIR / "vllm")).resolve()


def configure_storage(*, include_uv: bool = False, include_vllm: bool = False) -> None:
    if include_uv:
        os.environ.setdefault("UV_CACHE_DIR", str(UV_CACHE_DIR))
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE))
    os.environ.setdefault("HF_DATASETS_CACHE", str(HF_DATASETS_CACHE))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(TRANSFORMERS_CACHE))
    os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))
    if include_vllm:
        os.environ.setdefault("VLLM_CACHE_ROOT", str(VLLM_CACHE_ROOT))
