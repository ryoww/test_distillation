"""検証済み参照解つきの問題を、既存問題を雛形にして量産する。"""

from __future__ import annotations

from . import templates as _templates  # noqa: F401  登録の副作用が必要
from .base import TEMPLATES, SolveError, Template
from .pipeline import (
    GENERATOR_VERSION,
    ValidationError,
    generate_dataset,
    load_base,
    make_problem,
    shape_signature,
    validate_problem,
)

__all__ = [
    "GENERATOR_VERSION",
    "TEMPLATES",
    "SolveError",
    "Template",
    "ValidationError",
    "generate_dataset",
    "load_base",
    "make_problem",
    "shape_signature",
    "validate_problem",
]
