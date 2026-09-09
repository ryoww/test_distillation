"""検証済み参照解つきの問題を、既存問題を雛形にして量産する。"""

from __future__ import annotations

# 登録の副作用が必要なので、雛形モジュールは読み込むだけでよい。
from . import (  # noqa: F401
    templates,
    templates_assignment,
    templates_assignment_ext,
    templates_composite,
    templates_composite_ext,
    templates_covering_ext,
    templates_knapsack_ext,
    templates_network,
    templates_network_ext,
    templates_production,
    templates_production_ext,
    templates_routing,
    templates_routing_ext,
    templates_routing_ext2,
    templates_scheduling,
    templates_scheduling_ext,
    templates_scheduling_ext2,
)
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
