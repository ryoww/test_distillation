"""DSPy optimization modules for V3."""
from .lm_config import configure_lm, configure_qwen_default
from .best_known import BestKnownRegistry, registry, init_registry
from .metrics_v3 import evaluate_algorithm_v3, dspy_metric_v3, compute_v3_score
from .gepa_feedback_v3 import gepa_feedback_v3
from .signatures import GenerateOptimizationAlgorithm, ImproveAlgorithm, ParseInstance
from .modules import AlgorithmGenerator, AlgorithmModule, default_parse_code
from .requirement_builder import build_requirement, build_requirement_compact
from .data_loader import load_v3_data, convert_to_dspy_example, load_and_split, core_type_from_v3

__all__ = [
    "configure_lm", "configure_qwen_default",
    "BestKnownRegistry", "registry", "init_registry",
    "evaluate_algorithm_v3", "gepa_feedback_v3", "dspy_metric_v3", "compute_v3_score",
    "GenerateOptimizationAlgorithm", "ImproveAlgorithm", "ParseInstance",
    "AlgorithmGenerator", "AlgorithmModule", "default_parse_code",
    "build_requirement", "build_requirement_compact",
    "load_v3_data", "convert_to_dspy_example", "load_and_split", "core_type_from_v3",
]
