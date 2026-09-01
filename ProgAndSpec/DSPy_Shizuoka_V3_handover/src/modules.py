"""DSPy Module: AlgorithmGenerator。ChainOfThought で solve コードを生成する。

V2 v5: パースコードを学習可能なモジュールに。
- parse_instance: instanceのキーから変数抽出コードを生成
- generate: パースコード+問題文からアルゴリズムを生成
- improve: フィードバックで両方を改善
"""

from __future__ import annotations

import re

import dspy

from .signatures import GenerateOptimizationAlgorithm, ImproveAlgorithm, ParseInstance

_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_OPEN_FENCE = re.compile(r"```(?:python|py)?\s*\n?", re.IGNORECASE)


def strip_code_fence(text: str) -> str:
    """Extract Python code from LLM output. Handles ```python ... ``` and bare code."""
    text = text.strip()
    matches = _CODE_FENCE.findall(text)
    if matches:
        return max(matches, key=len).strip()
    # unclosed fence: take everything after the opening fence
    m = _OPEN_FENCE.search(text)
    if m:
        return text[m.end() :].strip().rstrip("`").strip()
    # bare 'python' language marker at start (no fence)
    if text.lower().startswith("python\n"):
        return text[7:].strip()
    return text.strip("`").strip()


def analyze_instance_structure(instance: dict) -> str:
    """Analyze instance structure and return a detailed description for the LLM.

    Returns a STRUCTURE ANALYSIS block that clearly describes:
    - Each key's type (list, dict, scalar, string)
    - For lists: count and element type (list of dicts vs list of primitives)
    - For dicts: nested keys
    - Sample values for context
    """
    lines = []
    lines.append("STRUCTURE ANALYSIS:")
    lines.append("This section describes the EXACT data structure of each field.")
    lines.append("Use this information to write correct access patterns.\n")

    for key in instance.keys():
        val = instance[key]
        if isinstance(val, list):
            if val and isinstance(val[0], dict):
                sub_keys = ", ".join(val[0].keys())
                sample_val = {k: val[0][k] for k in list(val[0].keys())[:3]}
                lines.append(f"- {key}: list[{len(val)}] of dicts with keys [{sub_keys}]")
                lines.append(f"  Sample element: {sample_val}")
                lines.append(f"  Access: for item in {key}: item['{list(val[0].keys())[0]}']")
            elif val:
                elem_type = type(val[0]).__name__
                lines.append(f"- {key}: list[{len(val)}] of {elem_type}s")
                lines.append(f"  Sample: {val[:3]}")
                lines.append(f"  Access: {key}[i] gives a {elem_type}")
            else:
                lines.append(f"- {key}: empty list")
        elif isinstance(val, dict):
            sub_keys = ", ".join(val.keys())
            lines.append(f"- {key}: dict with keys [{sub_keys}]")
            lines.append(f"  Access: {key}['{list(val.keys())[0]}']")
        elif isinstance(val, (int, float)):
            lines.append(f"- {key}: scalar ({val})")
            lines.append("  Access: use directly as number")
        elif isinstance(val, str):
            preview = val[:50] if len(val) > 50 else val
            lines.append(f'- {key}: string "{preview}"')
        else:
            lines.append(f"- {key}: {type(val).__name__}")

    return "\n".join(lines)


def default_parse_code(instance: dict) -> str:
    """Generate parse code from instance keys - covers ALL keys with type info and safe access.

    Returns parse code with:
    - Type-annotated variable extraction using .get() for safety
    - Safe access helpers (get_list, get_dict, get_scalar)
    - Structure hints for nested data
    """
    lines = []
    lines.append("# Safe access helpers")
    lines.append("def get_list(d, key, default=None):")
    lines.append("    val = d.get(key, default)")
    lines.append("    return val if isinstance(val, list) else default")
    lines.append("")
    lines.append("def get_dict(d, key, default=None):")
    lines.append("    val = d.get(key, default)")
    lines.append("    return val if isinstance(val, dict) else default")
    lines.append("")
    lines.append("def get_scalar(d, key, default=0):")
    lines.append("    val = d.get(key, default)")
    lines.append("    return val if isinstance(val, (int, float)) else default")
    lines.append("")
    lines.append("# Variable extraction with type info")

    for key in instance.keys():
        val = instance[key]
        # Sanitize key name for Python variable
        var_name = key.replace("-", "_").replace(" ", "_")
        if var_name and not (var_name[0].isalpha() or var_name[0] == "_"):
            var_name = "v_" + var_name

        if isinstance(val, list):
            if val and isinstance(val[0], dict):
                sub_keys = ", ".join(val[0].keys()) if val else "empty"
                comment = f"# list[{len(val)}] of dicts with keys [{sub_keys}]"
                lines.append(f"{var_name} = instance.get('{key}', [])  {comment}")
            else:
                comment = f"# list[{len(val)}]"
                lines.append(f"{var_name} = instance.get('{key}', [])  {comment}")
        elif isinstance(val, dict):
            sub_keys = ", ".join(val.keys())
            comment = f"# dict with keys [{sub_keys}]"
            lines.append(f"{var_name} = instance.get('{key}', {{}})  {comment}")
        elif isinstance(val, (int, float)):
            comment = f"# scalar ({val})"
            lines.append(f"{var_name} = instance.get('{key}', {val})  {comment}")
        elif isinstance(val, str):
            comment = "# string"
            lines.append(f"{var_name} = instance.get('{key}', '')  {comment}")
        else:
            comment = f"# {type(val).__name__}"
            lines.append(f"{var_name} = instance.get('{key}')  {comment}")
    return "\n".join(lines) if lines else "# No parse needed"


class AlgorithmGenerator(dspy.Module):
    """要件と問題タイプを受けて `solve(instance)` の Python コードを返す。

    v6: 2段階アプローチ。
    - Phase 1: parse_codeを生成して固定
    - Phase 2: 固定されたparse_codeでalgorithm_codeだけを学習
    """

    def __init__(self, parse_code_dict: dict = None):
        """
        Args:
            parse_code_dict: If provided, use these fixed parse codes per core_type.
                           Keys=core_type, Values=parse_code string.
                           Set in Phase 2 to freeze parsing.
        """
        super().__init__()
        self.parse_instance = dspy.ChainOfThought(ParseInstance)
        self.generate = dspy.ChainOfThought(GenerateOptimizationAlgorithm)
        self.improve = dspy.ChainOfThought(ImproveAlgorithm)
        self.parse_code_dict = parse_code_dict or {}

    def forward(self, requirement: str, core_type: str, **kwargs) -> dspy.Prediction:
        """Generate solve() code.

        NOTE: `instance` is NOT in with_inputs, so it's not passed during GEPA compile.
        Instance data is embedded in `requirement` via requirement_builder.
        We use a generic parse_code (LLM will figure out actual keys from requirement text).
        """
        # Static parse code — LLM sees actual instance keys/structure in requirement text
        parse_code = (
            "# Generic safe accessors for instance dict.\n"
            "# The requirement text contains the exact keys and structure.\n"
            "def get_list(d, key, default=None):\n"
            "    val = d.get(key, default if default is not None else [])\n"
            "    return val if isinstance(val, list) else (default if default is not None else [])\n"
            "def get_dict(d, key, default=None):\n"
            "    val = d.get(key, default if default is not None else {})\n"
            "    return val if isinstance(val, dict) else (default if default is not None else {})\n"
            "def get_scalar(d, key, default=0):\n"
            "    val = d.get(key, default)\n"
            "    return val if isinstance(val, (int, float)) else default\n"
        )
        # Allow override via parse_code_dict[core_type]
        if core_type in self.parse_code_dict:
            parse_code = self.parse_code_dict[core_type]

        # Generate algorithm with parse code
        out = self.generate(
            requirement=requirement,
            core_type=core_type,
            parse_code=parse_code,
        )
        code = strip_code_fence(out.algorithm_code)
        return dspy.Prediction(
            algorithm_code=code,
            parse_code=parse_code,
            rationale=getattr(out, "rationale", ""),
        )

    def improve_forward(
        self,
        original_code: str,
        parse_code: str,
        feedback: str,
        core_type: str,
        return_schema: str = "",
    ) -> dspy.Prediction:
        """フィードバックに基づいてアルゴリズムを改善（パースは固定）。

        return_schema は返却フィールド名と型だけを持ち、目的値は含まない。
        Why pass it here: 直す側は requirement を受け取らないため、これが無いと
        「どんな形で返すべきか」を知らないまま形式エラーを直すことになる。
        """
        out = self.improve(
            original_code=original_code,
            parse_code=parse_code,
            feedback=feedback,
            core_type=core_type,
            return_schema=return_schema,
        )
        improved_code = strip_code_fence(out.improved_code)
        return dspy.Prediction(
            algorithm_code=improved_code,
            parse_code=parse_code,  # Keep parse code fixed
            rationale=getattr(out, "rationale", ""),
        )


class AlgorithmModule(dspy.Module):
    """Alias for AlgorithmGenerator for backward compatibility."""

    def __init__(self):
        super().__init__()
        self._generator = AlgorithmGenerator()

    def forward(self, requirement: str, core_type: str) -> dspy.Prediction:
        return self._generator.forward(requirement, core_type)

    def improve_forward(self, original_code: str, feedback: str, core_type: str) -> dspy.Prediction:
        return self._generator.improve_forward(original_code, feedback, core_type)
