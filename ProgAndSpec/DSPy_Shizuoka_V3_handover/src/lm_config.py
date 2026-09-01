"""OpenAI-compatible DSPy LM configuration for local Qwen servers."""

from __future__ import annotations

import os
from dataclasses import dataclass

import dspy

DEFAULT_GENERATION_MODEL = "Qwen/Qwen3.6-27B"
DEFAULT_REFLECTION_MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_GENERATION_API_BASE = "http://127.0.0.1:7501/v1"
DEFAULT_REFLECTION_API_BASE = "http://127.0.0.1:7502/v1"


@dataclass(frozen=True)
class LMConfig:
    model: str
    api_base: str
    api_key: str = "local"
    temperature: float = 0.2
    max_tokens: int = 8192
    timeout: int = 1800


def _dspy_model_name(model: str) -> str:
    return model if model.startswith("openai/") else f"openai/{model}"


def create_lm(config: LMConfig) -> dspy.LM:
    """Build one DSPy client without changing global DSPy settings."""
    return dspy.LM(
        model=_dspy_model_name(config.model),
        api_base=config.api_base,
        api_key=config.api_key,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
    )


def configure_lm(lm_cfg: dict) -> bool:
    """Configure a single LM for compatibility with earlier entry points."""
    provider = lm_cfg.get("provider", "dummy")
    if provider == "dummy":
        return False
    if provider not in {"vllm", "openai"}:
        return False
    api_key = lm_cfg.get("api_key", "local")
    if key_env := lm_cfg.get("api_key_env"):
        api_key = os.environ.get(key_env, api_key)
    lm = create_lm(
        LMConfig(
            model=lm_cfg["model"],
            api_base=lm_cfg.get("api_base", DEFAULT_GENERATION_API_BASE),
            api_key=api_key,
            temperature=lm_cfg.get("temperature", 0.2),
            max_tokens=lm_cfg.get("max_tokens", 8192),
            timeout=lm_cfg.get("timeout", 1800),
        )
    )
    dspy.settings.configure(lm=lm)
    return True


def qwen_configs_from_env() -> tuple[LMConfig, LMConfig]:
    """Read generation and GEPA-reflection server settings from the environment."""
    generation = LMConfig(
        model=os.getenv("DSPY_GENERATION_MODEL", DEFAULT_GENERATION_MODEL),
        api_base=os.getenv("DSPY_GENERATION_API_BASE", DEFAULT_GENERATION_API_BASE),
        api_key=os.getenv("DSPY_GENERATION_API_KEY", "local"),
        temperature=float(os.getenv("DSPY_GENERATION_TEMPERATURE", "0.2")),
        max_tokens=int(os.getenv("DSPY_GENERATION_MAX_TOKENS", "8192")),
        timeout=int(os.getenv("DSPY_GENERATION_TIMEOUT", "1800")),
    )
    reflection = LMConfig(
        model=os.getenv("DSPY_REFLECTION_MODEL", DEFAULT_REFLECTION_MODEL),
        api_base=os.getenv("DSPY_REFLECTION_API_BASE", DEFAULT_REFLECTION_API_BASE),
        api_key=os.getenv("DSPY_REFLECTION_API_KEY", "local"),
        temperature=float(os.getenv("DSPY_REFLECTION_TEMPERATURE", "0.2")),
        max_tokens=int(os.getenv("DSPY_REFLECTION_MAX_TOKENS", "8192")),
        timeout=int(os.getenv("DSPY_REFLECTION_TIMEOUT", "1800")),
    )
    return generation, reflection


def configure_qwen_pair(
    generation: LMConfig | None = None,
    reflection: LMConfig | None = None,
) -> tuple[dspy.LM, dspy.LM]:
    """Configure Qwen3.6 for generation and return Qwen3.8 for GEPA reflection."""
    env_generation, env_reflection = qwen_configs_from_env()
    generation_lm = create_lm(generation or env_generation)
    reflection_lm = create_lm(reflection or env_reflection)
    dspy.settings.configure(lm=generation_lm)
    return generation_lm, reflection_lm


def configure_qwen_default() -> bool:
    """Configure the generation LM using the local Qwen defaults."""
    generation, _ = qwen_configs_from_env()
    dspy.settings.configure(lm=create_lm(generation))
    return True
