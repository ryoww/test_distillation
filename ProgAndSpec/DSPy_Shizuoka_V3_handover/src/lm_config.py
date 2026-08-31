"""LM 設定ヘルパ (vllm/openai-compat 対応)。

V1のlm_configをV2に移植。
- V1: openai/ollama/dummy
- V2: vllm (Qwen3.6-27B at http://10.81.105.102:11434/v1) + dummy
"""
from __future__ import annotations

import os

import dspy


def configure_lm(lm_cfg: dict) -> bool:
    """Return True if a real LM was configured, False for dummy fallback."""
    provider = lm_cfg.get("provider", "dummy")
    if provider == "dummy":
        return False
    if provider in ("vllm", "openai"):
        # vllmはOpenAI互換APIなので同じ扱い
        api_base = lm_cfg.get("api_base", "http://10.81.105.102:11434/v1")
        api_key = lm_cfg.get("api_key", "none")  # vllmはキー不要の場合も
        key_env = lm_cfg.get("api_key_env")
        if key_env:
            api_key = os.environ.get(key_env, api_key)

        lm = dspy.LM(
            model=f"openai/{lm_cfg['model']}",
            api_base=api_base,
            api_key=api_key,
            temperature=lm_cfg.get("temperature", 0.2),
            max_tokens=lm_cfg.get("max_tokens", 4096),
            timeout=lm_cfg.get("timeout", 1800),
        )
        dspy.settings.configure(lm=lm)
        return True
    print(f"[warn] unknown provider {provider}. Falling back to dummy.")
    return False


def configure_qwen_default() -> bool:
    """Qwen3.6-27B at default endpointで設定。"""
    return configure_lm({
        "provider": "vllm",
        "model": "Qwen3.6-27B",
        "api_base": "http://10.81.105.102:11434/v1",
        "temperature": 0.2,
        "max_tokens": 8192,
        "timeout": 1800,
    })
