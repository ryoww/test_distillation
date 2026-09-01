"""thinking の有効・無効と出力枠の設定を検証する。"""

from __future__ import annotations

import os
from unittest import mock

from src.lm_config import (
    DEFAULT_MAX_TOKENS,
    LMConfig,
    _thinking_from_env,
    create_lm,
    qwen_configs_from_env,
)


def _captured(config: LMConfig) -> dict:
    """create_lm が dspy.LM へ渡した引数を取り出す。"""
    with mock.patch("src.lm_config.dspy.LM") as lm:
        create_lm(config)
    return lm.call_args.kwargs


def _config(**overrides) -> LMConfig:
    base = {"model": "Qwen/Qwen3.6-27B", "api_base": "http://127.0.0.1:7501/v1"}
    return LMConfig(**{**base, **overrides})


def test_default_sends_no_chat_template_kwargs():
    """未指定なら chat template の既定に委ねる。Qwen3系はそこで thinking が開く。"""
    assert "extra_body" not in _captured(_config())


def test_disabling_thinking_sends_the_flag():
    kwargs = _captured(_config(enable_thinking=False))
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_enabling_thinking_explicitly_also_sends_the_flag():
    kwargs = _captured(_config(enable_thinking=True))
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}


def test_default_output_budget_covers_thinking_and_the_answer():
    assert DEFAULT_MAX_TOKENS == 32768
    assert _captured(_config())["max_tokens"] == DEFAULT_MAX_TOKENS


def test_explicit_max_tokens_is_passed_through():
    assert _captured(_config(max_tokens=65536))["max_tokens"] == 65536


def test_env_thinking_flag_parsing():
    for raw, expected in (
        (None, None),
        ("", None),
        ("0", False),
        ("false", False),
        ("FALSE", False),
        ("no", False),
        ("off", False),
        ("1", True),
        ("true", True),
        ("yes", True),
    ):
        env = {} if raw is None else {"X_THINK": raw}
        with mock.patch.dict(os.environ, env, clear=True):
            assert _thinking_from_env("X_THINK") is expected, raw


def test_env_configures_both_endpoints():
    env = {
        "DSPY_GENERATION_ENABLE_THINKING": "0",
        "DSPY_REFLECTION_MAX_TOKENS": "40960",
    }
    with mock.patch.dict(os.environ, env, clear=True):
        generation, reflection = qwen_configs_from_env()
    assert generation.enable_thinking is False
    assert generation.max_tokens == DEFAULT_MAX_TOKENS
    assert reflection.enable_thinking is None
    assert reflection.max_tokens == 40960
