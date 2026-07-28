"""Tokenizer-neutral assistant-only labeling.

Adapted from the reference adapter distributed with
r0b0tlab/qwen3.8-max-glm5.2-distillation-51389 at revision
2415a156cfaec47cab320d559dc4df2df0dfb103.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Mapping
from typing import Any

from jinja2 import TemplateError


def _tools(value: Any) -> list[dict[str, Any]] | None:
    if value in (None, "", []):
        return None
    tools = json.loads(value) if isinstance(value, str) else json.loads(json.dumps(value))
    for tool in tools:
        function = tool.get("function") or {}
        encoded = function.pop("parameters_json", None)
        if encoded is not None:
            function["parameters"] = json.loads(encoded)
    return tools


def _messages(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value)
    return json.loads(json.dumps(value))


def _ids(rendered: Any) -> list[int]:
    if isinstance(rendered, Mapping):
        rendered = rendered["input_ids"]
    if rendered and isinstance(rendered[0], list):
        if len(rendered) != 1:
            raise ValueError("unexpected batched chat-template output")
        rendered = rendered[0]
    return list(rendered)


def _render(tokenizer: Any, messages: list[dict[str, Any]], tools: Any) -> list[int]:
    kwargs = {"tokenize": True, "add_generation_prompt": False}
    if tools:
        kwargs["tools"] = tools
    return _ids(tokenizer.apply_chat_template(messages, **kwargs))


def _is_trainable(message: dict[str, Any]) -> bool:
    return message.get("role") == "assistant" and message.get("trainable", True)


def _prefix_labels(tokenizer: Any, messages: list[dict[str, Any]], tools: Any):
    previous: list[int] = []
    labels: list[int] = []
    for index, message in enumerate(messages):
        current = _render(tokenizer, messages[: index + 1], tools)
        if current[: len(previous)] != previous:
            raise ValueError(f"chat template is not prefix preserving at message {index}")
        suffix = current[len(previous) :]
        labels.extend(suffix if _is_trainable(message) else [-100] * len(suffix))
        previous = current
    return previous, labels, "prefix-differential"


def _redaction_labels(tokenizer: Any, messages: list[dict[str, Any]], tools: Any):
    full = _render(tokenizer, messages, tools)
    mask = [False] * len(full)
    assistant_indices = [index for index, message in enumerate(messages) if _is_trainable(message)]
    for index in assistant_indices:
        redacted = json.loads(json.dumps(messages))
        redacted[index]["content"] = f"__REDACTED_ASSISTANT_{index}__"
        redacted[index].pop("reasoning_content", None)
        redacted[index].pop("tool_calls", None)
        other = _render(tokenizer, redacted, tools)
        opcodes = difflib.SequenceMatcher(a=full, b=other, autojunk=False).get_opcodes()
        for tag, start, end, _, _ in opcodes:
            if tag in {"replace", "delete"}:
                for position in range(start, end):
                    mask[position] = True
    labels = [token if marked else -100 for token, marked in zip(full, mask, strict=True)]
    return full, labels, "structural-redaction"


def encode_example(example: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, Any]:
    messages = _messages(example["messages"])
    tools = _tools(example.get("tools"))
    try:
        input_ids, labels, method = _prefix_labels(tokenizer, messages, tools)
    except (ValueError, TemplateError):
        input_ids, labels, method = _redaction_labels(tokenizer, messages, tools)

    if len(input_ids) > max_length:
        return {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "mask_method": method,
            "valid": False,
            "invalid_reason": f"overlength:{len(input_ids)}",
        }
    if len(input_ids) != len(labels):
        raise AssertionError("input/label length mismatch")
    if not any(label != -100 for label in labels):
        return {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "mask_method": method,
            "valid": False,
            "invalid_reason": "no_supervised_tokens",
        }
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "mask_method": method,
        "valid": True,
        "invalid_reason": "",
    }
