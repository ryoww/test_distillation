#!/usr/bin/env python3
"""Compare base / LoRA / full-FT Agents-A1-4B on the same prompt.

Runs greedy decoding sequentially through all three variants to hold GPU
memory below one H200. Prints the per-variant assistant output; optionally
writes the comparison to JSON for downstream reporting.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_paths import configure_storage

configure_storage()

import torch
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)

BASE_ID = "InternScience/Agents-A1-4B"
BASE_REV = "945c40a4aa6f534d434a353207b8d42ecf7a5293"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-adapter", type=Path, required=True)
    parser.add_argument("--fullft-model", type=Path, required=True)
    parser.add_argument(
        "--variants",
        default="base,lora,fullft",
        help="Comma-separated subset of {base,lora,fullft}.",
    )
    parser.add_argument("--system", default=None)
    parser.add_argument("--user", default=None)
    parser.add_argument(
        "--test-index",
        type=int,
        default=None,
        help="Instead of --user/--system, pull the row at this index of "
        "r0b0tlab/qwen3.8-max-glm5.2-distillation-51389 test split.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def load_model(path_or_id: str, revision: str | None = None):
    config = AutoConfig.from_pretrained(
        path_or_id, **({"revision": revision} if revision else {})
    )
    cls = (
        AutoModelForImageTextToText
        if getattr(config, "vision_config", None) is not None
        else AutoModelForCausalLM
    )
    kw = {"dtype": torch.bfloat16, "device_map": "auto"}
    if revision:
        kw["revision"] = revision
    return cls.from_pretrained(path_or_id, **kw).eval()


def free(model) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def generate(model, tokenizer, system: str | None, user: str, max_new_tokens: int) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(model.device)
    with torch.inference_mode():
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(gen[0, input_ids.shape[-1] :], skip_special_tokens=True)


def load_test_row(index: int) -> tuple[str | None, str, str]:
    from datasets import load_dataset

    ds = load_dataset(
        "r0b0tlab/qwen3.8-max-glm5.2-distillation-51389",
        "sft_final",
        revision="2415a156cfaec47cab320d559dc4df2df0dfb103",
    )
    row = ds["test"][index]
    messages = row["messages"]
    if isinstance(messages, str):
        messages = json.loads(messages)
    system = next((m["content"] for m in messages if m["role"] == "system"), None)
    user = next((m["content"] for m in messages if m["role"] == "user"), "")
    assistant = next(
        (m["content"] for m in messages if m["role"] == "assistant"), ""
    )
    return system, user, assistant


def main() -> None:
    args = parse_args()
    if args.test_index is not None:
        system, user, teacher = load_test_row(args.test_index)
    else:
        if not args.user:
            raise SystemExit("Provide --user (with optional --system) or --test-index.")
        system, user, teacher = args.system, args.user, None
    print(f"SYSTEM: {system}\nUSER: {user}\n", flush=True)
    if teacher:
        print(f"TEACHER:\n{teacher}\n", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_ID, revision=BASE_REV)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    outputs: dict[str, str] = {}
    for tag in args.variants.split(","):
        tag = tag.strip()
        if not tag:
            continue
        t0 = time.time()
        if tag == "base":
            model = load_model(BASE_ID, BASE_REV)
        elif tag == "lora":
            model = PeftModel.from_pretrained(
                load_model(BASE_ID, BASE_REV), str(args.lora_adapter)
            )
        elif tag == "fullft":
            model = load_model(str(args.fullft_model))
        else:
            raise SystemExit(f"Unknown variant: {tag}")
        text = generate(model, tokenizer, system, user, args.max_new_tokens)
        outputs[tag] = text
        print(
            f"===== {tag.upper()} (len={len(text)}, {time.time() - t0:.1f}s) =====\n{text}\n",
            flush=True,
        )
        free(model)

    if args.output_json:
        args.output_json.write_text(
            json.dumps(
                {
                    "system": system,
                    "user": user,
                    "teacher": teacher,
                    "max_new_tokens": args.max_new_tokens,
                    "variants": outputs,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
