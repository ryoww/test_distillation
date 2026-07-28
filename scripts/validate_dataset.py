#!/usr/bin/env python3
"""Validate target-tokenizer rendering without loading model weights."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_paths import configure_storage

configure_storage()

from datasets import load_dataset
from transformers import AutoTokenizer

from distillation.semantic_labels import encode_example

DEFAULT_DATASET = "r0b0tlab/qwen3.8-max-glm5.2-distillation-51389"
DEFAULT_MODEL = "InternScience/Agents-A1-4B"
MODEL_REVISION = "945c40a4aa6f534d434a353207b8d42ecf7a5293"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default="sft_final")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--include-tools", action="store_true")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    rows = load_dataset(args.dataset, args.config, split=f"{args.split}[:{args.limit}]")
    counts: Counter[str] = Counter()
    examples = []
    for row in rows:
        if row.get("tools") and not args.include_tools:
            counts["skipped_tools"] += 1
            continue
        encoded = encode_example(row, tokenizer, args.max_length)
        key = "valid" if encoded["valid"] else encoded["invalid_reason"].split(":")[0]
        counts[key] += 1
        if encoded["valid"] and len(examples) < 3:
            examples.append(
                {
                    "id": row["id"],
                    "tokens": len(encoded["input_ids"]),
                    "supervised_tokens": sum(x != -100 for x in encoded["labels"]),
                    "mask_method": encoded["mask_method"],
                }
            )
    print(json.dumps({"counts": counts, "examples": examples}, indent=2, ensure_ascii=False))
    if counts["valid"] == 0:
        raise SystemExit("No valid examples were produced.")


if __name__ == "__main__":
    main()
