#!/usr/bin/env python3
"""Assistant-only LoRA/QLoRA sequence distillation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_paths import OUTPUT_DIR, configure_storage

configure_storage()

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

from distillation.collator import AssistantOnlyCollator
from distillation.semantic_labels import encode_example

DEFAULT_DATASET = "r0b0tlab/qwen3.8-max-glm5.2-distillation-51389"
DEFAULT_MODEL = "InternScience/Agents-A1-4B"
MODEL_REVISION = "945c40a4aa6f534d434a353207b8d42ecf7a5293"
DATASET_REVISION = "2415a156cfaec47cab320d559dc4df2df0dfb103"
LORA_SUFFIXES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_a",
    "in_proj_b",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--dataset-config", default="sft_final")
    parser.add_argument("--run-name")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int, default=256)
    parser.add_argument("--include-tools", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--full-finetune",
        action="store_true",
        help="Update all model parameters instead of adding LoRA adapters.",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() or None


def select_text_lora_targets(model: torch.nn.Module) -> list[str]:
    targets = []
    for name, module in model.named_modules():
        if (
            ("language_model.layers." in name or name.startswith("model.layers."))
            and isinstance(module, torch.nn.Linear)
            and name.rsplit(".", 1)[-1] in LORA_SUFFIXES
        ):
            targets.append(name)
    if not targets:
        raise RuntimeError("No language-model LoRA targets were found.")
    return targets


def load_model(args: argparse.Namespace, quantization: BitsAndBytesConfig | None):
    config = AutoConfig.from_pretrained(args.model, revision=args.model_revision)
    model_class = (
        AutoModelForImageTextToText
        if getattr(config, "vision_config", None) is not None
        else AutoModelForCausalLM
    )
    return model_class.from_pretrained(
        args.model,
        revision=args.model_revision,
        dtype=torch.bfloat16,
        quantization_config=quantization,
        device_map="auto" if args.load_in_4bit else None,
    )


def prepare_split(
    split: Dataset,
    tokenizer: Any,
    max_length: int,
    include_tools: bool,
    max_samples: int | None,
) -> tuple[Dataset, dict[str, int]]:
    if max_samples is not None:
        split = split.select(range(min(max_samples, len(split))))
    counts: Counter[str] = Counter()

    def encode(row: dict[str, Any]) -> dict[str, Any]:
        if row.get("tools") and not include_tools:
            counts["skipped_tools"] += 1
            return {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
                "mask_method": "",
                "valid": False,
                "invalid_reason": "tools_disabled",
            }
        result = encode_example(row, tokenizer, max_length)
        counts["valid" if result["valid"] else result["invalid_reason"].split(":")[0]] += 1
        return result

    encoded = split.map(
        encode,
        remove_columns=split.column_names,
        desc=f"Tokenizing {split.split or 'split'}",
        load_from_cache_file=False,
    )
    encoded = encoded.filter(lambda row: row["valid"], desc="Dropping unsupported/overlength rows")
    encoded = encoded.remove_columns(["valid", "invalid_reason", "mask_method"])
    if not len(encoded):
        raise RuntimeError("No trainable examples remain after preprocessing.")
    return encoded, dict(counts)


def main() -> None:
    args = parse_args()
    if args.full_finetune and args.load_in_4bit:
        raise ValueError("--full-finetune cannot be combined with --load-in-4bit")
    set_seed(args.seed)
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"agents-a1-4b-distill-{timestamp}"
    run_dir = (args.output_dir or OUTPUT_DIR / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"HF_HOME={os.environ['HF_HOME']}", flush=True)
    print(f"HF_DATASETS_CACHE={os.environ['HF_DATASETS_CACHE']}", flush=True)
    print(f"run_dir={run_dir}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        revision=args.dataset_revision,
    )
    train, train_stats = prepare_split(
        dataset["train"],
        tokenizer,
        args.max_length,
        args.include_tools,
        args.max_train_samples,
    )
    evaluation, eval_stats = prepare_split(
        dataset["validation"],
        tokenizer,
        args.max_length,
        args.include_tools,
        args.max_eval_samples,
    )
    (run_dir / "dataset_stats.json").write_text(
        json.dumps(
            {
                "train": train_stats,
                "validation": eval_stats,
                "train_rows_after_filter": len(train),
                "validation_rows_after_filter": len(evaluation),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    quantization = None
    if args.load_in_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = load_model(args, quantization)
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    targets = []
    if args.full_finetune:
        for parameter in model.parameters():
            parameter.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"trainable params: {trainable:,} || all params: {trainable:,} || trainable%: 100.0")
    else:
        targets = select_text_lora_targets(model)
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=targets,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        run_name=run_name,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="no",
        report_to=["tensorboard"],
        disable_tqdm=False,
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=evaluation,
        data_collator=AssistantOnlyCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    config = vars(args) | {
        "output_dir": str(run_dir),
        "git_revision": git_revision(),
        "visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "lora_target_count": len(targets),
        "lora_targets": targets,
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    adapter_dir = run_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    metrics = trainer.evaluate()
    (run_dir / "eval_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
