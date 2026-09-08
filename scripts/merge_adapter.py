#!/usr/bin/env python3
"""LoRA adapter をベースモデルに焼き込み、vLLM がそのまま読める通常のモデルとして保存する。

vLLM の LoRA 配信は対象モジュールの種類に制約があり、Agents-A1-4B の linear attention
（in_proj_*）を含む adapter が載る保証がない。マージ済みの重みなら通常モデルと同じ経路で
配信できる。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from project_paths import configure_storage

configure_storage()

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)

DEFAULT_MODEL = "InternScience/Agents-A1-4B"
MODEL_REVISION = "945c40a4aa6f534d434a353207b8d42ecf7a5293"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AutoConfig.from_pretrained(args.model, revision=args.model_revision)
    model_class = (
        AutoModelForImageTextToText
        if getattr(config, "vision_config", None) is not None
        else AutoModelForCausalLM
    )
    base = model_class.from_pretrained(
        args.model, revision=args.model_revision, dtype=torch.bfloat16
    )
    merged = PeftModel.from_pretrained(base, str(args.adapter)).merge_and_unload()
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output, safe_serialization=True)
    AutoTokenizer.from_pretrained(str(args.adapter)).save_pretrained(args.output)
    # 前処理設定やチャットテンプレートはベースのスナップショットから引き継ぐ。
    # save_pretrained は重みと config しか書かないので、無いと vLLM が起動時に見失う。
    source = Path(
        snapshot_download(args.model, revision=args.model_revision, local_files_only=True)
    )
    for name in (
        "generation_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "chat_template.json",
        "chat_template.jinja",
    ):
        if (source / name).is_file() and not (args.output / name).is_file():
            (args.output / name).write_bytes((source / name).read_bytes())
    print(f"merged model saved to {args.output}")


if __name__ == "__main__":
    main()
