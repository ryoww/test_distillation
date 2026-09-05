#!/usr/bin/env python3
"""指示文の変種を保存済み DSPy プログラムとして書き出す。

- compact: 共通原則だけの短い指示文。demo は Phase E と同じ 2 件を付けるので、
  Phase E との差は指示文の本文だけになる。
- modular: compact と同じ指示文に、問題の分野ごとの補足を実行時に選んで付ける。
  補足は `<program>.supplements.json` に置き、train_gepa_v3.py --eval-only が読む。
- before_demos: 初期指示文 + Phase E の demo 2 件。demo の効果だけを測る対照条件。

どちらも parse_instance / improve の指示文は初期のまま（Phase E も同じ）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.modules import AlgorithmGenerator, supplements_path_for

PROMPTS_DIR = BASE_DIR / "prompts"
PHASE_E_PATH = BASE_DIR / "compiled_program_v3_gepa_phaseE.json"
COMPACT_TEXT = PROMPTS_DIR / "compact_generate_instructions.md"
SUPPLEMENTS = PROMPTS_DIR / "domain_supplements.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-e", type=Path, default=PHASE_E_PATH)
    parser.add_argument("--instructions", type=Path, default=COMPACT_TEXT)
    parser.add_argument("--supplements", type=Path, default=SUPPLEMENTS)
    parser.add_argument("--output-dir", type=Path, default=PROMPTS_DIR)
    return parser.parse_args()


def phase_e_demos(path: Path) -> list[dict]:
    program = json.loads(path.read_text(encoding="utf-8"))
    return program["generate.predict"]["demos"]


def build_variant(instructions: str, demos: list[dict], output: Path) -> Path:
    """generate の指示文を差し替え、demo を付けた AlgorithmGenerator を保存する。"""
    program = AlgorithmGenerator()
    program.generate.predict.signature = program.generate.predict.signature.with_instructions(
        instructions
    )
    program.generate.predict.demos = list(demos)
    output.parent.mkdir(parents=True, exist_ok=True)
    program.save(str(output))
    return output


def main() -> None:
    args = parse_args()
    instructions = args.instructions.read_text(encoding="utf-8").strip()
    demos = phase_e_demos(args.phase_e)
    supplements = json.loads(args.supplements.read_text(encoding="utf-8"))

    # 対照条件: 初期指示文に Phase E と同じ demo だけを付ける。compact − before の差が
    # 指示文の本文によるものか demo によるものかを切り分けるために要る。
    original = AlgorithmGenerator().generate.predict.signature.instructions
    before_demos = build_variant(
        original, demos, args.output_dir / "compiled_program_v3_before_demos.json"
    )
    compact = build_variant(
        instructions, demos, args.output_dir / "compiled_program_v3_compact.json"
    )
    modular = build_variant(
        instructions, demos, args.output_dir / "compiled_program_v3_modular.json"
    )
    sidecar = supplements_path_for(modular)
    sidecar.write_text(
        json.dumps(supplements, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for path in (before_demos, compact, modular, sidecar):
        print(path.relative_to(BASE_DIR), path.stat().st_size, "bytes")
    print(
        f"instructions: {len(instructions)} chars; demos: {len(demos)}; domains: {len(supplements)}"
    )


if __name__ == "__main__":
    main()
