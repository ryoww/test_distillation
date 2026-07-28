"""Padding collator for pretokenized assistant-only examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class AssistantOnlyCollator:
    pad_token_id: int
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        target = max(len(item["input_ids"]) for item in features)
        target = ((target + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * (
            self.pad_to_multiple_of
        )
        ids, masks, labels = [], [], []
        for item in features:
            padding = target - len(item["input_ids"])
            ids.append(item["input_ids"] + [self.pad_token_id] * padding)
            masks.append(item["attention_mask"] + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)
        batch = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if torch.any((batch["labels"] != -100).sum(dim=1) == 0):
            raise AssertionError("batch contains an example with no supervised tokens")
        return batch
