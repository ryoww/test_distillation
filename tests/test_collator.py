from distillation.collator import AssistantOnlyCollator


def test_collator_masks_padding() -> None:
    batch = AssistantOnlyCollator(pad_token_id=0, pad_to_multiple_of=4)(
        [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
            {"input_ids": [3, 4, 5], "attention_mask": [1, 1, 1], "labels": [-100, 4, 5]},
        ]
    )
    assert batch["input_ids"].shape == (2, 4)
    assert batch["labels"].tolist() == [[-100, 2, -100, -100], [-100, 4, 5, -100]]
