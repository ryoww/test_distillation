import json

from scripts.merge_sft_datasets import merge, parse_input


def _write(directory, split, rows):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def _row(text, **extra):
    return {"messages": [{"role": "user", "content": text}], "tools": None, **extra}


def test_merge_repeats_train_only_and_keeps_training_columns(tmp_path):
    _write(tmp_path / "a", "train", [_row("a1", template_id=1)])
    _write(tmp_path / "a", "validation", [_row("av")])
    _write(tmp_path / "b", "train", [_row("b1", kind="x", gap=0.0)])
    _write(tmp_path / "b", "validation", [_row("bv")])

    merged = merge([("a", tmp_path / "a", 1), ("b", tmp_path / "b", 3)], seed=0)

    assert sorted(r["source"] for r in merged["train"]) == ["a", "b", "b", "b"]
    assert sorted(r["source"] for r in merged["validation"]) == ["a", "b"]
    assert all(set(r) == {"messages", "tools", "source"} for r in merged["train"])


def test_parse_input_defaults_repeat_to_one():
    assert parse_input("hard=data/x*2")[2] == 2
    assert parse_input("tpl=data/sft")[2] == 1


def test_merge_replaces_system_message_when_given(tmp_path):
    row = {
        "messages": [
            {"role": "system", "content": "OLD"},
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ],
        "tools": None,
    }
    _write(tmp_path / "a", "train", [row])
    _write(tmp_path / "a", "validation", [row])

    merged = merge([("a", tmp_path / "a", 1)], seed=0, system="NEW")

    assert [m["content"] for m in merged["train"][0]["messages"]] == ["NEW", "Q", "A"]


def test_merge_drops_excluded_kinds_but_keeps_rows_without_kind(tmp_path):
    _write(tmp_path / "a", "train", [_row("tpl", template_id=1)])
    _write(tmp_path / "a", "validation", [])
    _write(tmp_path / "b", "train", [_row("keep", kind="clsp"), _row("drop", kind="fjsp")])
    _write(tmp_path / "b", "validation", [_row("vdrop", kind="fjsp")])

    merged = merge(
        [("a", tmp_path / "a", 1), ("b", tmp_path / "b", 1)], seed=0, exclude_kinds=frozenset({"fjsp"})
    )

    texts = sorted(r["messages"][0]["content"] for r in merged["train"])
    assert texts == ["keep", "tpl"]
    assert merged["validation"] == []
