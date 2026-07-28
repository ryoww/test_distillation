from jinja2 import TemplateError

from distillation.semantic_labels import encode_example


class TinyTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        if len(messages) == 1 and messages[0]["role"] == "system":
            raise TemplateError("system-only prefixes are unsupported")
        rendered = "".join(f"<{item['role']}>{item['content']}" for item in messages)
        return [ord(character) for character in rendered]


def test_structural_redaction_labels_only_trainable_assistant() -> None:
    row = {
        "messages": [
            {"role": "system", "content": "rules", "trainable": False},
            {"role": "user", "content": "question", "trainable": False},
            {"role": "assistant", "content": "answer", "trainable": True},
        ],
        "tools": [],
    }
    encoded = encode_example(row, TinyTokenizer(), max_length=100)
    supervised = "".join(
        chr(token) for token, label in zip(encoded["input_ids"], encoded["labels"]) if label != -100
    )
    assert encoded["valid"]
    assert encoded["mask_method"] == "structural-redaction"
    assert "answer" in supervised
    assert "question" not in supervised
