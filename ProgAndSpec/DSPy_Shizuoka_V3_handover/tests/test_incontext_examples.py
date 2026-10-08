from scripts.evaluate_solver_model import with_incontext_example


def test_example_code_is_appended_after_the_requirement():
    out = with_incontext_example("REQ", "def solve(instance):\n    return {}\n")
    assert out.startswith("REQ\n\n## Reference solution for another instance")
    assert "```python\ndef solve(instance):\n    return {}\n```" in out


def test_no_example_keeps_the_requirement():
    assert with_incontext_example("REQ", None) == "REQ"
