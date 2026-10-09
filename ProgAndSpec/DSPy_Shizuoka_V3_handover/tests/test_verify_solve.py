from scripts.verify_solve import _run_inline


def test_inline_run_returns_the_solution_of_an_allowed_program():
    ok, result = _run_inline("def solve(instance):\n    return {'x': instance['n'] + 1}\n", {"n": 1})
    assert ok and result == {"x": 2}


def test_inline_run_keeps_the_scorer_ban_on_file_access():
    ok, message = _run_inline("def solve(instance):\n    return open('x').read()\n", {})
    assert not ok and "open" in str(message)
