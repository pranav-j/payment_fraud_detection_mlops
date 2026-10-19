"""TEMPORARY: proves CI actually fails on a failing test. Removed in the next commit."""


def test_ci_must_go_red() -> None:
    assert 1 == 2, "deliberate failure to verify the CI gate"
