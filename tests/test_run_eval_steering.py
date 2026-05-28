import pytest

from scripts.run_eval import _parse_float_list, _parse_steering_replacements


def test_parse_steering_replacements_json() -> None:
    assert _parse_steering_replacements('{"reward": "penalty", "rabbit": "mouse"}') == {
        "reward": "penalty",
        "rabbit": "mouse",
    }


def test_parse_steering_replacements_pairs() -> None:
    assert _parse_steering_replacements("reward=penalty, rabbit=mouse") == {
        "reward": "penalty",
        "rabbit": "mouse",
    }


def test_parse_steering_replacements_rejects_bad_value() -> None:
    with pytest.raises(ValueError):
        _parse_steering_replacements("reward")


def test_parse_float_list_uses_default() -> None:
    assert _parse_float_list(None, [0.1, 0.2]) == [0.1, 0.2]


def test_parse_float_list_parses_comma_values() -> None:
    assert _parse_float_list("0.05,0.1, 0.2", []) == [0.05, 0.1, 0.2]
