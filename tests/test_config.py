import pytest

from logit_cot_overthinking.config import ProbeConfig


def test_step_two_configuration_is_valid() -> None:
    ProbeConfig(
        selection="balanced-categories",
        num_rows=1000,
        max_num_seqs=64,
    ).validate()


def test_configuration_rejects_invalid_selection() -> None:
    with pytest.raises(ValueError, match="selection"):
        ProbeConfig(selection="random").validate()


def test_configuration_rejects_invalid_max_num_seqs() -> None:
    with pytest.raises(ValueError, match="max_num_seqs"):
        ProbeConfig(max_num_seqs=0).validate()
