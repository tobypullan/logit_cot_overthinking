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


def test_explicit_index_configuration_is_valid() -> None:
    ProbeConfig(
        selection="indices",
        row_indices=(3, 9),
        num_rows=2,
    ).validate()


def test_explicit_index_configuration_rejects_count_mismatch() -> None:
    with pytest.raises(ValueError, match="num_rows"):
        ProbeConfig(
            selection="indices",
            row_indices=(3, 9),
            num_rows=3,
        ).validate()


def test_gpqa_configuration_is_valid() -> None:
    ProbeConfig(
        dataset="fingertap/GPQA-Diamond",
        dataset_format="gpqa-diamond",
        split="test",
    ).validate()


def test_configuration_rejects_invalid_dataset_format() -> None:
    with pytest.raises(ValueError, match="dataset_format"):
        ProbeConfig(dataset_format="unknown").validate()


def test_configuration_rejects_invalid_max_num_seqs() -> None:
    with pytest.raises(ValueError, match="max_num_seqs"):
        ProbeConfig(max_num_seqs=0).validate()
