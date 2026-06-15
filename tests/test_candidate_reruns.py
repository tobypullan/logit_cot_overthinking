from pathlib import Path

import pandas as pd
import pytest

from logit_cot_overthinking.candidate_reruns import (
    CandidateSource,
    load_candidate_positions,
)
from logit_cot_overthinking.candidate_reruns_cli import _parse_seeds


def test_load_candidate_positions_filters_and_sorts(tmp_path: Path) -> None:
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    pd.DataFrame(
        [
            {"position": 9, "candidate": True},
            {"position": 2, "candidate": False},
            {"position": 4, "candidate": True},
        ]
    ).to_parquet(analysis_dir / "lost_cases.parquet", index=False)
    source = CandidateSource(
        name="test",
        analysis_dir=analysis_dir,
        dataset="test",
        dataset_format="mmlu-pro",
        flag_column="candidate",
    )

    assert load_candidate_positions(source) == [4, 9]


def test_load_candidate_positions_rejects_empty_selection(
    tmp_path: Path,
) -> None:
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    pd.DataFrame(
        [{"position": 1, "candidate": False}]
    ).to_parquet(analysis_dir / "lost_cases.parquet", index=False)
    source = CandidateSource(
        name="test",
        analysis_dir=analysis_dir,
        dataset="test",
        dataset_format="mmlu-pro",
        flag_column="candidate",
    )

    with pytest.raises(ValueError, match="No candidates"):
        load_candidate_positions(source)


def test_parse_seed_ranges() -> None:
    assert _parse_seeds("0-2,5,7-8") == [0, 1, 2, 5, 7, 8]
