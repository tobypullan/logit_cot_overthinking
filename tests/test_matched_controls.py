from __future__ import annotations

import pandas as pd

from logit_cot_overthinking.matched_controls import (
    select_matched_cohorts,
)


def test_select_matched_cohorts_balances_and_matches_categories() -> None:
    rows = []
    position = 0
    for category in ("a", "b"):
        for cohort, count in (
            ("loss", 3),
            ("final_correct", 5),
            ("stable_wrong", 4),
        ):
            for offset in range(count):
                rows.append(
                    {
                        "position": position,
                        "question_id": str(position),
                        "category": category,
                        "source": "test",
                        "answer": "A",
                        "trace_token_count": 100 * (offset + 1),
                        "baseline_cohort": cohort,
                    }
                )
                position += 1

    selected = select_matched_cohorts(
        pd.DataFrame(rows),
        per_cohort=4,
    )

    assert selected["baseline_cohort"].value_counts().to_dict() == {
        "loss": 4,
        "final_correct": 4,
        "stable_wrong": 4,
    }
    assert selected.groupby("match_id")["category"].nunique().eq(1).all()
    assert selected.groupby("match_id").size().eq(3).all()
    assert selected["position"].is_unique


def test_select_matched_cohorts_rejects_insufficient_triplets() -> None:
    table = pd.DataFrame(
        [
            {
                "position": index,
                "question_id": str(index),
                "category": "only",
                "source": "test",
                "answer": "A",
                "trace_token_count": 100,
                "baseline_cohort": cohort,
            }
            for index, cohort in enumerate(
                ["loss", "final_correct", "stable_wrong"]
            )
        ]
    )

    try:
        select_matched_cohorts(table, per_cohort=2)
    except ValueError as error:
        assert "fewer than requested" in str(error)
    else:
        raise AssertionError("Expected insufficient matching to fail")
