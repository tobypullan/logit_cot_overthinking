from __future__ import annotations

import numpy as np
import pandas as pd

from logit_cot_overthinking.matched_analysis import (
    auc_score,
    build_cohort_summary,
    build_matched_contrasts,
    grouped_cross_validated_predictions,
    repetition_score,
)


def test_auc_score_handles_ties() -> None:
    assert auc_score([0, 1, 0, 1], [0.1, 0.8, 0.2, 0.8]) == 1.0
    assert auc_score([0, 1], [0.5, 0.5]) == 0.5


def test_repetition_score_increases_for_repeated_ngrams() -> None:
    unique = repetition_score("one two three four five six seven")
    repeated = repetition_score(
        "one two three four one two three four one two three four"
    )
    assert repeated > unique


def test_grouped_predictions_are_finite() -> None:
    rows = []
    for position in range(20):
        for seed in range(3):
            rows.append(
                {
                    "dataset": "test",
                    "position": position,
                    "final_wrong": position % 2,
                    "signal": position % 2 + seed * 0.01,
                }
            )
    dataframe = pd.DataFrame(rows)
    predictions, coefficients = grouped_cross_validated_predictions(
        dataframe,
        ["signal"],
    )
    assert np.isfinite(predictions).all()
    assert len(coefficients) == 5
    assert auc_score(dataframe["final_wrong"], predictions) > 0.9


def test_cohort_summary_aggregates_loss_rates() -> None:
    attempts = pd.DataFrame(
        [
            {
                "dataset": "mmlu_pro",
                "dataset_label": "MMLU-Pro",
                "baseline_cohort": "loss",
                "broad_loss": True,
                "final_correct": False,
                "flip_count": 3,
                "trace_token_count": 100,
                "contains_simulated_retrieval_language": True,
                "contains_self_correction_language": False,
                "repetition_score": 0.2,
            },
            {
                "dataset": "mmlu_pro",
                "dataset_label": "MMLU-Pro",
                "baseline_cohort": "loss",
                "broad_loss": False,
                "final_correct": True,
                "flip_count": 1,
                "trace_token_count": 120,
                "contains_simulated_retrieval_language": False,
                "contains_self_correction_language": True,
                "repetition_score": 0.0,
            },
        ]
    )
    summary = build_cohort_summary(attempts).iloc[0]
    assert summary["broad_loss_rate"] == 0.5
    assert summary["final_accuracy"] == 0.5
    assert summary["mean_flips"] == 2


def test_matched_contrasts_use_triplet_risk_differences() -> None:
    attempts = []
    for match_id in ("a", "b"):
        for cohort, losses in (
            ("loss", [True, True]),
            ("final_correct", [False, True]),
            ("stable_wrong", [False, False]),
        ):
            for broad_loss in losses:
                attempts.append(
                    {
                        "dataset": "mmlu_pro",
                        "match_id": match_id,
                        "baseline_cohort": cohort,
                        "broad_loss": broad_loss,
                    }
                )
    contrasts = build_matched_contrasts(
        pd.DataFrame(attempts),
        bootstrap_iterations=100,
    ).set_index("comparison")
    assert (
        contrasts.loc[
            "loss_vs_final_correct",
            "risk_difference",
        ]
        == 0.5
    )
    assert (
        contrasts.loc[
            "loss_vs_stable_wrong",
            "risk_difference",
        ]
        == 1.0
    )
