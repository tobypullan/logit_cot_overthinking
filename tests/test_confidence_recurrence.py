from __future__ import annotations

import numpy as np
import pandas as pd

from logit_cot_overthinking.confidence_recurrence import (
    build_attempt_recurrence_table,
    build_recurrence_contrasts,
    build_recurrence_summary,
)


def _attempts() -> pd.DataFrame:
    rows = [
        {
            "dataset": "mmlu_pro",
            "dataset_label": "MMLU-Pro",
            "seed": 0,
            "position": 0,
            "question_id": "loss-high-final",
            "baseline_cohort": "loss",
            "match_id": "a",
            "final_correct": False,
            "forced_completion": False,
            "final_normalized_prediction_probability": 0.95,
        },
        {
            "dataset": "mmlu_pro",
            "dataset_label": "MMLU-Pro",
            "seed": 0,
            "position": 1,
            "question_id": "loss-low-final",
            "baseline_cohort": "loss",
            "match_id": "b",
            "final_correct": False,
            "forced_completion": True,
            "final_normalized_prediction_probability": 0.65,
        },
        {
            "dataset": "mmlu_pro",
            "dataset_label": "MMLU-Pro",
            "seed": 0,
            "position": 2,
            "question_id": "final-correct",
            "baseline_cohort": "final_correct",
            "match_id": "a",
            "final_correct": True,
            "forced_completion": False,
            "final_normalized_prediction_probability": 0.98,
        },
        {
            "dataset": "mmlu_pro",
            "dataset_label": "MMLU-Pro",
            "seed": 0,
            "position": 3,
            "question_id": "stable-wrong",
            "baseline_cohort": "stable_wrong",
            "match_id": "a",
            "final_correct": False,
            "forced_completion": False,
            "final_normalized_prediction_probability": 0.99,
        },
    ]
    return pd.DataFrame(rows)


def _checkpoints() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "mmlu_pro",
                "seed": 0,
                "position": 0,
                "question_id": "loss-high-final",
                "decile": 50,
                "current_correct": True,
                "current_normalized_correct_probability": 0.80,
            },
            {
                "dataset": "mmlu_pro",
                "seed": 0,
                "position": 1,
                "question_id": "loss-low-final",
                "decile": 40,
                "current_correct": True,
                "current_normalized_correct_probability": 0.95,
            },
            {
                "dataset": "mmlu_pro",
                "seed": 0,
                "position": 2,
                "question_id": "final-correct",
                "decile": 30,
                "current_correct": True,
                "current_normalized_correct_probability": 0.92,
            },
            {
                "dataset": "mmlu_pro",
                "seed": 0,
                "position": 3,
                "question_id": "stable-wrong",
                "decile": 60,
                "current_correct": False,
                "current_normalized_correct_probability": 0.99,
            },
        ]
    )


def test_attempt_recurrence_applies_intermediate_and_final_thresholds() -> None:
    recurrence = build_attempt_recurrence_table(
        _attempts(),
        _checkpoints(),
        correct_thresholds=(0.7, 0.9),
        final_thresholds=(0.0, 0.7, 0.9),
    )
    indexed = recurrence.set_index(
        ["question_id", "correct_threshold", "final_threshold"]
    )

    high_final = indexed.loc[("loss-high-final", 0.7, 0.9)]
    assert bool(high_final["qualified_intermediate_correct"])
    assert bool(high_final["final_confidence_qualified"])
    assert bool(high_final["qualified_loss"])
    assert high_final["first_qualified_decile"] == 50

    high_correct_required = indexed.loc[("loss-high-final", 0.9, 0.9)]
    assert not bool(high_correct_required["qualified_intermediate_correct"])
    assert not bool(high_correct_required["qualified_loss"])

    low_final_no_requirement = indexed.loc[("loss-low-final", 0.9, 0.0)]
    low_final_required = indexed.loc[("loss-low-final", 0.9, 0.7)]
    assert bool(low_final_no_requirement["qualified_loss"])
    assert not bool(low_final_required["qualified_loss"])

    final_correct = indexed.loc[("final-correct", 0.9, 0.9)]
    assert bool(final_correct["qualified_intermediate_correct"])
    assert not bool(final_correct["qualified_loss"])

    stable_wrong = indexed.loc[("stable-wrong", 0.7, 0.9)]
    assert not bool(stable_wrong["qualified_intermediate_correct"])
    assert not bool(stable_wrong["qualified_loss"])


def test_recurrence_summary_aggregates_by_dataset_cohort_and_thresholds() -> None:
    recurrence = build_attempt_recurrence_table(
        _attempts(),
        _checkpoints(),
        correct_thresholds=(0.9,),
        final_thresholds=(0.0, 0.7),
    )
    summary = build_recurrence_summary(recurrence).set_index(
        ["baseline_cohort", "correct_threshold", "final_threshold"]
    )

    loss_no_final_filter = summary.loc[("loss", 0.9, 0.0)]
    assert loss_no_final_filter["attempt_count"] == 2
    assert loss_no_final_filter["qualified_loss_count"] == 1
    assert loss_no_final_filter["qualified_loss_rate"] == 0.5
    assert loss_no_final_filter["forced_completion_count"] == 1

    loss_final_filtered = summary.loc[("loss", 0.9, 0.7)]
    assert loss_final_filtered["qualified_loss_count"] == 0

    final_correct = summary.loc[("final_correct", 0.9, 0.7)]
    assert final_correct["final_accuracy"] == 1.0
    assert final_correct["qualified_loss_count"] == 0


def test_recurrence_contrasts_use_matched_triplet_differences() -> None:
    rows = []
    for match_id, values in {
        "a": {
            "loss": True,
            "final_correct": False,
            "stable_wrong": False,
        },
        "b": {
            "loss": False,
            "final_correct": False,
            "stable_wrong": True,
        },
    }.items():
        for cohort, qualified_loss in values.items():
            rows.append(
                {
                    "dataset": "mmlu_pro",
                    "match_id": match_id,
                    "baseline_cohort": cohort,
                    "correct_threshold": 0.7,
                    "final_threshold": 0.9,
                    "qualified_loss": qualified_loss,
                }
            )
    contrasts = build_recurrence_contrasts(
        pd.DataFrame(rows),
        bootstrap_iterations=0,
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
        == 0.0
    )
    assert np.isnan(contrasts.loc["loss_vs_final_correct", "ci_low"])
