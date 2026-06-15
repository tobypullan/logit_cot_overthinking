from __future__ import annotations

import pandas as pd

from logit_cot_overthinking.extension_analysis import (
    summarize_extensions,
)


def test_summarize_extensions_tracks_correctness_transitions() -> None:
    rows = []
    for dataset, label, criterion in [
        (
            "gpqa_diamond",
            "GPQA Diamond",
            "normalized_reversal_candidate",
        ),
        ("mmlu_pro", "MMLU-Pro", "robust_loss"),
    ]:
        rows.extend(
            [
                {
                    "dataset": dataset,
                    "dataset_label": label,
                    "criterion": criterion,
                    "old_correct": False,
                    "new_correct": True,
                    "prediction_changed": True,
                    "forced_completion": False,
                    "ever_correct_final_wrong": False,
                    "criterion_reproduced": False,
                    "extended_trace_token_count": 20000,
                },
                {
                    "dataset": dataset,
                    "dataset_label": label,
                    "criterion": criterion,
                    "old_correct": True,
                    "new_correct": False,
                    "prediction_changed": True,
                    "forced_completion": True,
                    "ever_correct_final_wrong": True,
                    "criterion_reproduced": True,
                    "extended_trace_token_count": 32000,
                },
            ]
        )
    summary = summarize_extensions(pd.DataFrame(rows))

    gpqa = summary["datasets"]["gpqa_diamond"]
    assert gpqa["extended_trace_count"] == 2
    assert gpqa["wrong_to_correct_count"] == 1
    assert gpqa["correct_to_wrong_count"] == 1
    assert gpqa["forced_completion_count"] == 1
    assert gpqa["criterion_count"] == 1
