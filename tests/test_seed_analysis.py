from __future__ import annotations

import pandas as pd

from logit_cot_overthinking.seed_analysis import (
    build_candidate_summary,
    build_seed_summary,
)


def _attempts() -> pd.DataFrame:
    rows = []
    for dataset, criterion in [
        ("mmlu_pro", "robust_loss"),
        ("gpqa_diamond", "normalized_reversal_candidate"),
    ]:
        for seed in range(10):
            reproduced = seed < (6 if dataset == "mmlu_pro" else 2)
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": (
                        "MMLU-Pro"
                        if dataset == "mmlu_pro"
                        else "GPQA Diamond"
                    ),
                    "criterion": criterion,
                    "seed": seed,
                    "position": 1,
                    "question_id": f"{dataset}-1",
                    "category": "test",
                    "answer": "A",
                    "final_correct": not reproduced,
                    "generated_correct": not reproduced,
                    "outcome": (
                        "lost" if reproduced else "gained"
                    ),
                    "ever_correct_final_wrong": reproduced,
                    "criterion_reproduced": reproduced,
                    "forced_completion": seed == 9,
                    "trace_token_count": 100 + seed,
                    "flip_count": seed,
                }
            )
    return pd.DataFrame(rows)


def test_candidate_summary_counts_recurrence() -> None:
    candidates = build_candidate_summary(_attempts())

    mmlu = candidates[candidates["dataset"] == "mmlu_pro"].iloc[0]
    gpqa = candidates[
        candidates["dataset"] == "gpqa_diamond"
    ].iloc[0]
    assert mmlu["criterion_recurrence_count"] == 6
    assert mmlu["criterion_recurrence_rate"] == 0.6
    assert gpqa["criterion_recurrence_count"] == 2
    assert gpqa["forced_completion_count"] == 1


def test_seed_summary_separates_forced_attempts() -> None:
    attempts = _attempts()
    candidates = build_candidate_summary(attempts)
    summary = build_seed_summary(attempts, candidates)

    mmlu = summary["datasets"]["mmlu_pro"]
    assert mmlu["attempt_count"] == 10
    assert mmlu["natural_attempt_count"] == 9
    assert mmlu["criterion_recurrence_count"] == 6
    assert mmlu["candidates_with_majority_recurrence"] == 1
