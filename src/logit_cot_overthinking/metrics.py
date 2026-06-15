from __future__ import annotations

import json
import math
from string import ascii_uppercase

import numpy as np
import pandas as pd

from .gemma import DECILES


OUTCOME_LABELS = {
    (True, True): "stable_correct",
    (False, True): "gained",
    (True, False): "lost",
    (False, False): "stable_wrong",
}


def build_trajectory_dataframe(records: list[dict[str, object]]) -> pd.DataFrame:
    if not records:
        raise ValueError("No trajectory records were provided")

    dataframe = pd.DataFrame(records)
    dataframe.sort_values(["position", "decile"], inplace=True)
    dataframe.reset_index(drop=True, inplace=True)

    for label in ascii_uppercase[:10]:
        dataframe[f"logprob_{label}"] = dataframe["choice_logprobs"].apply(
            lambda values, current=label: values.get(current, np.nan)
        )
        dataframe[f"prob_{label}"] = dataframe["choice_probabilities"].apply(
            lambda values, current=label: values.get(current, np.nan)
        )

    dataframe["choice_logprobs_json"] = dataframe["choice_logprobs"].apply(
        lambda values: json.dumps(values, sort_keys=True)
    )
    dataframe["choice_probabilities_json"] = dataframe[
        "choice_probabilities"
    ].apply(lambda values: json.dumps(values, sort_keys=True))

    group_keys = ["position", "question_id"]
    dataframe["previous_prediction"] = dataframe.groupby(group_keys)[
        "prediction"
    ].shift()
    dataframe["prediction_flip"] = (
        dataframe["previous_prediction"].notna()
        & (dataframe["prediction"] != dataframe["previous_prediction"])
    )

    final_predictions = (
        dataframe[dataframe["decile"] == 100]
        .set_index(group_keys)["prediction"]
        .to_dict()
    )
    dataframe["final_prediction"] = [
        final_predictions[(row.position, row.question_id)]
        for row in dataframe.itertuples()
    ]
    dataframe["final_answer_commitment"] = dataframe.apply(
        lambda row: row["choice_probabilities"][row["final_prediction"]],
        axis=1,
    )

    endpoints = dataframe[dataframe["decile"].isin([0, 100])].pivot(
        index=group_keys,
        columns="decile",
        values="correct",
    )
    outcome_by_question = {
        index: OUTCOME_LABELS[(bool(row[0]), bool(row[100]))]
        for index, row in endpoints.iterrows()
    }
    dataframe["outcome"] = [
        outcome_by_question[(row.position, row.question_id)]
        for row in dataframe.itertuples()
    ]
    return dataframe


def build_summary(
    dataframe: pd.DataFrame,
    trace_records: list[dict[str, object]],
    config: dict[str, object],
) -> dict[str, object]:
    expected_rows = len(trace_records) * len(DECILES)
    valid_predictions = dataframe.apply(
        lambda row: row["prediction"] in row["valid_labels"],
        axis=1,
    )
    probability_columns = [f"prob_{label}" for label in ascii_uppercase[:10]]
    finite_choice_probabilities = dataframe.apply(
        lambda row: all(
            math.isfinite(float(row[column]))
            for column in probability_columns
            if not pd.isna(row[column])
        ),
        axis=1,
    )
    full_trace_rows = dataframe[dataframe["decile"] == 100]

    checks = {
        "trace_count_matches_selection": len(trace_records)
        == int(config["num_rows"]),
        "trajectory_row_count": len(dataframe) == expected_rows,
        "finite_valid_choice_probabilities": bool(
            finite_choice_probabilities.all()
        ),
        "predictions_within_valid_choices": bool(valid_predictions.all()),
        "full_trace_coverage_at_decile_100": bool(
            len(full_trace_rows) == len(trace_records)
            and full_trace_rows["is_full_trace"].all()
        ),
    }

    accuracy = (
        dataframe.groupby("decile", sort=True)["correct"]
        .mean()
        .rename(lambda value: str(int(value)))
        .to_dict()
    )
    outcome_counts = (
        dataframe[dataframe["decile"] == 100]["outcome"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    trace_categories = pd.Series(
        [str(record.get("category", "")) for record in trace_records],
        dtype="string",
    )
    category_counts = trace_categories.value_counts().sort_index().to_dict()
    truncated_trace_count = sum(
        bool(record.get("truncated", False)) for record in trace_records
    )
    return {
        "config": config,
        "trace_count": len(trace_records),
        "trajectory_row_count": len(dataframe),
        "category_counts": category_counts,
        "truncated_trace_count": truncated_trace_count,
        "per_decile_accuracy": accuracy,
        "outcome_counts": outcome_counts,
        "validation": {
            "passed": all(checks.values()),
            "checks": checks,
        },
    }
