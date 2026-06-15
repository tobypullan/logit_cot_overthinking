import math

from logit_cot_overthinking.metrics import (
    build_summary,
    build_trajectory_dataframe,
)


def make_record(question_id, position, answer, decile, prediction, probability):
    other = "B" if prediction == "A" else "A"
    probabilities = {prediction: probability, other: 0.1}
    return {
        "position": position,
        "question_id": question_id,
        "question": "Question",
        "options": ["one", "two"],
        "valid_labels": ["A", "B"],
        "answer": answer,
        "category": "test",
        "source": "test",
        "decile": decile,
        "prefix_token_count": decile,
        "trace_token_count": 100,
        "is_full_trace": decile == 100,
        "choice_logprobs": {
            label: math.log(value) for label, value in probabilities.items()
        },
        "choice_probabilities": probabilities,
        "choice_probability_mass": sum(probabilities.values()),
        "non_choice_probability": 1 - sum(probabilities.values()),
        "prediction": prediction,
        "prediction_probability": probability,
        "correct": prediction == answer,
        "sampled_token": prediction,
    }


def test_metrics_compute_flips_commitment_and_lost_outcome() -> None:
    records = []
    for decile in range(0, 101, 10):
        prediction = "A" if decile < 100 else "B"
        records.append(make_record("q1", 0, "A", decile, prediction, 0.7))

    dataframe = build_trajectory_dataframe(records)
    assert dataframe["prediction_flip"].sum() == 1
    assert set(dataframe["final_prediction"]) == {"B"}
    assert set(dataframe["outcome"]) == {"lost"}
    assert dataframe.iloc[0]["final_answer_commitment"] == 0.1

    summary = build_summary(
        dataframe,
        [{"question_id": "q1"}],
        {"num_rows": 1},
    )
    assert summary["trajectory_row_count"] == 11
    assert summary["outcome_counts"] == {"lost": 1}
    assert summary["category_counts"] == {"": 1}
    assert summary["truncated_trace_count"] == 0
    assert summary["validation"]["passed"] is True
