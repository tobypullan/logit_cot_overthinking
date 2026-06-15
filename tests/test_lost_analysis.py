from __future__ import annotations

from pathlib import Path

import pandas as pd

from logit_cot_overthinking.lost_analysis import (
    build_lost_case_table,
    build_lost_summary,
    plot_loss_timing,
)


def _question(
    question_id: str,
    position: int,
    predictions: list[str],
    answer: str = "A",
    final_mass: float = 0.9,
) -> list[dict[str, object]]:
    rows = []
    for decile, prediction in zip(range(0, 101, 10), predictions):
        mass = final_mass if decile == 100 else 0.9
        correct_probability = 0.8 if prediction == answer else 0.1
        wrong_probability = mass - correct_probability
        probabilities = {
            answer: correct_probability,
            "B": wrong_probability,
        }
        rows.append(
            {
                "position": position,
                "question_id": question_id,
                "question": f"Question {question_id}",
                "options": ["one", "two"],
                "answer": answer,
                "category": "test",
                "source": "unit",
                "decile": decile,
                "trace_token_count": 1000 + position,
                "choice_probabilities": probabilities,
                "choice_probability_mass": mass,
                "non_choice_probability": 1 - mass,
                "prediction": prediction,
                "correct": prediction == answer,
                "prediction_flip": (
                    decile > 0
                    and prediction != predictions[(decile // 10) - 1]
                ),
                "correct_answer_probability": correct_probability,
                "normalized_correct_probability": correct_probability / mass,
            }
        )
    return rows


def test_lost_case_table_distinguishes_endpoint_hidden_and_robust() -> None:
    endpoint = _question("endpoint", 0, ["A"] * 10 + ["B"])
    hidden = _question(
        "hidden",
        1,
        ["B", "A", "A", "B", "B", "B", "B", "B", "B", "B", "B"],
        final_mass=0.2,
    )
    final_correct = _question("correct", 2, ["B"] * 10 + ["A"])
    dataframe = pd.DataFrame(endpoint + hidden + final_correct)
    traces = [
        {
            "position": 0,
            "question_id": "endpoint",
            "generated_answer": "B",
            "reasoning_trace": "Wait, I found a test bank answer.",
            "generated_answer_text": "B",
        },
        {
            "position": 1,
            "question_id": "hidden",
            "generated_answer": "B",
            "reasoning_trace": "",
            "generated_answer_text": "B",
        },
        {
            "position": 2,
            "question_id": "correct",
            "generated_answer": "A",
            "reasoning_trace": "",
            "generated_answer_text": "A",
        },
    ]

    cases = build_lost_case_table(dataframe, traces)
    assert set(cases["question_id"]) == {"endpoint", "hidden"}
    by_id = cases.set_index("question_id")
    assert bool(by_id.loc["endpoint", "strict_endpoint_lost"])
    assert bool(by_id.loc["endpoint", "robust_loss"])
    assert bool(
        by_id.loc["endpoint", "contains_simulated_retrieval_language"]
    )
    assert not bool(by_id.loc["hidden", "strict_endpoint_lost"])
    assert not bool(by_id.loc["hidden", "robust_loss"])
    assert by_id.loc["hidden", "first_correct_decile"] == 10
    assert by_id.loc["hidden", "last_correct_decile"] == 20

    summary = build_lost_summary(dataframe, cases, 0.5, 0.5)
    assert summary["ever_correct_final_wrong_count"] == 2
    assert summary["endpoint_lost_count"] == 1
    assert summary["gained_then_lost_count"] == 1
    assert summary["robust_loss_count"] == 1
    assert summary["robust_endpoint_lost_count"] == 1
    assert summary["robust_gained_then_lost_count"] == 0


def test_plot_loss_timing_writes_png(tmp_path: Path) -> None:
    cases = pd.DataFrame(
        [
            {
                "first_correct_decile": 0,
                "last_correct_decile": 90,
                "flip_count": 2,
            }
        ]
    )
    output = plot_loss_timing(cases, tmp_path)
    assert output.exists()
    assert output.stat().st_size > 0
