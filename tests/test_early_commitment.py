from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from logit_cot_overthinking.early_commitment import (
    analyze_early_commitment,
    evaluate_early_commitment,
    evaluate_early_commitment_policies,
)


def _row(
    question_id: str,
    position: int,
    decile: int,
    prediction: str,
    answer_probability: float,
    prediction_probability: float,
    baseline_cohort: str,
    answer: str = "A",
) -> dict[str, object]:
    other = "B" if answer == "A" else "A"
    probabilities = {
        answer: answer_probability,
        other: prediction_probability
        if prediction == other
        else 1.0 - answer_probability,
    }
    if prediction == answer:
        probabilities[answer] = prediction_probability
        probabilities[other] = 1.0 - prediction_probability
    return {
        "dataset": "mmlu_pro",
        "seed": 0,
        "position": position,
        "question_id": question_id,
        "baseline_cohort": baseline_cohort,
        "answer": answer,
        "decile": decile,
        "choice_probabilities": probabilities,
        "choice_probability_mass": sum(probabilities.values()),
        "prediction": prediction,
        "prediction_probability": probabilities[prediction],
        "correct": prediction == answer,
    }


def _trajectory() -> pd.DataFrame:
    rows = []
    rows.extend(
        [
            _row("recoverable", 0, 0, "B", 0.10, 0.90, "loss"),
            _row("recoverable", 0, 10, "A", 0.91, 0.91, "loss"),
            _row("recoverable", 0, 20, "A", 0.92, 0.92, "loss"),
            _row("recoverable", 0, 100, "B", 0.20, 0.80, "loss"),
        ]
    )
    rows.extend(
        [
            _row("late_correct", 1, 0, "B", 0.30, 0.70, "final_correct"),
            _row("late_correct", 1, 10, "B", 0.30, 0.70, "final_correct"),
            _row("late_correct", 1, 20, "B", 0.30, 0.70, "final_correct"),
            _row("late_correct", 1, 100, "A", 0.95, 0.95, "final_correct"),
        ]
    )
    rows.extend(
        [
            _row("never_correct", 2, 0, "B", 0.15, 0.85, "loss"),
            _row("never_correct", 2, 10, "B", 0.15, 0.85, "loss"),
            _row("never_correct", 2, 20, "B", 0.15, 0.85, "loss"),
            _row("never_correct", 2, 100, "B", 0.15, 0.85, "loss"),
        ]
    )
    rows.extend(
        [
            _row("proxy_trap", 3, 0, "B", 0.08, 0.92, "final_correct"),
            _row("proxy_trap", 3, 10, "B", 0.07, 0.93, "final_correct"),
            _row("proxy_trap", 3, 20, "B", 0.06, 0.94, "final_correct"),
            _row("proxy_trap", 3, 100, "A", 0.96, 0.96, "final_correct"),
        ]
    )
    return pd.DataFrame(rows)


def test_policy_outcomes_pick_expected_checkpoints() -> None:
    outcomes = evaluate_early_commitment_policies(
        _trajectory(),
        thresholds=(0.9,),
        proxy_threshold=0.9,
        proxy_streak=2,
    )
    by_key = outcomes.set_index(["question_id", "policy"])

    assert not bool(by_key.loc[("recoverable", "final"), "selected_correct"])
    assert (
        by_key.loc[
            ("recoverable", "oracle_first_correct"),
            "stop_decile",
        ]
        == 10
    )
    assert (
        by_key.loc[
            ("recoverable", "threshold_first_0.9"),
            "stop_decile",
        ]
        == 10
    )
    proxy_policy = "proxy_confidence_streak_0.9_s2"
    assert (
        by_key.loc[("recoverable", proxy_policy), "stop_decile"]
        == 20
    )
    assert bool(by_key.loc[("recoverable", proxy_policy), "selected_correct"])
    assert (
        by_key.loc[("proxy_trap", proxy_policy), "stop_decile"]
        == 10
    )
    assert not bool(
        by_key.loc[("proxy_trap", proxy_policy), "selected_correct"]
    )
    assert outcomes[
        outcomes["policy"].isin(
            ["oracle_first_correct", "threshold_first_0.9"]
        )
    ]["oracle"].all()


def test_policy_summary_groups_by_dataset_and_baseline_cohort() -> None:
    outcomes, summary = evaluate_early_commitment(
        _trajectory(),
        thresholds=(0.9,),
        proxy_threshold=0.9,
        proxy_streak=2,
    )
    assert outcomes[outcomes["policy"] == "final"].shape[0] == 4

    by_key = summary.set_index(
        ["dataset", "baseline_cohort", "policy"]
    )
    row = by_key.loc[("mmlu_pro", "loss", "oracle_first_correct")]
    assert row["attempt_count"] == 2
    assert row["policy_accuracy"] == 0.5
    assert row["delta_vs_final"] == 0.5
    assert row["stop_rate"] == 0.5
    assert row["median_stop_decile"] == 55.0


def test_analyze_early_commitment_writes_outputs_without_pyarrow(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_root = tmp_path / "matched"
    run_dir = input_root / "mmlu_pro" / "seed_7"
    run_dir.mkdir(parents=True)
    (run_dir / "trajectory.parquet").write_text(
        "stub",
        encoding="utf-8",
    )

    def fake_read_parquet(path: Path) -> pd.DataFrame:
        assert path == run_dir / "trajectory.parquet"
        return _trajectory().drop(columns=["dataset", "seed"])

    def fake_to_parquet(
        self: pd.DataFrame,
        path: Path,
        index: bool = False,
    ) -> None:
        assert not index
        path.write_text(
            self.to_json(orient="records"),
            encoding="utf-8",
        )

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    result = analyze_early_commitment(
        input_root=input_root,
        selection_path=None,
        thresholds=(0.9,),
    )

    assert Path(result["outcomes_path"]).exists()
    assert Path(result["summary_path"]).exists()
    assert Path(result["broad_loss_summary_path"]).exists()
    summary = json.loads(
        Path(result["summary_json_path"]).read_text(encoding="utf-8")
    )
    assert summary["attempt_count"] == 4
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "upper bounds" in report
    assert "not deployable" in report
