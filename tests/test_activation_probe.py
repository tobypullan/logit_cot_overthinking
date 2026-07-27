from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from logit_cot_overthinking.activation_probe import (
    ProbeTrainingConfig,
    build_activation_probe_examples,
    evaluate_candidate_intervention,
    evaluate_probe_halting,
    resolve_layers,
    train_activation_probes,
)


def _trajectory_row(
    position: int,
    question_id: str,
    decile: int,
    prediction: str,
    correct: bool,
    final_prediction: str,
    answer: str = "A",
) -> dict[str, object]:
    probabilities = {"A": 0.1, "B": 0.1}
    probabilities[prediction] = 0.9
    if correct:
        probabilities[answer] = 0.9
    return {
        "position": position,
        "question_id": question_id,
        "question": f"Question {question_id}",
        "options": ["alpha", "beta"],
        "answer": answer,
        "category": "cat",
        "source": "test",
        "run_seed": 0,
        "decile": decile,
        "prefix_token_count": decile,
        "trace_token_count": 100,
        "choice_probabilities": probabilities,
        "choice_probability_mass": sum(probabilities.values()),
        "non_choice_probability": 0.0,
        "prediction": prediction,
        "prediction_probability": probabilities[prediction],
        "correct": correct,
        "baseline_cohort": "loss",
        "match_id": "m",
        "prediction_flip": False,
        "final_prediction": final_prediction,
        "outcome": "stable_wrong",
    }


def test_build_activation_probe_examples_labels_all_targets(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "matched" / "mmlu_pro" / "seed_0"
    run_dir.mkdir(parents=True)
    rows = [
        _trajectory_row(1, "loss", 10, "A", True, "B"),
        _trajectory_row(1, "loss", 100, "B", False, "B"),
        _trajectory_row(2, "flip_good", 10, "B", False, "A"),
        _trajectory_row(2, "flip_good", 100, "A", True, "A"),
        _trajectory_row(3, "stable_wrong", 10, "B", False, "B"),
        _trajectory_row(3, "stable_wrong", 100, "B", False, "B"),
    ]
    pd.DataFrame(rows).to_parquet(run_dir / "trajectory.parquet", index=False)
    (run_dir / "traces.jsonl").write_text("", encoding="utf-8")

    examples = build_activation_probe_examples(
        input_root=tmp_path / "matched",
        output_dir=tmp_path / "probe",
        deciles=(10,),
    ).set_index("question_id")

    loss = examples.loc["loss"]
    assert loss["current_correct"] == 1
    assert loss["future_loss"] == 1
    assert loss["future_change_to_wrong"] == 1
    assert loss["future_answer_flip"] == 1

    flip_good = examples.loc["flip_good"]
    assert flip_good["current_correct"] == 0
    assert flip_good["future_loss"] == 0
    assert flip_good["future_change_to_wrong"] == 0
    assert flip_good["future_answer_flip"] == 1

    stable_wrong = examples.loc["stable_wrong"]
    assert stable_wrong["current_correct"] == 0
    assert stable_wrong["future_loss"] == 0
    assert stable_wrong["future_change_to_wrong"] == 0
    assert stable_wrong["future_answer_flip"] == 0


def test_build_robust_loss_vs_no_loss_cohort(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        _trajectory_row(1, "robust", 10, "A", True, "B"),
        _trajectory_row(1, "robust", 100, "B", False, "B"),
        _trajectory_row(2, "no_loss", 10, "A", True, "A"),
        _trajectory_row(2, "no_loss", 100, "A", True, "A"),
        _trajectory_row(3, "nonrobust", 10, "A", True, "B"),
        _trajectory_row(3, "nonrobust", 100, "B", False, "B"),
    ]
    pd.DataFrame(rows).to_parquet(
        run_dir / "trajectory.parquet",
        index=False,
    )
    traces = [
        {
            "position": position,
            "question_id": question_id,
            "question": f"Question {question_id}",
            "options": ["alpha", "beta"],
            "answer": "A",
            "category": "cat",
            "source": "test",
            "reasoning_trace": "reasoning",
            "generated_answer_text": generated,
            "truncated": False,
        }
        for position, question_id, generated in (
            (1, "robust", "B"),
            (2, "no_loss", "A"),
            (3, "nonrobust", "A"),
        )
    ]
    (run_dir / "traces.jsonl").write_text(
        "\n".join(json.dumps(trace) for trace in traces) + "\n",
        encoding="utf-8",
    )

    examples = build_activation_probe_examples(
        input_root=run_dir,
        output_dir=tmp_path / "probe",
        deciles=(10,),
        example_cohort="robust_loss_vs_no_loss",
    ).set_index("question_id")

    assert set(examples.index) == {"robust", "no_loss"}
    assert examples.loc["robust", "robust_loss_case"] == 1
    assert examples.loc["no_loss", "robust_loss_case"] == 0
    assert set(examples["current_correct"]) == {1}


def test_build_intervention_candidates_includes_harmful_states(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        _trajectory_row(1, "beneficial", 10, "A", True, "B"),
        _trajectory_row(1, "beneficial", 100, "B", False, "B"),
        _trajectory_row(2, "harmful", 10, "B", False, "A"),
        _trajectory_row(2, "harmful", 100, "A", True, "A"),
        _trajectory_row(3, "neutral", 10, "B", False, "B"),
        _trajectory_row(3, "neutral", 100, "B", False, "B"),
    ]
    pd.DataFrame(rows).to_parquet(
        run_dir / "trajectory.parquet",
        index=False,
    )
    traces = [
        {
            "position": position,
            "question_id": question_id,
            "question": f"Question {question_id}",
            "options": ["alpha", "beta"],
            "answer": "A",
            "category": "cat",
            "source": "test",
            "reasoning_trace": "reasoning",
            "generated_answer_text": generated,
            "truncated": False,
        }
        for position, question_id, generated in (
            (1, "beneficial", "B"),
            (2, "harmful", "A"),
            (3, "neutral", "B"),
        )
    ]
    (run_dir / "traces.jsonl").write_text(
        "\n".join(json.dumps(trace) for trace in traces) + "\n",
        encoding="utf-8",
    )

    examples = build_activation_probe_examples(
        input_root=run_dir,
        output_dir=tmp_path / "probe",
        deciles=(10,),
        example_cohort="intervention_candidates",
        intervention_confidence_threshold=0.5,
    ).set_index("question_id")

    assert examples.loc["beneficial", "robust_stop_candidate"] == 1
    assert examples.loc["beneficial", "intervention_utility"] == "beneficial"
    assert examples.loc["harmful", "robust_stop_candidate"] == 0
    assert examples.loc["harmful", "intervention_utility"] == "harmful"
    assert examples.loc["neutral", "intervention_utility"] == "neutral"


def test_evaluate_probe_halting_combines_confidence_and_probe_score() -> None:
    examples = pd.DataFrame(
        [
            {
                "example_index": 0,
                "dataset": "mmlu_pro",
                "seed": 0,
                "position": 1,
                "question_id": "loss",
                "decile": 10,
                "normalized_prediction_probability": 0.95,
                "current_correct": True,
                "final_correct": False,
            },
            {
                "example_index": 1,
                "dataset": "mmlu_pro",
                "seed": 0,
                "position": 2,
                "question_id": "stable",
                "decile": 10,
                "normalized_prediction_probability": 0.95,
                "current_correct": True,
                "final_correct": True,
            },
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "example_index": 0,
                "layer": 4,
                "target": "future_loss",
                "score": 0.9,
                "label": 1,
                "fold": 0,
            },
            {
                "example_index": 1,
                "layer": 4,
                "target": "future_loss",
                "score": 0.1,
                "label": 0,
                "fold": 0,
            },
        ]
    )

    summary = evaluate_probe_halting(
        examples,
        predictions,
        confidence_thresholds=(0.9,),
        probe_thresholds=(0.5,),
    )
    probe_row = summary[
        summary["policy_family"] == "probe_confidence"
    ].iloc[0]
    probe_only_row = summary[
        summary["policy_family"] == "probe_only"
    ].iloc[0]

    assert probe_row["accuracy"] == 1.0
    assert probe_row["final_accuracy"] == 0.5
    assert probe_row["delta_vs_final"] == 0.5
    assert probe_row["stop_rate"] == 0.5
    assert probe_only_row["accuracy"] == 1.0
    assert probe_only_row["final_accuracy"] == 0.5
    assert probe_only_row["delta_vs_final"] == 0.5
    assert probe_only_row["stop_rate"] == 0.5


def test_evaluate_candidate_intervention_counts_benefit_and_harm() -> None:
    examples = pd.DataFrame(
        {
            "example_index": [0, 1],
            "decile": [20, 20],
            "normalized_prediction_probability": [0.95, 0.95],
            "current_correct": [True, False],
            "final_correct": [False, True],
            "intervention_utility": ["beneficial", "harmful"],
        }
    )
    predictions = pd.DataFrame(
        {
            "example_index": [0, 1],
            "layer": [36, 36],
            "target": ["robust_stop_candidate"] * 2,
            "score": [0.9, 0.1],
        }
    )

    result = evaluate_candidate_intervention(
        examples,
        predictions,
        probe_thresholds=(0.5,),
        fallback_attempt_count=1,
        fallback_correct_count=1,
    ).iloc[0]

    assert result["attempt_count"] == 3
    assert result["accuracy"] == 1.0
    assert result["final_accuracy"] == 2 / 3
    assert result["delta_vs_final"] == 1 / 3
    assert result["beneficial_stop_count"] == 1
    assert result["harmful_stop_count"] == 0


def test_train_activation_probes_numpy_backend_writes_metrics(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "probe"
    output_dir.mkdir()
    examples = pd.DataFrame(
        {
            "example_index": np.arange(8),
            "dataset": ["mmlu_pro"] * 8,
            "seed": [0] * 8,
            "position": np.arange(8),
            "question_id": [f"q{i}" for i in range(8)],
            "decile": [10] * 8,
            "normalized_prediction_probability": [0.9] * 8,
            "current_correct": [False, True] * 4,
            "final_correct": [True, False] * 4,
            "future_loss": [0, 1] * 4,
            "future_change_to_wrong": [0, 1] * 4,
            "future_answer_flip": [0, 1] * 4,
        }
    )
    examples.to_parquet(
        output_dir / "activation_probe_examples.parquet",
        index=False,
    )
    activations = np.zeros((8, 2, 3), dtype=np.float32)
    activations[:, :, 0] = np.asarray([0, 1] * 4)[:, None]
    activations[:, :, 1] = np.arange(8)[:, None] / 8
    np.save(output_dir / "activations.npy", activations)

    summary = train_activation_probes(
        ProbeTrainingConfig(
            output_dir=output_dir,
            layers=(0, 1),
            backend="numpy",
            folds=2,
            targets=(
                "current_correct",
                "future_loss",
                "future_change_to_wrong",
                "future_answer_flip",
            ),
        )
    )

    metrics = pd.read_parquet(summary["metrics_path"])
    assert set(metrics["target"]) == {
        "current_correct",
        "future_loss",
        "future_change_to_wrong",
        "future_answer_flip",
    }
    assert set(metrics["layer"]) == {0, 1}
    assert Path(summary["halting_summary_path"]).exists()


def test_train_activation_probes_reads_sharded_activations(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "probe"
    output_dir.mkdir()
    examples = pd.DataFrame(
        {
            "example_index": np.arange(8),
            "dataset": ["mmlu_pro"] * 8,
            "seed": [0] * 8,
            "position": np.arange(8),
            "question_id": [f"q{i}" for i in range(8)],
            "decile": [10] * 8,
            "normalized_prediction_probability": [0.9] * 8,
            "current_correct": [False, True] * 4,
            "final_correct": [True, False] * 4,
            "future_loss": [0, 1] * 4,
            "future_change_to_wrong": [0, 1] * 4,
            "future_answer_flip": [0, 1] * 4,
        }
    )
    examples.to_parquet(
        output_dir / "activation_probe_examples.parquet",
        index=False,
    )
    activations = np.zeros((8, 2, 3), dtype=np.float16)
    activations[:, :, 0] = np.asarray([0, 1] * 4)[:, None]
    activations[:, :, 1] = np.arange(8)[:, None] / 8

    shard_dir = output_dir / "activations_shards"
    shard_dir.mkdir()
    np.save(shard_dir / "activations_00000_00004.npy", activations[:4])
    np.save(shard_dir / "activations_00004_00008.npy", activations[4:])
    (shard_dir / "manifest.json").write_text(
        """{
  "format": "activation_shards_v1",
  "shape": [8, 2, 3],
  "dtype": "float16",
  "axis": 0,
  "chunk_size": 4,
  "shards": [
    {"path": "activations_00000_00004.npy", "start": 0, "end": 4},
    {"path": "activations_00004_00008.npy", "start": 4, "end": 8}
  ]
}
""",
        encoding="utf-8",
    )

    summary = train_activation_probes(
        ProbeTrainingConfig(
            output_dir=output_dir,
            layers=(0, 1),
            backend="numpy",
            folds=2,
            targets=("future_loss",),
        )
    )

    assert summary["activations_path"] == shard_dir / "manifest.json"
    metrics = pd.read_parquet(summary["metrics_path"])
    assert set(metrics["layer"]) == {0, 1}


def test_resolve_layers_auto_includes_embedding_and_final() -> None:
    layers = resolve_layers(None, num_hidden_layers=12)
    assert layers[0] == 0
    assert layers[-1] == 12
