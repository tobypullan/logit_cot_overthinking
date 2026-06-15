from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm


COLORS = {
    "blue": "#2563EB",
    "green": "#16A34A",
    "orange": "#EA580C",
    "red": "#DC2626",
    "purple": "#7C3AED",
    "gray": "#64748B",
    "light_gray": "#CBD5E1",
}


def _save_figure(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def plot_overview(dataframe: pd.DataFrame, output_dir: Path) -> Path:
    grouped = dataframe.groupby("decile", sort=True)
    deciles = np.array(sorted(dataframe["decile"].unique()))
    accuracy = grouped["correct"].mean().reindex(deciles)
    commitment = grouped["final_answer_commitment"].mean().reindex(deciles)
    non_choice = grouped["non_choice_probability"].mean().reindex(deciles)
    flips = grouped["prediction_flip"].mean().reindex(deciles)

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    panels = (
        (axes[0, 0], accuracy, "Accuracy", COLORS["green"], False),
        (
            axes[0, 1],
            commitment,
            "Probability on eventual final answer",
            COLORS["blue"],
            False,
        ),
        (
            axes[1, 0],
            non_choice,
            "Probability outside valid answer letters",
            COLORS["orange"],
            True,
        ),
        (
            axes[1, 1],
            flips,
            "Prediction flip rate",
            COLORS["purple"],
            False,
        ),
    )
    for axis, values, title, color, logarithmic in panels:
        axis.plot(deciles, values, marker="o", linewidth=2.3, color=color)
        axis.set_title(title)
        axis.set_xticks(deciles)
        axis.grid(alpha=0.25)
        if logarithmic:
            axis.set_yscale("log")
            axis.set_ylim(max(values[values > 0].min() / 2, 1e-6), 1.2)
        else:
            axis.set_ylim(-0.04, 1.04)
        axis.set_ylabel("Probability")
    for axis in axes[1]:
        axis.set_xlabel("Reasoning trace revealed (%)")

    figure.suptitle(
        f"Gemma 4 12B smoke-run trajectory overview (n={dataframe['question_id'].nunique()})",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "Descriptive smoke-test results only; the sample is too small for aggregate inference.",
        ha="center",
        color=COLORS["gray"],
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    return _save_figure(figure, output_dir / "trajectory_overview.png")


def plot_correct_answer_heatmap(dataframe: pd.DataFrame, output_dir: Path) -> Path:
    ordered = dataframe.sort_values(["position", "decile"]).copy()
    ordered["correct_answer_probability"] = ordered.apply(
        lambda row: row["choice_probabilities"][row["answer"]],
        axis=1,
    )
    probability_matrix = ordered.pivot(
        index="question_id",
        columns="decile",
        values="correct_answer_probability",
    )
    prediction_matrix = ordered.pivot(
        index="question_id",
        columns="decile",
        values="prediction",
    ).reindex(probability_matrix.index)
    answer_by_question = (
        ordered.drop_duplicates("question_id").set_index("question_id")["answer"]
    )

    positive_values = probability_matrix.to_numpy()
    positive_values = positive_values[positive_values > 0]
    minimum = max(float(positive_values.min()), 1e-8)

    figure, axis = plt.subplots(figsize=(12, 3.8))
    image = axis.imshow(
        probability_matrix.to_numpy(),
        aspect="auto",
        cmap="viridis",
        norm=LogNorm(vmin=minimum, vmax=1),
    )
    for row_index, question_id in enumerate(probability_matrix.index):
        answer = answer_by_question.loc[question_id]
        for column_index, decile in enumerate(probability_matrix.columns):
            prediction = prediction_matrix.loc[question_id, decile]
            probability = probability_matrix.loc[question_id, decile]
            text_color = "white" if probability < 0.15 else "black"
            weight = "bold" if prediction == answer else "normal"
            axis.text(
                column_index,
                row_index,
                prediction,
                ha="center",
                va="center",
                color=text_color,
                fontweight=weight,
                fontsize=10,
            )

    axis.set_xticks(range(len(probability_matrix.columns)))
    axis.set_xticklabels(probability_matrix.columns)
    axis.set_yticks(range(len(probability_matrix.index)))
    axis.set_yticklabels(
        [
            f"Question {question_id} (answer {answer_by_question.loc[question_id]})"
            for question_id in probability_matrix.index
        ]
    )
    axis.set_xlabel("Reasoning trace revealed (%)")
    axis.set_title(
        "Correct-answer probability and argmax prediction",
        fontsize=14,
        fontweight="bold",
    )
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Raw probability on correct answer (log scale)")
    figure.text(
        0.5,
        0.01,
        "Cell letter is the argmax answer; bold letters are correct.",
        ha="center",
        color=COLORS["gray"],
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    return _save_figure(figure, output_dir / "correct_answer_heatmap.png")


def plot_choice_trajectories(dataframe: pd.DataFrame, output_dir: Path) -> Path:
    question_rows = list(
        dataframe.sort_values("position").drop_duplicates("question_id").itertuples()
    )
    figure, axes = plt.subplots(
        len(question_rows),
        1,
        figsize=(11, 3.4 * len(question_rows)),
        sharex=True,
        squeeze=False,
    )

    for axis, question in zip(axes[:, 0], question_rows):
        subset = dataframe[dataframe["question_id"] == question.question_id].sort_values(
            "decile"
        )
        labels = list(question.valid_labels)
        for label in labels:
            values = subset[f"prob_{label}"]
            is_answer = label == question.answer
            axis.plot(
                subset["decile"],
                values,
                marker="o" if is_answer else None,
                linewidth=2.8 if is_answer else 1,
                alpha=1 if is_answer else 0.35,
                color=COLORS["green"] if is_answer else COLORS["light_gray"],
                label=f"{label} (correct)" if is_answer else label,
                zorder=3 if is_answer else 1,
            )
        axis.plot(
            subset["decile"],
            subset["non_choice_probability"],
            linestyle="--",
            linewidth=1.8,
            color=COLORS["orange"],
            label="Non-choice",
            zorder=2,
        )
        axis.set_yscale("symlog", linthresh=1e-6)
        axis.set_ylim(0, 1.2)
        axis.set_ylabel("Raw probability")
        axis.set_title(
            f"Question {question.question_id}: {question.outcome.replace('_', ' ')}",
            loc="left",
            fontweight="bold",
        )
        axis.grid(alpha=0.2)
        axis.legend(
            ncols=min(6, len(labels) + 1),
            fontsize=8,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
        )

    axes[-1, 0].set_xlabel("Reasoning trace revealed (%)")
    axes[-1, 0].set_xticks(sorted(dataframe["decile"].unique()))
    figure.suptitle(
        "Per-question answer distributions",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return _save_figure(figure, output_dir / "choice_probability_trajectories.png")


def plot_runtime_and_trace_lengths(
    traces_path: Path,
    summary_path: Path,
    output_dir: Path,
) -> Path:
    traces = [
        json.loads(line)
        for line in traces_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    timings = summary.get("timings_seconds", {})
    stage_names = [
        "dataset_loading",
        "model_initialization",
        "trace_generation",
        "trajectory_probing",
        "output_processing",
    ]
    stage_labels = [
        "Dataset",
        "Model init",
        "Trace generation",
        "33 probes",
        "Outputs",
    ]
    stage_values = [float(timings.get(name, 0)) for name in stage_names]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    trace_ids = [str(trace["question_id"]) for trace in traces]
    trace_lengths = [int(trace["trace_token_count"]) for trace in traces]
    bars = axes[0].bar(trace_ids, trace_lengths, color=COLORS["blue"])
    axes[0].bar_label(bars, padding=3, fmt="%d")
    axes[0].set_title("Generated reasoning trace lengths")
    axes[0].set_xlabel("Question ID")
    axes[0].set_ylabel("Tokens")
    axes[0].grid(axis="y", alpha=0.25)

    runtime_bars = axes[1].barh(
        stage_labels,
        stage_values,
        color=[
            COLORS["gray"],
            COLORS["purple"],
            COLORS["blue"],
            COLORS["green"],
            COLORS["gray"],
        ],
    )
    axes[1].bar_label(runtime_bars, padding=3, fmt="%.2fs")
    axes[1].set_title("Cached smoke-run time by stage")
    axes[1].set_xlabel("Seconds")
    axes[1].set_xlim(0, max(stage_values) * 1.15)
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].invert_yaxis()

    figure.suptitle("Smoke-run compute diagnostics", fontsize=15, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return _save_figure(figure, output_dir / "runtime_and_trace_lengths.png")


def create_visualizations(input_dir: Path, output_dir: Path) -> list[Path]:
    trajectory_path = input_dir / "trajectory.parquet"
    traces_path = input_dir / "traces.jsonl"
    summary_path = input_dir / "summary.json"
    for path in (trajectory_path, traces_path, summary_path):
        if not path.exists():
            raise FileNotFoundError(f"Required smoke-run artifact not found: {path}")

    dataframe = pd.read_parquet(trajectory_path)
    required_columns = {
        "question_id",
        "position",
        "decile",
        "answer",
        "prediction",
        "correct",
        "choice_probabilities",
        "final_answer_commitment",
        "non_choice_probability",
        "prediction_flip",
        "outcome",
    }
    missing = required_columns.difference(dataframe.columns)
    if missing:
        raise KeyError(f"trajectory.parquet is missing columns: {sorted(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_overview(dataframe, output_dir),
        plot_correct_answer_heatmap(dataframe, output_dir),
        plot_choice_trajectories(dataframe, output_dir),
        plot_runtime_and_trace_lengths(
            traces_path,
            summary_path,
            output_dir,
        ),
    ]
