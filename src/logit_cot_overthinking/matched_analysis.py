from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .lost_analysis import (
    SELF_CORRECTION_PATTERN,
    SIMULATED_RETRIEVAL_PATTERN,
)


MODEL_FEATURES = {
    "confidence": [
        "current_normalized_correct_probability",
    ],
    "instability": [
        "current_normalized_correct_probability",
        "flips_so_far",
        "normalized_confidence_decline",
        "time_since_first_correct",
        "stable_correct_streak",
    ],
    "instability_plus_length": [
        "current_normalized_correct_probability",
        "flips_so_far",
        "normalized_confidence_decline",
        "time_since_first_correct",
        "stable_correct_streak",
        "log_trace_token_count",
    ],
    "full_prefix": [
        "current_normalized_correct_probability",
        "flips_so_far",
        "normalized_confidence_decline",
        "time_since_first_correct",
        "stable_correct_streak",
        "log_trace_token_count",
        "prefix_self_correction",
        "prefix_simulated_retrieval",
        "prefix_repetition_score",
    ],
}


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def repetition_score(text: str, n: int = 4) -> float:
    words = text.lower().split()
    if len(words) < n:
        return 0.0
    ngrams = [
        tuple(words[index : index + n])
        for index in range(len(words) - n + 1)
    ]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def _normalized_correct_probability(row: pd.Series) -> float:
    mass = float(row["choice_probability_mass"])
    if mass <= 0:
        return 0.0
    return float(row["choice_probabilities"][row["answer"]]) / mass


def build_matched_attempt_tables(
    input_root: Path,
    selection_path: Path,
    model: str = "google/gemma-4-12B-it",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from transformers import AutoTokenizer

    selection = pd.read_parquet(selection_path)
    metadata = {
        (str(row.dataset), int(row.position)): row._asdict()
        for row in selection.itertuples(index=False)
    }
    tokenizer = AutoTokenizer.from_pretrained(model)
    attempts: list[dict[str, object]] = []
    checkpoints: list[dict[str, object]] = []
    for dataset in ("mmlu_pro", "gpqa_diamond"):
        for seed in range(10):
            run_dir = input_root / dataset / f"seed_{seed}"
            dataframe = pd.read_parquet(
                run_dir / "trajectory.parquet"
            )
            traces = {
                (int(trace["position"]), str(trace["question_id"])): trace
                for trace in _read_jsonl(run_dir / "traces.jsonl")
            }
            for (position, question_id), group in dataframe.groupby(
                ["position", "question_id"],
                sort=False,
            ):
                group = group.sort_values("decile").copy()
                group["correct_answer_probability"] = group.apply(
                    lambda row: float(
                        row["choice_probabilities"][row["answer"]]
                    ),
                    axis=1,
                )
                group["normalized_correct_probability"] = group.apply(
                    _normalized_correct_probability,
                    axis=1,
                )
                key = (int(position), str(question_id))
                trace = traces[key]
                meta = metadata[(dataset, int(position))]
                final = group.iloc[-1]
                correct_rows = group[group["correct"]]
                broad_loss = (
                    not correct_rows.empty and not bool(final["correct"])
                )
                trace_text = str(trace["reasoning_trace"])
                token_ids = tokenizer.encode(
                    trace_text,
                    add_special_tokens=False,
                )
                attempt = {
                    "dataset": dataset,
                    "dataset_label": meta["dataset_label"],
                    "seed": seed,
                    "position": int(position),
                    "question_id": str(question_id),
                    "category": str(final["category"]),
                    "baseline_cohort": meta["baseline_cohort"],
                    "match_id": meta["match_id"],
                    "broad_loss": broad_loss,
                    "final_correct": bool(final["correct"]),
                    "ever_correct": not correct_rows.empty,
                    "outcome": str(final["outcome"]),
                    "flip_count": int(
                        group["prediction_flip"].sum()
                    ),
                    "trace_token_count": int(
                        final["trace_token_count"]
                    ),
                    "first_correct_decile": (
                        int(correct_rows["decile"].min())
                        if not correct_rows.empty
                        else None
                    ),
                    "last_correct_decile": (
                        int(correct_rows["decile"].max())
                        if not correct_rows.empty
                        else None
                    ),
                    "correct_decile_count": len(correct_rows),
                    "peak_correct_probability": float(
                        group.loc[
                            group["decile"] < 100,
                            "correct_answer_probability",
                        ].max()
                    ),
                    "peak_normalized_correct_probability": float(
                        group.loc[
                            group["decile"] < 100,
                            "normalized_correct_probability",
                        ].max()
                    ),
                    "final_correct_probability": float(
                        final["correct_answer_probability"]
                    ),
                    "final_normalized_correct_probability": float(
                        final["normalized_correct_probability"]
                    ),
                    "contains_self_correction_language": bool(
                        SELF_CORRECTION_PATTERN.search(trace_text)
                    ),
                    "contains_simulated_retrieval_language": bool(
                        SIMULATED_RETRIEVAL_PATTERN.search(trace_text)
                    ),
                    "repetition_score": repetition_score(trace_text),
                    "extended": bool(trace.get("extended", False)),
                    "forced_completion": bool(
                        trace.get("forced_completion", False)
                    ),
                }
                attempts.append(attempt)

                previous_prediction: str | None = None
                flips = 0
                first_correct: int | None = None
                streak = 0
                peak_normalized = 0.0
                previous_normalized: float | None = None
                for row in group.itertuples(index=False):
                    if previous_prediction is not None and (
                        str(row.prediction) != previous_prediction
                    ):
                        flips += 1
                    previous_prediction = str(row.prediction)
                    current_correct = bool(row.correct)
                    if current_correct:
                        if first_correct is None:
                            first_correct = int(row.decile)
                        streak += 1
                    else:
                        streak = 0
                    current_normalized = float(
                        row.normalized_correct_probability
                    )
                    peak_normalized = max(
                        peak_normalized,
                        current_normalized,
                    )
                    if (
                        int(row.decile) in range(10, 100, 10)
                        and current_correct
                    ):
                        prefix_count = int(row.prefix_token_count)
                        prefix = tokenizer.decode(
                            token_ids[:prefix_count],
                            skip_special_tokens=False,
                        )
                        checkpoints.append(
                            {
                                **{
                                    key: attempt[key]
                                    for key in (
                                        "dataset",
                                        "dataset_label",
                                        "seed",
                                        "position",
                                        "question_id",
                                        "category",
                                        "baseline_cohort",
                                        "match_id",
                                    )
                                },
                                "decile": int(row.decile),
                                "final_wrong": not bool(
                                    final["correct"]
                                ),
                                "flips_so_far": flips,
                                "time_since_first_correct": int(
                                    row.decile
                                )
                                - int(first_correct),
                                "stable_correct_streak": streak,
                                "current_correct_probability": float(
                                    row.correct_answer_probability
                                ),
                                "current_normalized_correct_probability": (
                                    current_normalized
                                ),
                                "normalized_confidence_decline": (
                                    peak_normalized
                                    - current_normalized
                                ),
                                "recent_normalized_decline": (
                                    max(
                                        0.0,
                                        float(previous_normalized)
                                        - current_normalized,
                                    )
                                    if previous_normalized is not None
                                    else 0.0
                                ),
                                "choice_probability_mass": float(
                                    row.choice_probability_mass
                                ),
                                "prefix_token_count": prefix_count,
                                "trace_token_count": int(
                                    row.trace_token_count
                                ),
                                "log_trace_token_count": math.log1p(
                                    int(row.trace_token_count)
                                ),
                                "prefix_self_correction": int(
                                    bool(
                                        SELF_CORRECTION_PATTERN.search(
                                            prefix
                                        )
                                    )
                                ),
                                "prefix_simulated_retrieval": int(
                                    bool(
                                        SIMULATED_RETRIEVAL_PATTERN.search(
                                            prefix
                                        )
                                    )
                                ),
                                "prefix_repetition_score": (
                                    repetition_score(prefix)
                                ),
                            }
                        )
                    previous_normalized = current_normalized
    return pd.DataFrame(attempts), pd.DataFrame(checkpoints)


def auc_score(y_true: Sequence[int], scores: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    values = np.asarray(scores, dtype=float)
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while (
            end < len(values)
            and values[order[end]] == values[order[start]]
        ):
            end += 1
        average_rank = (start + 1 + end) / 2
        ranks[order[start:end]] = average_rank
        start = end
    positive_rank_sum = ranks[y == 1].sum()
    return float(
        (
            positive_rank_sum
            - positives * (positives + 1) / 2
        )
        / (positives * negatives)
    )


def _fold_for_question(
    dataset: str,
    position: int,
    folds: int,
) -> int:
    digest = hashlib.sha256(
        f"{dataset}:{position}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") % folds


def _prepare_matrix(
    dataframe: pd.DataFrame,
    features: Sequence[str],
    medians: np.ndarray | None = None,
    means: np.ndarray | None = None,
    scales: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix = dataframe[list(features)].astype(float).to_numpy()
    if medians is None:
        medians = np.nanmedian(matrix, axis=0)
    missing = np.where(np.isnan(matrix))
    matrix[missing] = medians[missing[1]]
    if means is None:
        means = matrix.mean(axis=0)
    if scales is None:
        scales = matrix.std(axis=0)
        scales[scales < 1e-8] = 1.0
    matrix = (matrix - means) / scales
    return matrix, medians, means, scales


def _fit_logistic(
    matrix: np.ndarray,
    target: np.ndarray,
    ridge: float = 1.0,
) -> np.ndarray:
    design = np.column_stack([np.ones(len(matrix)), matrix])
    coefficients = np.zeros(design.shape[1])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    for _ in range(80):
        logits = np.clip(design @ coefficients, -30, 30)
        probabilities = 1 / (1 + np.exp(-logits))
        weights = np.clip(
            probabilities * (1 - probabilities),
            1e-6,
            None,
        )
        gradient = (
            design.T @ (target - probabilities)
            - penalty @ coefficients
        )
        hessian = (
            (design.T * weights) @ design + penalty
        )
        step = np.linalg.solve(hessian, gradient)
        coefficients += step
        if np.max(np.abs(step)) < 1e-7:
            break
    return coefficients


def grouped_cross_validated_predictions(
    dataframe: pd.DataFrame,
    features: Sequence[str],
    folds: int = 5,
) -> tuple[np.ndarray, list[np.ndarray]]:
    fold_ids = np.asarray(
        [
            _fold_for_question(
                str(row.dataset),
                int(row.position),
                folds,
            )
            for row in dataframe.itertuples(index=False)
        ]
    )
    target = dataframe["final_wrong"].astype(int).to_numpy()
    predictions = np.full(len(dataframe), np.nan)
    coefficients: list[np.ndarray] = []
    for fold in range(folds):
        train = fold_ids != fold
        test = fold_ids == fold
        if not test.any() or len(np.unique(target[train])) < 2:
            continue
        train_matrix, medians, means, scales = _prepare_matrix(
            dataframe.loc[train],
            features,
        )
        test_matrix, _, _, _ = _prepare_matrix(
            dataframe.loc[test],
            features,
            medians=medians,
            means=means,
            scales=scales,
        )
        coefficient = _fit_logistic(
            train_matrix,
            target[train],
        )
        logits = np.clip(
            np.column_stack(
                [np.ones(test.sum()), test_matrix]
            )
            @ coefficient,
            -30,
            30,
        )
        predictions[test] = 1 / (1 + np.exp(-logits))
        coefficients.append(coefficient)
    return predictions, coefficients


def evaluate_predictors(
    checkpoints: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, object]] = []
    coefficients: list[dict[str, object]] = []
    for dataset in ("combined", "mmlu_pro", "gpqa_diamond"):
        dataset_rows = (
            checkpoints
            if dataset == "combined"
            else checkpoints[checkpoints["dataset"] == dataset]
        )
        for decile in range(10, 100, 10):
            subset = dataset_rows[
                dataset_rows["decile"] == decile
            ].reset_index(drop=True)
            if (
                len(subset) < 30
                or subset["final_wrong"].nunique() < 2
            ):
                continue
            for model_name, features in MODEL_FEATURES.items():
                predictions, fold_coefficients = (
                    grouped_cross_validated_predictions(
                        subset,
                        features,
                    )
                )
                valid = np.isfinite(predictions)
                target = (
                    subset.loc[valid, "final_wrong"]
                    .astype(int)
                    .to_numpy()
                )
                scores = predictions[valid]
                metrics.append(
                    {
                        "dataset": dataset,
                        "decile": decile,
                        "model": model_name,
                        "row_count": int(valid.sum()),
                        "question_count": int(
                            subset.loc[
                                valid,
                                ["dataset", "position"],
                            ]
                            .drop_duplicates()
                            .shape[0]
                        ),
                        "loss_rate": float(target.mean()),
                        "auc": auc_score(target, scores),
                        "brier": float(
                            np.mean((scores - target) ** 2)
                        ),
                    }
                )
                if model_name == "full_prefix":
                    for coefficient in fold_coefficients:
                        for feature, value in zip(
                            features,
                            coefficient[1:],
                        ):
                            coefficients.append(
                                {
                                    "dataset": dataset,
                                    "decile": decile,
                                    "feature": feature,
                                    "coefficient": float(value),
                                }
                            )
    coefficient_frame = pd.DataFrame(coefficients)
    if not coefficient_frame.empty:
        coefficient_frame = (
            coefficient_frame.groupby(
                ["dataset", "decile", "feature"],
                as_index=False,
            )["coefficient"]
            .mean()
        )
    return pd.DataFrame(metrics), coefficient_frame


def build_cohort_summary(
    attempts: pd.DataFrame,
) -> pd.DataFrame:
    return (
        attempts.groupby(
            ["dataset", "dataset_label", "baseline_cohort"],
            as_index=False,
        )
        .agg(
            attempt_count=("broad_loss", "size"),
            broad_loss_count=("broad_loss", "sum"),
            broad_loss_rate=("broad_loss", "mean"),
            final_accuracy=("final_correct", "mean"),
            mean_flips=("flip_count", "mean"),
            median_trace_tokens=("trace_token_count", "median"),
            retrieval_rate=(
                "contains_simulated_retrieval_language",
                "mean",
            ),
            self_correction_rate=(
                "contains_self_correction_language",
                "mean",
            ),
            mean_repetition_score=("repetition_score", "mean"),
        )
    )


def build_matched_contrasts(
    attempts: pd.DataFrame,
    bootstrap_iterations: int = 5000,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for dataset, subset in attempts.groupby("dataset"):
        per_match = (
            subset.groupby(
                ["match_id", "baseline_cohort"],
                as_index=False,
            )["broad_loss"]
            .mean()
            .pivot(
                index="match_id",
                columns="baseline_cohort",
                values="broad_loss",
            )
        )
        for control in ("final_correct", "stable_wrong"):
            differences = (
                per_match["loss"] - per_match[control]
            ).to_numpy()
            samples = rng.choice(
                differences,
                size=(bootstrap_iterations, len(differences)),
                replace=True,
            ).mean(axis=1)
            records.append(
                {
                    "dataset": dataset,
                    "comparison": f"loss_vs_{control}",
                    "risk_difference": float(differences.mean()),
                    "ci_low": float(np.quantile(samples, 0.025)),
                    "ci_high": float(np.quantile(samples, 0.975)),
                    "match_count": len(differences),
                }
            )
    return pd.DataFrame(records)


def _save_figure(figure: plt.Figure, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_cohort_loss_rates(
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    output_dir: Path,
) -> Path:
    cohorts = ["loss", "final_correct", "stable_wrong"]
    labels = ["Seed-0 loss", "Final-correct", "Stable-wrong"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, (dataset, label) in zip(
        axes,
        (("mmlu_pro", "MMLU-Pro"), ("gpqa_diamond", "GPQA Diamond")),
    ):
        subset = summary.set_index(
            ["dataset", "baseline_cohort"]
        )
        values = [
            float(subset.loc[(dataset, cohort), "broad_loss_rate"])
            for cohort in cohorts
        ]
        axis.bar(labels, values, color=["#b91c1c", "#15803d", "#6b7280"])
        axis.set_title(label)
        axis.set_ylim(0, 1)
        axis.set_ylabel("Broad-loss rate across reruns")
        axis.tick_params(axis="x", rotation=15)
        for index, value in enumerate(values):
            axis.text(
                index,
                value + 0.02,
                f"{value:.1%}",
                ha="center",
            )
        contrast_rows = contrasts[
            contrasts["dataset"] == dataset
        ]
        annotation = "\n".join(
            (
                comparison.replace("loss_vs_", "Δ vs ")
                .replace("_", " ")
                + f": {risk_difference:+.1%} "
                f"[{ci_low:+.1%}, {ci_high:+.1%}]"
            )
            for comparison, risk_difference, ci_low, ci_high in (
                contrast_rows[
                    [
                        "comparison",
                        "risk_difference",
                        "ci_low",
                        "ci_high",
                    ]
                ].itertuples(index=False, name=None)
            )
        )
        axis.text(
            0.02,
            0.98,
            annotation,
            transform=axis.transAxes,
            va="top",
            fontsize=8,
        )
    return _save_figure(
        figure,
        output_dir / "cohort_loss_rates.png",
    )


def plot_predictive_auc(
    metrics: pd.DataFrame,
    output_dir: Path,
) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    colors = {
        "confidence": "#6b7280",
        "instability": "#dc2626",
        "instability_plus_length": "#2563eb",
        "full_prefix": "#7c3aed",
    }
    for axis, dataset in zip(
        axes,
        ("combined", "mmlu_pro", "gpqa_diamond"),
    ):
        subset = metrics[metrics["dataset"] == dataset]
        for model_name in MODEL_FEATURES:
            rows = subset[subset["model"] == model_name]
            axis.plot(
                rows["decile"],
                rows["auc"],
                marker="o",
                label=model_name.replace("_", " "),
                color=colors[model_name],
            )
        axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
        axis.set_title(dataset.replace("_", " ").title())
        axis.set_xlabel("Currently-correct checkpoint (%)")
        axis.set_ylim(0.45, 1.0)
        axis.set_ylabel("Question-grouped CV AUC")
    axes[-1].legend(frameon=False, fontsize=8)
    return _save_figure(
        figure,
        output_dir / "predictive_auc_by_decile.png",
    )


def plot_feature_coefficients(
    coefficients: pd.DataFrame,
    output_dir: Path,
) -> Path:
    subset = coefficients[
        (coefficients["dataset"] == "combined")
        & coefficients["decile"].isin([50, 80])
    ]
    pivot = subset.pivot(
        index="feature",
        columns="decile",
        values="coefficient",
    ).fillna(0)
    figure, axis = plt.subplots(figsize=(8, 5))
    image = axis.imshow(
        pivot.to_numpy(),
        aspect="auto",
        cmap="coolwarm",
        vmin=-max(0.1, abs(pivot.to_numpy()).max()),
        vmax=max(0.1, abs(pivot.to_numpy()).max()),
    )
    axis.set_yticks(
        range(len(pivot.index)),
        [value.replace("_", " ") for value in pivot.index],
    )
    axis.set_xticks(
        range(len(pivot.columns)),
        [f"{value}%" for value in pivot.columns],
    )
    axis.set_title("Standardized full-model coefficients")
    figure.colorbar(image, ax=axis, label="Higher means more loss risk")
    return _save_figure(
        figure,
        output_dir / "feature_coefficients.png",
    )


def _write_report(
    cohort_summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    metrics: pd.DataFrame,
    coefficients: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# Matched-control overthinking experiment",
        "",
        "Each dataset contributes 25 seed-0 loss questions, 25 "
        "final-correct controls, and 25 never-correct controls, matched "
        "within category and by seed-0 trace length. Every question was "
        "rerun at seeds 0 through 9.",
        "",
        "## Cohort recurrence",
        "",
        "| Dataset | Seed-0 cohort | Attempts | Broad losses | Rate | "
        "Final accuracy | Mean flips |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in cohort_summary.itertuples(index=False):
        lines.append(
            f"| {row.dataset_label} | "
            f"{row.baseline_cohort.replace('_', ' ')} | "
            f"{row.attempt_count} | {int(row.broad_loss_count)} | "
            f"{row.broad_loss_rate:.1%} | "
            f"{row.final_accuracy:.1%} | {row.mean_flips:.2f} |"
        )
    lines.extend(
        [
            "",
            "Matched risk differences use a 5,000-sample bootstrap over "
            "the 25 matched triplets:",
            "",
        ]
    )
    for row in contrasts.itertuples(index=False):
        lines.append(
            f"- **{row.dataset.replace('_', ' ').title()} "
            f"{row.comparison.replace('loss_vs_', 'loss vs ').replace('_', ' ')}:** "
            f"{row.risk_difference:+.1%} "
            f"(95% CI {row.ci_low:+.1%} to {row.ci_high:+.1%})."
        )

    lines.extend(
        [
            "",
            "## Prediction while currently correct",
            "",
            "AUCs use five-fold cross-validation grouped by question, so "
            "seeds of the same question cannot appear in both train and "
            "test folds.",
            "",
            "| Dataset | Checkpoint | Confidence AUC | Instability AUC | "
            "+ length AUC | + prefix language AUC |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in ("combined", "mmlu_pro", "gpqa_diamond"):
        for decile in (50, 80):
            subset = metrics[
                (metrics["dataset"] == dataset)
                & (metrics["decile"] == decile)
            ].set_index("model")
            if len(subset) != len(MODEL_FEATURES):
                continue
            lines.append(
                f"| {dataset.replace('_', ' ').title()} | {decile}% | "
                f"{subset.loc['confidence', 'auc']:.3f} | "
                f"{subset.loc['instability', 'auc']:.3f} | "
                f"{subset.loc['instability_plus_length', 'auc']:.3f} | "
                f"{subset.loc['full_prefix', 'auc']:.3f} |"
            )

    combined = coefficients[
        (coefficients["dataset"] == "combined")
        & (coefficients["decile"] == 80)
    ].sort_values("coefficient", ascending=False)
    lines.extend(
        [
            "",
            "## Strongest late-stage signals",
            "",
        ]
    )
    for row in combined.head(4).itertuples(index=False):
        lines.append(
            f"- `{row.feature}`: standardized coefficient "
            f"{row.coefficient:+.3f}."
        )
    lines.extend(
        [
            "",
            "The coefficients describe association within the matched "
            "sample, not a causal effect. Trace length is also an oracle "
            "feature because the eventual total length is unknown at an "
            "online stopping point.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_matched_controls(
    input_root: Path = Path(
        "outputs/matched_controls_gemma4_12b_extended"
    ),
    selection_path: Path = Path(
        "outputs/matched_controls_gemma4_12b/cohort_selection.parquet"
    ),
    output_dir: Path | None = None,
) -> dict[str, object]:
    output_dir = output_dir or input_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    attempts, checkpoints = build_matched_attempt_tables(
        input_root,
        selection_path,
    )
    cohort_summary = build_cohort_summary(attempts)
    contrasts = build_matched_contrasts(attempts)
    metrics, coefficients = evaluate_predictors(checkpoints)

    attempts.to_parquet(
        output_dir / "attempts.parquet",
        index=False,
    )
    checkpoints.to_parquet(
        output_dir / "currently_correct_checkpoints.parquet",
        index=False,
    )
    cohort_summary.to_parquet(
        output_dir / "cohort_summary.parquet",
        index=False,
    )
    contrasts.to_parquet(
        output_dir / "matched_contrasts.parquet",
        index=False,
    )
    metrics.to_parquet(
        output_dir / "predictive_metrics.parquet",
        index=False,
    )
    coefficients.to_parquet(
        output_dir / "predictive_coefficients.parquet",
        index=False,
    )
    figures = [
        plot_cohort_loss_rates(
            cohort_summary,
            contrasts,
            output_dir,
        ),
        plot_predictive_auc(metrics, output_dir),
        plot_feature_coefficients(coefficients, output_dir),
    ]
    report_path = output_dir / "matched_control_report.md"
    _write_report(
        cohort_summary,
        contrasts,
        metrics,
        coefficients,
        report_path,
    )
    summary = {
        "attempt_count": len(attempts),
        "checkpoint_count": len(checkpoints),
        "question_count": int(
            attempts[["dataset", "position"]]
            .drop_duplicates()
            .shape[0]
        ),
        "forced_completion_count": int(
            attempts["forced_completion"].sum()
        ),
        "cohorts": cohort_summary.to_dict(orient="records"),
        "matched_contrasts": contrasts.to_dict(orient="records"),
        "predictive_metrics": metrics.to_dict(orient="records"),
        "predictive_coefficients": coefficients.to_dict(
            orient="records"
        ),
    }
    with (output_dir / "matched_control_summary.json").open(
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(summary, output, indent=2, sort_keys=True)
        output.write("\n")
    return {
        "report_path": str(report_path),
        "summary_path": str(
            output_dir / "matched_control_summary.json"
        ),
        "figure_paths": [str(path) for path in figures],
        "summary": summary,
    }
