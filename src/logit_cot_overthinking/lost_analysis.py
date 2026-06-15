from __future__ import annotations

import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .gemma import extract_answer_letter
from .visualization import COLORS, _read_trace_records, _save_figure


DECILES = list(range(0, 101, 10))
SELF_CORRECTION_PATTERN = re.compile(
    r"\b(wait|re-?check|reconsider|double-check|one more (check|thought)|"
    r"self-correction)\b",
    re.IGNORECASE,
)
SIMULATED_RETRIEVAL_PATTERN = re.compile(
    r"(quizlet|test bank|search(?:ing)?|"
    r"found (?:a|the|another|several|multiple) "
    r"(?:source|reference|question|version)|"
    r"source (?:says|states)|study\.com)",
    re.IGNORECASE,
)


def _key(position: object, question_id: object) -> tuple[int, str]:
    return int(position), str(question_id)


def _native(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def load_complete_run(input_dir: Path) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    trajectory_path = input_dir / "trajectory.parquet"
    traces_path = input_dir / "traces.jsonl"
    for path in (trajectory_path, traces_path):
        if not path.exists():
            raise FileNotFoundError(f"Required run artifact not found: {path}")

    dataframe = pd.read_parquet(trajectory_path)
    traces = _read_trace_records(traces_path)
    for trace in traces:
        labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(trace["options"])])
        trace["generated_answer"] = extract_answer_letter(
            str(trace.get("generated_answer_text", "")),
            labels,
        )
    truncated_keys = {
        _key(trace["position"], trace["question_id"])
        for trace in traces
        if bool(trace.get("truncated", False))
    }
    complete = dataframe[
        ~dataframe.apply(
            lambda row: _key(row["position"], row["question_id"])
            in truncated_keys,
            axis=1,
        )
    ].copy()
    complete["question_id"] = complete["question_id"].astype(str)
    complete["correct_answer_probability"] = complete.apply(
        lambda row: float(row["choice_probabilities"][row["answer"]]),
        axis=1,
    )
    complete["normalized_correct_probability"] = np.where(
        complete["choice_probability_mass"] > 0,
        complete["correct_answer_probability"]
        / complete["choice_probability_mass"],
        np.nan,
    )
    return complete, traces


def build_lost_case_table(
    dataframe: pd.DataFrame,
    traces: list[dict[str, object]],
    confidence_threshold: float = 0.5,
    final_choice_mass_threshold: float = 0.5,
) -> pd.DataFrame:
    trace_by_key = {
        _key(trace["position"], trace["question_id"]): trace for trace in traces
    }
    cases: list[dict[str, object]] = []
    for (position, question_id), group in dataframe.groupby(
        ["position", "question_id"],
        sort=False,
    ):
        group = group.sort_values("decile")
        final = group.iloc[-1]
        correct_rows = group[group["correct"]]
        if correct_rows.empty or bool(final["correct"]):
            continue

        trace = trace_by_key[_key(position, question_id)]
        prefinal = group[group["decile"] < 100]
        peak_index = prefinal["correct_answer_probability"].idxmax()
        peak = prefinal.loc[peak_index]
        normalized_peak_index = prefinal[
            "normalized_correct_probability"
        ].idxmax()
        normalized_peak = prefinal.loc[normalized_peak_index]
        final_prediction = str(final["prediction"])
        final_prediction_probability = float(
            final["choice_probabilities"][final_prediction]
        )
        generated_answer = trace.get("generated_answer")
        strict_endpoint_lost = bool(group.iloc[0]["correct"])
        high_confidence_before_final = (
            float(peak["correct_answer_probability"]) >= confidence_threshold
        )
        reliable_final_probe = (
            float(final["choice_probability_mass"])
            >= final_choice_mass_threshold
        )
        generation_agrees_with_probe = generated_answer == final_prediction
        robust_loss = (
            high_confidence_before_final
            and reliable_final_probe
            and generation_agrees_with_probe
        )
        final_normalized_prediction_probability = (
            final_prediction_probability
            / float(final["choice_probability_mass"])
            if float(final["choice_probability_mass"]) > 0
            else np.nan
        )
        normalized_reversal_candidate = (
            float(normalized_peak["normalized_correct_probability"]) >= 0.9
            and final_normalized_prediction_probability >= 0.9
            and generation_agrees_with_probe
        )
        reasoning_trace = str(trace.get("reasoning_trace", ""))

        cases.append(
            {
                "position": int(position),
                "question_id": str(question_id),
                "category": str(final["category"]),
                "source": str(final["source"]),
                "question": str(final["question"]),
                "options": list(final["options"]),
                "answer": str(final["answer"]),
                "final_prediction": final_prediction,
                "generated_answer": generated_answer,
                "loss_type": (
                    "endpoint_lost"
                    if strict_endpoint_lost
                    else "gained_then_lost"
                ),
                "strict_endpoint_lost": strict_endpoint_lost,
                "robust_loss": robust_loss,
                "normalized_reversal_candidate": (
                    normalized_reversal_candidate
                ),
                "high_confidence_before_final": high_confidence_before_final,
                "reliable_final_probe": reliable_final_probe,
                "generation_agrees_with_probe": generation_agrees_with_probe,
                "first_correct_decile": int(correct_rows["decile"].min()),
                "last_correct_decile": int(correct_rows["decile"].max()),
                "correct_decile_count": int(len(correct_rows)),
                "flip_count": int(group["prediction_flip"].sum()),
                "prediction_path": group["prediction"].astype(str).tolist(),
                "correct_deciles": correct_rows["decile"].astype(int).tolist(),
                "trace_token_count": int(final["trace_token_count"]),
                "peak_correct_probability": float(
                    peak["correct_answer_probability"]
                ),
                "peak_normalized_correct_probability": float(
                    normalized_peak["normalized_correct_probability"]
                ),
                "peak_correct_decile": int(peak["decile"]),
                "peak_normalized_correct_decile": int(
                    normalized_peak["decile"]
                ),
                "final_correct_probability": float(
                    final["correct_answer_probability"]
                ),
                "final_normalized_correct_probability": float(
                    final["normalized_correct_probability"]
                ),
                "final_prediction_probability": final_prediction_probability,
                "final_normalized_prediction_probability": (
                    final_normalized_prediction_probability
                ),
                "final_choice_probability_mass": float(
                    final["choice_probability_mass"]
                ),
                "final_non_choice_probability": float(
                    final["non_choice_probability"]
                ),
                "contains_self_correction_language": bool(
                    SELF_CORRECTION_PATTERN.search(reasoning_trace)
                ),
                "contains_simulated_retrieval_language": bool(
                    SIMULATED_RETRIEVAL_PATTERN.search(reasoning_trace)
                ),
                "reasoning_trace": reasoning_trace,
                "generated_answer_text": str(
                    trace.get("generated_answer_text", "")
                ),
            }
        )

    result = pd.DataFrame(cases)
    if not result.empty:
        result.sort_values(
            [
                "robust_loss",
                "last_correct_decile",
                "peak_correct_probability",
            ],
            ascending=[False, False, False],
            inplace=True,
        )
        result.reset_index(drop=True, inplace=True)
    return result


def _question_metrics(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (position, question_id), group in dataframe.groupby(
        ["position", "question_id"],
        sort=False,
    ):
        group = group.sort_values("decile")
        any_correct = bool(group["correct"].any())
        final_correct = bool(group.iloc[-1]["correct"])
        if final_correct:
            comparison_group = "final correct"
        elif any_correct:
            comparison_group = "ever correct, final wrong"
        else:
            comparison_group = "never correct"
        rows.append(
            {
                "position": int(position),
                "question_id": str(question_id),
                "category": str(group.iloc[-1]["category"]),
                "comparison_group": comparison_group,
                "trace_token_count": int(group.iloc[-1]["trace_token_count"]),
                "flip_count": int(group["prediction_flip"].sum()),
            }
        )
    return pd.DataFrame(rows)


def build_lost_summary(
    dataframe: pd.DataFrame,
    cases: pd.DataFrame,
    confidence_threshold: float,
    final_choice_mass_threshold: float,
) -> dict[str, object]:
    question_metrics = _question_metrics(dataframe)
    category_totals = (
        question_metrics["category"].value_counts().sort_index().to_dict()
    )

    category_rows: list[dict[str, object]] = []
    for category, total in category_totals.items():
        subset = cases[cases["category"] == category]
        strict = int(subset["strict_endpoint_lost"].sum())
        robust = int(subset["robust_loss"].sum())
        broad = len(subset)
        category_rows.append(
            {
                "category": category,
                "complete_questions": int(total),
                "ever_correct_final_wrong": broad,
                "ever_correct_final_wrong_rate": broad / total,
                "endpoint_lost": strict,
                "endpoint_lost_rate": strict / total,
                "robust_loss": robust,
                "robust_loss_rate": robust / total,
            }
        )
    category_rows.sort(
        key=lambda row: row["ever_correct_final_wrong_rate"],
        reverse=True,
    )

    comparison: dict[str, dict[str, object]] = {}
    for name, group in question_metrics.groupby("comparison_group", sort=False):
        comparison[name] = {
            "count": len(group),
            "trace_tokens_mean": group["trace_token_count"].mean(),
            "trace_tokens_median": group["trace_token_count"].median(),
            "flips_mean": group["flip_count"].mean(),
            "flips_median": group["flip_count"].median(),
        }

    return _native(
        {
            "definitions": {
                "ever_correct_final_wrong": (
                    "Correct at one or more probe deciles and wrong at decile 100."
                ),
                "endpoint_lost": (
                    "Correct at decile 0 and wrong at decile 100."
                ),
                "robust_loss": (
                    "Ever-correct/final-wrong with a pre-final raw correct-answer "
                    f"probability >= {confidence_threshold}, final valid-letter "
                    f"mass >= {final_choice_mass_threshold}, and agreement between "
                    "the final probe and generated answer."
                ),
            },
            "complete_question_count": question_metrics.shape[0],
            "ever_correct_final_wrong_count": len(cases),
            "endpoint_lost_count": int(cases["strict_endpoint_lost"].sum()),
            "gained_then_lost_count": int(
                (~cases["strict_endpoint_lost"]).sum()
            ),
            "robust_loss_count": int(cases["robust_loss"].sum()),
            "normalized_reversal_candidate_count": int(
                cases["normalized_reversal_candidate"].sum()
            ),
            "generated_final_correct_count": int(
                (
                    cases["generated_answer"]
                    == cases["answer"]
                ).sum()
            ),
            "generated_final_wrong_count": int(
                (
                    cases["generated_answer"].notna()
                    & (cases["generated_answer"] != cases["answer"])
                ).sum()
            ),
            "robust_endpoint_lost_count": int(
                (
                    cases["robust_loss"]
                    & cases["strict_endpoint_lost"]
                ).sum()
            ),
            "robust_gained_then_lost_count": int(
                (
                    cases["robust_loss"]
                    & ~cases["strict_endpoint_lost"]
                ).sum()
            ),
            "category_rates": category_rows,
            "comparison_groups": comparison,
            "first_correct_decile_counts": (
                cases["first_correct_decile"]
                .value_counts()
                .sort_index()
                .to_dict()
            ),
            "last_correct_decile_counts": (
                cases["last_correct_decile"]
                .value_counts()
                .sort_index()
                .to_dict()
            ),
            "flip_count_distribution": (
                cases["flip_count"].value_counts().sort_index().to_dict()
            ),
            "probe_generation_agreement_rate": cases[
                "generation_agrees_with_probe"
            ].mean(),
            "median_final_choice_probability_mass": cases[
                "final_choice_probability_mass"
            ].median(),
            "robust_self_correction_language_count": int(
                cases.loc[
                    cases["robust_loss"],
                    "contains_self_correction_language",
                ].sum()
            ),
            "robust_simulated_retrieval_language_count": int(
                cases.loc[
                    cases["robust_loss"],
                    "contains_simulated_retrieval_language",
                ].sum()
            ),
        }
    )


def plot_loss_timing(cases: pd.DataFrame, output_dir: Path) -> Path:
    first = cases["first_correct_decile"].value_counts().reindex(DECILES, fill_value=0)
    last = cases["last_correct_decile"].value_counts().reindex(DECILES, fill_value=0)
    flips = cases["flip_count"].value_counts().sort_index()

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(DECILES))
    axes[0].bar(
        x - 0.2,
        first.to_numpy(),
        width=0.4,
        color=COLORS["blue"],
        label="First correct",
    )
    axes[0].bar(
        x + 0.2,
        last.to_numpy(),
        width=0.4,
        color=COLORS["red"],
        label="Last correct",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(DECILES)
    axes[0].set_xlabel("Reasoning trace revealed (%)")
    axes[0].set_ylabel("Cases")
    axes[0].set_title("When correctness appears and disappears")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(
        flips.index.astype(str),
        flips.to_numpy(),
        color=COLORS["purple"],
    )
    axes[1].set_xlabel("Prediction flips across adjacent deciles")
    axes[1].set_ylabel("Cases")
    axes[1].set_title("Lost cases often oscillate")
    axes[1].grid(axis="y", alpha=0.2)

    figure.suptitle(
        f"Timing of ever-correct, final-wrong cases (n={len(cases)})",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return _save_figure(figure, output_dir / "loss_timing.png")


def plot_category_loss_rates(
    summary: dict[str, object],
    output_dir: Path,
) -> Path:
    rates = pd.DataFrame(summary["category_rates"]).sort_values(
        "ever_correct_final_wrong_rate"
    )
    y = np.arange(len(rates))
    figure, axis = plt.subplots(figsize=(10, 7))
    axis.barh(
        y,
        rates["ever_correct_final_wrong_rate"],
        color=COLORS["orange"],
        alpha=0.82,
        label="Ever correct, final wrong",
    )
    axis.barh(
        y,
        rates["robust_loss_rate"],
        color=COLORS["red"],
        label="Robust loss",
    )
    axis.set_yticks(y)
    axis.set_yticklabels(rates["category"])
    axis.set_xlabel("Share of complete category questions")
    axis.set_title("Loss rate by category", fontweight="bold")
    axis.grid(axis="x", alpha=0.2)
    axis.legend()
    for index, row in enumerate(rates.itertuples()):
        axis.text(
            row.ever_correct_final_wrong_rate + 0.003,
            index,
            f"{row.ever_correct_final_wrong}/{row.complete_questions}",
            va="center",
            fontsize=8,
        )
    figure.tight_layout()
    return _save_figure(figure, output_dir / "lost_category_rates.png")


def plot_group_comparison(
    dataframe: pd.DataFrame,
    output_dir: Path,
) -> Path:
    metrics = _question_metrics(dataframe)
    order = ["final correct", "ever correct, final wrong", "never correct"]
    colors = [COLORS["green"], COLORS["red"], COLORS["gray"]]
    labels = [
        f"{name}\n(n={(metrics['comparison_group'] == name).sum()})"
        for name in order
    ]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, column, title, logarithmic in (
        (axes[0], "trace_token_count", "Reasoning trace length", True),
        (axes[1], "flip_count", "Prediction instability", False),
    ):
        groups = [
            metrics.loc[metrics["comparison_group"] == name, column].to_numpy()
            for name in order
        ]
        box = axis.boxplot(
            groups,
            tick_labels=labels,
            patch_artist=True,
            showfliers=False,
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        if logarithmic:
            axis.set_yscale("log")
            axis.set_ylabel("Trace tokens (log scale)")
        else:
            axis.set_ylabel("Number of flips")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Lost cases reason longer and change answers more often",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return _save_figure(figure, output_dir / "lost_vs_other_cases.png")


def plot_lost_case_heatmap(
    dataframe: pd.DataFrame,
    cases: pd.DataFrame,
    output_dir: Path,
) -> Path:
    order = [
        _key(row.position, row.question_id) for row in cases.itertuples()
    ]
    working = dataframe.copy()
    working["case_key"] = [
        _key(row.position, row.question_id) for row in working.itertuples()
    ]
    matrix = working.pivot(
        index="case_key",
        columns="decile",
        values="normalized_correct_probability",
    ).reindex(order)

    height = max(8, len(cases) * 0.16)
    figure, axis = plt.subplots(figsize=(11, height))
    image = axis.imshow(
        matrix.to_numpy(),
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    axis.axvline(9.5, color="white", linewidth=1.4, linestyle="--")
    axis.set_xticks(range(len(matrix.columns)))
    axis.set_xticklabels(matrix.columns)
    axis.set_yticks(range(len(cases)))
    axis.set_yticklabels(
        [
            (
                f"{'*' if row.robust_loss else ' '} {row.question_id}  "
                f"{row.category}  {row.answer}->{row.final_prediction}"
            )
            for row in cases.itertuples()
        ],
        fontsize=5,
    )
    axis.set_xlabel("Reasoning trace revealed (%)")
    axis.set_title(
        "Correct-answer share of valid-letter probability in lost cases",
        fontweight="bold",
    )
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Normalized probability on correct answer")
    figure.text(
        0.5,
        0.005,
        "* robust loss; dashed line separates the final probe.",
        ha="center",
        color=COLORS["gray"],
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.015, 1, 1))
    return _save_figure(figure, output_dir / "lost_case_heatmap.png")


def plot_reversal_trajectories(
    dataframe: pd.DataFrame,
    cases: pd.DataFrame,
    output_dir: Path,
    flag_column: str,
    filename: str,
    title: str,
) -> Path | None:
    selected = cases[cases[flag_column]]
    if selected.empty:
        return None

    columns = 4
    rows = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(15, rows * 3.1),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for axis, case in zip(axes.flat, selected.itertuples()):
        subset = dataframe[
            (dataframe["position"] == case.position)
            & (dataframe["question_id"] == case.question_id)
        ].sort_values("decile")
        final_wrong_probability = subset["choice_probabilities"].apply(
            lambda values: float(values[case.final_prediction])
        )
        mass = subset["choice_probability_mass"].astype(float)
        normalized_wrong = np.where(mass > 0, final_wrong_probability / mass, np.nan)
        axis.plot(
            subset["decile"],
            subset["normalized_correct_probability"],
            marker="o",
            linewidth=2.2,
            color=COLORS["green"],
            label="Correct answer",
        )
        axis.plot(
            subset["decile"],
            normalized_wrong,
            marker="o",
            linewidth=2.2,
            color=COLORS["red"],
            label="Final wrong answer",
        )
        axis.axvline(case.last_correct_decile, color=COLORS["gray"], alpha=0.35)
        axis.set_title(
            f"Q{case.question_id} | {case.category} | "
            f"{case.answer}->{case.final_prediction}\n"
            f"last correct {case.last_correct_decile}% | "
            f"{case.trace_token_count:,} tokens",
            fontsize=9,
        )
        axis.set_ylim(-0.03, 1.03)
        axis.set_xticks([0, 20, 40, 60, 80, 100])
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(selected) :]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.974),
        ncols=2,
    )
    figure.supxlabel("Reasoning trace revealed (%)")
    figure.supylabel("Probability normalized over valid answer letters")
    figure.suptitle(
        f"{title} (n={len(selected)})",
        fontsize=15,
        fontweight="bold",
        y=0.998,
    )
    figure.tight_layout(rect=(0.02, 0.02, 1, 0.93))
    return _save_figure(figure, output_dir / filename)


def _write_report(
    summary: dict[str, object],
    cases: pd.DataFrame,
    output_path: Path,
) -> Path:
    comparison = summary["comparison_groups"]
    broad = comparison["ever correct, final wrong"]
    final_correct = comparison["final correct"]
    category_rows = summary["category_rates"]
    top_categories = category_rows[:5]
    robust_categories = sorted(
        (
            row
            for row in category_rows
            if int(row["robust_loss"]) > 0
        ),
        key=lambda row: int(row["robust_loss"]),
        reverse=True,
    )
    top_robust = robust_categories[:3]
    top_robust_count = sum(int(row["robust_loss"]) for row in top_robust)
    broad_only_leaders = [
        row["category"]
        for row in category_rows[:4]
        if int(row["robust_loss"]) == 0
    ]
    late_losses = int((cases["last_correct_decile"] >= 80).sum())
    robust = cases[cases["robust_loss"]]

    lines = [
        "# Lost-case analysis",
        "",
        "Complete traces only.",
        "",
        "## Main counts",
        "",
        f"- Ever correct, final wrong: {summary['ever_correct_final_wrong_count']}",
        f"- Endpoint lost (correct at 0%, wrong at 100%): {summary['endpoint_lost_count']}",
        f"- Gained then lost: {summary['gained_then_lost_count']}",
        f"- Confidence-filtered robust losses: {summary['robust_loss_count']}",
        (
            f"- Normalized reversal candidates: "
            f"{summary['normalized_reversal_candidate_count']}"
        ),
        (
            f"- Broad losses ending with a correct generated answer: "
            f"{summary['generated_final_correct_count']}"
        ),
        (
            f"- Broad losses ending with an explicit wrong generated answer: "
            f"{summary['generated_final_wrong_count']}"
        ),
        (
            f"- Robust losses hidden by the endpoint label: "
            f"{summary['robust_gained_then_lost_count']}"
        ),
        "",
        "## Quantitative observations",
        "",
        (
            f"- {late_losses} cases were still correct at 80% or 90% of the "
            "trace before ending wrong."
        ),
        (
            f"- Lost cases had a median {broad['trace_tokens_median']:.0f} "
            f"reasoning tokens versus {final_correct['trace_tokens_median']:.0f} "
            "for final-correct cases."
        ),
        (
            f"- Lost cases had a median {broad['flips_median']:.0f} prediction "
            f"flips versus {final_correct['flips_median']:.0f} for final-correct cases."
        ),
        (
            f"- The median final valid-letter probability mass was only "
            f"{summary['median_final_choice_probability_mass']:.2%}; probe and "
            f"generated answer agreed in "
            f"{summary['probe_generation_agreement_rate']:.1%} of broad losses."
        ),
        *(
            [
                (
                    f"- {summary['robust_simulated_retrieval_language_count']} "
                    f"of {summary['robust_loss_count']} robust traces used "
                    "simulated retrieval language such as searching for a "
                    "source, test bank, or exact answer."
                )
            ]
            if summary["robust_loss_count"]
            else []
        ),
        "- Highest broad loss rates: "
        + ", ".join(
            f"{row['category']} {row['ever_correct_final_wrong_rate']:.1%}"
            for row in top_categories
        )
        + ".",
        *(
            [
                (
                    f"- {top_robust_count} of "
                    f"{summary['robust_loss_count']} robust losses came from "
                    + ", ".join(
                        str(row["category"]) for row in top_robust
                    )
                    + "."
                )
            ]
            if top_robust
            else []
        ),
        (
            "- The leading broad-loss categories had no robust cases: "
            + ", ".join(str(category) for category in broad_only_leaders)
            + ". This indicates that low final answer-letter mass explains much "
            "of their apparent loss rate."
        ),
        "",
        "## Robust cases",
        "",
        "| Question | Category | Change | Last correct | Peak correct p | Tokens |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in robust.itertuples():
        lines.append(
            f"| {row.question_id} | {row.category} | "
            f"{row.answer}->{row.final_prediction} | {row.last_correct_decile}% | "
            f"{row.peak_correct_probability:.3f} | {row.trace_token_count:,} |"
        )
    normalized = cases[cases["normalized_reversal_candidate"]]
    lines.extend(
        [
            "",
            "The robust filter requires high pre-final raw answer probability, "
            "substantial final probability mass on valid letters, and agreement "
            "between the final probe and the generated answer.",
            "",
            "## Normalized reversal candidates",
            "",
            "| Question | Change | First correct | Last correct | Peak raw p | "
            "Final letter mass |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in normalized.itertuples():
        lines.append(
            f"| {row.question_id} | {row.answer}->{row.final_prediction} | "
            f"{row.first_correct_decile}% | {row.last_correct_decile}% | "
            f"{row.peak_correct_probability:.3f} | "
            f"{row.final_choice_probability_mass:.3f} |"
        )
    lines.extend(
        [
            "",
            "Normalized candidates require at least 90% of valid-letter "
            "probability on the correct answer before the end, at least 90% on "
            "the final wrong answer at the end, and agreement with the generated "
            "answer. They can still have low absolute answer-letter probability.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def analyze_lost_cases(
    input_dir: Path,
    output_dir: Path,
    confidence_threshold: float = 0.5,
    final_choice_mass_threshold: float = 0.5,
) -> list[Path]:
    dataframe, traces = load_complete_run(input_dir)
    cases = build_lost_case_table(
        dataframe,
        traces,
        confidence_threshold=confidence_threshold,
        final_choice_mass_threshold=final_choice_mass_threshold,
    )
    if cases.empty:
        raise ValueError("No ever-correct, final-wrong cases were found")
    summary = build_lost_summary(
        dataframe,
        cases,
        confidence_threshold,
        final_choice_mass_threshold,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "lost_cases.parquet"
    cases.to_parquet(cases_path, index=False)
    robust_csv_path = output_dir / "robust_lost_cases.csv"
    robust_columns = [
        "position",
        "question_id",
        "category",
        "source",
        "answer",
        "final_prediction",
        "last_correct_decile",
        "peak_correct_probability",
        "final_choice_probability_mass",
        "flip_count",
        "trace_token_count",
        "question",
    ]
    cases.loc[cases["robust_loss"], robust_columns].to_csv(
        robust_csv_path,
        index=False,
    )
    summary_path = output_dir / "lost_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = _write_report(summary, cases, output_dir / "lost_report.md")

    paths: list[Path] = [
        cases_path,
        robust_csv_path,
        summary_path,
        report_path,
        plot_loss_timing(cases, output_dir),
        plot_category_loss_rates(summary, output_dir),
        plot_group_comparison(dataframe, output_dir),
        plot_lost_case_heatmap(dataframe, cases, output_dir),
    ]
    robust_plot = plot_reversal_trajectories(
        dataframe,
        cases,
        output_dir,
        flag_column="robust_loss",
        filename="robust_loss_trajectories.png",
        title="Confidence-filtered answer reversals",
    )
    normalized_plot = plot_reversal_trajectories(
        dataframe,
        cases,
        output_dir,
        flag_column="normalized_reversal_candidate",
        filename="normalized_reversal_trajectories.png",
        title="Normalized answer-reversal candidates",
    )
    for plot in (robust_plot, normalized_plot):
        if plot is not None:
            paths.append(plot)
    return paths
