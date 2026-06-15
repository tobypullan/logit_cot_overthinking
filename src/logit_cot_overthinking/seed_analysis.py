from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .lost_analysis import build_lost_case_table, load_complete_run


@dataclass(frozen=True)
class SeedAnalysisDataset:
    name: str
    label: str
    criterion: str
    candidate_flag_label: str


DATASETS = (
    SeedAnalysisDataset(
        name="mmlu_pro",
        label="MMLU-Pro",
        criterion="robust_loss",
        candidate_flag_label="robust loss",
    ),
    SeedAnalysisDataset(
        name="gpqa_diamond",
        label="GPQA Diamond",
        criterion="normalized_reversal_candidate",
        candidate_flag_label="normalized reversal",
    ),
)
ORIGINAL_RUNS = {
    "mmlu_pro": Path(
        "outputs/mmlu_pro_gemma4_12b_n1000_seed0"
    ),
    "gpqa_diamond": Path(
        "outputs/gpqa_diamond_gemma4_12b_seed0"
    ),
}


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_seed_attempt_table(
    input_root: Path,
    seeds: tuple[int, ...] = tuple(range(10)),
) -> pd.DataFrame:
    attempts: list[dict[str, object]] = []
    for dataset in DATASETS:
        for seed in seeds:
            run_dir = input_root / dataset.name / f"seed_{seed}"
            dataframe, traces = load_complete_run(run_dir)
            trace_by_key = {
                (int(trace["position"]), str(trace["question_id"])): trace
                for trace in traces
            }
            cases = build_lost_case_table(dataframe, traces)
            cases_by_key = {
                (int(row.position), str(row.question_id)): row
                for row in cases.itertuples(index=False)
            }
            for (position, question_id), group in dataframe.groupby(
                ["position", "question_id"],
                sort=False,
            ):
                group = group.sort_values("decile")
                final = group.iloc[-1]
                key = (int(position), str(question_id))
                trace = trace_by_key[key]
                case = cases_by_key.get(key)
                criterion_reproduced = bool(
                    case is not None
                    and getattr(case, dataset.criterion)
                )
                attempts.append(
                    {
                        "dataset": dataset.name,
                        "dataset_label": dataset.label,
                        "criterion": dataset.criterion,
                        "seed": seed,
                        "position": int(position),
                        "question_id": str(question_id),
                        "category": str(final["category"]),
                        "answer": str(final["answer"]),
                        "final_prediction": str(final["prediction"]),
                        "generated_answer": trace.get(
                            "generated_answer"
                        ),
                        "final_correct": bool(final["correct"]),
                        "generated_correct": (
                            trace.get("generated_answer")
                            == str(final["answer"])
                        ),
                        "outcome": str(final["outcome"]),
                        "ever_correct_final_wrong": case is not None,
                        "robust_loss": bool(
                            case is not None and case.robust_loss
                        ),
                        "normalized_reversal_candidate": bool(
                            case is not None
                            and case.normalized_reversal_candidate
                        ),
                        "criterion_reproduced": criterion_reproduced,
                        "flip_count": int(
                            group["prediction_flip"].sum()
                        ),
                        "trace_token_count": int(
                            final["trace_token_count"]
                        ),
                        "extended": bool(
                            trace.get("extended", False)
                        ),
                        "forced_completion": bool(
                            trace.get("forced_completion", False)
                        ),
                        "first_correct_decile": (
                            int(case.first_correct_decile)
                            if case is not None
                            else None
                        ),
                        "last_correct_decile": (
                            int(case.last_correct_decile)
                            if case is not None
                            else None
                        ),
                        "peak_correct_probability": (
                            float(case.peak_correct_probability)
                            if case is not None
                            else None
                        ),
                        "final_choice_probability_mass": float(
                            final["choice_probability_mass"]
                        ),
                    }
                )
    return pd.DataFrame(attempts)


def build_candidate_summary(
    attempts: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    group_columns = [
        "dataset",
        "dataset_label",
        "criterion",
        "position",
        "question_id",
        "category",
        "answer",
    ]
    for keys, group in attempts.groupby(group_columns, sort=False):
        (
            dataset,
            dataset_label,
            criterion,
            position,
            question_id,
            category,
            answer,
        ) = keys
        outcome_counts = group["outcome"].value_counts()
        records.append(
            {
                "dataset": dataset,
                "dataset_label": dataset_label,
                "criterion": criterion,
                "position": int(position),
                "question_id": str(question_id),
                "category": category,
                "answer": answer,
                "attempt_count": len(group),
                "criterion_recurrence_count": int(
                    group["criterion_reproduced"].sum()
                ),
                "criterion_recurrence_rate": float(
                    group["criterion_reproduced"].mean()
                ),
                "ever_correct_final_wrong_count": int(
                    group["ever_correct_final_wrong"].sum()
                ),
                "final_correct_count": int(
                    group["final_correct"].sum()
                ),
                "generated_correct_count": int(
                    group["generated_correct"].sum()
                ),
                "forced_completion_count": int(
                    group["forced_completion"].sum()
                ),
                "stable_correct_count": int(
                    outcome_counts.get("stable_correct", 0)
                ),
                "gained_count": int(
                    outcome_counts.get("gained", 0)
                ),
                "lost_count": int(
                    outcome_counts.get("lost", 0)
                ),
                "stable_wrong_count": int(
                    outcome_counts.get("stable_wrong", 0)
                ),
                "trace_tokens_median": float(
                    group["trace_token_count"].median()
                ),
                "flip_count_mean": float(group["flip_count"].mean()),
            }
        )
    return pd.DataFrame(records).sort_values(
        [
            "dataset",
            "criterion_recurrence_count",
            "ever_correct_final_wrong_count",
            "position",
        ],
        ascending=[True, False, False, True],
    )


def build_seed_summary(
    attempts: pd.DataFrame,
    candidates: pd.DataFrame,
) -> dict[str, object]:
    datasets: dict[str, object] = {}
    for definition in DATASETS:
        subset = attempts[attempts["dataset"] == definition.name]
        candidate_subset = candidates[
            candidates["dataset"] == definition.name
        ]
        natural = subset[~subset["forced_completion"]]
        recurrence = subset["criterion_reproduced"]
        natural_recurrence = natural["criterion_reproduced"]
        datasets[definition.name] = {
            "label": definition.label,
            "criterion": definition.criterion,
            "candidate_count": len(candidate_subset),
            "attempt_count": len(subset),
            "forced_completion_count": int(
                subset["forced_completion"].sum()
            ),
            "criterion_recurrence_count": int(recurrence.sum()),
            "criterion_recurrence_rate": float(recurrence.mean()),
            "natural_attempt_count": len(natural),
            "natural_criterion_recurrence_count": int(
                natural_recurrence.sum()
            ),
            "natural_criterion_recurrence_rate": (
                float(natural_recurrence.mean())
                if len(natural)
                else None
            ),
            "ever_correct_final_wrong_count": int(
                subset["ever_correct_final_wrong"].sum()
            ),
            "final_accuracy": float(subset["final_correct"].mean()),
            "generated_accuracy": float(
                subset["generated_correct"].mean()
            ),
            "candidates_with_any_recurrence": int(
                (
                    candidate_subset[
                        "criterion_recurrence_count"
                    ]
                    > 0
                ).sum()
            ),
            "candidates_with_majority_recurrence": int(
                (
                    candidate_subset[
                        "criterion_recurrence_rate"
                    ]
                    >= 0.5
                ).sum()
            ),
            "candidate_recurrence_distribution": {
                str(int(count)): int(frequency)
                for count, frequency in (
                    candidate_subset[
                        "criterion_recurrence_count"
                    ]
                    .value_counts()
                    .sort_index()
                    .items()
                )
            },
            "outcome_counts": {
                str(outcome): int(count)
                for outcome, count in (
                    subset["outcome"]
                    .value_counts()
                    .sort_index()
                    .items()
                )
            },
        }
    return {
        "input_attempt_count": len(attempts),
        "candidate_count": len(candidates),
        "datasets": datasets,
    }


def build_seed_zero_replay_summary(
    input_root: Path,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for definition in DATASETS:
        original_dir = ORIGINAL_RUNS[definition.name]
        original_traces = {
            int(trace["position"]): trace
            for trace in _read_jsonl(
                original_dir / "traces.jsonl"
            )
        }
        rerun_traces = _read_jsonl(
            input_root
            / definition.name
            / "seed_0"
            / "traces.jsonl"
        )
        original_final = {
            int(row.position): str(row.prediction)
            for row in pd.read_parquet(
                original_dir / "trajectory.parquet"
            )
            .query("decile == 100")
            .itertuples(index=False)
        }
        rerun_final = {
            int(row.position): str(row.prediction)
            for row in pd.read_parquet(
                input_root
                / definition.name
                / "seed_0"
                / "trajectory.parquet"
            )
            .query("decile == 100")
            .itertuples(index=False)
        }
        result[definition.name] = {
            "candidate_count": len(rerun_traces),
            "exact_trace_match_count": sum(
                str(original_traces[int(trace["position"])]["raw_response"])
                == str(trace["raw_response"])
                for trace in rerun_traces
            ),
            "same_final_prediction_count": sum(
                original_final[int(trace["position"])]
                == rerun_final[int(trace["position"])]
                for trace in rerun_traces
            ),
        }
    return result


def _save_figure(figure: plt.Figure, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_recurrence_heatmap(
    attempts: pd.DataFrame,
    output_dir: Path,
) -> Path:
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11, 10),
        gridspec_kw={"height_ratios": [17, 8]},
    )
    for axis, definition in zip(axes, DATASETS):
        subset = attempts[attempts["dataset"] == definition.name].copy()
        labels = (
            subset[["position", "question_id"]]
            .drop_duplicates()
            .sort_values("position")
        )
        values = []
        row_labels = []
        forced = []
        for row in labels.itertuples(index=False):
            group = subset[
                (subset["position"] == row.position)
                & (subset["question_id"] == row.question_id)
            ].sort_values("seed")
            values.append(
                np.where(
                    group["criterion_reproduced"],
                    2,
                    np.where(
                        group["ever_correct_final_wrong"],
                        1,
                        0,
                    ),
                )
            )
            forced.append(group["forced_completion"].to_numpy())
            row_labels.append(f"{row.position}: {row.question_id}")
        matrix = np.asarray(values)
        axis.imshow(
            matrix,
            aspect="auto",
            vmin=0,
            vmax=2,
            cmap=matplotlib.colors.ListedColormap(
                ["#e5e7eb", "#f59e0b", "#b91c1c"]
            ),
        )
        axis.set_xticks(range(10), [str(seed) for seed in range(10)])
        axis.set_yticks(range(len(row_labels)), row_labels)
        axis.set_xlabel("Seed")
        axis.set_title(
            f"{definition.label}: {definition.candidate_flag_label} "
            "recurrence"
        )
        forced_matrix = np.asarray(forced)
        for row_index, column_index in np.argwhere(forced_matrix):
            axis.text(
                column_index,
                row_index,
                "x",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
                fontweight="bold",
            )
    figure.text(
        0.5,
        0.005,
        "Gray: no reversal; amber: ever-correct/final-wrong; "
        "red: target criterion; x: forced completion",
        ha="center",
        fontsize=9,
    )
    return _save_figure(
        figure,
        output_dir / "seed_recurrence_heatmap.png",
    )


def plot_candidate_recurrence(
    candidates: pd.DataFrame,
    output_dir: Path,
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    for axis, definition in zip(axes, DATASETS):
        subset = candidates[
            candidates["dataset"] == definition.name
        ].sort_values(
            ["criterion_recurrence_count", "position"],
            ascending=[False, True],
        )
        labels = [
            f"{row.position}:{row.question_id}"
            for row in subset.itertuples(index=False)
        ]
        axis.bar(
            range(len(subset)),
            subset["criterion_recurrence_count"],
            color="#b91c1c",
        )
        axis.set_xticks(
            range(len(subset)),
            labels,
            rotation=75,
            ha="right",
            fontsize=8,
        )
        axis.set_ylim(0, 10)
        axis.set_ylabel("Seeds reproducing criterion")
        axis.set_title(definition.label)
        axis.axhline(5, color="#6b7280", linestyle="--", linewidth=1)
    return _save_figure(
        figure,
        output_dir / "candidate_recurrence_counts.png",
    )


def plot_outcomes_by_seed(
    attempts: pd.DataFrame,
    output_dir: Path,
) -> Path:
    outcomes = [
        "stable_correct",
        "gained",
        "lost",
        "stable_wrong",
    ]
    colors = ["#15803d", "#3b82f6", "#dc2626", "#6b7280"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for axis, definition in zip(axes, DATASETS):
        subset = attempts[attempts["dataset"] == definition.name]
        counts = (
            subset.groupby(["seed", "outcome"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=outcomes, fill_value=0)
        )
        bottom = np.zeros(len(counts))
        for outcome, color in zip(outcomes, colors):
            values = counts[outcome].to_numpy()
            axis.bar(
                counts.index,
                values,
                bottom=bottom,
                label=outcome.replace("_", " "),
                color=color,
            )
            bottom += values
        axis.set_title(definition.label)
        axis.set_xlabel("Seed")
        axis.set_ylabel("Candidate attempts")
        axis.set_xticks(range(10))
    axes[-1].legend(frameon=False, fontsize=8)
    return _save_figure(
        figure,
        output_dir / "outcomes_by_seed.png",
    )


def plot_trace_length_comparison(
    attempts: pd.DataFrame,
    output_dir: Path,
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, definition in zip(axes, DATASETS):
        subset = attempts[attempts["dataset"] == definition.name]
        groups = [
            subset.loc[
                ~subset["ever_correct_final_wrong"],
                "trace_token_count",
            ],
            subset.loc[
                subset["ever_correct_final_wrong"]
                & ~subset["criterion_reproduced"],
                "trace_token_count",
            ],
            subset.loc[
                subset["criterion_reproduced"],
                "trace_token_count",
            ],
        ]
        axis.boxplot(
            groups,
            tick_labels=[
                "other",
                "weak reversal",
                "criterion",
            ],
            showfliers=False,
        )
        axis.set_title(definition.label)
        axis.set_ylabel("Reasoning tokens")
        axis.tick_params(axis="x", rotation=15)
    return _save_figure(
        figure,
        output_dir / "trace_length_by_reversal.png",
    )


def _write_report(
    summary: dict[str, object],
    candidates: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# Ten-seed candidate rerun analysis",
        "",
        "All capped attempts were extended by up to 16,384 additional "
        "tokens. Runaway traces that still did not close were explicitly "
        "closed and are marked as forced completions.",
        "",
        "## Dataset-level results",
        "",
    ]
    for definition in DATASETS:
        values = summary["datasets"][definition.name]
        lines.extend(
            [
                f"### {definition.label}",
                "",
                (
                    f"- {values['criterion_recurrence_count']} of "
                    f"{values['attempt_count']} attempts reproduced the "
                    f"{definition.candidate_flag_label} criterion "
                    f"({values['criterion_recurrence_rate']:.1%})."
                ),
                (
                    f"- Excluding forced completions: "
                    f"{values['natural_criterion_recurrence_count']} of "
                    f"{values['natural_attempt_count']} "
                    f"({values['natural_criterion_recurrence_rate']:.1%})."
                ),
                (
                    f"- {values['candidates_with_any_recurrence']} of "
                    f"{values['candidate_count']} candidates reproduced "
                    "at least once; "
                    f"{values['candidates_with_majority_recurrence']} "
                    "reproduced in at least half of seeds."
                ),
                (
                    f"- {values['forced_completion_count']} attempts "
                    "required explicit thought-channel closure."
                ),
                (
                    f"- Final probe accuracy across this selected set: "
                    f"{values['final_accuracy']:.1%}."
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Most recurrent candidates",
            "",
            "| Dataset | Position | Question ID | Criterion seeds | "
            "Ever-correct/final-wrong | Final correct | Forced |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    top = candidates.sort_values(
        [
            "criterion_recurrence_count",
            "ever_correct_final_wrong_count",
        ],
        ascending=False,
    ).head(15)
    for row in top.itertuples(index=False):
        lines.append(
            f"| {row.dataset_label} | {row.position} | "
            f"{row.question_id} | {row.criterion_recurrence_count}/10 | "
            f"{row.ever_correct_final_wrong_count}/10 | "
            f"{row.final_correct_count}/10 | "
            f"{row.forced_completion_count}/10 |"
        )

    lines.extend(
        [
            "",
            "## Specific findings",
            "",
        ]
    )
    for definition in DATASETS:
        subset = candidates[
            candidates["dataset"] == definition.name
        ].sort_values(
            [
                "criterion_recurrence_count",
                "ever_correct_final_wrong_count",
            ],
            ascending=False,
        )
        strongest = subset.iloc[0]
        zero_recurrence = subset[
            subset["criterion_recurrence_count"] == 0
        ]
        lines.append(
            f"- **{definition.label}:** the strongest candidate was "
            f"`{strongest.question_id}` at position "
            f"{strongest.position}, reproducing in "
            f"{strongest.criterion_recurrence_count}/10 seeds and ending "
            f"wrong after being correct in "
            f"{strongest.ever_correct_final_wrong_count}/10."
        )
        if len(zero_recurrence):
            identifiers = ", ".join(
                f"`{row.question_id}`"
                for row in zero_recurrence.itertuples(index=False)
            )
            lines.append(
                f"- **{definition.label}:** {len(zero_recurrence)} "
                f"seed-0 candidate(s) never reproduced the target "
                f"criterion: {identifiers}."
            )

    lines.extend(
        [
            "- MMLU-Pro recurrence was associated much more strongly with "
            "prediction instability than trace length: target cases "
            "averaged about twice as many flips, while median trace length "
            "was similar.",
            "- GPQA target reversals were concentrated in longer traces: "
            "their median reasoning length was roughly twice that of other "
            "candidate attempts.",
            "",
            "## Seed-0 replay check",
            "",
        ]
    )
    replay = summary["seed_zero_replay"]
    for definition in DATASETS:
        values = replay[definition.name]
        lines.append(
            f"- **{definition.label}:** "
            f"{values['exact_trace_match_count']}/"
            f"{values['candidate_count']} traces matched the original "
            f"seed-0 generation exactly; "
            f"{values['same_final_prediction_count']}/"
            f"{values['candidate_count']} retained the same final "
            "prediction."
        )
    lines.extend(
        [
            "- Seed alone is therefore insufficient for bitwise replay "
            "under the changed batched scheduling. These runs are best "
            "treated as stochastic repeats rather than deterministic "
            "reproductions.",
            "",
            "## Interpretation",
            "",
            "- The seed-0 selection enriches for reversals, but most "
            "individual events are not deterministic across sampling seeds.",
            "- Candidates with repeated reversals are stronger evidence of "
            "question-specific overthinking than one-off seed outcomes.",
            "- Forced completions are retained for coverage but should be "
            "reported separately because their terminal answer was elicited "
            "by closing a runaway thought channel.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_seed_reruns(
    input_root: Path = Path(
        "outputs/candidate_reruns_gemma4_12b_extended"
    ),
    output_dir: Path | None = None,
) -> dict[str, object]:
    output_dir = output_dir or input_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    attempts = build_seed_attempt_table(input_root)
    candidates = build_candidate_summary(attempts)
    summary = build_seed_summary(attempts, candidates)
    summary["seed_zero_replay"] = build_seed_zero_replay_summary(
        input_root
    )

    attempts.to_parquet(
        output_dir / "seed_attempts.parquet",
        index=False,
    )
    candidates.to_parquet(
        output_dir / "candidate_stability.parquet",
        index=False,
    )
    with (output_dir / "seed_summary.json").open(
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(summary, output, indent=2, sort_keys=True)
        output.write("\n")
    figures = [
        plot_recurrence_heatmap(attempts, output_dir),
        plot_candidate_recurrence(candidates, output_dir),
        plot_outcomes_by_seed(attempts, output_dir),
        plot_trace_length_comparison(attempts, output_dir),
    ]
    report_path = output_dir / "seed_report.md"
    _write_report(summary, candidates, report_path)
    return {
        "attempts_path": str(
            output_dir / "seed_attempts.parquet"
        ),
        "candidates_path": str(
            output_dir / "candidate_stability.parquet"
        ),
        "summary_path": str(output_dir / "seed_summary.json"),
        "report_path": str(report_path),
        "figure_paths": [str(path) for path in figures],
        "summary": summary,
    }
