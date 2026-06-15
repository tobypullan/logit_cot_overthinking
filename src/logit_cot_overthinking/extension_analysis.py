from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExtensionPair:
    name: str
    label: str
    original_dir: Path
    extended_dir: Path
    criterion: str


PAIRS = (
    ExtensionPair(
        name="gpqa_diamond",
        label="GPQA Diamond",
        original_dir=Path(
            "outputs/gpqa_diamond_gemma4_12b_seed0"
        ),
        extended_dir=Path(
            "outputs/gpqa_diamond_gemma4_12b_seed0_extended"
        ),
        criterion="normalized_reversal_candidate",
    ),
    ExtensionPair(
        name="mmlu_pro",
        label="MMLU-Pro",
        original_dir=Path(
            "outputs/mmlu_pro_gemma4_12b_n1000_seed0"
        ),
        extended_dir=Path(
            "outputs/mmlu_pro_gemma4_12b_n1000_seed0_extended"
        ),
        criterion="robust_loss",
    ),
)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_extension_table(
    pairs: tuple[ExtensionPair, ...] = PAIRS,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for pair in pairs:
        original_traces = _read_jsonl(
            pair.original_dir / "traces.jsonl"
        )
        extended_traces = {
            (int(trace["position"]), str(trace["question_id"])): trace
            for trace in _read_jsonl(
                pair.extended_dir / "traces.jsonl"
            )
        }
        original_trajectory = pd.read_parquet(
            pair.original_dir / "trajectory.parquet"
        )
        extended_trajectory = pd.read_parquet(
            pair.extended_dir / "trajectory.parquet"
        )
        original_final = {
            (int(row.position), str(row.question_id)): row
            for row in original_trajectory[
                original_trajectory["decile"] == 100
            ].itertuples(index=False)
        }
        extended_final = {
            (int(row.position), str(row.question_id)): row
            for row in extended_trajectory[
                extended_trajectory["decile"] == 100
            ].itertuples(index=False)
        }
        lost_cases = pd.read_parquet(
            pair.extended_dir
            / "lost_analysis"
            / "lost_cases.parquet"
        )
        case_by_key = {
            (int(row.position), str(row.question_id)): row
            for row in lost_cases.itertuples(index=False)
        }
        for original_trace in original_traces:
            if not bool(original_trace.get("truncated", False)):
                continue
            key = (
                int(original_trace["position"]),
                str(original_trace["question_id"]),
            )
            trace = extended_traces[key]
            old = original_final[key]
            new = extended_final[key]
            case = case_by_key.get(key)
            records.append(
                {
                    "dataset": pair.name,
                    "dataset_label": pair.label,
                    "criterion": pair.criterion,
                    "position": key[0],
                    "question_id": key[1],
                    "category": trace["category"],
                    "answer": trace["answer"],
                    "old_prediction": old.prediction,
                    "new_prediction": new.prediction,
                    "old_correct": bool(old.correct),
                    "new_correct": bool(new.correct),
                    "prediction_changed": (
                        old.prediction != new.prediction
                    ),
                    "new_outcome": new.outcome,
                    "generated_answer": trace.get(
                        "generated_answer"
                    ),
                    "generated_correct": (
                        trace.get("generated_answer")
                        == trace["answer"]
                    ),
                    "original_trace_token_count": int(
                        trace["original_trace_token_count"]
                    ),
                    "extended_trace_token_count": int(
                        trace["trace_token_count"]
                    ),
                    "added_reasoning_tokens": int(
                        trace["trace_token_count"]
                    )
                    - int(trace["original_trace_token_count"]),
                    "forced_completion": bool(
                        trace.get("forced_completion", False)
                    ),
                    "ever_correct_final_wrong": case is not None,
                    "criterion_reproduced": bool(
                        case is not None
                        and getattr(case, pair.criterion)
                    ),
                }
            )
    return pd.DataFrame(records)


def summarize_extensions(
    table: pd.DataFrame,
) -> dict[str, object]:
    datasets: dict[str, object] = {}
    for pair in PAIRS:
        subset = table[table["dataset"] == pair.name]
        transitions = (
            subset.groupby(["old_correct", "new_correct"])
            .size()
            .to_dict()
        )
        datasets[pair.name] = {
            "label": pair.label,
            "criterion": pair.criterion,
            "extended_trace_count": len(subset),
            "naturally_completed_count": int(
                (~subset["forced_completion"]).sum()
            ),
            "forced_completion_count": int(
                subset["forced_completion"].sum()
            ),
            "old_final_correct_count": int(
                subset["old_correct"].sum()
            ),
            "new_final_correct_count": int(
                subset["new_correct"].sum()
            ),
            "prediction_changed_count": int(
                subset["prediction_changed"].sum()
            ),
            "prediction_changed_rate": float(
                subset["prediction_changed"].mean()
            ),
            "wrong_to_correct_count": int(
                transitions.get((False, True), 0)
            ),
            "correct_to_wrong_count": int(
                transitions.get((True, False), 0)
            ),
            "ever_correct_final_wrong_count": int(
                subset["ever_correct_final_wrong"].sum()
            ),
            "criterion_count": int(
                subset["criterion_reproduced"].sum()
            ),
            "median_extended_trace_tokens": float(
                subset["extended_trace_token_count"].median()
            ),
            "maximum_extended_trace_tokens": int(
                subset["extended_trace_token_count"].max()
            ),
        }
    return {
        "extended_trace_count": len(table),
        "forced_completion_count": int(
            table["forced_completion"].sum()
        ),
        "datasets": datasets,
    }


def _save_figure(figure: plt.Figure, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_correctness_transitions(
    table: pd.DataFrame,
    output_dir: Path,
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, pair in zip(axes, PAIRS):
        subset = table[table["dataset"] == pair.name]
        matrix = np.zeros((2, 2), dtype=int)
        for row in subset.itertuples(index=False):
            matrix[int(row.old_correct), int(row.new_correct)] += 1
        axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    fontsize=14,
                )
        axis.set_xticks([0, 1], ["Wrong", "Correct"])
        axis.set_yticks([0, 1], ["Wrong", "Correct"])
        axis.set_xlabel("After extension")
        axis.set_ylabel("At original 16K cap")
        axis.set_title(pair.label)
    return _save_figure(
        figure,
        output_dir / "correctness_transitions.png",
    )


def plot_extension_outcomes(
    table: pd.DataFrame,
    output_dir: Path,
) -> Path:
    outcomes = [
        "stable_correct",
        "gained",
        "lost",
        "stable_wrong",
    ]
    colors = ["#15803d", "#3b82f6", "#dc2626", "#6b7280"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, pair in zip(axes, PAIRS):
        subset = table[table["dataset"] == pair.name]
        counts = subset["new_outcome"].value_counts()
        values = [int(counts.get(outcome, 0)) for outcome in outcomes]
        axis.bar(
            [outcome.replace("_", "\n") for outcome in outcomes],
            values,
            color=colors,
        )
        axis.set_title(pair.label)
        axis.set_ylabel("Previously capped traces")
    return _save_figure(
        figure,
        output_dir / "extended_trace_outcomes.png",
    )


def _write_report(
    summary: dict[str, object],
    path: Path,
) -> None:
    lines = [
        "# Capped-trace extension analysis",
        "",
        (
            f"Extended {summary['extended_trace_count']} traces. "
            f"{summary['forced_completion_count']} still failed to close "
            "within the additional 16,384-token budget and were explicitly "
            "closed before sampling an answer."
        ),
        "",
    ]
    for pair in PAIRS:
        values = summary["datasets"][pair.name]
        net = (
            values["new_final_correct_count"]
            - values["old_final_correct_count"]
        )
        lines.extend(
            [
                f"## {pair.label}",
                "",
                (
                    f"- Extended {values['extended_trace_count']} traces; "
                    f"{values['naturally_completed_count']} stopped "
                    f"naturally and {values['forced_completion_count']} "
                    "required explicit closure."
                ),
                (
                    f"- The final prediction changed for "
                    f"{values['prediction_changed_count']} traces "
                    f"({values['prediction_changed_rate']:.1%})."
                ),
                (
                    f"- Correctness at the cap versus after extension: "
                    f"{values['old_final_correct_count']} -> "
                    f"{values['new_final_correct_count']} "
                    f"(net {net:+d}); "
                    f"{values['wrong_to_correct_count']} changed wrong to "
                    f"correct and {values['correct_to_wrong_count']} "
                    "changed correct to wrong."
                ),
                (
                    f"- {values['ever_correct_final_wrong_count']} "
                    "extended traces were ever-correct/final-wrong; "
                    f"{values['criterion_count']} met the dataset's target "
                    f"`{values['criterion']}` criterion."
                ),
                (
                    f"- Median completed trace length was "
                    f"{values['median_extended_trace_tokens']:,.0f} tokens; "
                    f"the maximum was "
                    f"{values['maximum_extended_trace_tokens']:,}."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "- A 16K endpoint is not a reliable proxy for these long "
            "traces: many predictions changed after continuation.",
            "- Continuation improved net accuracy among capped traces, but "
            "also exposed additional correct-to-wrong reversals. Excluding "
            "the capped subset therefore biased both accuracy and loss-case "
            "counts.",
            "- Forced completions are a small, separately labeled tail of "
            "pathological repetitive or unfinished reasoning.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_extensions(
    output_dir: Path = Path(
        "outputs/trace_extension_analysis_gemma4_12b"
    ),
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    table = build_extension_table()
    summary = summarize_extensions(table)
    table.to_parquet(
        output_dir / "extended_traces.parquet",
        index=False,
    )
    with (output_dir / "extension_summary.json").open(
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(summary, output, indent=2, sort_keys=True)
        output.write("\n")
    figures = [
        plot_correctness_transitions(table, output_dir),
        plot_extension_outcomes(table, output_dir),
    ]
    report_path = output_dir / "extension_report.md"
    _write_report(summary, report_path)
    return {
        "table_path": str(output_dir / "extended_traces.parquet"),
        "summary_path": str(output_dir / "extension_summary.json"),
        "report_path": str(report_path),
        "figure_paths": [str(path) for path in figures],
        "summary": summary,
    }
