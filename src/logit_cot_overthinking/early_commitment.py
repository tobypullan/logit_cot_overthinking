from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_THRESHOLDS = (0.5, 0.7, 0.9)
DEFAULT_PROXY_THRESHOLD = 0.9
DEFAULT_PROXY_STREAK = 2
IDENTITY_COLUMNS = (
    "dataset",
    "dataset_label",
    "seed",
    "run_seed",
    "position",
    "question_id",
)
METADATA_COLUMNS = (
    "dataset",
    "dataset_label",
    "seed",
    "run_seed",
    "position",
    "question_id",
    "category",
    "source",
    "baseline_cohort",
    "match_id",
    "answer",
)


def _choice_probability(
    probabilities: object,
    label: object,
) -> float:
    if isinstance(probabilities, str):
        probabilities = json.loads(probabilities)
    if not isinstance(probabilities, Mapping):
        return float("nan")
    return float(probabilities.get(str(label), np.nan))


def _normalized_probability(
    probability: object,
    mass: object,
) -> float:
    try:
        mass_value = float(mass)
        probability_value = float(probability)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(mass_value) or mass_value <= 0:
        return 0.0
    if not np.isfinite(probability_value):
        return 0.0
    return probability_value / mass_value


def _format_threshold(threshold: float) -> str:
    return f"{threshold:g}"


def _attempt_group_columns(dataframe: pd.DataFrame) -> list[str]:
    columns = [
        column for column in IDENTITY_COLUMNS if column in dataframe.columns
    ]
    if not {"position", "question_id"}.intersection(columns):
        raise ValueError(
            "Trajectory data must include position or question_id so "
            "checkpoint rows can be grouped into attempts."
        )
    return columns


def _validate_trajectory(dataframe: pd.DataFrame) -> None:
    required = {
        "answer",
        "choice_probabilities",
        "choice_probability_mass",
        "correct",
        "decile",
        "prediction",
    }
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(
            "Trajectory data is missing required columns: "
            + ", ".join(missing)
        )


def _prepare_trajectory(dataframe: pd.DataFrame) -> pd.DataFrame:
    _validate_trajectory(dataframe)
    prepared = dataframe.copy()
    if "normalized_correct_probability" not in prepared.columns:
        prepared["normalized_correct_probability"] = prepared.apply(
            lambda row: _normalized_probability(
                _choice_probability(
                    row["choice_probabilities"],
                    row["answer"],
                ),
                row["choice_probability_mass"],
            ),
            axis=1,
        )
    if "prediction_probability" not in prepared.columns:
        prepared["prediction_probability"] = prepared.apply(
            lambda row: _choice_probability(
                row["choice_probabilities"],
                row["prediction"],
            ),
            axis=1,
        )
    prepared["normalized_prediction_probability"] = prepared.apply(
        lambda row: _normalized_probability(
            row["prediction_probability"],
            row["choice_probability_mass"],
        ),
        axis=1,
    )
    return prepared


def _prediction_streaks(group: pd.DataFrame) -> list[int]:
    streaks: list[int] = []
    previous: str | None = None
    streak = 0
    for prediction in group["prediction"].astype(str):
        if prediction == previous:
            streak += 1
        else:
            streak = 1
        streaks.append(streak)
        previous = prediction
    return streaks


def _row_metadata(row: pd.Series) -> dict[str, object]:
    return {
        column: row[column]
        for column in METADATA_COLUMNS
        if column in row.index
    }


def _policy_record(
    metadata: dict[str, object],
    policy: str,
    family: str,
    selected: pd.Series,
    final: pd.Series,
    threshold: float | None = None,
    streak: int | None = None,
    oracle: bool = False,
) -> dict[str, object]:
    selected_correct = bool(selected["correct"])
    final_correct = bool(final["correct"])
    stop_decile = int(selected["decile"])
    final_decile = int(final["decile"])
    return {
        **metadata,
        "policy": policy,
        "policy_family": family,
        "threshold": threshold,
        "streak": streak,
        "oracle": oracle,
        "deployable": not oracle,
        "stop_decile": stop_decile,
        "stopped_early": stop_decile < final_decile,
        "selected_prediction": str(selected["prediction"]),
        "selected_correct": selected_correct,
        "selected_normalized_correct_probability": float(
            selected["normalized_correct_probability"]
        ),
        "selected_normalized_prediction_probability": float(
            selected["normalized_prediction_probability"]
        ),
        "final_decile": final_decile,
        "final_prediction": str(final["prediction"]),
        "final_correct": final_correct,
        "delta_correct": int(selected_correct) - int(final_correct),
    }


def evaluate_early_commitment_policies(
    dataframe: pd.DataFrame,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    proxy_threshold: float = DEFAULT_PROXY_THRESHOLD,
    proxy_streak: int = DEFAULT_PROXY_STREAK,
) -> pd.DataFrame:
    """Evaluate early-stopping policies for each attempt trajectory.

    Oracle policies use the true answer/correctness and should be treated as
    upper bounds rather than deployable interventions.
    """

    prepared = _prepare_trajectory(dataframe)
    group_columns = _attempt_group_columns(prepared)
    records: list[dict[str, object]] = []

    for _, group in prepared.groupby(group_columns, sort=False):
        group = group.sort_values("decile").reset_index(drop=True)
        group["prediction_streak"] = _prediction_streaks(group)
        final = group.iloc[-1]
        prefinal = group[group["decile"] < final["decile"]]
        metadata = _row_metadata(final)
        correct_rows = group[group["correct"].astype(bool)]
        broad_loss = (
            not correct_rows.empty and not bool(final["correct"])
        )
        metadata["broad_loss"] = broad_loss

        records.append(
            _policy_record(
                metadata,
                "final",
                "final",
                final,
                final,
                oracle=False,
            )
        )

        oracle_selected = (
            correct_rows.iloc[0] if not correct_rows.empty else final
        )
        records.append(
            _policy_record(
                metadata,
                "oracle_first_correct",
                "oracle_first_correct",
                oracle_selected,
                final,
                oracle=True,
            )
        )

        # This threshold policy is deliberately oracle-ish: it requires
        # knowing whether the current prediction equals the true answer.
        for threshold in thresholds:
            candidates = prefinal[
                prefinal["correct"].astype(bool)
                & (
                    prefinal["normalized_correct_probability"]
                    >= float(threshold)
                )
            ]
            selected = (
                candidates.iloc[0] if not candidates.empty else final
            )
            records.append(
                _policy_record(
                    metadata,
                    f"threshold_first_{_format_threshold(float(threshold))}",
                    "threshold_first",
                    selected,
                    final,
                    threshold=float(threshold),
                    oracle=True,
                )
            )

        proxy_candidates = prefinal[
            (
                prefinal["normalized_prediction_probability"]
                >= float(proxy_threshold)
            )
            & (prefinal["prediction_streak"] >= int(proxy_streak))
        ]
        proxy_selected = (
            proxy_candidates.iloc[0]
            if not proxy_candidates.empty
            else final
        )
        records.append(
            _policy_record(
                metadata,
                (
                    "proxy_confidence_streak_"
                    f"{_format_threshold(float(proxy_threshold))}"
                    f"_s{int(proxy_streak)}"
                ),
                "proxy_confidence_streak",
                proxy_selected,
                final,
                threshold=float(proxy_threshold),
                streak=int(proxy_streak),
                oracle=False,
            )
        )

    return pd.DataFrame(records)


def summarize_policy_outcomes(
    outcomes: pd.DataFrame,
    group_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    if group_columns is None:
        group_columns = [
            column
            for column in ("dataset", "baseline_cohort")
            if column in outcomes.columns
        ]
    group_columns = list(group_columns)
    records: list[dict[str, object]] = []
    for keys, group in outcomes.groupby(
        group_columns + ["policy"],
        sort=False,
        dropna=False,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = dict(zip(group_columns + ["policy"], keys))
        policy_accuracy = float(group["selected_correct"].mean())
        final_accuracy = float(group["final_correct"].mean())
        records.append(
            {
                **values,
                "attempt_count": int(len(group)),
                "policy_accuracy": policy_accuracy,
                "final_accuracy": final_accuracy,
                "delta_vs_final": policy_accuracy - final_accuracy,
                "stop_rate": float(group["stopped_early"].mean()),
                "median_stop_decile": float(
                    group["stop_decile"].median()
                ),
            }
        )
    return pd.DataFrame(records)


def evaluate_early_commitment(
    dataframe: pd.DataFrame,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    proxy_threshold: float = DEFAULT_PROXY_THRESHOLD,
    proxy_streak: int = DEFAULT_PROXY_STREAK,
    group_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = evaluate_early_commitment_policies(
        dataframe,
        thresholds=thresholds,
        proxy_threshold=proxy_threshold,
        proxy_streak=proxy_streak,
    )
    summary = summarize_policy_outcomes(
        outcomes,
        group_columns=group_columns,
    )
    return outcomes, summary


def _trajectory_paths(input_root: Path) -> list[Path]:
    direct = input_root / "trajectory.parquet"
    if direct.exists():
        return [direct]
    paths = sorted(input_root.glob("*/*/trajectory.parquet"))
    return [
        path
        for path in paths
        if path.parent.name.startswith("seed_")
    ]


def _load_trajectories(input_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in _trajectory_paths(input_root):
        dataframe = pd.read_parquet(path)
        if (
            "dataset" not in dataframe.columns
            and path.parent.parent != input_root
        ):
            dataframe = dataframe.copy()
            dataframe["dataset"] = path.parent.parent.name
        if (
            "seed" not in dataframe.columns
            and "run_seed" not in dataframe.columns
            and path.parent.name.startswith("seed_")
        ):
            dataframe = dataframe.copy()
            dataframe["seed"] = int(path.parent.name.removeprefix("seed_"))
        frames.append(dataframe)
    if not frames:
        raise FileNotFoundError(
            f"No trajectory.parquet files found under {input_root}"
        )
    return pd.concat(frames, ignore_index=True)


def _attach_selection_metadata(
    trajectories: pd.DataFrame,
    selection_path: Path | None,
) -> pd.DataFrame:
    if selection_path is None or not selection_path.exists():
        return trajectories
    if "dataset" not in trajectories.columns or "position" not in trajectories.columns:
        return trajectories

    selection = pd.read_parquet(selection_path)
    metadata_columns = [
        column
        for column in (
            "dataset_label",
            "baseline_cohort",
            "match_id",
            "category",
            "source",
            "answer",
        )
        if column in selection.columns
    ]
    if not metadata_columns:
        return trajectories

    left = trajectories.copy()
    left["dataset"] = left["dataset"].astype(str)
    selection = selection[["dataset", "position", *metadata_columns]].copy()
    selection["dataset"] = selection["dataset"].astype(str)
    selection["position"] = selection["position"].astype(int)
    merged = left.merge(
        selection,
        on=["dataset", "position"],
        how="left",
        suffixes=("", "_selection"),
        validate="many_to_one",
    )
    for column in metadata_columns:
        selection_column = f"{column}_selection"
        if selection_column not in merged.columns:
            continue
        if column in left.columns:
            merged[column] = merged[column].combine_first(
                merged[selection_column]
            )
        else:
            merged[column] = merged[selection_column]
        merged.drop(columns=[selection_column], inplace=True)
    return merged


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if pd.isna(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as output:
        json.dump(_json_safe(dict(payload)), output, indent=2, sort_keys=True)
        output.write("\n")


def _write_report(
    summary: pd.DataFrame,
    broad_loss_summary: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# Early-commitment intervention analysis",
        "",
        "This analysis asks whether an answer would improve if the trace "
        "were stopped at an earlier checkpoint.",
        "",
        "`oracle_first_correct` and `threshold_first_*` use the true "
        "answer or correctness while choosing a checkpoint. They are "
        "upper bounds for recoverable accuracy, not deployable policies.",
        "",
        "## Policy summary",
        "",
        "| Dataset | Baseline cohort | Policy | Attempts | Accuracy | "
        "Delta vs final | Stop rate | Median stop decile |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        dataset = getattr(row, "dataset", "all")
        cohort = getattr(row, "baseline_cohort", "all")
        lines.append(
            f"| {dataset} | {cohort} | `{row.policy}` | "
            f"{row.attempt_count} | {row.policy_accuracy:.1%} | "
            f"{row.delta_vs_final:+.1%} | {row.stop_rate:.1%} | "
            f"{row.median_stop_decile:.0f} |"
        )

    if not broad_loss_summary.empty:
        lines.extend(
            [
                "",
                "## Broad-loss slice",
                "",
                "Broad-loss attempts are those that were correct at some "
                "checkpoint but wrong at the final checkpoint.",
                "",
                "| Dataset | Baseline cohort | Broad loss | Policy | "
                "Attempts | Accuracy | Delta vs final | Stop rate | "
                "Median stop decile |",
                "|---|---|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in broad_loss_summary.itertuples(index=False):
            dataset = getattr(row, "dataset", "all")
            cohort = getattr(row, "baseline_cohort", "all")
            lines.append(
                f"| {dataset} | {cohort} | {bool(row.broad_loss)} | "
                f"`{row.policy}` | {row.attempt_count} | "
                f"{row.policy_accuracy:.1%} | "
                f"{row.delta_vs_final:+.1%} | {row.stop_rate:.1%} | "
                f"{row.median_stop_decile:.0f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `final` is the observed baseline endpoint.",
            "- `oracle_first_correct` estimates the maximum accuracy "
            "recoverable by stopping exactly when an attempt first becomes "
            "correct.",
            "- `threshold_first_*` asks whether high normalized true-answer "
            "probability would have been enough to stop earlier; it still "
            "uses correctness labels and is therefore an upper bound.",
            "- `proxy_confidence_streak_*` uses only the predicted answer's "
            "normalized probability and prediction stability, then evaluates "
            "whether the chosen prediction was correct.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_early_commitment(
    input_root: Path = Path(
        "outputs/matched_controls_gemma4_12b_extended"
    ),
    selection_path: Path | None = Path(
        "outputs/matched_controls_gemma4_12b/cohort_selection.parquet"
    ),
    output_dir: Path | None = None,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    proxy_threshold: float = DEFAULT_PROXY_THRESHOLD,
    proxy_streak: int = DEFAULT_PROXY_STREAK,
) -> dict[str, object]:
    output_dir = output_dir or input_root / "analysis" / "early_commitment"
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories = _attach_selection_metadata(
        _load_trajectories(input_root),
        selection_path,
    )
    outcomes, summary = evaluate_early_commitment(
        trajectories,
        thresholds=thresholds,
        proxy_threshold=proxy_threshold,
        proxy_streak=proxy_streak,
    )
    broad_loss_group_columns = [
        column
        for column in ("dataset", "baseline_cohort", "broad_loss")
        if column in outcomes.columns
    ]
    broad_loss_summary = (
        summarize_policy_outcomes(outcomes, broad_loss_group_columns)
        if "broad_loss" in broad_loss_group_columns
        else pd.DataFrame()
    )

    outcomes_path = output_dir / "policy_outcomes.parquet"
    summary_path = output_dir / "policy_summary.parquet"
    broad_loss_summary_path = (
        output_dir / "broad_loss_policy_summary.parquet"
    )
    json_path = output_dir / "early_commitment_summary.json"
    report_path = output_dir / "early_commitment_report.md"

    outcomes.to_parquet(outcomes_path, index=False)
    summary.to_parquet(summary_path, index=False)
    if not broad_loss_summary.empty:
        broad_loss_summary.to_parquet(
            broad_loss_summary_path,
            index=False,
        )

    payload = {
        "attempt_count": int(
            outcomes[outcomes["policy"] == "final"].shape[0]
        ),
        "trajectory_row_count": int(len(trajectories)),
        "policies": sorted(outcomes["policy"].unique().tolist()),
        "thresholds": [float(value) for value in thresholds],
        "proxy_threshold": float(proxy_threshold),
        "proxy_streak": int(proxy_streak),
        "summary": summary.to_dict(orient="records"),
        "broad_loss_summary": broad_loss_summary.to_dict(
            orient="records"
        ),
        "notes": [
            "oracle_first_correct and threshold_first_* use true answers "
            "or correctness labels and are upper bounds, not deployable "
            "policies.",
            "proxy_confidence_streak_* uses prediction confidence and "
            "stability only, then evaluates correctness after selection.",
        ],
    }
    _write_json(json_path, payload)
    _write_report(summary, broad_loss_summary, report_path)

    result = {
        "outcomes_path": str(outcomes_path),
        "summary_path": str(summary_path),
        "summary_json_path": str(json_path),
        "report_path": str(report_path),
        "summary": payload,
    }
    if not broad_loss_summary.empty:
        result["broad_loss_summary_path"] = str(
            broad_loss_summary_path
        )
    return result
