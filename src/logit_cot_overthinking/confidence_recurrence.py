from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_CORRECT_THRESHOLDS = (0.5, 0.7, 0.9)
DEFAULT_FINAL_THRESHOLDS = (0.0, 0.7, 0.9)

ATTEMPT_KEY_COLUMNS = ("dataset", "seed", "position", "question_id")
SUMMARY_COLUMNS = (
    "dataset",
    "dataset_label",
    "baseline_cohort",
    "correct_threshold",
    "final_threshold",
)


def _native(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _sorted_thresholds(thresholds: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(threshold) for threshold in thresholds)
    if not values:
        raise ValueError("At least one threshold is required")
    if any(threshold < 0.0 or threshold > 1.0 for threshold in values):
        raise ValueError("Thresholds must be between 0.0 and 1.0")
    return tuple(sorted(set(values)))


def _require_columns(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    name: str,
) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _attempts_with_labels(attempts: pd.DataFrame) -> pd.DataFrame:
    result = attempts.copy()
    if "dataset_label" not in result.columns:
        result["dataset_label"] = result["dataset"].astype(str)
    if "forced_completion" not in result.columns:
        result["forced_completion"] = False
    return result


def _ensure_final_prediction_confidence(
    attempts: pd.DataFrame,
    final_thresholds: Sequence[float],
) -> pd.DataFrame:
    result = attempts.copy()
    if "final_normalized_prediction_probability" in result.columns:
        return result

    probability_columns = {
        "final_prediction_probability",
        "final_choice_probability_mass",
    }
    if probability_columns.issubset(result.columns):
        result["final_normalized_prediction_probability"] = np.where(
            result["final_choice_probability_mass"].astype(float) > 0,
            result["final_prediction_probability"].astype(float)
            / result["final_choice_probability_mass"].astype(float),
            np.nan,
        )
        return result

    if max(final_thresholds) <= 0.0:
        result["final_normalized_prediction_probability"] = np.nan
        return result

    raise ValueError(
        "final_normalized_prediction_probability is required when "
        "any final threshold is greater than 0.0"
    )


def _checkpoint_key_columns(
    attempts: pd.DataFrame,
    checkpoints: pd.DataFrame,
) -> list[str]:
    columns = [
        column
        for column in ATTEMPT_KEY_COLUMNS
        if column in attempts.columns and column in checkpoints.columns
    ]
    if "dataset" not in columns or "position" not in columns:
        raise ValueError(
            "attempts and checkpoints must share at least dataset and position"
        )
    return columns


def attach_final_prediction_confidence(
    attempts: pd.DataFrame,
    input_root: Path,
) -> pd.DataFrame:
    """Join final normalized prediction probabilities from run trajectories."""

    if "final_normalized_prediction_probability" in attempts.columns:
        return attempts.copy()

    _require_columns(attempts, ATTEMPT_KEY_COLUMNS, "attempts")
    records: list[dict[str, object]] = []
    for dataset, seed in (
        attempts[["dataset", "seed"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ):
        path = (
            input_root
            / str(dataset)
            / f"seed_{int(seed)}"
            / "trajectory.parquet"
        )
        if not path.exists():
            raise FileNotFoundError(f"Required trajectory not found: {path}")
        trajectory = pd.read_parquet(path)
        final_rows = trajectory[trajectory["decile"] == 100]
        if final_rows.empty:
            final_rows = (
                trajectory.sort_values("decile")
                .groupby(["position", "question_id"], as_index=False)
                .tail(1)
            )
        for row in final_rows.itertuples(index=False):
            prediction = str(row.prediction)
            mass = float(row.choice_probability_mass)
            if hasattr(row, "prediction_probability"):
                prediction_probability = float(row.prediction_probability)
            else:
                prediction_probability = float(
                    row.choice_probabilities[prediction]
                )
            normalized = (
                prediction_probability / mass if mass > 0.0 else np.nan
            )
            records.append(
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "position": int(row.position),
                    "question_id": str(row.question_id),
                    "final_prediction": prediction,
                    "final_prediction_probability": prediction_probability,
                    "final_choice_probability_mass": mass,
                    "final_normalized_prediction_probability": normalized,
                }
            )

    confidence = pd.DataFrame(records)
    result = attempts.copy()
    result["dataset"] = result["dataset"].astype(str)
    result["question_id"] = result["question_id"].astype(str)
    merged = result.merge(
        confidence,
        on=list(ATTEMPT_KEY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    if merged["final_normalized_prediction_probability"].isna().any():
        missing = merged[
            merged["final_normalized_prediction_probability"].isna()
        ][list(ATTEMPT_KEY_COLUMNS)]
        raise ValueError(
            "Could not attach final prediction confidence for "
            f"{len(missing)} attempts"
        )
    return merged


def load_pre_final_correct_checkpoints(
    input_root: Path,
    attempts: pd.DataFrame,
) -> pd.DataFrame:
    """Load all pre-final correct probes with normalized correct confidence."""

    _require_columns(attempts, ATTEMPT_KEY_COLUMNS, "attempts")
    attempt_keys = {
        (
            str(row.dataset),
            int(row.seed),
            int(row.position),
            str(row.question_id),
        )
        for row in attempts.itertuples(index=False)
    }
    records: list[dict[str, object]] = []
    for dataset, seed in (
        attempts[["dataset", "seed"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ):
        path = (
            input_root
            / str(dataset)
            / f"seed_{int(seed)}"
            / "trajectory.parquet"
        )
        if not path.exists():
            raise FileNotFoundError(f"Required trajectory not found: {path}")
        trajectory = pd.read_parquet(path)
        if "normalized_correct_probability" not in trajectory.columns:
            trajectory = trajectory.copy()
            trajectory["correct_answer_probability"] = trajectory.apply(
                lambda row: float(
                    row["choice_probabilities"][row["answer"]]
                ),
                axis=1,
            )
            trajectory["normalized_correct_probability"] = np.where(
                trajectory["choice_probability_mass"].astype(float) > 0.0,
                trajectory["correct_answer_probability"].astype(float)
                / trajectory["choice_probability_mass"].astype(float),
                np.nan,
            )
        prefinal = trajectory[
            (trajectory["decile"] < 100)
            & trajectory["correct"].astype(bool)
        ]
        for row in prefinal.itertuples(index=False):
            key = (
                str(dataset),
                int(seed),
                int(row.position),
                str(row.question_id),
            )
            if key not in attempt_keys:
                continue
            records.append(
                {
                    "dataset": str(dataset),
                    "seed": int(seed),
                    "position": int(row.position),
                    "question_id": str(row.question_id),
                    "decile": int(row.decile),
                    "current_correct": True,
                    "current_normalized_correct_probability": float(
                        row.normalized_correct_probability
                    ),
                }
            )
    return pd.DataFrame(
        records,
        columns=[
            "dataset",
            "seed",
            "position",
            "question_id",
            "decile",
            "current_correct",
            "current_normalized_correct_probability",
        ],
    )


def load_confidence_recurrence_inputs(
    input_root: Path = Path("outputs/matched_controls_gemma4_12b_extended"),
    selection_path: Path = Path(
        "outputs/matched_controls_gemma4_12b/cohort_selection.parquet"
    ),
    prefer_cached: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached matched-analysis tables, or rebuild them from runs."""

    analysis_dir = input_root / "analysis"
    attempts_path = analysis_dir / "attempts.parquet"
    if prefer_cached and attempts_path.exists():
        attempts = pd.read_parquet(attempts_path)
    else:
        from .matched_analysis import build_matched_attempt_tables

        attempts, _ = build_matched_attempt_tables(
            input_root,
            selection_path,
        )
    attempts = attach_final_prediction_confidence(attempts, input_root)
    checkpoints = load_pre_final_correct_checkpoints(input_root, attempts)
    return attempts, checkpoints


def build_attempt_recurrence_table(
    attempts: pd.DataFrame,
    checkpoints: pd.DataFrame,
    correct_thresholds: Sequence[float] = DEFAULT_CORRECT_THRESHOLDS,
    final_thresholds: Sequence[float] = DEFAULT_FINAL_THRESHOLDS,
) -> pd.DataFrame:
    """Expand attempts by threshold and flag qualified recurrence losses."""

    correct_thresholds = _sorted_thresholds(correct_thresholds)
    final_thresholds = _sorted_thresholds(final_thresholds)
    attempts = _attempts_with_labels(
        _ensure_final_prediction_confidence(attempts, final_thresholds)
    )
    _require_columns(
        attempts,
        [
            "dataset",
            "dataset_label",
            "baseline_cohort",
            "final_correct",
            "forced_completion",
            "final_normalized_prediction_probability",
        ],
        "attempts",
    )
    _require_columns(
        checkpoints,
        ["current_normalized_correct_probability"],
        "checkpoints",
    )
    key_columns = _checkpoint_key_columns(attempts, checkpoints)

    checkpoint_rows: list[dict[str, object]] = []
    working_checkpoints = checkpoints.copy()
    if "decile" in working_checkpoints.columns:
        working_checkpoints = working_checkpoints[
            working_checkpoints["decile"] < 100
        ]
    if "current_correct" in working_checkpoints.columns:
        working_checkpoints = working_checkpoints[
            working_checkpoints["current_correct"].astype(bool)
        ]

    for threshold in correct_thresholds:
        qualified = working_checkpoints[
            working_checkpoints[
                "current_normalized_correct_probability"
            ].astype(float)
            >= threshold
        ]
        if qualified.empty:
            continue
        aggregations: dict[str, tuple[str, str]] = {
            "qualified_checkpoint_count": (
                "current_normalized_correct_probability",
                "size",
            ),
            "peak_qualified_normalized_correct_probability": (
                "current_normalized_correct_probability",
                "max",
            ),
        }
        if "decile" in qualified.columns:
            aggregations["first_qualified_decile"] = ("decile", "min")
            aggregations["last_qualified_decile"] = ("decile", "max")
        grouped = (
            qualified.groupby(key_columns, as_index=False)
            .agg(**aggregations)
            .assign(correct_threshold=threshold)
        )
        checkpoint_rows.extend(grouped.to_dict(orient="records"))

    if checkpoint_rows:
        checkpoint_summary = pd.DataFrame(checkpoint_rows)
    else:
        checkpoint_summary = pd.DataFrame(
            columns=[
                *key_columns,
                "correct_threshold",
                "qualified_checkpoint_count",
                "peak_qualified_normalized_correct_probability",
                "first_qualified_decile",
                "last_qualified_decile",
            ]
        )

    expanded_rows: list[pd.DataFrame] = []
    for threshold in correct_thresholds:
        chunk = attempts.merge(
            checkpoint_summary[
                checkpoint_summary["correct_threshold"] == threshold
            ],
            on=key_columns,
            how="left",
            suffixes=("", "_checkpoint"),
        )
        chunk["correct_threshold"] = threshold
        expanded_rows.append(chunk)
    expanded = pd.concat(expanded_rows, ignore_index=True)

    expanded["qualified_checkpoint_count"] = (
        expanded["qualified_checkpoint_count"].fillna(0).astype(int)
    )
    expanded["qualified_intermediate_correct"] = (
        expanded["qualified_checkpoint_count"] > 0
    )
    for column in (
        "first_qualified_decile",
        "last_qualified_decile",
        "peak_qualified_normalized_correct_probability",
    ):
        if column not in expanded.columns:
            expanded[column] = np.nan

    rows: list[pd.DataFrame] = []
    for threshold in final_thresholds:
        chunk = expanded.copy()
        chunk["final_threshold"] = threshold
        if threshold <= 0.0:
            chunk["final_confidence_qualified"] = True
        else:
            chunk["final_confidence_qualified"] = (
                chunk["final_normalized_prediction_probability"].astype(float)
                >= threshold
            )
        rows.append(chunk)

    recurrence = pd.concat(rows, ignore_index=True)
    recurrence["final_wrong"] = ~recurrence["final_correct"].astype(bool)
    recurrence["qualified_loss"] = (
        recurrence["qualified_intermediate_correct"]
        & recurrence["final_wrong"]
        & recurrence["final_confidence_qualified"]
    )

    preferred = [
        *ATTEMPT_KEY_COLUMNS,
        "dataset_label",
        "baseline_cohort",
        "match_id",
        "correct_threshold",
        "final_threshold",
        "qualified_intermediate_correct",
        "qualified_checkpoint_count",
        "first_qualified_decile",
        "last_qualified_decile",
        "peak_qualified_normalized_correct_probability",
        "final_correct",
        "final_wrong",
        "final_normalized_prediction_probability",
        "final_confidence_qualified",
        "qualified_loss",
        "forced_completion",
    ]
    existing = [column for column in preferred if column in recurrence.columns]
    remaining = [
        column for column in recurrence.columns if column not in set(existing)
    ]
    return recurrence[existing + remaining]


def build_recurrence_summary(
    recurrence: pd.DataFrame,
) -> pd.DataFrame:
    return (
        recurrence.groupby(list(SUMMARY_COLUMNS), as_index=False)
        .agg(
            attempt_count=("qualified_loss", "size"),
            qualified_loss_count=("qualified_loss", "sum"),
            qualified_loss_rate=("qualified_loss", "mean"),
            final_accuracy=("final_correct", "mean"),
            forced_completion_count=("forced_completion", "sum"),
        )
        .sort_values(list(SUMMARY_COLUMNS))
        .reset_index(drop=True)
    )


def build_recurrence_contrasts(
    recurrence: pd.DataFrame,
    bootstrap_iterations: int = 5000,
    seed: int = 0,
) -> pd.DataFrame:
    _require_columns(
        recurrence,
        [
            "dataset",
            "match_id",
            "baseline_cohort",
            "correct_threshold",
            "final_threshold",
            "qualified_loss",
        ],
        "recurrence",
    )
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    group_columns = ["dataset", "correct_threshold", "final_threshold"]
    for group_values, subset in recurrence.groupby(group_columns, sort=True):
        dataset, correct_threshold, final_threshold = group_values
        per_match = (
            subset.groupby(
                ["match_id", "baseline_cohort"],
                as_index=False,
            )["qualified_loss"]
            .mean()
            .pivot(
                index="match_id",
                columns="baseline_cohort",
                values="qualified_loss",
            )
        )
        for control in ("final_correct", "stable_wrong"):
            if "loss" not in per_match.columns or control not in per_match:
                continue
            pairs = per_match[["loss", control]].dropna()
            if pairs.empty:
                continue
            differences = (pairs["loss"] - pairs[control]).to_numpy()
            if bootstrap_iterations > 0 and len(differences) > 0:
                samples = rng.choice(
                    differences,
                    size=(bootstrap_iterations, len(differences)),
                    replace=True,
                ).mean(axis=1)
                ci_low = float(np.quantile(samples, 0.025))
                ci_high = float(np.quantile(samples, 0.975))
            else:
                ci_low = float("nan")
                ci_high = float("nan")
            records.append(
                {
                    "dataset": dataset,
                    "correct_threshold": float(correct_threshold),
                    "final_threshold": float(final_threshold),
                    "comparison": f"loss_vs_{control}",
                    "risk_difference": float(differences.mean()),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "match_count": int(len(differences)),
                }
            )
    return pd.DataFrame(records)


def _write_report(
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# Confidence-threshold recurrence analysis",
        "",
        "Qualified losses require a pre-final correct probe with normalized "
        "correct-answer probability at or above the listed threshold, a "
        "wrong final answer, and the listed minimum final normalized "
        "prediction probability.",
        "",
        "## Cohort summary",
        "",
        "| Dataset | Seed-0 cohort | Correct threshold | Final threshold | "
        "Attempts | Qualified losses | Rate | Final accuracy | Forced completions |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.dataset_label} | "
            f"{row.baseline_cohort.replace('_', ' ')} | "
            f"{row.correct_threshold:.1f} | {row.final_threshold:.1f} | "
            f"{int(row.attempt_count)} | {int(row.qualified_loss_count)} | "
            f"{row.qualified_loss_rate:.1%} | "
            f"{row.final_accuracy:.1%} | "
            f"{int(row.forced_completion_count)} |"
        )

    lines.extend(
        [
            "",
            "## Matched risk differences",
            "",
            "Risk differences average within matched triplets and bootstrap "
            "over match IDs.",
            "",
        ]
    )
    if contrasts.empty:
        lines.append("No matched contrasts could be computed.")
    else:
        for row in contrasts.itertuples(index=False):
            comparison = (
                row.comparison.replace("loss_vs_", "loss vs ")
                .replace("_", " ")
            )
            lines.append(
                f"- **{row.dataset.replace('_', ' ').title()} "
                f"correct >= {row.correct_threshold:.1f}, "
                f"final >= {row.final_threshold:.1f}, {comparison}:** "
                f"{row.risk_difference:+.1%} "
                f"(95% CI {row.ci_low:+.1%} to {row.ci_high:+.1%}; "
                f"n={int(row.match_count)} matches)."
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _summary_payload(
    recurrence: pd.DataFrame,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    correct_thresholds: Sequence[float],
    final_thresholds: Sequence[float],
) -> dict[str, object]:
    key_columns = [
        column for column in ATTEMPT_KEY_COLUMNS if column in recurrence.columns
    ]
    attempt_count = (
        int(recurrence[key_columns].drop_duplicates().shape[0])
        if key_columns
        else int(len(recurrence))
    )
    return _native(
        {
            "attempt_count": attempt_count,
            "recurrence_row_count": int(len(recurrence)),
            "correct_thresholds": list(correct_thresholds),
            "final_thresholds": list(final_thresholds),
            "cohorts": summary.to_dict(orient="records"),
            "matched_contrasts": contrasts.to_dict(orient="records"),
        }
    )


def analyze_confidence_recurrence(
    input_root: Path = Path("outputs/matched_controls_gemma4_12b_extended"),
    selection_path: Path = Path(
        "outputs/matched_controls_gemma4_12b/cohort_selection.parquet"
    ),
    output_dir: Path | None = None,
    attempts: pd.DataFrame | None = None,
    checkpoints: pd.DataFrame | None = None,
    correct_thresholds: Sequence[float] = DEFAULT_CORRECT_THRESHOLDS,
    final_thresholds: Sequence[float] = DEFAULT_FINAL_THRESHOLDS,
    bootstrap_iterations: int = 5000,
    seed: int = 0,
    prefer_cached: bool = True,
) -> dict[str, object]:
    correct_thresholds = _sorted_thresholds(correct_thresholds)
    final_thresholds = _sorted_thresholds(final_thresholds)
    if (attempts is None) != (checkpoints is None):
        raise ValueError(
            "attempts and checkpoints must be provided together, or neither"
        )
    if attempts is None and checkpoints is None:
        attempts, checkpoints = load_confidence_recurrence_inputs(
            input_root=input_root,
            selection_path=selection_path,
            prefer_cached=prefer_cached,
        )

    output_dir = output_dir or input_root / "analysis" / "confidence_recurrence"
    output_dir.mkdir(parents=True, exist_ok=True)

    recurrence = build_attempt_recurrence_table(
        attempts,
        checkpoints,
        correct_thresholds=correct_thresholds,
        final_thresholds=final_thresholds,
    )
    summary = build_recurrence_summary(recurrence)
    contrasts = build_recurrence_contrasts(
        recurrence,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )

    recurrence_path = output_dir / "attempt_recurrence.parquet"
    summary_path = output_dir / "recurrence_summary.parquet"
    contrasts_path = output_dir / "recurrence_contrasts.parquet"
    json_path = output_dir / "confidence_recurrence_summary.json"
    report_path = output_dir / "confidence_recurrence_report.md"

    recurrence.to_parquet(recurrence_path, index=False)
    summary.to_parquet(summary_path, index=False)
    contrasts.to_parquet(contrasts_path, index=False)
    _write_report(summary, contrasts, report_path)

    payload = _summary_payload(
        recurrence,
        summary,
        contrasts,
        correct_thresholds,
        final_thresholds,
    )
    with json_path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")

    return {
        "recurrence_path": str(recurrence_path),
        "summary_path": str(summary_path),
        "contrasts_path": str(contrasts_path),
        "json_path": str(json_path),
        "report_path": str(report_path),
        "summary": payload,
    }
