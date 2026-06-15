from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import ProbeConfig
from .data import MultipleChoiceQuestion, load_questions
from .gemma import DECILES, GemmaProbeRunner
from .metrics import build_summary, build_trajectory_dataframe


@dataclass(frozen=True)
class MatchedDataset:
    name: str
    label: str
    input_dir: Path
    dataset: str
    dataset_format: str


DATASETS = (
    MatchedDataset(
        name="mmlu_pro",
        label="MMLU-Pro",
        input_dir=Path(
            "outputs/mmlu_pro_gemma4_12b_n1000_seed0_extended"
        ),
        dataset="TIGER-Lab/MMLU-Pro",
        dataset_format="mmlu-pro",
    ),
    MatchedDataset(
        name="gpqa_diamond",
        label="GPQA Diamond",
        input_dir=Path(
            "outputs/gpqa_diamond_gemma4_12b_seed0_extended"
        ),
        dataset="fingertap/GPQA-Diamond",
        dataset_format="gpqa-diamond",
    ),
)


def build_source_question_table(
    input_dir: Path,
) -> pd.DataFrame:
    dataframe = pd.read_parquet(input_dir / "trajectory.parquet")
    rows: list[dict[str, object]] = []
    for (position, question_id), group in dataframe.groupby(
        ["position", "question_id"],
        sort=False,
    ):
        group = group.sort_values("decile")
        final = group.iloc[-1]
        ever_correct = bool(group["correct"].any())
        final_correct = bool(final["correct"])
        cohort = (
            "final_correct"
            if final_correct
            else ("loss" if ever_correct else "stable_wrong")
        )
        rows.append(
            {
                "position": int(position),
                "question_id": str(question_id),
                "category": str(final["category"]),
                "source": str(final["source"]),
                "answer": str(final["answer"]),
                "trace_token_count": int(final["trace_token_count"]),
                "baseline_cohort": cohort,
            }
        )
    return pd.DataFrame(rows)


def _allocate_category_quotas(
    table: pd.DataFrame,
    count: int,
) -> dict[str, int]:
    capacities: dict[str, int] = {}
    for category, group in table.groupby("category"):
        sizes = group["baseline_cohort"].value_counts()
        capacities[str(category)] = min(
            int(sizes.get("loss", 0)),
            int(sizes.get("final_correct", 0)),
            int(sizes.get("stable_wrong", 0)),
        )
    capacities = {
        category: capacity
        for category, capacity in capacities.items()
        if capacity > 0
    }
    if sum(capacities.values()) < count:
        raise ValueError(
            f"Only {sum(capacities.values())} matched triplets are "
            f"available, fewer than requested {count}"
        )

    quotas = {category: 0 for category in sorted(capacities)}
    allocated = 0
    while allocated < count:
        progressed = False
        for category in sorted(capacities):
            if quotas[category] < capacities[category]:
                quotas[category] += 1
                allocated += 1
                progressed = True
                if allocated == count:
                    break
        if not progressed:
            raise RuntimeError("Could not allocate matched category quotas")
    return quotas


def _nearest_unused(
    target_tokens: int,
    candidates: pd.DataFrame,
    used_positions: set[int],
) -> pd.Series:
    available = candidates[
        ~candidates["position"].isin(used_positions)
    ].copy()
    if available.empty:
        raise RuntimeError("No unmatched control remains")
    available["distance"] = (
        np.log1p(available["trace_token_count"])
        - math.log1p(target_tokens)
    ).abs()
    return available.sort_values(
        ["distance", "position"]
    ).iloc[0]


def _match_category(
    category_rows: pd.DataFrame,
    quota: int,
) -> list[tuple[pd.Series, pd.Series, pd.Series]]:
    losses = category_rows[
        category_rows["baseline_cohort"] == "loss"
    ].copy()
    final_controls = category_rows[
        category_rows["baseline_cohort"] == "final_correct"
    ].copy()
    wrong_controls = category_rows[
        category_rows["baseline_cohort"] == "stable_wrong"
    ].copy()
    used_losses: set[int] = set()
    used_final: set[int] = set()
    used_wrong: set[int] = set()
    matches: list[tuple[pd.Series, pd.Series, pd.Series]] = []
    for _ in range(quota):
        choices: list[
            tuple[float, int, pd.Series, pd.Series, pd.Series]
        ] = []
        for _, loss in losses[
            ~losses["position"].isin(used_losses)
        ].iterrows():
            final = _nearest_unused(
                int(loss["trace_token_count"]),
                final_controls,
                used_final,
            )
            wrong = _nearest_unused(
                int(loss["trace_token_count"]),
                wrong_controls,
                used_wrong,
            )
            score = (
                abs(
                    math.log1p(int(final["trace_token_count"]))
                    - math.log1p(int(loss["trace_token_count"]))
                )
                + abs(
                    math.log1p(int(wrong["trace_token_count"]))
                    - math.log1p(int(loss["trace_token_count"]))
                )
            )
            choices.append(
                (
                    score,
                    int(loss["position"]),
                    loss,
                    final,
                    wrong,
                )
            )
        if not choices:
            raise RuntimeError("No matched triplet remains")
        _, _, loss, final, wrong = min(
            choices,
            key=lambda value: (value[0], value[1]),
        )
        used_losses.add(int(loss["position"]))
        used_final.add(int(final["position"]))
        used_wrong.add(int(wrong["position"]))
        matches.append((loss, final, wrong))
    return matches


def select_matched_cohorts(
    table: pd.DataFrame,
    per_cohort: int = 25,
) -> pd.DataFrame:
    quotas = _allocate_category_quotas(table, per_cohort)
    selected_rows: list[dict[str, object]] = []
    match_number = 0
    for category, quota in quotas.items():
        if quota == 0:
            continue
        category_rows = table[table["category"] == category]
        for loss, final, wrong in _match_category(
            category_rows,
            quota,
        ):
            match_id = f"{category}-{match_number:03d}"
            match_number += 1
            for cohort, row in (
                ("loss", loss),
                ("final_correct", final),
                ("stable_wrong", wrong),
            ):
                values = row.to_dict()
                selected_rows.append(
                    {
                        **values,
                        "baseline_cohort": cohort,
                        "match_id": match_id,
                        "matched_loss_tokens": int(
                            loss["trace_token_count"]
                        ),
                        "absolute_log_token_distance": abs(
                            math.log1p(int(values["trace_token_count"]))
                            - math.log1p(
                                int(loss["trace_token_count"])
                            )
                        ),
                    }
                )
    selected = pd.DataFrame(selected_rows)
    counts = selected["baseline_cohort"].value_counts().to_dict()
    expected = {
        "loss": per_cohort,
        "final_correct": per_cohort,
        "stable_wrong": per_cohort,
    }
    if counts != expected:
        raise RuntimeError(
            f"Matched cohort counts differ from expected: {counts}"
        )
    return selected.sort_values(
        ["position", "baseline_cohort"]
    ).reset_index(drop=True)


def build_matched_selection(
    per_cohort: int = 25,
) -> pd.DataFrame:
    selections: list[pd.DataFrame] = []
    for dataset in DATASETS:
        table = build_source_question_table(dataset.input_dir)
        selected = select_matched_cohorts(
            table,
            per_cohort=per_cohort,
        )
        selected.insert(0, "dataset", dataset.name)
        selected.insert(1, "dataset_label", dataset.label)
        selections.append(selected)
    return pd.concat(selections, ignore_index=True)


def _write_jsonl(
    path: Path,
    records: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def run_matched_controls(
    output_root: Path,
    seeds: list[int],
    per_cohort: int = 25,
    model: str = "google/gemma-4-12B-it",
    trace_max_tokens: int = 16384,
    max_model_len: int = 20480,
    max_num_seqs: int = 32,
    gpu_memory_utilization: float = 0.9,
) -> dict[str, object]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be non-empty and unique")
    started_at = time.perf_counter()
    selection = build_matched_selection(per_cohort)
    output_root.mkdir(parents=True, exist_ok=True)
    selection.to_parquet(
        output_root / "cohort_selection.parquet",
        index=False,
    )

    flattened_questions: list[MultipleChoiceQuestion] = []
    flattened_seeds: list[int] = []
    flattened_metadata: list[dict[str, object]] = []
    run_slices: list[
        tuple[MatchedDataset, int, int, int, list[int]]
    ] = []
    for dataset in DATASETS:
        subset = selection[
            selection["dataset"] == dataset.name
        ].sort_values("position")
        positions = subset["position"].astype(int).tolist()
        metadata_by_position = {
            int(row.position): row._asdict()
            for row in subset.itertuples(index=False)
        }
        questions = load_questions(
            dataset_name=dataset.dataset,
            dataset_format=dataset.dataset_format,
            split="test",
            start_row=0,
            num_rows=len(positions),
            selection="indices",
            seed=0,
            row_indices=positions,
        )
        for seed in seeds:
            start = len(flattened_questions)
            flattened_questions.extend(questions)
            flattened_seeds.extend([seed] * len(questions))
            flattened_metadata.extend(
                metadata_by_position[question.position]
                for question in questions
            )
            run_slices.append(
                (
                    dataset,
                    seed,
                    start,
                    len(flattened_questions),
                    positions,
                )
            )

    runner_config = ProbeConfig(
        model=model,
        num_rows=len(flattened_questions),
        seed=seeds[0],
        trace_max_tokens=trace_max_tokens,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        output_dir=output_root,
    )
    runner = GemmaProbeRunner(runner_config)
    prompts = runner.build_base_prompts(flattened_questions)
    model_ready_at = time.perf_counter()
    traces = runner.generate_traces(
        flattened_questions,
        prompts,
        seeds=flattened_seeds,
    )
    for trace, metadata in zip(traces, flattened_metadata):
        trace["baseline_cohort"] = metadata["baseline_cohort"]
        trace["match_id"] = metadata["match_id"]
        trace["baseline_trace_token_count"] = int(
            metadata["trace_token_count"]
        )
    traces_ready_at = time.perf_counter()

    trajectory_records = runner.probe_trajectories(
        flattened_questions,
        prompts,
        traces,
        seeds=flattened_seeds,
    )
    for question_index, metadata in enumerate(flattened_metadata):
        start = question_index * len(DECILES)
        end = start + len(DECILES)
        for record in trajectory_records[start:end]:
            record["baseline_cohort"] = metadata[
                "baseline_cohort"
            ]
            record["match_id"] = metadata["match_id"]
            record["baseline_trace_token_count"] = int(
                metadata["trace_token_count"]
            )
    probes_ready_at = time.perf_counter()

    run_results: dict[str, object] = {}
    for dataset, seed, start, end, positions in run_slices:
        output_dir = output_root / dataset.name / f"seed_{seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        run_traces = traces[start:end]
        record_start = start * len(DECILES)
        record_end = end * len(DECILES)
        dataframe = build_trajectory_dataframe(
            trajectory_records[record_start:record_end]
        )
        _write_jsonl(output_dir / "traces.jsonl", run_traces)
        dataframe.to_parquet(
            output_dir / "trajectory.parquet",
            index=False,
        )
        config = {
            **runner_config.to_dict(),
            "dataset": dataset.dataset,
            "dataset_format": dataset.dataset_format,
            "split": "test",
            "selection": "indices",
            "row_indices": positions,
            "start_row": 0,
            "num_rows": len(positions),
            "seed": seed,
            "output_dir": str(output_dir),
            "matched_control_experiment": True,
        }
        summary = build_summary(dataframe, run_traces, config)
        summary["matched_control_experiment"] = True
        with (output_dir / "summary.json").open(
            "w",
            encoding="utf-8",
        ) as output:
            json.dump(summary, output, indent=2, sort_keys=True)
            output.write("\n")
        if not summary["validation"]["passed"]:
            raise RuntimeError(
                f"Matched run failed validation: {output_dir}"
            )
        run_results[f"{dataset.name}/seed_{seed}"] = {
            "trace_count": len(run_traces),
            "truncated_trace_count": summary[
                "truncated_trace_count"
            ],
            "validation_passed": True,
        }

    finished_at = time.perf_counter()
    manifest = {
        "model": model,
        "seeds": seeds,
        "per_cohort_per_dataset": per_cohort,
        "question_count": len(selection),
        "total_trace_count": len(traces),
        "total_probe_row_count": len(trajectory_records),
        "trace_max_tokens": trace_max_tokens,
        "max_model_len": max_model_len,
        "runs": run_results,
        "matching": {
            dataset.name: {
                "cohort_counts": (
                    selection[selection["dataset"] == dataset.name][
                        "baseline_cohort"
                    ]
                    .value_counts()
                    .sort_index()
                    .to_dict()
                ),
                "median_absolute_log_token_distance": float(
                    selection[
                        selection["dataset"] == dataset.name
                    ]["absolute_log_token_distance"].median()
                ),
                "max_absolute_log_token_distance": float(
                    selection[
                        selection["dataset"] == dataset.name
                    ]["absolute_log_token_distance"].max()
                ),
            }
            for dataset in DATASETS
        },
        "timings_seconds": {
            "selection_and_model_initialization": round(
                model_ready_at - started_at,
                3,
            ),
            "trace_generation": round(
                traces_ready_at - model_ready_at,
                3,
            ),
            "trajectory_probing": round(
                probes_ready_at - traces_ready_at,
                3,
            ),
            "output_processing": round(
                finished_at - probes_ready_at,
                3,
            ),
            "total": round(finished_at - started_at, 3),
        },
    }
    with (output_root / "manifest.json").open(
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(manifest, output, indent=2, sort_keys=True)
        output.write("\n")
    return manifest


def matched_control_extension_specs(
    input_root: Path = Path(
        "outputs/matched_controls_gemma4_12b"
    ),
    output_root: Path = Path(
        "outputs/matched_controls_gemma4_12b_extended"
    ),
    seeds: Sequence[int] = tuple(range(10)),
):
    from .trace_extension import RunExtensionSpec

    return tuple(
        RunExtensionSpec(
            input_dir=input_root / dataset.name / f"seed_{seed}",
            output_dir=output_root / dataset.name / f"seed_{seed}",
            name=f"{dataset.name}/seed_{seed}",
        )
        for dataset in DATASETS
        for seed in seeds
    )
