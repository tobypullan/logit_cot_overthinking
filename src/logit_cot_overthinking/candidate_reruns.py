from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import ProbeConfig
from .data import MultipleChoiceQuestion, load_questions
from .gemma import DECILES, GemmaProbeRunner
from .metrics import build_summary, build_trajectory_dataframe


@dataclass(frozen=True)
class CandidateSource:
    name: str
    analysis_dir: Path
    dataset: str
    dataset_format: str
    flag_column: str


DEFAULT_SOURCES = (
    CandidateSource(
        name="mmlu_pro",
        analysis_dir=Path(
            "outputs/mmlu_pro_gemma4_12b_n1000_seed0/lost_analysis"
        ),
        dataset="TIGER-Lab/MMLU-Pro",
        dataset_format="mmlu-pro",
        flag_column="robust_loss",
    ),
    CandidateSource(
        name="gpqa_diamond",
        analysis_dir=Path(
            "outputs/gpqa_diamond_gemma4_12b_seed0/lost_analysis"
        ),
        dataset="fingertap/GPQA-Diamond",
        dataset_format="gpqa-diamond",
        flag_column="normalized_reversal_candidate",
    ),
)


def load_candidate_positions(source: CandidateSource) -> list[int]:
    path = source.analysis_dir / "lost_cases.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Candidate table not found: {path}")
    dataframe = pd.read_parquet(path)
    if source.flag_column not in dataframe:
        raise KeyError(
            f"{path} does not contain candidate flag "
            f"{source.flag_column!r}"
        )
    selected = dataframe[dataframe[source.flag_column]].sort_values("position")
    positions = selected["position"].astype(int).tolist()
    if not positions:
        raise ValueError(
            f"No candidates selected by {source.flag_column!r} in {path}"
        )
    if len(set(positions)) != len(positions):
        raise ValueError(f"Duplicate candidate positions in {path}")
    return positions


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_run(
    output_dir: Path,
    source: CandidateSource,
    positions: list[int],
    seed: int,
    traces: list[dict[str, object]],
    trajectory_records: list[dict[str, object]],
    runner_config: ProbeConfig,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "traces.jsonl", traces)
    dataframe = build_trajectory_dataframe(trajectory_records)
    dataframe.to_parquet(output_dir / "trajectory.parquet", index=False)
    config = {
        **runner_config.to_dict(),
        "dataset": source.dataset,
        "dataset_format": source.dataset_format,
        "split": "test",
        "selection": "indices",
        "row_indices": positions,
        "start_row": 0,
        "num_rows": len(positions),
        "seed": seed,
        "output_dir": str(output_dir),
        "candidate_flag": source.flag_column,
    }
    summary = build_summary(dataframe, traces, config)
    summary["candidate_rerun"] = True
    with (output_dir / "summary.json").open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, sort_keys=True)
        output.write("\n")
    if not summary["validation"]["passed"]:
        raise RuntimeError(f"Candidate rerun failed validation: {output_dir}")
    return summary


def run_candidate_reruns(
    output_root: Path,
    seeds: list[int],
    sources: tuple[CandidateSource, ...] = DEFAULT_SOURCES,
    model: str = "google/gemma-4-12B-it",
    trace_max_tokens: int = 16384,
    max_model_len: int = 20480,
    max_num_seqs: int = 16,
    gpu_memory_utilization: float = 0.9,
) -> dict[str, object]:
    if not seeds:
        raise ValueError("At least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique")

    started_at = time.perf_counter()
    positions_by_source = {
        source.name: load_candidate_positions(source) for source in sources
    }
    questions_by_source: dict[str, list[MultipleChoiceQuestion]] = {}
    for source in sources:
        positions = positions_by_source[source.name]
        questions_by_source[source.name] = load_questions(
            dataset_name=source.dataset,
            dataset_format=source.dataset_format,
            split="test",
            start_row=0,
            num_rows=len(positions),
            selection="indices",
            seed=0,
            row_indices=positions,
        )

    flattened_questions: list[MultipleChoiceQuestion] = []
    flattened_seeds: list[int] = []
    run_slices: list[
        tuple[CandidateSource, int, int, int]
    ] = []
    for source in sources:
        questions = questions_by_source[source.name]
        for seed in seeds:
            start = len(flattened_questions)
            flattened_questions.extend(questions)
            flattened_seeds.extend([seed] * len(questions))
            run_slices.append(
                (source, seed, start, len(flattened_questions))
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
    base_prompts = runner.build_base_prompts(flattened_questions)
    model_ready_at = time.perf_counter()
    traces = runner.generate_traces(
        flattened_questions,
        base_prompts,
        seeds=flattened_seeds,
    )
    traces_ready_at = time.perf_counter()

    for source, seed, start, end in run_slices:
        _write_jsonl(
            output_root / source.name / f"seed_{seed}" / "traces.jsonl",
            traces[start:end],
        )

    trajectory_records = runner.probe_trajectories(
        flattened_questions,
        base_prompts,
        traces,
        seeds=flattened_seeds,
    )
    probes_ready_at = time.perf_counter()

    summaries: dict[str, dict[str, object]] = {}
    decile_count = len(DECILES)
    for source, seed, start, end in run_slices:
        output_dir = output_root / source.name / f"seed_{seed}"
        record_start = start * decile_count
        record_end = end * decile_count
        key = f"{source.name}/seed_{seed}"
        summaries[key] = _write_run(
            output_dir=output_dir,
            source=source,
            positions=positions_by_source[source.name],
            seed=seed,
            traces=traces[start:end],
            trajectory_records=trajectory_records[
                record_start:record_end
            ],
            runner_config=runner_config,
        )

    finished_at = time.perf_counter()
    manifest = {
        "model": model,
        "seeds": seeds,
        "trace_max_tokens": trace_max_tokens,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "total_trace_count": len(flattened_questions),
        "total_probe_row_count": len(trajectory_records),
        "sources": {
            source.name: {
                "dataset": source.dataset,
                "dataset_format": source.dataset_format,
                "candidate_flag": source.flag_column,
                "positions": positions_by_source[source.name],
                "question_ids": [
                    question.question_id
                    for question in questions_by_source[source.name]
                ],
                "candidate_count": len(
                    questions_by_source[source.name]
                ),
            }
            for source in sources
        },
        "timings_seconds": {
            "model_initialization": round(model_ready_at - started_at, 3),
            "trace_generation": round(traces_ready_at - model_ready_at, 3),
            "trajectory_probing": round(
                probes_ready_at - traces_ready_at,
                3,
            ),
            "output_processing": round(finished_at - probes_ready_at, 3),
            "total": round(finished_at - started_at, 3),
        },
        "runs": {
            key: {
                "trace_count": summary["trace_count"],
                "complete_trace_count": summary[
                    "complete_trace_count"
                ],
                "validation_passed": summary["validation"]["passed"],
            }
            for key, summary in summaries.items()
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "manifest.json").open(
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(manifest, output, indent=2, sort_keys=True)
        output.write("\n")
    return manifest
