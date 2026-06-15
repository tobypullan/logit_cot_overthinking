from __future__ import annotations

import json
import time
from pathlib import Path

from .config import ProbeConfig
from .data import load_mmlu_pro_questions
from .gemma import GemmaProbeRunner
from .metrics import build_summary, build_trajectory_dataframe


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_probe(config: ProbeConfig) -> dict[str, object]:
    started_at = time.perf_counter()
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    questions = load_mmlu_pro_questions(
        dataset_name=config.dataset,
        split=config.split,
        start_row=config.start_row,
        num_rows=config.num_rows,
        selection=config.selection,
        seed=config.seed,
    )
    if len(questions) != config.num_rows:
        raise RuntimeError(
            f"Requested {config.num_rows} rows but loaded {len(questions)}"
        )
    dataset_ready_at = time.perf_counter()

    runner = GemmaProbeRunner(config)
    base_prompts = runner.build_base_prompts(questions)
    model_ready_at = time.perf_counter()
    traces = runner.generate_traces(questions, base_prompts)
    traces_ready_at = time.perf_counter()
    _write_jsonl(config.output_dir / "traces.jsonl", traces)

    trajectory_records = runner.probe_trajectories(
        questions,
        base_prompts,
        traces,
    )
    probes_ready_at = time.perf_counter()
    dataframe = build_trajectory_dataframe(trajectory_records)
    dataframe.to_parquet(config.output_dir / "trajectory.parquet", index=False)
    outputs_ready_at = time.perf_counter()

    summary = build_summary(dataframe, traces, config.to_dict())
    summary["timings_seconds"] = {
        "dataset_loading": round(dataset_ready_at - started_at, 3),
        "model_initialization": round(model_ready_at - dataset_ready_at, 3),
        "trace_generation": round(traces_ready_at - model_ready_at, 3),
        "trajectory_probing": round(probes_ready_at - traces_ready_at, 3),
        "output_processing": round(outputs_ready_at - probes_ready_at, 3),
        "total_before_summary_write": round(outputs_ready_at - started_at, 3),
    }
    with (config.output_dir / "summary.json").open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, sort_keys=True)
        output.write("\n")

    if not summary["validation"]["passed"]:
        raise RuntimeError(
            f"Probe outputs failed validation; see {config.output_dir / 'summary.json'}"
        )
    return summary
