from __future__ import annotations

import json
import time
from pathlib import Path

from .config import ProbeConfig
from .data import MultipleChoiceQuestion, load_questions
from .gemma import GemmaProbeRunner
from .metrics import build_summary, build_trajectory_dataframe


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_reusable_traces(
    config: ProbeConfig,
    questions: list[MultipleChoiceQuestion],
) -> dict[tuple[int, str], dict[str, object]]:
    traces_path = config.output_dir / "traces.jsonl"
    summary_path = config.output_dir / "summary.json"
    if not traces_path.exists() or not summary_path.exists():
        return {}

    previous_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    previous_config = previous_summary.get("config", {})
    identity_keys = (
        "model",
        "dataset",
        "dataset_format",
        "split",
        "selection",
        "row_indices",
        "start_row",
        "num_rows",
        "seed",
    )
    mismatches = {
        key: (
            previous_config.get(
                key,
                "auto" if key == "dataset_format" else None,
            ),
            config.to_dict().get(key),
        )
        for key in identity_keys
        if previous_config.get(
            key,
            "auto" if key == "dataset_format" else None,
        )
        != config.to_dict().get(key)
    }
    if mismatches:
        raise ValueError(
            "Cannot reuse traces from a different run configuration: "
            f"{mismatches}"
        )

    selected_keys = {
        (int(question.position), str(question.question_id))
        for question in questions
    }
    reusable: dict[tuple[int, str], dict[str, object]] = {}
    for record in _read_jsonl(traces_path):
        key = (int(record["position"]), str(record["question_id"]))
        if key in selected_keys and not bool(record.get("truncated", False)):
            reusable[key] = record
    return reusable


def run_probe(config: ProbeConfig) -> dict[str, object]:
    started_at = time.perf_counter()
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(
        dataset_name=config.dataset,
        dataset_format=config.dataset_format,
        split=config.split,
        start_row=config.start_row,
        num_rows=config.num_rows,
        selection=config.selection,
        seed=config.seed,
        row_indices=config.row_indices,
    )
    if len(questions) != config.num_rows:
        raise RuntimeError(
            f"Requested {config.num_rows} rows but loaded {len(questions)}"
        )
    dataset_ready_at = time.perf_counter()

    runner = GemmaProbeRunner(config)
    base_prompts = runner.build_base_prompts(questions)
    model_ready_at = time.perf_counter()
    reusable = (
        _load_reusable_traces(config, questions)
        if config.resume_traces
        else {}
    )
    pending_indices = [
        index
        for index, question in enumerate(questions)
        if (question.position, question.question_id) not in reusable
    ]
    generated = (
        runner.generate_traces(
            [questions[index] for index in pending_indices],
            [base_prompts[index] for index in pending_indices],
        )
        if pending_indices
        else []
    )
    generated_by_key = {
        (int(record["position"]), str(record["question_id"])): record
        for record in generated
    }
    traces = [
        reusable[(question.position, question.question_id)]
        if (question.position, question.question_id) in reusable
        else generated_by_key[(question.position, question.question_id)]
        for question in questions
    ]
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
    summary["trace_reuse"] = {
        "reused": len(reusable),
        "generated": len(generated),
    }
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
