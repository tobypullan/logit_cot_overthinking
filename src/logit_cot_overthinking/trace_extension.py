from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import ProbeConfig
from .data import MultipleChoiceQuestion, load_questions
from .gemma import DECILES, THOUGHT_END, GemmaProbeRunner
from .metrics import build_summary, build_trajectory_dataframe


@dataclass(frozen=True)
class RunExtensionSpec:
    input_dir: Path
    output_dir: Path
    name: str


def default_extension_specs() -> tuple[RunExtensionSpec, ...]:
    specs = [
        RunExtensionSpec(
            input_dir=Path(
                "outputs/gpqa_diamond_gemma4_12b_seed0"
            ),
            output_dir=Path(
                "outputs/gpqa_diamond_gemma4_12b_seed0_extended"
            ),
            name="gpqa_diamond_full",
        ),
        RunExtensionSpec(
            input_dir=Path(
                "outputs/mmlu_pro_gemma4_12b_n1000_seed0"
            ),
            output_dir=Path(
                "outputs/mmlu_pro_gemma4_12b_n1000_seed0_extended"
            ),
            name="mmlu_pro_n1000",
        ),
    ]
    candidate_input = Path(
        "outputs/candidate_reruns_gemma4_12b"
    )
    candidate_output = Path(
        "outputs/candidate_reruns_gemma4_12b_extended"
    )
    for dataset in ("mmlu_pro", "gpqa_diamond"):
        for seed in range(10):
            specs.append(
                RunExtensionSpec(
                    input_dir=(
                        candidate_input / dataset / f"seed_{seed}"
                    ),
                    output_dir=(
                        candidate_output / dataset / f"seed_{seed}"
                    ),
                    name=f"{dataset}/seed_{seed}",
                )
            )
    return tuple(specs)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


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


def _key(record: object) -> tuple[int, str]:
    if isinstance(record, MultipleChoiceQuestion):
        return int(record.position), str(record.question_id)
    return int(record["position"]), str(record["question_id"])


def _load_run(
    spec: RunExtensionSpec,
) -> dict[str, object]:
    summary_path = spec.input_dir / "summary.json"
    traces_path = spec.input_dir / "traces.jsonl"
    trajectory_path = spec.input_dir / "trajectory.parquet"
    for path in (summary_path, traces_path, trajectory_path):
        if not path.exists():
            raise FileNotFoundError(f"Run artifact not found: {path}")

    summary = _read_json(summary_path)
    config = dict(summary["config"])
    row_indices = tuple(
        int(value) for value in config.get("row_indices", [])
    )
    questions = load_questions(
        dataset_name=str(config["dataset"]),
        dataset_format=str(config.get("dataset_format", "auto")),
        split=str(config.get("split", "test")),
        start_row=int(config.get("start_row", 0)),
        num_rows=int(config["num_rows"]),
        selection=str(config.get("selection", "contiguous")),
        seed=int(config.get("seed", 0)),
        row_indices=row_indices,
    )
    traces = _read_jsonl(traces_path)
    if [_key(question) for question in questions] != [
        _key(trace) for trace in traces
    ]:
        raise RuntimeError(
            f"Question and trace order differ in {spec.input_dir}"
        )
    return {
        "spec": spec,
        "summary": summary,
        "config": config,
        "questions": questions,
        "traces": traces,
        "trajectory": pd.read_parquet(trajectory_path),
    }


def _replace_by_key(
    records: list[dict[str, object]],
    replacements: dict[tuple[int, str], dict[str, object]],
) -> list[dict[str, object]]:
    return [
        replacements.get(_key(record), record)
        for record in records
    ]


def extend_capped_runs(
    specs: tuple[RunExtensionSpec, ...] | None = None,
    model: str = "google/gemma-4-12B-it",
    extension_max_tokens: int = 16384,
    max_extension_rounds: int = 1,
    max_model_len: int = 49152,
    max_num_seqs: int = 16,
    gpu_memory_utilization: float = 0.9,
    manifest_path: Path = Path(
        "outputs/trace_extensions_gemma4_12b_manifest.json"
    ),
) -> dict[str, object]:
    specs = specs or default_extension_specs()
    started_at = time.perf_counter()
    runs = [_load_run(spec) for spec in specs]
    capped_count = sum(
        bool(trace.get("truncated", False))
        for run in runs
        for trace in run["traces"]
    )
    if capped_count == 0:
        raise ValueError("No capped traces were found")

    runner_config = ProbeConfig(
        model=model,
        num_rows=capped_count,
        seed=0,
        trace_max_tokens=extension_max_tokens,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        output_dir=Path("outputs"),
    )
    runner_config.validate()
    runner = GemmaProbeRunner(runner_config)
    model_ready_at = time.perf_counter()

    extension_entries: list[
        tuple[
            dict[str, object],
            MultipleChoiceQuestion,
            str,
            dict[str, object],
        ]
    ] = []
    for run in runs:
        prompts = runner.build_base_prompts(run["questions"])
        for question, prompt, trace in zip(
            run["questions"],
            prompts,
            run["traces"],
        ):
            if bool(trace.get("truncated", False)):
                extension_entries.append(
                    (run, question, prompt, trace)
                )

    for _ in range(max_extension_rounds):
        pending = [
            entry
            for entry in extension_entries
            if bool(entry[3].get("truncated", False))
        ]
        if not pending:
            break
        extended = runner.extend_traces(
            [entry[1] for entry in pending],
            [entry[2] for entry in pending],
            [entry[3] for entry in pending],
            extension_max_tokens=extension_max_tokens,
        )
        replacement_by_identity = {
            id(entry[3]): record
            for entry, record in zip(pending, extended)
        }
        extension_entries = [
            (
                run,
                question,
                prompt,
                replacement_by_identity.get(id(trace), trace),
            )
            for run, question, prompt, trace in extension_entries
        ]

    still_capped = [
        entry
        for entry in extension_entries
        if bool(entry[3].get("truncated", False))
    ]
    if still_capped:
        forced = runner.force_close_traces(
            [entry[1] for entry in still_capped],
            [entry[2] for entry in still_capped],
            [entry[3] for entry in still_capped],
        )
        replacement_by_identity = {
            id(entry[3]): record
            for entry, record in zip(still_capped, forced)
        }
        extension_entries = [
            (
                run,
                question,
                prompt,
                replacement_by_identity.get(id(trace), trace),
            )
            for run, question, prompt, trace in extension_entries
        ]
    extensions_ready_at = time.perf_counter()

    extended_questions = [entry[1] for entry in extension_entries]
    extended_prompts = [entry[2] for entry in extension_entries]
    extended_traces = [entry[3] for entry in extension_entries]
    extended_seeds = [
        int(trace.get("run_seed", 0)) for trace in extended_traces
    ]
    probe_records = runner.probe_trajectories(
        extended_questions,
        extended_prompts,
        extended_traces,
        seeds=extended_seeds,
    )
    probes_ready_at = time.perf_counter()

    replacement_traces_by_run: dict[
        str,
        dict[tuple[int, str], dict[str, object]],
    ] = {}
    probe_records_by_run: dict[str, list[dict[str, object]]] = {}
    decile_count = len(DECILES)
    for index, entry in enumerate(extension_entries):
        run, question, _, trace = entry
        name = run["spec"].name
        replacement_traces_by_run.setdefault(name, {})[
            _key(question)
        ] = trace
        probe_records_by_run.setdefault(name, []).extend(
            probe_records[
                index * decile_count : (index + 1) * decile_count
            ]
        )

    run_results: dict[str, dict[str, object]] = {}
    for run in runs:
        spec = run["spec"]
        replacements = replacement_traces_by_run.get(
            spec.name,
            {},
        )
        traces = _replace_by_key(run["traces"], replacements)
        changed_keys = set(replacements)
        retained = run["trajectory"][
            ~run["trajectory"].apply(
                lambda row: _key(row) in changed_keys,
                axis=1,
            )
        ]
        combined_records = retained.to_dict(orient="records")
        combined_records.extend(
            probe_records_by_run.get(spec.name, [])
        )
        dataframe = build_trajectory_dataframe(combined_records)

        output_config = {
            **run["config"],
            "output_dir": str(spec.output_dir),
            "trace_max_tokens": (
                int(run["config"]["trace_max_tokens"])
                + extension_max_tokens * max_extension_rounds
            ),
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
        }
        summary = build_summary(dataframe, traces, output_config)
        summary["trace_extension"] = {
            "source_dir": str(spec.input_dir),
            "extended_trace_count": len(replacements),
            "extension_max_tokens": extension_max_tokens,
            "max_extension_rounds": max_extension_rounds,
            "remaining_truncated_trace_count": sum(
                bool(trace.get("truncated", False))
                for trace in traces
            ),
            "forced_completion_count": sum(
                bool(trace.get("forced_completion", False))
                for trace in replacements.values()
            ),
        }
        if not summary["validation"]["passed"]:
            raise RuntimeError(
                f"Extended run failed validation: {spec.name}"
            )
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(spec.output_dir / "traces.jsonl", traces)
        dataframe.to_parquet(
            spec.output_dir / "trajectory.parquet",
            index=False,
        )
        _write_json(spec.output_dir / "summary.json", summary)
        run_results[spec.name] = {
            "input_dir": str(spec.input_dir),
            "output_dir": str(spec.output_dir),
            "trace_count": len(traces),
            "extended_trace_count": len(replacements),
            "truncated_before": int(
                run["summary"]["truncated_trace_count"]
            ),
            "truncated_after": summary["truncated_trace_count"],
            "forced_completion_count": summary[
                "trace_extension"
            ]["forced_completion_count"],
            "validation_passed": summary["validation"]["passed"],
        }

    finished_at = time.perf_counter()
    manifest = {
        "model": model,
        "extension_max_tokens": extension_max_tokens,
        "max_extension_rounds": max_extension_rounds,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "run_count": len(runs),
        "extended_trace_count": len(extension_entries),
        "forced_completion_count": sum(
            bool(entry[3].get("forced_completion", False))
            for entry in extension_entries
        ),
        "runs": run_results,
        "timings_seconds": {
            "input_loading_and_model_initialization": round(
                model_ready_at - started_at,
                3,
            ),
            "trace_extension": round(
                extensions_ready_at - model_ready_at,
                3,
            ),
            "trajectory_probing": round(
                probes_ready_at - extensions_ready_at,
                3,
            ),
            "output_processing": round(
                finished_at - probes_ready_at,
                3,
            ),
            "total": round(finished_at - started_at, 3),
        },
    }
    _write_json(manifest_path, manifest)
    return manifest


def _strip_forced_answer(
    trace: dict[str, object],
) -> dict[str, object]:
    raw_response = str(trace["raw_response"])
    if THOUGHT_END not in raw_response:
        raise ValueError("Forced trace does not contain a thought closure")
    record = dict(trace)
    record.update(
        {
            "raw_response": raw_response.split(THOUGHT_END, maxsplit=1)[0],
            "generated_answer_text": "",
            "generated_answer": None,
            "generated_token_count": int(
                trace["generated_token_count"]
            )
            - int(trace.get("forced_answer_token_count", 0)),
            "finish_reason": "length",
            "truncated": True,
            "forced_completion": False,
            "forced_answer_token_count": 0,
        }
    )
    return record


def repair_forced_answers(
    specs: tuple[RunExtensionSpec, ...] | None = None,
    model: str = "google/gemma-4-12B-it",
    max_model_len: int = 49152,
    max_num_seqs: int = 8,
    gpu_memory_utilization: float = 0.9,
) -> dict[str, object]:
    source_specs = specs or default_extension_specs()
    repair_specs = tuple(
        RunExtensionSpec(
            input_dir=spec.output_dir,
            output_dir=spec.output_dir,
            name=spec.name,
        )
        for spec in source_specs
        if spec.output_dir.exists()
    )
    runs = [_load_run(spec) for spec in repair_specs]
    forced_count = sum(
        bool(trace.get("forced_completion", False))
        for run in runs
        for trace in run["traces"]
    )
    if not forced_count:
        return {"repaired_trace_count": 0, "runs": {}}

    config = ProbeConfig(
        model=model,
        num_rows=forced_count,
        seed=0,
        trace_max_tokens=1,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        output_dir=Path("outputs"),
    )
    runner = GemmaProbeRunner(config)
    entries: list[
        tuple[
            dict[str, object],
            MultipleChoiceQuestion,
            str,
            dict[str, object],
        ]
    ] = []
    for run in runs:
        prompts = runner.build_base_prompts(run["questions"])
        for question, prompt, trace in zip(
            run["questions"],
            prompts,
            run["traces"],
        ):
            if bool(trace.get("forced_completion", False)):
                entries.append(
                    (
                        run,
                        question,
                        prompt,
                        _strip_forced_answer(trace),
                    )
                )

    repaired = runner.force_close_traces(
        [entry[1] for entry in entries],
        [entry[2] for entry in entries],
        [entry[3] for entry in entries],
    )
    probes = runner.probe_trajectories(
        [entry[1] for entry in entries],
        [entry[2] for entry in entries],
        repaired,
        seeds=[
            int(trace.get("run_seed", 0)) for trace in repaired
        ],
    )
    by_run: dict[str, dict[tuple[int, str], dict[str, object]]] = {}
    probe_by_run: dict[str, list[dict[str, object]]] = {}
    for index, (entry, trace) in enumerate(zip(entries, repaired)):
        run, question, _, _ = entry
        by_run.setdefault(run["spec"].name, {})[
            _key(question)
        ] = trace
        probe_by_run.setdefault(run["spec"].name, []).extend(
            probes[index * len(DECILES) : (index + 1) * len(DECILES)]
        )

    results: dict[str, object] = {}
    for run in runs:
        replacements = by_run.get(run["spec"].name, {})
        if not replacements:
            continue
        traces = _replace_by_key(run["traces"], replacements)
        keys = set(replacements)
        retained = run["trajectory"][
            ~run["trajectory"].apply(
                lambda row: _key(row) in keys,
                axis=1,
            )
        ]
        records = retained.to_dict(orient="records")
        records.extend(probe_by_run[run["spec"].name])
        dataframe = build_trajectory_dataframe(records)
        summary = build_summary(
            dataframe,
            traces,
            run["config"],
        )
        summary["trace_extension"] = {
            **run["summary"].get("trace_extension", {}),
            "forced_answers_constrained_to_valid_labels": True,
        }
        if not summary["validation"]["passed"]:
            raise RuntimeError(
                f"Repaired run failed validation: "
                f"{run['spec'].name}"
            )
        _write_jsonl(
            run["spec"].output_dir / "traces.jsonl",
            traces,
        )
        dataframe.to_parquet(
            run["spec"].output_dir / "trajectory.parquet",
            index=False,
        )
        _write_json(
            run["spec"].output_dir / "summary.json",
            summary,
        )
        results[run["spec"].name] = {
            "repaired_trace_count": len(replacements),
            "validation_passed": True,
        }
    return {
        "repaired_trace_count": forced_count,
        "runs": results,
    }
