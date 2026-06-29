from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import ProbeConfig
from .data import MultipleChoiceQuestion
from .gemma import (
    THOUGHT_END,
    THOUGHT_START,
    GemmaProbeRunner,
    extract_answer_letter,
    parse_gemma_response,
    validate_answer_tokens,
)


BRANCH_MODE_INSTRUCTIONS = {
    "answer_only": "",
    "normal": "",
    "short_verification": (
        "\n\nCheck the last step briefly before choosing the answer.\n"
    ),
    "preserve_unless_decisive": (
        "\n\nKeep the current answer unless there is a decisive "
        "contradiction. Check only the key premise before choosing the "
        "answer.\n"
    ),
}
BRANCH_MODES = tuple(BRANCH_MODE_INSTRUCTIONS)
FINAL_OUTCOMES = ("loss", "correct", "all")


@dataclass(frozen=True)
class BranchingInterventionConfig:
    input_root: Path = Path(
        "outputs/matched_controls_gemma4_12b_extended"
    )
    selection_path: Path | None = Path(
        "outputs/matched_controls_gemma4_12b/cohort_selection.parquet"
    )
    output_dir: Path = Path(
        "outputs/branching_intervention_gemma4_12b"
    )
    model: str = "google/gemma-4-12B-it"
    deciles: tuple[int, ...] = (30, 40, 50, 60, 70, 80, 90)
    cohorts: tuple[str, ...] = ("loss",)
    final_outcome: str = "loss"
    min_current_normalized_correct_probability: float = 0.90
    max_candidates_per_dataset: int = 25
    branch_modes: tuple[str, ...] = BRANCH_MODES
    branch_seeds: tuple[int, ...] = (0, 1, 2, 3)
    branch_max_tokens: int = 512
    dry_run: bool = True
    trace_max_tokens: int = 512
    max_model_len: int = 49152
    max_num_seqs: int = 16
    gpu_memory_utilization: float = 0.90
    continuation_temperature: float = 1.0
    continuation_top_p: float = 0.95
    continuation_top_k: int = 64

    def validate(self) -> None:
        if self.final_outcome not in FINAL_OUTCOMES:
            raise ValueError(
                f"final_outcome must be one of {FINAL_OUTCOMES}"
            )
        unknown = sorted(set(self.branch_modes) - set(BRANCH_MODES))
        if unknown:
            raise ValueError(f"Unsupported branch modes: {unknown}")
        if not self.deciles:
            raise ValueError("At least one decile is required")
        invalid_deciles = [
            decile
            for decile in self.deciles
            if decile <= 0 or decile >= 100 or decile % 10 != 0
        ]
        if invalid_deciles:
            raise ValueError(
                "Branch deciles must be pre-final deciles in 10-point "
                f"steps: {invalid_deciles}"
            )
        if not 0 <= self.min_current_normalized_correct_probability <= 1:
            raise ValueError(
                "min_current_normalized_correct_probability must be in [0, 1]"
            )
        if self.max_candidates_per_dataset < 1:
            raise ValueError("max_candidates_per_dataset must be at least 1")
        if not self.branch_seeds:
            raise ValueError("At least one branch seed is required")
        if self.branch_max_tokens < 1:
            raise ValueError("branch_max_tokens must be at least 1")
        if self.max_model_len <= self.branch_max_tokens:
            raise ValueError("max_model_len must be greater than branch_max_tokens")
        if self.max_num_seqs < 1:
            raise ValueError("max_num_seqs must be at least 1")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(_json_ready(value), output, indent=2, sort_keys=True)
        output.write("\n")


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(_json_ready(record)) + "\n")


def _json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        as_float = float(value)
        return None if math.isnan(as_float) else as_float
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _slugify(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return text.lower() or "item"


def load_currently_correct_checkpoints(
    input_root: Path,
    selection_path: Path | None = None,
    model: str = "google/gemma-4-12B-it",
) -> pd.DataFrame:
    checkpoint_path = input_root / "analysis" / "currently_correct_checkpoints.parquet"
    if checkpoint_path.exists():
        return pd.read_parquet(checkpoint_path)

    if selection_path is None:
        raise FileNotFoundError(
            f"{checkpoint_path} does not exist and no selection_path was provided"
        )

    from .matched_analysis import build_matched_attempt_tables

    _, checkpoints = build_matched_attempt_tables(
        input_root=input_root,
        selection_path=selection_path,
        model=model,
    )
    return checkpoints


def select_branch_candidates(
    checkpoints: pd.DataFrame,
    input_root: Path,
    deciles: Sequence[int] = (30, 40, 50, 60, 70, 80, 90),
    cohorts: Sequence[str] = ("loss",),
    final_outcome: str = "loss",
    min_current_normalized_correct_probability: float = 0.90,
    max_candidates_per_dataset: int = 25,
) -> pd.DataFrame:
    required = {
        "dataset",
        "seed",
        "position",
        "question_id",
        "baseline_cohort",
        "decile",
        "final_wrong",
        "current_normalized_correct_probability",
        "prefix_token_count",
    }
    missing = sorted(required - set(checkpoints.columns))
    if missing:
        raise ValueError(f"Checkpoint table is missing columns: {missing}")
    if final_outcome not in FINAL_OUTCOMES:
        raise ValueError(f"final_outcome must be one of {FINAL_OUTCOMES}")

    subset = checkpoints.copy()
    subset = subset[subset["decile"].astype(int).isin([int(d) for d in deciles])]
    if cohorts:
        subset = subset[subset["baseline_cohort"].isin(list(cohorts))]
    if final_outcome == "loss":
        subset = subset[subset["final_wrong"].astype(bool)]
    elif final_outcome == "correct":
        subset = subset[~subset["final_wrong"].astype(bool)]
    subset = subset[
        subset["current_normalized_correct_probability"].astype(float)
        >= min_current_normalized_correct_probability
    ]
    subset = subset[subset["prefix_token_count"].astype(int) > 0]
    if subset.empty:
        return subset.assign(
            candidate_id=pd.Series(dtype="string"),
            trace_path=pd.Series(dtype="string"),
        )

    sort_columns = [
        "dataset",
        "seed",
        "position",
        "question_id",
        "decile",
        "current_normalized_correct_probability",
    ]
    subset = subset.sort_values(
        sort_columns,
        ascending=[True, True, True, True, True, False],
    )
    subset = subset.drop_duplicates(
        ["dataset", "seed", "position", "question_id"],
        keep="first",
    )
    subset = subset.sort_values(
        [
            "dataset",
            "current_normalized_correct_probability",
            "decile",
            "seed",
            "position",
        ],
        ascending=[True, False, True, True, True],
    )
    subset = (
        subset.groupby("dataset", group_keys=False)
        .head(max_candidates_per_dataset)
        .reset_index(drop=True)
    )

    rows = []
    for row in subset.to_dict(orient="records"):
        candidate_id = "_".join(
            [
                _slugify(row["dataset"]),
                f"seed{int(row['seed'])}",
                f"pos{int(row['position'])}",
                _slugify(row["question_id"]),
                f"d{int(row['decile'])}",
            ]
        )
        row["candidate_id"] = candidate_id
        row["trace_path"] = str(
            input_root
            / str(row["dataset"])
            / f"seed_{int(row['seed'])}"
            / "traces.jsonl"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_branch_requests(
    candidates: pd.DataFrame,
    branch_modes: Sequence[str] = BRANCH_MODES,
    branch_seeds: Sequence[int] = (0, 1, 2, 3),
    branch_max_tokens: int = 512,
) -> pd.DataFrame:
    unknown = sorted(set(branch_modes) - set(BRANCH_MODES))
    if unknown:
        raise ValueError(f"Unsupported branch modes: {unknown}")

    rows: list[dict[str, object]] = []
    request_index = 0
    for candidate in candidates.to_dict(orient="records"):
        for mode in branch_modes:
            for seed in branch_seeds:
                branch_id = "_".join(
                    [
                        str(candidate["candidate_id"]),
                        _slugify(mode),
                        f"s{int(seed)}",
                    ]
                )
                rows.append(
                    {
                        **candidate,
                        "request_index": request_index,
                        "branch_id": branch_id,
                        "branch_mode": mode,
                        "branch_seed": int(seed),
                        "branch_max_tokens": int(branch_max_tokens),
                    }
                )
                request_index += 1
    return pd.DataFrame(rows)


def decode_trace_prefix(
    trace: dict[str, object],
    prefix_token_count: int,
    tokenizer: Any,
) -> str:
    if prefix_token_count < 0:
        raise ValueError("prefix_token_count must be non-negative")
    reasoning = str(trace["reasoning_trace"])
    token_ids = tokenizer.encode(reasoning, add_special_tokens=False)
    if prefix_token_count > len(token_ids):
        raise ValueError(
            "prefix_token_count exceeds trace length: "
            f"{prefix_token_count} > {len(token_ids)}"
        )
    if prefix_token_count == 0:
        return ""
    return tokenizer.decode(
        token_ids[:prefix_token_count],
        skip_special_tokens=False,
    )


def branch_reasoning_prefix(prefix: str, mode: str) -> str:
    if mode not in BRANCH_MODE_INSTRUCTIONS:
        raise ValueError(f"Unsupported branch mode: {mode}")
    return f"{prefix}{BRANCH_MODE_INSTRUCTIONS[mode]}"


def build_branch_prompt(base_prompt: str, prefix: str, mode: str) -> str:
    reasoning_prefix = branch_reasoning_prefix(prefix, mode)
    if mode == "answer_only":
        return f"{base_prompt}{THOUGHT_START}{reasoning_prefix}{THOUGHT_END}"
    return f"{base_prompt}{THOUGHT_START}{reasoning_prefix}"


def _question_from_trace(trace: dict[str, object]) -> MultipleChoiceQuestion:
    return MultipleChoiceQuestion(
        position=int(trace["position"]),
        question_id=str(trace["question_id"]),
        question=str(trace["question"]),
        options=tuple(str(option) for option in trace["options"]),
        answer=str(trace["answer"]),
        category=str(trace.get("category", "")),
        source=str(trace.get("source", "")),
    )


def _load_trace_cache(candidates: pd.DataFrame) -> dict[Path, dict[tuple[int, str], dict[str, object]]]:
    cache: dict[Path, dict[tuple[int, str], dict[str, object]]] = {}
    for path_text in sorted(candidates["trace_path"].astype(str).unique()):
        path = Path(path_text)
        traces = _read_jsonl(path)
        cache[path] = {
            (int(trace["position"]), str(trace["question_id"])): trace
            for trace in traces
        }
    return cache


def _lookup_trace(
    trace_cache: dict[Path, dict[tuple[int, str], dict[str, object]]],
    row: dict[str, object],
) -> dict[str, object]:
    path = Path(str(row["trace_path"]))
    key = (int(row["position"]), str(row["question_id"]))
    try:
        return trace_cache[path][key]
    except KeyError as error:
        raise KeyError(f"Trace {key} not found in {path}") from error


def _materialize_entries(
    requests: pd.DataFrame,
    runner: GemmaProbeRunner,
) -> list[dict[str, object]]:
    trace_cache = _load_trace_cache(requests)
    prompt_cache: dict[tuple[str, int, str], str] = {}
    entries: list[dict[str, object]] = []
    for request in requests.to_dict(orient="records"):
        trace = _lookup_trace(trace_cache, request)
        question = _question_from_trace(trace)
        prompt_key = (
            str(request["dataset"]),
            int(request["position"]),
            str(request["question_id"]),
        )
        if prompt_key not in prompt_cache:
            prompt_cache[prompt_key] = runner.build_base_prompts([question])[0]
        prefix = decode_trace_prefix(
            trace,
            int(request["prefix_token_count"]),
            runner.tokenizer,
        )
        mode = str(request["branch_mode"])
        reasoning_prefix = branch_reasoning_prefix(prefix, mode)
        prompt = build_branch_prompt(
            prompt_cache[prompt_key],
            prefix,
            mode,
        )
        entries.append(
            {
                "request": request,
                "trace": trace,
                "question": question,
                "base_prompt": prompt_cache[prompt_key],
                "prompt": prompt,
                "reasoning_prefix": reasoning_prefix,
                "answer_tokens": validate_answer_tokens(
                    runner.tokenizer,
                    question.labels,
                ),
            }
        )
    return entries


def _initial_branch_results(
    entries: list[dict[str, object]],
    runner: GemmaProbeRunner,
    config: BranchingInterventionConfig,
) -> list[dict[str, object]]:
    from vllm import SamplingParams

    prompts = [str(entry["prompt"]) for entry in entries]
    sampling_params = []
    for entry in entries:
        request = entry["request"]
        answer_token_ids = list(entry["answer_tokens"].values())
        if request["branch_mode"] == "answer_only":
            sampling_params.append(
                SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    top_k=0,
                    max_tokens=1,
                    seed=int(request["branch_seed"]),
                    allowed_token_ids=answer_token_ids,
                    skip_special_tokens=False,
                    spaces_between_special_tokens=False,
                )
            )
        else:
            sampling_params.append(
                SamplingParams(
                    temperature=config.continuation_temperature,
                    top_p=config.continuation_top_p,
                    top_k=config.continuation_top_k,
                    max_tokens=config.branch_max_tokens,
                    seed=int(request["branch_seed"]),
                    skip_special_tokens=False,
                    spaces_between_special_tokens=False,
                )
            )

    outputs = runner.llm.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    if len(outputs) != len(entries):
        raise RuntimeError(
            f"Expected {len(entries)} branch outputs, got {len(outputs)}"
        )

    rows: list[dict[str, object]] = []
    for entry, output in zip(entries, outputs):
        if not output.outputs:
            raise RuntimeError(
                f"No branch output for {entry['request']['branch_id']}"
            )
        request = dict(entry["request"])
        question = entry["question"]
        completion = output.outputs[0]
        if request["branch_mode"] == "answer_only":
            raw_response = (
                f"{THOUGHT_START}{entry['reasoning_prefix']}"
                f"{THOUGHT_END}{completion.text}"
            )
        else:
            raw_response = f"{THOUGHT_START}{entry['reasoning_prefix']}{completion.text}"
        parsed = parse_gemma_response(raw_response)
        answer = extract_answer_letter(parsed.answer_text, question.labels)
        rows.append(
            {
                **request,
                "branch_completion_text": completion.text,
                "branch_generated_token_count": len(completion.token_ids),
                "branch_finish_reason": completion.finish_reason,
                "raw_branch_response": raw_response,
                "branch_answer_text": parsed.answer_text,
                "branch_answer": answer,
                "branch_correct": answer == question.answer,
                "branch_forced_close": False,
                "forced_answer_text": "",
            }
        )
    return rows


def _force_close_missing_answers(
    rows: list[dict[str, object]],
    entries: list[dict[str, object]],
    runner: GemmaProbeRunner,
) -> list[dict[str, object]]:
    from vllm import SamplingParams

    pending: list[tuple[int, dict[str, object]]] = []
    for index, row in enumerate(rows):
        if row["branch_mode"] != "answer_only" and row["branch_answer"] is None:
            pending.append((index, entries[index]))
    if not pending:
        return rows

    prompts = []
    sampling_params = []
    for row_index, entry in pending:
        raw_response = str(rows[row_index]["raw_branch_response"])
        closure = "" if THOUGHT_END in raw_response else THOUGHT_END
        prompts.append(f"{entry['base_prompt']}{raw_response}{closure}")
        sampling_params.append(
            SamplingParams(
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                max_tokens=1,
                seed=int(entry["request"]["branch_seed"]),
                allowed_token_ids=list(entry["answer_tokens"].values()),
                skip_special_tokens=False,
                spaces_between_special_tokens=False,
            )
        )

    outputs = runner.llm.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    if len(outputs) != len(pending):
        raise RuntimeError(
            f"Expected {len(pending)} forced branch answers, got {len(outputs)}"
        )

    for (row_index, entry), output in zip(pending, outputs):
        if not output.outputs:
            raise RuntimeError(
                f"No forced answer for {rows[row_index]['branch_id']}"
            )
        completion = output.outputs[0]
        raw_response = str(rows[row_index]["raw_branch_response"])
        closure = "" if THOUGHT_END in raw_response else THOUGHT_END
        merged = f"{raw_response}{closure}{completion.text}"
        parsed = parse_gemma_response(merged)
        question = entry["question"]
        answer = extract_answer_letter(parsed.answer_text, question.labels)
        rows[row_index].update(
            {
                "raw_branch_response": merged,
                "branch_answer_text": parsed.answer_text,
                "branch_answer": answer,
                "branch_correct": answer == question.answer,
                "branch_forced_close": True,
                "forced_answer_text": completion.text,
                "branch_generated_token_count": (
                    int(rows[row_index]["branch_generated_token_count"])
                    + len(completion.token_ids)
                ),
            }
        )
    return rows


def summarize_branch_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_columns = [
        "dataset",
        "baseline_cohort",
        "final_wrong",
        "branch_mode",
    ]
    for key, group in results.groupby(group_columns, dropna=False):
        dataset, cohort, final_wrong, mode = key
        rows.append(
            {
                "dataset": dataset,
                "baseline_cohort": cohort,
                "final_wrong": bool(final_wrong),
                "branch_mode": mode,
                "request_count": len(group),
                "candidate_count": group["candidate_id"].nunique(),
                "branch_accuracy": float(group["branch_correct"].mean()),
                "missing_answer_rate": float(group["branch_answer"].isna().mean()),
                "forced_close_rate": float(group["branch_forced_close"].mean()),
            }
        )
    return pd.DataFrame(rows)


def setup_branching_intervention(
    config: BranchingInterventionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = load_currently_correct_checkpoints(
        input_root=config.input_root,
        selection_path=config.selection_path,
        model=config.model,
    )
    candidates = select_branch_candidates(
        checkpoints=checkpoints,
        input_root=config.input_root,
        deciles=config.deciles,
        cohorts=config.cohorts,
        final_outcome=config.final_outcome,
        min_current_normalized_correct_probability=(
            config.min_current_normalized_correct_probability
        ),
        max_candidates_per_dataset=config.max_candidates_per_dataset,
    )
    requests = build_branch_requests(
        candidates,
        branch_modes=config.branch_modes,
        branch_seeds=config.branch_seeds,
        branch_max_tokens=config.branch_max_tokens,
    )

    candidates_path = config.output_dir / "branch_candidates.parquet"
    requests_path = config.output_dir / "branch_requests.parquet"
    requests_jsonl_path = config.output_dir / "branch_requests.jsonl"
    candidates.to_parquet(candidates_path, index=False)
    requests.to_parquet(requests_path, index=False)
    _write_jsonl(requests_jsonl_path, requests.to_dict(orient="records"))

    manifest = {
        "experiment": "branching_intervention",
        "input_root": config.input_root,
        "selection_path": config.selection_path,
        "output_dir": config.output_dir,
        "model": config.model,
        "deciles": config.deciles,
        "cohorts": config.cohorts,
        "final_outcome": config.final_outcome,
        "min_current_normalized_correct_probability": (
            config.min_current_normalized_correct_probability
        ),
        "max_candidates_per_dataset": config.max_candidates_per_dataset,
        "candidate_count": len(candidates),
        "branch_request_count": len(requests),
        "branch_modes": config.branch_modes,
        "branch_seeds": config.branch_seeds,
        "branch_max_tokens": config.branch_max_tokens,
        "dry_run": config.dry_run,
        "candidate_path": candidates_path,
        "request_path": requests_path,
        "request_jsonl_path": requests_jsonl_path,
    }
    _write_json(config.output_dir / "branching_manifest.json", manifest)
    return candidates, requests, manifest


def run_branching_intervention(
    config: BranchingInterventionConfig,
) -> dict[str, object]:
    started_at = time.perf_counter()
    candidates, requests, manifest = setup_branching_intervention(config)
    setup_ready_at = time.perf_counter()
    if config.dry_run:
        manifest["timings_seconds"] = {
            "setup": round(setup_ready_at - started_at, 3),
            "total": round(setup_ready_at - started_at, 3),
        }
        _write_json(config.output_dir / "branching_manifest.json", manifest)
        return manifest
    if requests.empty:
        raise ValueError("No branch requests were selected")

    runner_config = ProbeConfig(
        model=config.model,
        num_rows=len(requests),
        seed=config.branch_seeds[0],
        trace_max_tokens=config.trace_max_tokens,
        max_model_len=config.max_model_len,
        max_num_seqs=config.max_num_seqs,
        gpu_memory_utilization=config.gpu_memory_utilization,
        output_dir=config.output_dir,
    )
    runner_config.validate()
    runner = GemmaProbeRunner(runner_config)
    model_ready_at = time.perf_counter()
    entries = _materialize_entries(requests, runner)
    materialized_at = time.perf_counter()
    rows = _initial_branch_results(entries, runner, config)
    rows = _force_close_missing_answers(rows, entries, runner)
    branches_ready_at = time.perf_counter()

    results = pd.DataFrame(rows).sort_values("request_index")
    summary = summarize_branch_results(results)
    results_path = config.output_dir / "branch_results.parquet"
    results_jsonl_path = config.output_dir / "branch_results.jsonl"
    summary_path = config.output_dir / "branch_summary.parquet"
    results.to_parquet(results_path, index=False)
    summary.to_parquet(summary_path, index=False)
    _write_jsonl(results_jsonl_path, results.to_dict(orient="records"))

    finished_at = time.perf_counter()
    manifest.update(
        {
            "dry_run": False,
            "result_path": results_path,
            "result_jsonl_path": results_jsonl_path,
            "summary_path": summary_path,
            "answered_request_count": int(results["branch_answer"].notna().sum()),
            "branch_accuracy": (
                float(results["branch_correct"].mean())
                if not results.empty
                else float("nan")
            ),
            "forced_close_count": int(results["branch_forced_close"].sum()),
            "timings_seconds": {
                "setup": round(setup_ready_at - started_at, 3),
                "model_initialization": round(
                    model_ready_at - setup_ready_at,
                    3,
                ),
                "prompt_materialization": round(
                    materialized_at - model_ready_at,
                    3,
                ),
                "branch_generation": round(
                    branches_ready_at - materialized_at,
                    3,
                ),
                "output_processing": round(
                    finished_at - branches_ready_at,
                    3,
                ),
                "total": round(finished_at - started_at, 3),
            },
        }
    )
    _write_json(config.output_dir / "branching_manifest.json", manifest)
    return manifest
