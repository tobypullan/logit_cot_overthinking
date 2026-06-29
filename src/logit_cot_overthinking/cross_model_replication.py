from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .branching_intervention import BRANCH_MODES


@dataclass(frozen=True)
class CrossModelReplicationConfig:
    models: tuple[str, ...]
    output_root: Path = Path("outputs/cross_model_replication")
    seeds: tuple[int, ...] = tuple(range(10))
    per_cohort: int = 25
    adapter: str = "auto"
    trace_max_tokens: int = 16384
    matched_max_model_len: int = 20480
    matched_max_num_seqs: int = 32
    extension_max_tokens: int = 16384
    extension_max_model_len: int = 49152
    extension_max_num_seqs: int = 16
    gpu_memory_utilization: float = 0.90
    include_branching_setup: bool = True
    branch_min_confidence: float = 0.90
    branch_max_candidates_per_dataset: int = 25
    branch_modes: tuple[str, ...] = BRANCH_MODES
    branch_seeds: tuple[int, ...] = (0, 1, 2, 3)

    def validate(self) -> None:
        if not self.models:
            raise ValueError("At least one model is required")
        if any(not model.strip() for model in self.models):
            raise ValueError("Model names must not be empty")
        if self.adapter not in {"auto", "gemma"}:
            raise ValueError("adapter must be 'auto' or 'gemma'")
        if not self.seeds:
            raise ValueError("At least one seed is required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds must be unique")
        if self.per_cohort < 1:
            raise ValueError("per_cohort must be at least 1")
        if self.matched_max_model_len <= self.trace_max_tokens:
            raise ValueError(
                "matched_max_model_len must be greater than trace_max_tokens"
            )
        if self.extension_max_tokens < 1:
            raise ValueError("extension_max_tokens must be at least 1")
        if self.extension_max_num_seqs < 1 or self.matched_max_num_seqs < 1:
            raise ValueError("max_num_seqs values must be at least 1")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        unknown = sorted(set(self.branch_modes) - set(BRANCH_MODES))
        if unknown:
            raise ValueError(f"Unsupported branch modes: {unknown}")


def slugify_model_id(model: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_").lower()
    return slug or "model"


def _seed_spec(seeds: Sequence[int]) -> str:
    ordered = list(seeds)
    if ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"{ordered[0]}-{ordered[-1]}"
    return ",".join(str(seed) for seed in ordered)


def _csv(values: Sequence[object]) -> str:
    return ",".join(str(value) for value in values)


def _command(parts: Sequence[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def infer_adapter_status(model: str, adapter: str = "auto") -> dict[str, object]:
    if adapter == "gemma":
        return {
            "adapter": "gemma",
            "runnable": True,
            "note": "Forced to use the existing Gemma thought-channel adapter.",
        }
    lower = model.lower()
    if "gemma" in lower:
        return {
            "adapter": "gemma",
            "runnable": True,
            "note": "Detected as Gemma-compatible from the model name.",
        }
    return {
        "adapter": "adapter_required",
        "runnable": False,
        "note": (
            "The current runner hard-codes Gemma chat templates and thought "
            "channel markers; add a model adapter before running this model."
        ),
    }


def build_model_replication_plan(
    model: str,
    config: CrossModelReplicationConfig,
) -> dict[str, object]:
    slug = slugify_model_id(model)
    model_root = config.output_root / slug
    matched_root = model_root / "matched_controls"
    extended_root = model_root / "matched_controls_extended"
    branch_root = model_root / "branching_intervention"
    selection_path = matched_root / "cohort_selection.parquet"
    seed_spec = _seed_spec(config.seeds)
    status = infer_adapter_status(model, config.adapter)

    commands = [
        {
            "stage": "matched_controls",
            "command": _command(
                [
                    "trajectory-run-matched-controls",
                    "--model",
                    model,
                    "--seeds",
                    seed_spec,
                    "--per-cohort",
                    config.per_cohort,
                    "--output-root",
                    matched_root,
                    "--trace-max-tokens",
                    config.trace_max_tokens,
                    "--max-model-len",
                    config.matched_max_model_len,
                    "--max-num-seqs",
                    config.matched_max_num_seqs,
                    "--gpu-memory-utilization",
                    config.gpu_memory_utilization,
                ]
            ),
        },
        {
            "stage": "extend_matched_controls",
            "command": _command(
                [
                    "trajectory-extend-matched-controls",
                    "--model",
                    model,
                    "--seeds",
                    seed_spec,
                    "--input-root",
                    matched_root,
                    "--output-root",
                    extended_root,
                    "--extension-max-tokens",
                    config.extension_max_tokens,
                    "--max-model-len",
                    config.extension_max_model_len,
                    "--max-num-seqs",
                    config.extension_max_num_seqs,
                    "--gpu-memory-utilization",
                    config.gpu_memory_utilization,
                ]
            ),
        },
        {
            "stage": "matched_analysis",
            "command": _command(
                [
                    "trajectory-analyze-matched-controls",
                    "--input-root",
                    extended_root,
                    "--selection",
                    selection_path,
                    "--output-dir",
                    extended_root / "analysis",
                ]
            ),
        },
        {
            "stage": "confidence_recurrence",
            "command": _command(
                [
                    "trajectory-analyze-confidence-recurrence",
                    "--input-root",
                    extended_root,
                    "--selection",
                    selection_path,
                    "--output-dir",
                    extended_root / "analysis" / "confidence_recurrence",
                ]
            ),
        },
        {
            "stage": "early_commitment",
            "command": _command(
                [
                    "trajectory-analyze-early-commitment",
                    "--input-root",
                    extended_root,
                    "--selection",
                    selection_path,
                    "--output-dir",
                    extended_root / "analysis" / "early_commitment",
                ]
            ),
        },
    ]
    if config.include_branching_setup:
        commands.append(
            {
                "stage": "branching_setup",
                "command": _command(
                    [
                        "trajectory-run-branching-intervention",
                        "--model",
                        model,
                        "--input-root",
                        extended_root,
                        "--selection",
                        selection_path,
                        "--output-dir",
                        branch_root,
                        "--min-current-normalized-correct-probability",
                        config.branch_min_confidence,
                        "--max-candidates-per-dataset",
                        config.branch_max_candidates_per_dataset,
                        "--branch-modes",
                        _csv(config.branch_modes),
                        "--branch-seeds",
                        _seed_spec(config.branch_seeds),
                    ]
                ),
            }
        )

    return {
        "model": model,
        "slug": slug,
        "model_root": str(model_root),
        "matched_root": str(matched_root),
        "extended_root": str(extended_root),
        "selection_path": str(selection_path),
        "adapter": status["adapter"],
        "runnable": status["runnable"],
        "adapter_note": status["note"],
        "commands": commands if status["runnable"] else [],
        "blocked_commands": [] if status["runnable"] else commands,
    }


def build_cross_model_replication_plan(
    config: CrossModelReplicationConfig,
) -> dict[str, object]:
    config.validate()
    models = [
        build_model_replication_plan(model.strip(), config)
        for model in config.models
    ]
    return {
        "experiment": "cross_model_replication",
        "output_root": str(config.output_root),
        "models": models,
        "seeds": list(config.seeds),
        "per_cohort": config.per_cohort,
        "replication_mode": (
            "same matched question positions; baseline cohorts come from "
            "the source matched-selection run"
        ),
        "adapter": config.adapter,
        "include_branching_setup": config.include_branching_setup,
        "runnable_model_count": sum(bool(model["runnable"]) for model in models),
        "model_count": len(models),
    }


def render_commands_markdown(plan: dict[str, object]) -> str:
    lines = [
        "# Cross-Model Replication Commands",
        "",
        (
            "This plan repeats the matched-control, extension, analysis, "
            "confidence-recurrence, early-commitment, and optional branching "
            "setup pipeline for each runnable model."
        ),
        "",
        f"Replication mode: {plan['replication_mode']}",
        "",
    ]
    for model_plan in plan["models"]:
        lines.extend(
            [
                f"## {model_plan['model']}",
                "",
                f"Adapter: {model_plan['adapter']}",
                f"Runnable now: {model_plan['runnable']}",
                f"Note: {model_plan['adapter_note']}",
                "",
            ]
        )
        commands = model_plan["commands"] or model_plan["blocked_commands"]
        for item in commands:
            prefix = "" if model_plan["runnable"] else "# "
            lines.append(f"### {item['stage']}")
            lines.append("")
            lines.append("```bash")
            lines.append(f"{prefix}{item['command']}")
            lines.append("```")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_cross_model_replication_plan(
    config: CrossModelReplicationConfig,
) -> dict[str, object]:
    plan = build_cross_model_replication_plan(config)
    config.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_root / "manifest.json"
    commands_path = config.output_root / "commands.md"
    with manifest_path.open("w", encoding="utf-8") as output:
        json.dump(plan, output, indent=2, sort_keys=True)
        output.write("\n")
    commands_path.write_text(
        render_commands_markdown(plan),
        encoding="utf-8",
    )
    return {
        **plan,
        "manifest_path": str(manifest_path),
        "commands_path": str(commands_path),
    }
