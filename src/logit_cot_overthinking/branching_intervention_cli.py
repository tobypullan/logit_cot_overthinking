from __future__ import annotations

import argparse
import json
from pathlib import Path

from .branching_intervention import (
    BRANCH_MODES,
    BranchingInterventionConfig,
    run_branching_intervention,
)
from .candidate_reruns_cli import _parse_seeds


def _parse_deciles(value: str) -> tuple[int, ...]:
    deciles: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)
            step = 10 if end >= start else -10
            deciles.extend(range(start, end + step, step))
        else:
            deciles.append(int(part))
    return tuple(dict.fromkeys(deciles))


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Set up or run branch interventions from high-confidence "
            "currently-correct checkpoints."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/matched_controls_gemma4_12b_extended"),
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path(
            "outputs/matched_controls_gemma4_12b/"
            "cohort_selection.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/branching_intervention_gemma4_12b"),
    )
    parser.add_argument("--model", default="google/gemma-4-12B-it")
    parser.add_argument(
        "--deciles",
        default="30-90",
        help="Comma-separated deciles or inclusive 10-point ranges.",
    )
    parser.add_argument(
        "--cohorts",
        default="loss",
        help="Comma-separated baseline cohorts to include.",
    )
    parser.add_argument(
        "--final-outcome",
        choices=("loss", "correct", "all"),
        default="loss",
    )
    parser.add_argument(
        "--min-current-normalized-correct-probability",
        type=float,
        default=0.90,
    )
    parser.add_argument("--max-candidates-per-dataset", type=int, default=25)
    parser.add_argument(
        "--branch-modes",
        default=",".join(BRANCH_MODES),
        help=f"Comma-separated branch modes: {', '.join(BRANCH_MODES)}",
    )
    parser.add_argument("--branch-seeds", default="0-3")
    parser.add_argument("--branch-max-tokens", type=int, default=512)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually generate branch continuations with vLLM.",
    )
    parser.add_argument("--trace-max-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=49152)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--continuation-temperature", type=float, default=1.0)
    parser.add_argument("--continuation-top-p", type=float, default=0.95)
    parser.add_argument("--continuation-top-k", type=int, default=64)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = BranchingInterventionConfig(
        input_root=args.input_root,
        selection_path=args.selection,
        output_dir=args.output_dir,
        model=args.model,
        deciles=_parse_deciles(args.deciles),
        cohorts=_parse_csv(args.cohorts),
        final_outcome=args.final_outcome,
        min_current_normalized_correct_probability=(
            args.min_current_normalized_correct_probability
        ),
        max_candidates_per_dataset=args.max_candidates_per_dataset,
        branch_modes=_parse_csv(args.branch_modes),
        branch_seeds=tuple(_parse_seeds(args.branch_seeds)),
        branch_max_tokens=args.branch_max_tokens,
        dry_run=not args.execute,
        trace_max_tokens=args.trace_max_tokens,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        continuation_temperature=args.continuation_temperature,
        continuation_top_p=args.continuation_top_p,
        continuation_top_k=args.continuation_top_k,
    )
    manifest = run_branching_intervention(config)
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
