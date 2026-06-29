from __future__ import annotations

import argparse
import json
from pathlib import Path

from .branching_intervention_cli import _parse_csv
from .candidate_reruns_cli import _parse_seeds
from .cross_model_replication import (
    CrossModelReplicationConfig,
    write_cross_model_replication_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write a command manifest for cross-model replication of the "
            "matched-control analyses."
        )
    )
    parser.add_argument(
        "--models",
        default="google/gemma-4-12B-it",
        help="Comma-separated model IDs to plan.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/cross_model_replication"),
    )
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--per-cohort", type=int, default=25)
    parser.add_argument(
        "--adapter",
        choices=("auto", "gemma"),
        default="auto",
        help="Use 'gemma' to force the current Gemma adapter for all models.",
    )
    parser.add_argument("--trace-max-tokens", type=int, default=16384)
    parser.add_argument("--matched-max-model-len", type=int, default=20480)
    parser.add_argument("--matched-max-num-seqs", type=int, default=32)
    parser.add_argument("--extension-max-tokens", type=int, default=16384)
    parser.add_argument("--extension-max-model-len", type=int, default=49152)
    parser.add_argument("--extension-max-num-seqs", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--skip-branching-setup",
        action="store_true",
        help="Do not include branching-intervention setup commands.",
    )
    parser.add_argument("--branch-min-confidence", type=float, default=0.90)
    parser.add_argument(
        "--branch-max-candidates-per-dataset",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--branch-modes",
        default="answer_only,normal,short_verification,preserve_unless_decisive",
    )
    parser.add_argument("--branch-seeds", default="0-3")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = CrossModelReplicationConfig(
        models=_parse_csv(args.models),
        output_root=args.output_root,
        seeds=tuple(_parse_seeds(args.seeds)),
        per_cohort=args.per_cohort,
        adapter=args.adapter,
        trace_max_tokens=args.trace_max_tokens,
        matched_max_model_len=args.matched_max_model_len,
        matched_max_num_seqs=args.matched_max_num_seqs,
        extension_max_tokens=args.extension_max_tokens,
        extension_max_model_len=args.extension_max_model_len,
        extension_max_num_seqs=args.extension_max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        include_branching_setup=not args.skip_branching_setup,
        branch_min_confidence=args.branch_min_confidence,
        branch_max_candidates_per_dataset=(
            args.branch_max_candidates_per_dataset
        ),
        branch_modes=_parse_csv(args.branch_modes),
        branch_seeds=tuple(_parse_seeds(args.branch_seeds)),
    )
    result = write_cross_model_replication_plan(config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
