from __future__ import annotations

import argparse
import json
from pathlib import Path

from .candidate_reruns_cli import _parse_seeds
from .matched_controls import run_matched_controls


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run matched loss, final-correct, and stable-wrong controls "
            "across sampling seeds."
        )
    )
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--per-cohort", type=int, default=25)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/matched_controls_gemma4_12b"),
    )
    parser.add_argument("--model", default="google/gemma-4-12B-it")
    parser.add_argument("--trace-max-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=20480)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_matched_controls(
        output_root=args.output_root,
        seeds=_parse_seeds(args.seeds),
        per_cohort=args.per_cohort,
        model=args.model,
        trace_max_tokens=args.trace_max_tokens,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
