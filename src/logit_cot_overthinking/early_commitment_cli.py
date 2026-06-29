from __future__ import annotations

import argparse
import json
from pathlib import Path

from .early_commitment import (
    DEFAULT_PROXY_STREAK,
    DEFAULT_PROXY_THRESHOLD,
    DEFAULT_THRESHOLDS,
    analyze_early_commitment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate oracle and practical early-commitment policies on "
            "matched-control trajectory parquet files."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(
            "outputs/matched_controls_gemma4_12b_extended"
        ),
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path(
            "outputs/matched_controls_gemma4_12b/"
            "cohort_selection.parquet"
        ),
        help=(
            "Matched-control cohort selection table used to restore "
            "metadata for extended traces."
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--threshold",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
        help=(
            "Normalized correct-probability thresholds for oracle "
            "threshold_first policies."
        ),
    )
    parser.add_argument(
        "--proxy-threshold",
        type=float,
        default=DEFAULT_PROXY_THRESHOLD,
        help=(
            "Normalized prediction-probability threshold for the "
            "non-oracle confidence-streak proxy."
        ),
    )
    parser.add_argument(
        "--proxy-streak",
        type=int,
        default=DEFAULT_PROXY_STREAK,
        help=(
            "Minimum consecutive checkpoints with the same prediction "
            "for the non-oracle confidence-streak proxy."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = analyze_early_commitment(
        input_root=args.input_root,
        selection_path=args.selection,
        output_dir=args.output_dir,
        thresholds=tuple(args.threshold),
        proxy_threshold=args.proxy_threshold,
        proxy_streak=args.proxy_streak,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
