from __future__ import annotations

import argparse
import json
from pathlib import Path

from .confidence_recurrence import (
    DEFAULT_CORRECT_THRESHOLDS,
    DEFAULT_FINAL_THRESHOLDS,
    analyze_confidence_recurrence,
)


def _thresholds(value: str) -> tuple[float, ...]:
    return tuple(
        float(part.strip())
        for part in value.split(",")
        if part.strip()
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether high-confidence intermediate correct probes "
            "recur as final wrong answers in matched-control reruns."
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
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--correct-thresholds",
        type=_thresholds,
        default=DEFAULT_CORRECT_THRESHOLDS,
        help="Comma-separated pre-final normalized correct thresholds.",
    )
    parser.add_argument(
        "--final-thresholds",
        type=_thresholds,
        default=DEFAULT_FINAL_THRESHOLDS,
        help="Comma-separated final normalized prediction thresholds.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=5000,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild matched-analysis tables instead of using cached ones.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = analyze_confidence_recurrence(
        input_root=args.input_root,
        selection_path=args.selection,
        output_dir=args.output_dir,
        correct_thresholds=args.correct_thresholds,
        final_thresholds=args.final_thresholds,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
        prefer_cached=not args.rebuild,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
