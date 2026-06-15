from __future__ import annotations

import argparse
from pathlib import Path

from .lost_analysis import analyze_lost_cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze cases that become correct but finish wrong."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing trajectory.parquet and traces.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <input-dir>/lost_analysis).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum pre-final raw probability for a robust loss.",
    )
    parser.add_argument(
        "--final-choice-mass-threshold",
        type=float,
        default=0.5,
        help="Minimum final valid-letter probability mass for a robust loss.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir or args.input_dir / "lost_analysis"
    paths = analyze_lost_cases(
        args.input_dir,
        output_dir,
        confidence_threshold=args.confidence_threshold,
        final_choice_mass_threshold=args.final_choice_mass_threshold,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
