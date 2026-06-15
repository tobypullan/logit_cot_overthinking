from __future__ import annotations

import argparse
from pathlib import Path

from .visualization import create_visualizations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create visualizations from trajectory-probe artifacts."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/smoke"),
        help="Directory containing trajectory.parquet, traces.jsonl, and summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figure directory (default: <input-dir>/figures).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir or args.input_dir / "figures"
    paths = create_visualizations(args.input_dir, output_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

