from __future__ import annotations

import argparse
import json
from pathlib import Path

from .matched_analysis import analyze_matched_controls


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze matched ten-seed controls and evaluate trajectory "
            "features that predict a later loss."
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = analyze_matched_controls(
        input_root=args.input_root,
        selection_path=args.selection,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
