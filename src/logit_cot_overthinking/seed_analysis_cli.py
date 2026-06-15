from __future__ import annotations

import argparse
import json
from pathlib import Path

from .seed_analysis import analyze_seed_reruns


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze candidate recurrence across the ten extended "
            "sampling-seed reruns."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(
            "outputs/candidate_reruns_gemma4_12b_extended"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = analyze_seed_reruns(
        input_root=args.input_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
