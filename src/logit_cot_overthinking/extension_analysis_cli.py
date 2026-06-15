from __future__ import annotations

import argparse
import json
from pathlib import Path

from .extension_analysis import analyze_extensions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the original 16K endpoints with completed extended "
            "GPQA and MMLU-Pro traces."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/trace_extension_analysis_gemma4_12b"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = analyze_extensions(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
