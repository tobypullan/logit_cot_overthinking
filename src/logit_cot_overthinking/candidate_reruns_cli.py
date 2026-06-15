from __future__ import annotations

import argparse
import json
from pathlib import Path

from .candidate_reruns import run_candidate_reruns


def _parse_seeds(value: str) -> list[int]:
    seeds: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(part))
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun MMLU-Pro and GPQA reversal candidates across seeds "
            "using one model initialization."
        )
    )
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/candidate_reruns_gemma4_12b"),
    )
    parser.add_argument("--model", default="google/gemma-4-12B-it")
    parser.add_argument("--trace-max-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=20480)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = run_candidate_reruns(
        output_root=args.output_root,
        seeds=_parse_seeds(args.seeds),
        model=args.model,
        trace_max_tokens=args.trace_max_tokens,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
