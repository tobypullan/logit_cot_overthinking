from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ProbeConfig
from .pipeline import run_probe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe Gemma reasoning trajectories using next-token logits."
    )
    parser.add_argument("--model", default="google/gemma-4-12B-it")
    parser.add_argument("--dataset", default="TIGER-Lab/MMLU-Pro")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--selection",
        choices=("contiguous", "balanced-categories"),
        default="contiguous",
        help="Row selection policy. Balanced selection samples evenly across categories.",
    )
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--num-rows", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace-max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=64,
        help="Maximum sequences scheduled concurrently by vLLM.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/smoke"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ProbeConfig(
        model=args.model,
        dataset=args.dataset,
        split=args.split,
        selection=args.selection,
        start_row=args.start_row,
        num_rows=args.num_rows,
        seed=args.seed,
        trace_max_tokens=args.trace_max_tokens,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        output_dir=args.output_dir,
    )
    summary = run_probe(config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
