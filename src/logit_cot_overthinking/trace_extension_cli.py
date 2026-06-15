from __future__ import annotations

import argparse
import json

from .trace_extension import extend_capped_runs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Continue capped full-run and candidate-rerun traces from "
            "their stored prefixes, then re-probe the changed trajectories."
        )
    )
    parser.add_argument("--model", default="google/gemma-4-12B-it")
    parser.add_argument(
        "--extension-max-tokens",
        type=int,
        default=16384,
    )
    parser.add_argument(
        "--max-extension-rounds",
        type=int,
        default=1,
    )
    parser.add_argument("--max-model-len", type=int, default=49152)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = extend_capped_runs(
        model=args.model,
        extension_max_tokens=args.extension_max_tokens,
        max_extension_rounds=args.max_extension_rounds,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
