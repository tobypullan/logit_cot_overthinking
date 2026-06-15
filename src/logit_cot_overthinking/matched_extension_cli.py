from __future__ import annotations

import argparse
import json
from pathlib import Path

from .candidate_reruns_cli import _parse_seeds
from .matched_controls import matched_control_extension_specs
from .trace_extension import extend_capped_runs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extend capped traces from the matched-control experiment."
        )
    )
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/matched_controls_gemma4_12b"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "outputs/matched_controls_gemma4_12b_extended"
        ),
    )
    parser.add_argument("--model", default="google/gemma-4-12B-it")
    parser.add_argument(
        "--extension-max-tokens",
        type=int,
        default=16384,
    )
    parser.add_argument("--max-model-len", type=int, default=49152)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seeds = _parse_seeds(args.seeds)
    result = extend_capped_runs(
        specs=matched_control_extension_specs(
            input_root=args.input_root,
            output_root=args.output_root,
            seeds=seeds,
        ),
        model=args.model,
        extension_max_tokens=args.extension_max_tokens,
        max_extension_rounds=1,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        manifest_path=args.output_root / "extension_manifest.json",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
