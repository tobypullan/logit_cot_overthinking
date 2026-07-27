from __future__ import annotations

import argparse
import json
from pathlib import Path

from .activation_probe import (
    TwoHeadInterventionConfig,
    apply_two_head_intervention,
    train_two_head_intervention,
)
from .activation_probe_cli import _parse_floats, _parse_layers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train or apply a two-head correctness and conditional-loss "
            "intervention probe."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("train", "apply"),
        default="train",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(
            "outputs/activation_probe_mmlu_pro_n2000_intervention"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/two_head_intervention_mmlu_pro_n2000"),
    )
    parser.add_argument("--layers", default="auto")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--correctness-thresholds",
        default="0.5,0.7,0.8,0.9",
    )
    parser.add_argument(
        "--loss-thresholds",
        default="0.5,0.7,0.8,0.9",
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=Path(
            "outputs/activation_probe_mmlu_pro_n1000_intervention_eval"
        ),
    )
    parser.add_argument("--layer", type=int, default=36)
    parser.add_argument("--correctness-threshold", type=float, default=0.9)
    parser.add_argument("--loss-threshold", type=float, default=0.9)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(
            "outputs/activation_probe_mmlu_pro_n1000_intervention_eval/"
            "external_two_head_summary.json"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "apply":
        result = apply_two_head_intervention(
            model_dir=args.output_dir,
            evaluation_dir=args.evaluation_dir,
            output_path=args.output_path,
            layer=args.layer,
            current_correct_threshold=args.correctness_threshold,
            conditional_loss_threshold=args.loss_threshold,
        )
    else:
        result = train_two_head_intervention(
            TwoHeadInterventionConfig(
                source_dir=args.source_dir,
                output_dir=args.output_dir,
                layers=_parse_layers(args.layers),
                folds=args.folds,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                seed=args.seed,
                correctness_thresholds=_parse_floats(
                    args.correctness_thresholds
                ),
                loss_thresholds=_parse_floats(args.loss_thresholds),
            )
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
