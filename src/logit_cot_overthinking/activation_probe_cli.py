from __future__ import annotations

import argparse
import json
from pathlib import Path

from .activation_probe import (
    DEFAULT_CONFIDENCE_THRESHOLDS,
    DEFAULT_DECILES,
    DEFAULT_PROBE_THRESHOLDS,
    PROBE_TARGETS,
    ActivationExtractionConfig,
    ProbeTrainingConfig,
    build_activation_probe_examples,
    extract_probe_activations,
    train_activation_probes,
)
from .branching_intervention_cli import _parse_csv, _parse_deciles
from .candidate_reruns_cli import _parse_seeds


def _parse_layers(value: str) -> tuple[int, ...] | None:
    value = value.strip().lower()
    if value == "auto":
        return None
    if value == "all":
        raise ValueError("Pass the explicit layer range, for example 0-48.")
    layers: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            layers.extend(range(start, end + step, step))
        else:
            layers.append(int(part))
    return tuple(dict.fromkeys(layers))


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(
        float(part.strip())
        for part in value.split(",")
        if part.strip()
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build hidden-state activation probes for future loss, future "
            "change-to-wrong, and future answer flip labels."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("examples", "extract", "train", "all"),
        default="all",
        help=(
            "examples writes labels only; extract writes labels and hidden "
            "activations; train trains from existing activations; all does "
            "extract then train."
        ),
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/matched_controls_gemma4_12b_extended"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/activation_probe_gemma4_12b"),
    )
    parser.add_argument("--model", default="google/gemma-4-12B-it")
    parser.add_argument(
        "--deciles",
        default=",".join(str(decile) for decile in DEFAULT_DECILES),
    )
    parser.add_argument(
        "--datasets",
        default="",
        help="Optional comma-separated dataset folder names.",
    )
    parser.add_argument(
        "--seeds",
        default="",
        help="Optional seed list/range such as 0-9.",
    )
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--layers",
        default="auto",
        help=(
            "auto chooses a spaced grid including embeddings and final layer; "
            "or pass comma-separated layer indices / integer ranges."
        ),
    )
    parser.add_argument("--activation-batch-size", type=int, default=1)
    parser.add_argument(
        "--activation-dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--model-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--no-reuse-examples",
        action="store_true",
        help="Rebuild activation_probe_examples.parquet even if it exists.",
    )
    parser.add_argument(
        "--no-reuse-activations",
        action="store_true",
        help="Re-extract activations even if activations.npy exists.",
    )
    parser.add_argument(
        "--targets",
        default=",".join(PROBE_TARGETS),
        help=f"Comma-separated targets: {', '.join(PROBE_TARGETS)}",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "torch", "numpy"),
        default="auto",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument(
        "--confidence-thresholds",
        default=",".join(str(value) for value in DEFAULT_CONFIDENCE_THRESHOLDS),
    )
    parser.add_argument(
        "--probe-thresholds",
        default=",".join(str(value) for value in DEFAULT_PROBE_THRESHOLDS),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    deciles = _parse_deciles(args.deciles)
    datasets = _parse_csv(args.datasets)
    seeds = tuple(_parse_seeds(args.seeds)) if args.seeds.strip() else ()
    layers = _parse_layers(args.layers)

    if args.stage == "examples":
        examples = build_activation_probe_examples(
            input_root=args.input_root,
            output_dir=args.output_dir,
            deciles=deciles,
            datasets=datasets,
            seeds=seeds,
            max_examples=args.max_examples,
            random_seed=args.random_seed,
        )
        print(
            json.dumps(
                {
                    "examples_path": str(
                        args.output_dir / "activation_probe_examples.parquet"
                    ),
                    "example_count": len(examples),
                    "target_positive_counts": {
                        target: int(examples[target].sum())
                        for target in PROBE_TARGETS
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    extraction_config = ActivationExtractionConfig(
        input_root=args.input_root,
        output_dir=args.output_dir,
        model=args.model,
        deciles=deciles,
        datasets=datasets,
        seeds=seeds,
        max_examples=args.max_examples,
        random_seed=args.random_seed,
        layers=layers,
        batch_size=args.activation_batch_size,
        activation_dtype=args.activation_dtype,
        model_dtype=args.model_dtype,
        device_map=args.device_map,
        reuse_examples=not args.no_reuse_examples,
        reuse_activations=not args.no_reuse_activations,
    )
    training_config = ProbeTrainingConfig(
        output_dir=args.output_dir,
        layers=layers,
        targets=_parse_csv(args.targets),
        backend=args.backend,
        folds=args.folds,
        epochs=args.epochs,
        batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        ridge=args.ridge,
        seed=args.random_seed,
        confidence_thresholds=_parse_floats(args.confidence_thresholds),
        probe_thresholds=_parse_floats(args.probe_thresholds),
    )

    if args.stage == "extract":
        result = extract_probe_activations(extraction_config)
    elif args.stage == "train":
        result = train_activation_probes(training_config)
    else:
        extraction = extract_probe_activations(extraction_config)
        result = train_activation_probes(training_config)
        result = {"extraction": extraction, "training": result}
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
