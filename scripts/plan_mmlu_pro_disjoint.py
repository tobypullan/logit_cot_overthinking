from __future__ import annotations

import argparse
import json
from collections import Counter

from datasets import load_dataset

from logit_cot_overthinking.data import (
    select_balanced_category_indices,
    select_balanced_category_indices_excluding,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan a category-balanced MMLU-Pro sample disjoint from an "
            "earlier deterministic balanced sample."
        )
    )
    parser.add_argument("--exclude-num-rows", type=int, default=1000)
    parser.add_argument("--exclude-selection-seed", type=int, default=0)
    parser.add_argument("--num-rows", type=int, default=2000)
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument(
        "--format",
        choices=("csv", "summary"),
        default="summary",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    categories = [str(category) for category in dataset["category"]]
    excluded = select_balanced_category_indices(
        categories,
        num_rows=args.exclude_num_rows,
        seed=args.exclude_selection_seed,
    )
    selected = select_balanced_category_indices_excluding(
        categories,
        num_rows=args.num_rows,
        seed=args.selection_seed,
        excluded_indices=excluded,
    )

    if args.format == "csv":
        print(",".join(str(index) for index in selected))
        return

    counts = Counter(categories[index] for index in selected)
    summary = {
        "dataset": "TIGER-Lab/MMLU-Pro",
        "split": "test",
        "dataset_rows": len(dataset),
        "excluded_rows": len(excluded),
        "selected_rows": len(selected),
        "overlap": len(set(excluded) & set(selected)),
        "exclude_selection": {
            "num_rows": args.exclude_num_rows,
            "seed": args.exclude_selection_seed,
        },
        "new_selection": {
            "num_rows": args.num_rows,
            "seed": args.selection_seed,
        },
        "category_counts": dict(sorted(counts.items())),
        "first_ten_indices": selected[:10],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
