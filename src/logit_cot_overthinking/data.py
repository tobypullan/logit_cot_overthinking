from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from string import ascii_uppercase
from typing import Sequence


@dataclass(frozen=True)
class MultipleChoiceQuestion:
    position: int
    question_id: str
    question: str
    options: tuple[str, ...]
    answer: str
    category: str
    source: str

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(ascii_uppercase[: len(self.options)])

    @property
    def prompt(self) -> str:
        return format_question(self.question, self.options)


def format_question(question: str, options: Sequence[str]) -> str:
    if not options:
        raise ValueError("A multiple-choice question must have at least one option")
    if len(options) > len(ascii_uppercase):
        raise ValueError("Only up to 26 answer choices are supported")

    choices = "\n".join(
        f"{ascii_uppercase[index]}. {option}" for index, option in enumerate(options)
    )
    return f"{question.strip()}\n\n{choices}"


def select_balanced_category_indices(
    categories: Sequence[str],
    num_rows: int,
    seed: int,
) -> list[int]:
    if num_rows < 1:
        raise ValueError("num_rows must be at least 1")
    if num_rows > len(categories):
        raise ValueError(
            f"Requested {num_rows} rows from a split with {len(categories)} rows"
        )

    indices_by_category: dict[str, list[int]] = defaultdict(list)
    for index, category in enumerate(categories):
        indices_by_category[str(category)].append(index)
    if not indices_by_category:
        raise ValueError("Cannot balance an empty category sequence")

    ordered_categories = sorted(indices_by_category)
    base_quota, remainder = divmod(num_rows, len(ordered_categories))
    quotas = {
        category: base_quota + (offset < remainder)
        for offset, category in enumerate(ordered_categories)
    }
    undersized = {
        category: (quotas[category], len(indices))
        for category, indices in indices_by_category.items()
        if len(indices) < quotas[category]
    }
    if undersized:
        raise ValueError(
            "Not enough rows for balanced category selection: "
            f"{undersized}"
        )

    rng = random.Random(seed)
    selected: list[int] = []
    for category in ordered_categories:
        selected.extend(
            rng.sample(indices_by_category[category], quotas[category])
        )
    return sorted(selected)


def load_mmlu_pro_questions(
    dataset_name: str,
    split: str,
    start_row: int,
    num_rows: int,
    selection: str = "contiguous",
    seed: int = 0,
) -> list[MultipleChoiceQuestion]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)
    if selection == "contiguous":
        end_row = min(start_row + num_rows, len(dataset))
        if start_row >= len(dataset):
            raise IndexError(
                f"start_row {start_row} is outside split {split!r} "
                f"with {len(dataset)} rows"
            )
        selected_indices = list(range(start_row, end_row))
    elif selection == "balanced-categories":
        if start_row != 0:
            raise ValueError(
                "start_row must be 0 when using balanced-categories selection"
            )
        selected_indices = select_balanced_category_indices(
            dataset["category"],
            num_rows,
            seed,
        )
    else:
        raise ValueError(f"Unsupported selection policy: {selection!r}")

    selected = dataset.select(selected_indices)
    questions: list[MultipleChoiceQuestion] = []
    for position, row in zip(selected_indices, selected):
        options = tuple(str(option) for option in row["options"])
        labels = tuple(ascii_uppercase[: len(options)])
        answer = str(row["answer"]).strip().upper()
        if answer not in labels:
            raise ValueError(
                f"Question {row['question_id']} has answer {answer!r} outside {labels}"
            )
        questions.append(
            MultipleChoiceQuestion(
                position=position,
                question_id=str(row["question_id"]),
                question=str(row["question"]),
                options=options,
                answer=answer,
                category=str(row.get("category", "")),
                source=str(row.get("src", "")),
            )
        )
    return questions
