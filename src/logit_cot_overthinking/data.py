from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass
from string import ascii_uppercase
from typing import Sequence


GPQA_OPTION_PATTERN = re.compile(
    r"(?m)^([A-Z])[.)]\s+(.+?)(?=\n[A-Z][.)]\s+|\Z)",
    re.DOTALL,
)


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


def parse_gpqa_diamond_question(
    formatted_question: str,
) -> tuple[str, tuple[str, ...]]:
    text = formatted_question.strip()
    matches = list(GPQA_OPTION_PATTERN.finditer(text))
    for start_index in range(len(matches)):
        option_matches = matches[start_index:]
        labels = [match.group(1) for match in option_matches]
        expected = list(ascii_uppercase[: len(option_matches)])
        if (
            labels == expected
            and len(option_matches) >= 2
            and option_matches[-1].end() == len(text)
        ):
            question = text[: option_matches[0].start()].strip()
            options = tuple(
                match.group(2).strip() for match in option_matches
            )
            if not question:
                raise ValueError("GPQA question stem is empty")
            return question, options
    raise ValueError(
        "Could not find a trailing contiguous answer-choice block in GPQA row"
    )


def _resolve_dataset_format(
    dataset_format: str,
    column_names: Sequence[str],
) -> str:
    if dataset_format != "auto":
        return dataset_format
    columns = set(column_names)
    if {"question_id", "question", "options", "answer"}.issubset(columns):
        return "mmlu-pro"
    if columns == {"question", "answer"}:
        return "gpqa-diamond"
    raise ValueError(
        "Could not auto-detect dataset format from columns "
        f"{sorted(columns)}; pass --dataset-format explicitly"
    )


def load_questions(
    dataset_name: str,
    dataset_format: str,
    split: str,
    start_row: int,
    num_rows: int,
    selection: str = "contiguous",
    seed: int = 0,
    row_indices: Sequence[int] = (),
) -> list[MultipleChoiceQuestion]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)
    resolved_format = _resolve_dataset_format(
        dataset_format,
        dataset.column_names,
    )
    if selection == "contiguous":
        end_row = min(start_row + num_rows, len(dataset))
        if start_row >= len(dataset):
            raise IndexError(
                f"start_row {start_row} is outside split {split!r} "
                f"with {len(dataset)} rows"
            )
        selected_indices = list(range(start_row, end_row))
    elif selection == "indices":
        selected_indices = [int(index) for index in row_indices]
        if len(selected_indices) != num_rows:
            raise ValueError(
                "num_rows must equal the number of explicit row indices"
            )
        if len(set(selected_indices)) != len(selected_indices):
            raise ValueError("Explicit row indices must not contain duplicates")
        invalid = [
            index
            for index in selected_indices
            if index < 0 or index >= len(dataset)
        ]
        if invalid:
            raise IndexError(
                f"Row indices outside split {split!r} with "
                f"{len(dataset)} rows: {invalid}"
            )
    elif selection == "balanced-categories":
        if resolved_format != "mmlu-pro":
            raise ValueError(
                "balanced-categories selection is only supported for MMLU-Pro"
            )
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
        if resolved_format == "mmlu-pro":
            question_id = str(row["question_id"])
            question = str(row["question"])
            options = tuple(str(option) for option in row["options"])
            category = str(row.get("category", ""))
            source = str(row.get("src", ""))
        else:
            question, options = parse_gpqa_diamond_question(
                str(row["question"])
            )
            question_id = f"gpqa-diamond-{position:03d}"
            category = "gpqa-diamond"
            source = dataset_name

        labels = tuple(ascii_uppercase[: len(options)])
        answer = str(row["answer"]).strip().upper()
        if answer not in labels:
            raise ValueError(
                f"Question {question_id} has answer {answer!r} outside {labels}"
            )
        questions.append(
            MultipleChoiceQuestion(
                position=position,
                question_id=question_id,
                question=question,
                options=options,
                answer=answer,
                category=category,
                source=source,
            )
        )
    return questions


def load_mmlu_pro_questions(
    dataset_name: str,
    split: str,
    start_row: int,
    num_rows: int,
    selection: str = "contiguous",
    seed: int = 0,
) -> list[MultipleChoiceQuestion]:
    return load_questions(
        dataset_name=dataset_name,
        dataset_format="mmlu-pro",
        split=split,
        start_row=start_row,
        num_rows=num_rows,
        selection=selection,
        seed=seed,
        row_indices=(),
    )
