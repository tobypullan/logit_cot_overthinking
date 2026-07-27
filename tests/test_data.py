import sys
from types import SimpleNamespace

import pytest

from logit_cot_overthinking.data import (
    format_question,
    load_questions,
    parse_gpqa_diamond_question,
    parse_swe_qa_options,
    select_balanced_category_indices,
    select_balanced_category_indices_excluding,
)


def test_format_question_preserves_variable_choice_count() -> None:
    prompt = format_question("Pick one.", ["first", "second", "third"])
    assert prompt == "Pick one.\n\nA. first\nB. second\nC. third"
    assert "D." not in prompt


def test_format_question_rejects_empty_options() -> None:
    with pytest.raises(ValueError, match="at least one option"):
        format_question("Impossible.", [])


def test_format_question_includes_code_context() -> None:
    prompt = format_question(
        "What does f return?",
        ["one", "two"],
        context="def f():\n    return 1",
    )

    assert prompt.startswith("Code context:\n<code>\ndef f()")
    assert "</code>\n\nQuestion:\nWhat does f return?" in prompt
    assert prompt.endswith("A. one\nB. two")


def test_parse_gpqa_diamond_question_extracts_final_choice_block() -> None:
    question, options = parse_gpqa_diamond_question(
        "Which original option is correct?\n\n"
        "a) Alpha\n"
        "b) Beta\n"
        "c) Gamma\n"
        "d) Delta\n\n"
        "A. d\n"
        "B. a\n"
        "C. b\n"
        "D. c"
    )

    assert question.endswith("d) Delta")
    assert options == ("d", "a", "b", "c")


def test_parse_gpqa_diamond_question_rejects_missing_choices() -> None:
    with pytest.raises(ValueError, match="answer-choice block"):
        parse_gpqa_diamond_question("Question without final choices")


def test_balanced_category_selection_is_deterministic_and_ordered() -> None:
    categories = ["physics"] * 8 + ["biology"] * 8 + ["law"] * 8
    first = select_balanced_category_indices(categories, num_rows=8, seed=17)
    second = select_balanced_category_indices(categories, num_rows=8, seed=17)

    assert first == second
    assert first == sorted(first)
    selected_categories = [categories[index] for index in first]
    assert selected_categories.count("biology") == 3
    assert selected_categories.count("law") == 3
    assert selected_categories.count("physics") == 2


def test_balanced_category_selection_rejects_undersized_categories() -> None:
    with pytest.raises(ValueError, match="Not enough rows"):
        select_balanced_category_indices(
            ["large"] * 5 + ["small"],
            num_rows=6,
            seed=0,
        )


def test_balanced_selection_excluding_is_disjoint_and_balanced() -> None:
    categories = ["physics"] * 8 + ["biology"] * 8 + ["law"] * 8
    excluded = select_balanced_category_indices(
        categories,
        num_rows=6,
        seed=4,
    )

    selected = select_balanced_category_indices_excluding(
        categories,
        num_rows=9,
        seed=5,
        excluded_indices=excluded,
    )

    assert set(selected).isdisjoint(excluded)
    selected_categories = [categories[index] for index in selected]
    assert selected_categories.count("biology") == 3
    assert selected_categories.count("law") == 3
    assert selected_categories.count("physics") == 3


def test_balanced_selection_excluding_rejects_duplicate_indices() -> None:
    with pytest.raises(ValueError, match="must not contain duplicates"):
        select_balanced_category_indices_excluding(
            ["physics"] * 4,
            num_rows=1,
            seed=0,
            excluded_indices=[0, 0],
        )


def test_parse_swe_qa_options_orders_labels() -> None:
    assert parse_swe_qa_options(
        {"C": "third", "A": "first", "D": "fourth", "B": "second"}
    ) == ("first", "second", "third", "fourth")


def test_parse_swe_qa_options_rejects_noncontiguous_labels() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        parse_swe_qa_options({"A": "first", "C": "third"})


class _FakeDataset:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.column_names = list(rows[0])

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, key: str):
        return [row[key] for row in self.rows]

    def select(self, indices: list[int]) -> "_FakeDataset":
        return _FakeDataset([self.rows[index] for index in indices])


def _swe_qa_row(
    index: int,
    repo: str,
    category: str,
) -> dict[str, object]:
    return {
        "question": f"What does helper_{index} return?",
        "options": {
            "A": "None",
            "B": str(index),
            "C": "True",
            "D": "False",
        },
        "code": f"def helper_{index}():\n    return {index}",
        "correct_answer": "B",
        "repo": repo,
        "category": category,
        "chunks": [],
        "entities": [f"helper_{index}"],
    }


def test_load_swe_qa_question_preserves_context_and_metadata(
    monkeypatch,
) -> None:
    rows = [_swe_qa_row(0, "owner/repo", "interacting_entities")]
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *_args, **_kwargs: _FakeDataset(rows)),
    )

    questions = load_questions(
        dataset_name="lailaelkoussy/swe-qa",
        dataset_format="auto",
        split="oracle",
        start_row=0,
        num_rows=1,
    )

    question = questions[0]
    assert question.question_id == "swe-qa-00000"
    assert question.options == ("None", "0", "True", "False")
    assert question.answer == "B"
    assert question.context.startswith("def helper_0")
    assert question.repository == "owner/repo"
    assert question.question_type == "interacting_entities"
    assert question.category == "owner/repo::interacting_entities"
    assert "Code context:" in question.prompt


def test_balanced_swe_qa_selection_uses_repo_and_question_type(
    monkeypatch,
) -> None:
    rows = [
        _swe_qa_row(index, repo, category)
        for index, (repo, category) in enumerate(
            [
                ("repo-a", "declaration_call"),
                ("repo-a", "declaration_call"),
                ("repo-a", "interacting_entities"),
                ("repo-a", "interacting_entities"),
                ("repo-b", "declaration_call"),
                ("repo-b", "declaration_call"),
                ("repo-b", "interacting_entities"),
                ("repo-b", "interacting_entities"),
            ]
        )
    ]
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *_args, **_kwargs: _FakeDataset(rows)),
    )

    questions = load_questions(
        dataset_name="lailaelkoussy/swe-qa",
        dataset_format="swe-qa",
        split="oracle",
        start_row=0,
        num_rows=4,
        selection="balanced-categories",
        seed=3,
    )

    assert {
        (question.repository, question.question_type)
        for question in questions
    } == {
        ("repo-a", "declaration_call"),
        ("repo-a", "interacting_entities"),
        ("repo-b", "declaration_call"),
        ("repo-b", "interacting_entities"),
    }
