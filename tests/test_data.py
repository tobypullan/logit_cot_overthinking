import pytest

from logit_cot_overthinking.data import (
    format_question,
    parse_gpqa_diamond_question,
    select_balanced_category_indices,
)


def test_format_question_preserves_variable_choice_count() -> None:
    prompt = format_question("Pick one.", ["first", "second", "third"])
    assert prompt == "Pick one.\n\nA. first\nB. second\nC. third"
    assert "D." not in prompt


def test_format_question_rejects_empty_options() -> None:
    with pytest.raises(ValueError, match="at least one option"):
        format_question("Impossible.", [])


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
