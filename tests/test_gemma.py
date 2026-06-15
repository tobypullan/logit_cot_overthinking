import math

import pytest

from logit_cot_overthinking.gemma import (
    DECILES,
    THOUGHT_END,
    THOUGHT_START,
    build_decile_prefixes,
    extract_answer_letter,
    parse_gemma_response,
    probabilities_from_logprobs,
    validate_answer_tokens,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(token_id) for token_id in token_ids)


class FakeLogprob:
    def __init__(self, logprob):
        self.logprob = logprob


def test_parse_complete_gemma_response() -> None:
    parsed = parse_gemma_response(
        f"{THOUGHT_START}reasoning here{THOUGHT_END}C<turn|>"
    )
    assert parsed.reasoning == "reasoning here"
    assert parsed.answer_text == "C"


def test_parse_truncated_gemma_response() -> None:
    parsed = parse_gemma_response(f"{THOUGHT_START}unfinished")
    assert parsed.reasoning == "unfinished"
    assert parsed.answer_text == ""


def test_decile_prefixes_use_ceil_and_full_original_text() -> None:
    tokenizer = FakeTokenizer()
    prefixes = build_decile_prefixes("abcdefghijk", tokenizer)
    assert tuple(prefixes) == DECILES
    assert prefixes[0] == ("", 0)
    assert prefixes[10] == ("ab", 2)
    assert prefixes[50] == ("abcdef", 6)
    assert prefixes[100] == ("abcdefghijk", 11)


def test_answer_token_validation_and_extraction() -> None:
    tokenizer = FakeTokenizer()
    assert validate_answer_tokens(tokenizer, ("A", "B", "C")) == {
        "A": ord("A"),
        "B": ord("B"),
        "C": ord("C"),
    }
    assert extract_answer_letter("C", ("A", "B", "C")) == "C"
    assert extract_answer_letter("Answer: B", ("A", "B", "C")) == "B"
    assert extract_answer_letter("D", ("A", "B", "C")) is None


def test_probability_calculation_preserves_raw_choice_mass() -> None:
    token_map = {"A": ord("A"), "B": ord("B")}
    values = {
        ord("A"): FakeLogprob(math.log(0.25)),
        ord("B"): FakeLogprob(math.log(0.50)),
    }
    logprobs, probabilities, non_choice = probabilities_from_logprobs(
        values, token_map
    )
    assert logprobs["A"] == pytest.approx(math.log(0.25))
    assert probabilities == pytest.approx({"A": 0.25, "B": 0.50})
    assert non_choice == pytest.approx(0.25)

