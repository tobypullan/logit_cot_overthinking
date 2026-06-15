from __future__ import annotations

from logit_cot_overthinking.gemma import (
    THOUGHT_END,
    THOUGHT_START,
    force_close_trace,
    merge_trace_extension,
)
from logit_cot_overthinking.trace_extension import (
    _replace_by_key,
    _strip_forced_answer,
)


class CharacterTokenizer:
    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> list[int]:
        return [ord(character) for character in text]


def test_merge_trace_extension_preserves_prefix_and_parses_answer() -> None:
    prefix = f"{THOUGHT_START}Work in progress"
    trace = {
        "position": 2,
        "question_id": "q2",
        "raw_response": prefix,
        "reasoning_trace": "Work in progress",
        "trace_token_count": 16,
        "generated_token_count": 20,
        "finish_reason": "length",
        "truncated": True,
        "run_seed": 3,
    }

    merged = merge_trace_extension(
        trace=trace,
        continuation_text=f" and done{THOUGHT_END}B",
        continuation_token_count=7,
        finish_reason="stop",
        tokenizer=CharacterTokenizer(),
        valid_labels=("A", "B", "C", "D"),
    )

    assert merged["raw_response"].startswith(prefix)
    assert merged["reasoning_trace"] == "Work in progress and done"
    assert merged["generated_answer"] == "B"
    assert merged["generated_token_count"] == 27
    assert merged["original_generated_token_count"] == 20
    assert merged["extension_generated_token_count"] == 7
    assert merged["extension_rounds"] == 1
    assert merged["truncated"] is False


def test_merge_trace_extension_accumulates_multiple_rounds() -> None:
    trace = {
        "position": 1,
        "question_id": "q1",
        "raw_response": f"{THOUGHT_START}first",
        "reasoning_trace": "first",
        "trace_token_count": 5,
        "generated_token_count": 8,
        "finish_reason": "length",
        "truncated": True,
        "run_seed": 0,
    }
    first = merge_trace_extension(
        trace,
        " second",
        4,
        "length",
        CharacterTokenizer(),
        ("A", "B"),
    )
    second = merge_trace_extension(
        first,
        f" third{THOUGHT_END}A",
        6,
        "stop",
        CharacterTokenizer(),
        ("A", "B"),
    )

    assert second["extension_rounds"] == 2
    assert second["extension_generated_token_count"] == 10
    assert second["original_generated_token_count"] == 8
    assert second["generated_token_count"] == 18


def test_replace_by_key_retains_input_order() -> None:
    records = [
        {"position": 1, "question_id": "a", "value": "old"},
        {"position": 2, "question_id": "b", "value": "same"},
    ]
    replacements = {
        (1, "a"): {
            "position": 1,
            "question_id": "a",
            "value": "new",
        }
    }

    result = _replace_by_key(records, replacements)

    assert [record["question_id"] for record in result] == ["a", "b"]
    assert [record["value"] for record in result] == ["new", "same"]


def test_force_close_trace_marks_runaway_reasoning() -> None:
    trace = {
        "position": 4,
        "question_id": "q4",
        "raw_response": f"{THOUGHT_START}long reasoning",
        "reasoning_trace": "long reasoning",
        "trace_token_count": 14,
        "generated_token_count": 20,
        "finish_reason": "length",
        "truncated": True,
    }

    closed = force_close_trace(
        trace,
        "D<turn|>",
        2,
        CharacterTokenizer(),
        ("A", "B", "C", "D"),
    )

    assert closed["reasoning_trace"] == "long reasoning"
    assert closed["generated_answer"] == "D"
    assert closed["forced_completion"] is True
    assert closed["truncated"] is False
    assert closed["finish_reason"] == "forced_close"


def test_strip_forced_answer_restores_capped_prefix() -> None:
    trace = {
        "raw_response": f"{THOUGHT_START}reason{THOUGHT_END}B",
        "generated_answer_text": "B",
        "generated_answer": "B",
        "generated_token_count": 12,
        "forced_answer_token_count": 1,
        "finish_reason": "forced_close",
        "truncated": False,
        "forced_completion": True,
    }

    stripped = _strip_forced_answer(trace)

    assert stripped["raw_response"] == f"{THOUGHT_START}reason"
    assert stripped["generated_token_count"] == 11
    assert stripped["generated_answer"] is None
    assert stripped["truncated"] is True
