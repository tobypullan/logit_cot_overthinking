from __future__ import annotations

from pathlib import Path

import pandas as pd

from logit_cot_overthinking.branching_intervention import (
    BRANCH_MODE_INSTRUCTIONS,
    build_branch_prompt,
    build_branch_requests,
    branch_reasoning_prefix,
    decode_trace_prefix,
    select_branch_candidates,
)
from logit_cot_overthinking.gemma import THOUGHT_END, THOUGHT_START


def _checkpoint(
    question_id: str,
    position: int,
    decile: int,
    confidence: float,
    final_wrong: bool = True,
    cohort: str = "loss",
    dataset: str = "mmlu_pro",
    seed: int = 0,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "dataset_label": "MMLU-Pro",
        "seed": seed,
        "position": position,
        "question_id": question_id,
        "category": "biology",
        "baseline_cohort": cohort,
        "match_id": "match-1",
        "decile": decile,
        "final_wrong": final_wrong,
        "flips_so_far": 1,
        "time_since_first_correct": 0,
        "stable_correct_streak": 1,
        "current_correct_probability": confidence,
        "current_normalized_correct_probability": confidence,
        "normalized_confidence_decline": 0.0,
        "recent_normalized_decline": 0.0,
        "choice_probability_mass": 1.0,
        "prefix_token_count": 10 * decile,
        "trace_token_count": 2000,
        "log_trace_token_count": 7.6,
        "prefix_self_correction": 0,
        "prefix_simulated_retrieval": 0,
        "prefix_repetition_score": 0.0,
    }


def test_select_branch_candidates_filters_and_keeps_earliest_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoints = pd.DataFrame(
        [
            _checkpoint("kept", 1, 30, 0.95),
            _checkpoint("kept", 1, 50, 0.99),
            _checkpoint("low-confidence", 2, 40, 0.70),
            _checkpoint("final-correct", 3, 40, 0.97, final_wrong=False),
            _checkpoint("wrong-cohort", 4, 40, 0.97, cohort="stable_wrong"),
            _checkpoint(
                "gpqa-kept",
                5,
                40,
                0.96,
                dataset="gpqa_diamond",
            ),
        ]
    )

    candidates = select_branch_candidates(
        checkpoints,
        input_root=tmp_path / "matched",
        deciles=(30, 40, 50),
        cohorts=("loss",),
        final_outcome="loss",
        min_current_normalized_correct_probability=0.9,
        max_candidates_per_dataset=5,
    )

    by_question = candidates.set_index("question_id")
    assert set(by_question.index) == {"kept", "gpqa-kept"}
    assert by_question.loc["kept", "decile"] == 30
    assert by_question.loc["kept", "candidate_id"] == "mmlu_pro_seed0_pos1_kept_d30"
    assert by_question.loc["gpqa-kept", "trace_path"].endswith(
        "gpqa_diamond/seed_0/traces.jsonl"
    )


def test_build_branch_requests_crosses_modes_and_seeds() -> None:
    candidates = pd.DataFrame(
        [
            {
                **_checkpoint("q1", 1, 40, 0.95),
                "candidate_id": "mmlu_seed0_pos1_q1_d40",
                "trace_path": "traces.jsonl",
            }
        ]
    )

    requests = build_branch_requests(
        candidates,
        branch_modes=("answer_only", "normal"),
        branch_seeds=(7, 8),
        branch_max_tokens=128,
    )

    assert requests["branch_id"].tolist() == [
        "mmlu_seed0_pos1_q1_d40_answer_only_s7",
        "mmlu_seed0_pos1_q1_d40_answer_only_s8",
        "mmlu_seed0_pos1_q1_d40_normal_s7",
        "mmlu_seed0_pos1_q1_d40_normal_s8",
    ]
    assert requests["request_index"].tolist() == [0, 1, 2, 3]
    assert requests["branch_max_tokens"].tolist() == [128, 128, 128, 128]


def test_branch_prompt_modes_use_thought_channel_closure() -> None:
    base_prompt = "<chat>"
    prefix = "The answer appears to be A."

    answer_prompt = build_branch_prompt(base_prompt, prefix, "answer_only")
    assert answer_prompt == f"{base_prompt}{THOUGHT_START}{prefix}{THOUGHT_END}"

    normal_prompt = build_branch_prompt(base_prompt, prefix, "normal")
    assert normal_prompt == f"{base_prompt}{THOUGHT_START}{prefix}"

    verification_prefix = branch_reasoning_prefix(
        prefix,
        "short_verification",
    )
    assert verification_prefix.endswith(
        BRANCH_MODE_INSTRUCTIONS["short_verification"]
    )


def test_decode_trace_prefix_uses_token_boundary() -> None:
    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False):
            assert not add_special_tokens
            return text.split()

        def decode(self, token_ids, skip_special_tokens: bool = False):
            assert not skip_special_tokens
            return " ".join(token_ids)

    trace = {"reasoning_trace": "alpha beta gamma delta"}

    assert decode_trace_prefix(trace, 3, FakeTokenizer()) == "alpha beta gamma"
