from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .config import ProbeConfig
from .data import MultipleChoiceQuestion


SYSTEM_PROMPT = (
    "Solve the following problem. Please make sure that your response only "
    "consists of a single letter corresponding to the correct answer choice. "
    "Do not include anything else in your final response."
)
THOUGHT_START = "<|channel>thought\n"
THOUGHT_END = "<channel|>"
DECILES = tuple(range(0, 101, 10))


@dataclass(frozen=True)
class ParsedResponse:
    reasoning: str
    answer_text: str


def parse_gemma_response(response: str) -> ParsedResponse:
    start = response.find(THOUGHT_START)
    if start < 0:
        return ParsedResponse(reasoning="", answer_text=_clean_answer_text(response))

    reasoning_start = start + len(THOUGHT_START)
    end = response.find(THOUGHT_END, reasoning_start)
    if end < 0:
        return ParsedResponse(
            reasoning=response[reasoning_start:],
            answer_text="",
        )
    return ParsedResponse(
        reasoning=response[reasoning_start:end],
        answer_text=_clean_answer_text(response[end + len(THOUGHT_END) :]),
    )


def _clean_answer_text(text: str) -> str:
    cleaned = text
    for marker in ("<turn|>", "<eos>", "<pad>"):
        cleaned = cleaned.replace(marker, "")
    return cleaned.strip()


def extract_answer_letter(answer_text: str, valid_labels: Sequence[str]) -> str | None:
    valid = set(valid_labels)
    stripped = answer_text.strip()
    if stripped in valid:
        return stripped
    for match in re.finditer(r"\b([A-Z])\b", stripped):
        if match.group(1) in valid:
            return match.group(1)
    return None


def build_decile_prefixes(
    reasoning: str,
    tokenizer: Any,
) -> dict[int, tuple[str, int]]:
    token_ids = tokenizer.encode(reasoning, add_special_tokens=False)
    total_tokens = len(token_ids)
    prefixes: dict[int, tuple[str, int]] = {0: ("", 0)}
    for decile in DECILES[1:]:
        cutoff = math.ceil(total_tokens * decile / 100)
        if decile == 100:
            prefix = reasoning
        else:
            prefix = tokenizer.decode(
                token_ids[:cutoff],
                skip_special_tokens=False,
            )
        prefixes[decile] = (prefix, cutoff)
    return prefixes


def validate_answer_tokens(
    tokenizer: Any,
    labels: Sequence[str],
) -> dict[str, int]:
    answer_tokens: dict[str, int] = {}
    seen_ids: set[int] = set()
    for label in labels:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"Answer label {label!r} must tokenize to exactly one token; got {token_ids}"
            )
        token_id = int(token_ids[0])
        if token_id in seen_ids:
            raise ValueError(f"Answer labels do not have unique token IDs: {label!r}")
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        if decoded != label:
            raise ValueError(
                f"Answer token {token_id} decodes to {decoded!r}, expected {label!r}"
            )
        seen_ids.add(token_id)
        answer_tokens[label] = token_id
    return answer_tokens


def probabilities_from_logprobs(
    first_position_logprobs: Any,
    answer_tokens: dict[str, int],
) -> tuple[dict[str, float], dict[str, float], float]:
    if not first_position_logprobs:
        raise RuntimeError("Probe generation returned no next-token log-probabilities")

    logprobs: dict[str, float] = {}
    probabilities: dict[str, float] = {}
    for label, token_id in answer_tokens.items():
        if token_id not in first_position_logprobs:
            raise RuntimeError(
                f"Probe response omitted requested token {label!r} ({token_id})"
            )
        entry = first_position_logprobs[token_id]
        logprob = float(entry.logprob if hasattr(entry, "logprob") else entry)
        probability = math.exp(logprob)
        if not math.isfinite(probability):
            raise RuntimeError(f"Non-finite probability for answer {label!r}")
        logprobs[label] = logprob
        probabilities[label] = probability

    choice_mass = sum(probabilities.values())
    non_choice_probability = max(0.0, min(1.0, 1.0 - choice_mass))
    return logprobs, probabilities, non_choice_probability


class GemmaProbeRunner:
    def __init__(self, config: ProbeConfig) -> None:
        executable_dir = str(Path(sys.executable).absolute().parent)
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        if executable_dir not in path_entries:
            os.environ["PATH"] = os.pathsep.join([executable_dir, *path_entries])

        from transformers import AutoTokenizer
        from vllm import LLM

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        self.llm = LLM(
            model=config.model,
            dtype="bfloat16",
            max_model_len=config.max_model_len,
            max_num_seqs=config.max_num_seqs,
            gpu_memory_utilization=config.gpu_memory_utilization,
            trust_remote_code=True,
            # vLLM 0.23's Gemma Unified warmup currently needs the video
            # profile path even for text-only requests.
            limit_mm_per_prompt={"image": 0, "audio": 0},
            max_logprobs=10,
        )

    def build_base_prompts(
        self,
        questions: Sequence[MultipleChoiceQuestion],
    ) -> list[str]:
        messages = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question.prompt},
            ]
            for question in questions
        ]
        prompts = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        return list(prompts)

    def generate_traces(
        self,
        questions: Sequence[MultipleChoiceQuestion],
        base_prompts: Sequence[str],
    ) -> list[dict[str, object]]:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            max_tokens=self.config.trace_max_tokens,
            seed=self.config.seed,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
        )
        outputs = self.llm.generate(
            list(base_prompts),
            sampling_params=sampling_params,
            use_tqdm=True,
        )
        if len(outputs) != len(questions):
            raise RuntimeError(
                f"Expected {len(questions)} trace generations, got {len(outputs)}"
            )

        records: list[dict[str, object]] = []
        for question, output in zip(questions, outputs):
            if not output.outputs:
                raise RuntimeError(f"No trace generated for question {question.question_id}")
            completion = output.outputs[0]
            parsed = parse_gemma_response(completion.text)
            reasoning_token_ids = self.tokenizer.encode(
                parsed.reasoning,
                add_special_tokens=False,
            )
            records.append(
                {
                    "position": question.position,
                    "question_id": question.question_id,
                    "question": question.question,
                    "options": list(question.options),
                    "answer": question.answer,
                    "category": question.category,
                    "source": question.source,
                    "prompt": question.prompt,
                    "raw_response": completion.text,
                    "reasoning_trace": parsed.reasoning,
                    "generated_answer_text": parsed.answer_text,
                    "generated_answer": extract_answer_letter(
                        parsed.answer_text,
                        question.labels,
                    ),
                    "trace_token_count": len(reasoning_token_ids),
                    "generated_token_count": len(completion.token_ids),
                    "finish_reason": completion.finish_reason,
                    "truncated": completion.finish_reason == "length",
                }
            )
        return records

    def probe_trajectories(
        self,
        questions: Sequence[MultipleChoiceQuestion],
        base_prompts: Sequence[str],
        traces: Sequence[dict[str, object]],
    ) -> list[dict[str, object]]:
        from vllm import SamplingParams

        prompts: list[str] = []
        sampling_params: list[Any] = []
        metadata: list[tuple[MultipleChoiceQuestion, int, int, int, dict[str, int]]] = []

        for question, base_prompt, trace in zip(questions, base_prompts, traces):
            reasoning = str(trace["reasoning_trace"])
            prefixes = build_decile_prefixes(reasoning, self.tokenizer)
            answer_tokens = validate_answer_tokens(self.tokenizer, question.labels)
            token_ids = list(answer_tokens.values())
            for decile in DECILES:
                prefix, prefix_token_count = prefixes[decile]
                prompts.append(f"{base_prompt}{THOUGHT_START}{prefix}{THOUGHT_END}")
                sampling_params.append(
                    SamplingParams(
                        temperature=0.0,
                        top_p=1.0,
                        top_k=0,
                        max_tokens=1,
                        seed=self.config.seed,
                        logprobs=len(token_ids),
                        logprob_token_ids=token_ids,
                        skip_special_tokens=False,
                        spaces_between_special_tokens=False,
                    )
                )
                metadata.append(
                    (
                        question,
                        decile,
                        prefix_token_count,
                        int(trace["trace_token_count"]),
                        answer_tokens,
                    )
                )

        outputs = self.llm.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=True,
        )
        if len(outputs) != len(metadata):
            raise RuntimeError(
                f"Expected {len(metadata)} probe generations, got {len(outputs)}"
            )

        records: list[dict[str, object]] = []
        for output, item in zip(outputs, metadata):
            question, decile, prefix_token_count, trace_token_count, answer_tokens = item
            if not output.outputs:
                raise RuntimeError(f"No probe output for question {question.question_id}")
            completion = output.outputs[0]
            first_position_logprobs = (
                completion.logprobs[0] if completion.logprobs else None
            )
            logprobs, probabilities, non_choice_probability = (
                probabilities_from_logprobs(
                    first_position_logprobs,
                    answer_tokens,
                )
            )
            prediction = max(probabilities, key=probabilities.get)
            records.append(
                {
                    "position": question.position,
                    "question_id": question.question_id,
                    "question": question.question,
                    "options": list(question.options),
                    "valid_labels": list(question.labels),
                    "answer": question.answer,
                    "category": question.category,
                    "source": question.source,
                    "decile": decile,
                    "prefix_token_count": prefix_token_count,
                    "trace_token_count": trace_token_count,
                    "is_full_trace": (
                        decile == 100 and prefix_token_count == trace_token_count
                    ),
                    "choice_logprobs": logprobs,
                    "choice_probabilities": probabilities,
                    "choice_probability_mass": sum(probabilities.values()),
                    "non_choice_probability": non_choice_probability,
                    "prediction": prediction,
                    "prediction_probability": probabilities[prediction],
                    "correct": prediction == question.answer,
                    "sampled_token": completion.text,
                }
            )
        return records
