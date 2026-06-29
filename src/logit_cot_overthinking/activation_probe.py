from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import MultipleChoiceQuestion
from .gemma import SYSTEM_PROMPT, THOUGHT_END, THOUGHT_START
from .matched_analysis import auc_score


DEFAULT_DECILES = tuple(range(10, 100, 10))
DEFAULT_CONFIDENCE_THRESHOLDS = (0.7, 0.8, 0.9, 0.95)
DEFAULT_PROBE_THRESHOLDS = (0.3, 0.5, 0.7, 0.9)
PROBE_TARGETS = {
    "future_loss": (
        "Current prediction is correct and the final prediction is wrong."
    ),
    "future_change_to_wrong": (
        "The final prediction is wrong and differs from the current prediction."
    ),
    "future_answer_flip": (
        "The final prediction differs from the current prediction."
    ),
}


@dataclass(frozen=True)
class ActivationExtractionConfig:
    input_root: Path = Path("outputs/matched_controls_gemma4_12b_extended")
    output_dir: Path = Path("outputs/activation_probe_gemma4_12b")
    model: str = "google/gemma-4-12B-it"
    deciles: tuple[int, ...] = DEFAULT_DECILES
    datasets: tuple[str, ...] = ()
    seeds: tuple[int, ...] = ()
    max_examples: int | None = None
    random_seed: int = 0
    layers: tuple[int, ...] | None = None
    batch_size: int = 1
    activation_dtype: str = "float16"
    model_dtype: str = "bfloat16"
    device_map: str = "auto"
    reuse_examples: bool = True
    reuse_activations: bool = True

    def validate(self) -> None:
        if not self.deciles:
            raise ValueError("At least one decile is required")
        invalid = [
            decile
            for decile in self.deciles
            if decile < 0 or decile >= 100 or decile % 10 != 0
        ]
        if invalid:
            raise ValueError(
                "Deciles must be pre-final 10-point checkpoints: "
                f"{invalid}"
            )
        if self.max_examples is not None and self.max_examples < 1:
            raise ValueError("max_examples must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.activation_dtype not in {"float16", "float32"}:
            raise ValueError("activation_dtype must be float16 or float32")
        if self.model_dtype not in {"float16", "bfloat16", "float32", "auto"}:
            raise ValueError(
                "model_dtype must be float16, bfloat16, float32, or auto"
            )


@dataclass(frozen=True)
class ProbeTrainingConfig:
    output_dir: Path = Path("outputs/activation_probe_gemma4_12b")
    examples_path: Path | None = None
    activations_path: Path | None = None
    layers: tuple[int, ...] | None = None
    targets: tuple[str, ...] = tuple(PROBE_TARGETS)
    backend: str = "auto"
    folds: int = 5
    epochs: int = 8
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    ridge: float = 1.0
    seed: int = 0
    confidence_thresholds: tuple[float, ...] = DEFAULT_CONFIDENCE_THRESHOLDS
    probe_thresholds: tuple[float, ...] = DEFAULT_PROBE_THRESHOLDS

    def validate(self) -> None:
        unknown = sorted(set(self.targets) - set(PROBE_TARGETS))
        if unknown:
            raise ValueError(f"Unknown probe targets: {unknown}")
        if self.backend not in {"auto", "torch", "numpy"}:
            raise ValueError("backend must be auto, torch, or numpy")
        if self.folds < 2:
            raise ValueError("folds must be at least 2")
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(_json_ready(value), output, indent=2, sort_keys=True)
        output.write("\n")


def _json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        as_float = float(value)
        return None if math.isnan(as_float) else as_float
    if isinstance(value, (np.bool_,)):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _choice_probability(
    probabilities: object,
    label: object,
) -> float:
    if isinstance(probabilities, str):
        probabilities = json.loads(probabilities)
    if not isinstance(probabilities, Mapping):
        return float("nan")
    return float(probabilities.get(str(label), np.nan))


def _normalized_probability(
    probability: object,
    mass: object,
) -> float:
    try:
        probability_value = float(probability)
        mass_value = float(mass)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(probability_value) or not np.isfinite(mass_value):
        return 0.0
    if mass_value <= 0:
        return 0.0
    return probability_value / mass_value


def _run_identity_from_path(path: Path) -> tuple[str, int]:
    seed_dir = path.parent.name
    dataset = path.parent.parent.name
    match = re.fullmatch(r"seed_(\d+)", seed_dir)
    if not match:
        raise ValueError(f"Could not infer seed from {path}")
    return dataset, int(match.group(1))


def _trajectory_paths(input_root: Path) -> list[Path]:
    return sorted(input_root.glob("*/*/trajectory.parquet"))


def _fold_for_attempt(dataset: str, position: int, folds: int) -> int:
    digest = hashlib.sha256(
        f"{dataset}:{position}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") % folds


def _attempt_columns(dataframe: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in ("dataset", "seed", "position", "question_id")
        if column in dataframe.columns
    ]
    if "position" not in columns or "question_id" not in columns:
        raise ValueError(
            "Trajectory rows must include position and question_id"
        )
    return columns


def _build_example_rows(
    trajectory: pd.DataFrame,
    trace_path: Path,
    deciles: set[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for _, group in trajectory.groupby(_attempt_columns(trajectory), sort=False):
        group = group.sort_values("decile").reset_index(drop=True)
        final_rows = group[group["decile"].astype(int) == 100]
        final = final_rows.iloc[-1] if not final_rows.empty else group.iloc[-1]
        final_prediction = str(final["prediction"])
        final_correct = bool(final["correct"])
        prefinal = group[
            group["decile"].astype(int).isin(deciles)
            & (group["decile"].astype(int) < int(final["decile"]))
        ]
        for row in prefinal.itertuples(index=False):
            row_dict = row._asdict()
            prediction = str(row_dict["prediction"])
            answer = str(row_dict["answer"])
            prediction_probability = row_dict.get("prediction_probability")
            if prediction_probability is None:
                prediction_probability = _choice_probability(
                    row_dict["choice_probabilities"],
                    prediction,
                )
            correct_probability = _choice_probability(
                row_dict["choice_probabilities"],
                answer,
            )
            current_correct = bool(row_dict["correct"])
            final_wrong = not final_correct
            rows.append(
                {
                    "dataset": str(row_dict.get("dataset", "")),
                    "dataset_label": str(
                        row_dict.get("dataset_label", row_dict.get("dataset", ""))
                    ),
                    "seed": int(row_dict.get("seed", row_dict.get("run_seed", 0))),
                    "run_seed": int(row_dict.get("run_seed", row_dict.get("seed", 0))),
                    "position": int(row_dict["position"]),
                    "question_id": str(row_dict["question_id"]),
                    "category": str(row_dict.get("category", "")),
                    "source": str(row_dict.get("source", "")),
                    "baseline_cohort": str(row_dict.get("baseline_cohort", "")),
                    "match_id": str(row_dict.get("match_id", "")),
                    "decile": int(row_dict["decile"]),
                    "prefix_token_count": int(row_dict["prefix_token_count"]),
                    "trace_token_count": int(row_dict["trace_token_count"]),
                    "answer": answer,
                    "current_prediction": prediction,
                    "final_prediction": final_prediction,
                    "current_correct": current_correct,
                    "final_correct": final_correct,
                    "final_wrong": final_wrong,
                    "future_loss": int(current_correct and final_wrong),
                    "future_change_to_wrong": int(
                        final_wrong and final_prediction != prediction
                    ),
                    "future_answer_flip": int(final_prediction != prediction),
                    "prediction_probability": float(prediction_probability),
                    "choice_probability_mass": float(
                        row_dict["choice_probability_mass"]
                    ),
                    "normalized_prediction_probability": (
                        _normalized_probability(
                            prediction_probability,
                            row_dict["choice_probability_mass"],
                        )
                    ),
                    "correct_answer_probability": float(correct_probability),
                    "normalized_correct_probability": (
                        _normalized_probability(
                            correct_probability,
                            row_dict["choice_probability_mass"],
                        )
                    ),
                    "prediction_flip": bool(
                        row_dict.get("prediction_flip", False)
                    ),
                    "outcome": str(row_dict.get("outcome", "")),
                    "trace_path": str(trace_path),
                }
            )
    return rows


def build_activation_probe_examples(
    input_root: Path,
    output_dir: Path,
    deciles: Sequence[int] = DEFAULT_DECILES,
    datasets: Sequence[str] = (),
    seeds: Sequence[int] = (),
    max_examples: int | None = None,
    random_seed: int = 0,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_filter = set(datasets)
    seed_filter = {int(seed) for seed in seeds}
    selected_deciles = {int(decile) for decile in deciles}
    rows: list[dict[str, object]] = []
    for trajectory_path in _trajectory_paths(input_root):
        dataset, seed = _run_identity_from_path(trajectory_path)
        if dataset_filter and dataset not in dataset_filter:
            continue
        if seed_filter and seed not in seed_filter:
            continue
        trajectory = pd.read_parquet(trajectory_path).copy()
        trajectory["dataset"] = trajectory.get("dataset", dataset)
        trajectory["seed"] = trajectory.get("seed", seed)
        trajectory["run_seed"] = trajectory.get("run_seed", seed)
        rows.extend(
            _build_example_rows(
                trajectory,
                trace_path=trajectory_path.parent / "traces.jsonl",
                deciles=selected_deciles,
            )
        )

    examples = pd.DataFrame(rows)
    if examples.empty:
        raise ValueError(f"No activation-probe examples found under {input_root}")
    examples = examples.sort_values(
        ["dataset", "seed", "position", "question_id", "decile"]
    ).reset_index(drop=True)
    if max_examples is not None and len(examples) > max_examples:
        examples = (
            examples.sample(n=max_examples, random_state=random_seed)
            .sort_values(["dataset", "seed", "position", "question_id", "decile"])
            .reset_index(drop=True)
        )
    examples.insert(0, "example_index", np.arange(len(examples), dtype=int))
    examples["fold_key"] = [
        f"{row.dataset}:{int(row.position)}"
        for row in examples.itertuples(index=False)
    ]
    examples.to_parquet(
        output_dir / "activation_probe_examples.parquet",
        index=False,
    )
    return examples


def _question_from_trace(trace: dict[str, object]) -> MultipleChoiceQuestion:
    return MultipleChoiceQuestion(
        position=int(trace["position"]),
        question_id=str(trace["question_id"]),
        question=str(trace["question"]),
        options=tuple(str(option) for option in trace["options"]),
        answer=str(trace["answer"]),
        category=str(trace.get("category", "")),
        source=str(trace.get("source", "")),
    )


def _format_base_prompt(
    trace: dict[str, object],
    tokenizer: Any,
) -> str:
    question = _question_from_trace(trace)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.prompt},
    ]
    return str(
        tokenizer.apply_chat_template(
            [messages],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )[0]
    )


def _trace_cache_for_examples(
    examples: pd.DataFrame,
) -> dict[Path, dict[tuple[int, str], dict[str, object]]]:
    cache: dict[Path, dict[tuple[int, str], dict[str, object]]] = {}
    for path_text in sorted(examples["trace_path"].astype(str).unique()):
        path = Path(path_text)
        traces = _read_jsonl(path)
        cache[path] = {
            (int(trace["position"]), str(trace["question_id"])): trace
            for trace in traces
        }
    return cache


def _prompt_for_example(
    row: pd.Series,
    trace_cache: dict[Path, dict[tuple[int, str], dict[str, object]]],
    base_prompt_cache: dict[tuple[str, int, str], str],
    tokenizer: Any,
) -> str:
    path = Path(str(row["trace_path"]))
    key = (int(row["position"]), str(row["question_id"]))
    trace = trace_cache[path][key]
    prompt_key = (str(row["dataset"]), int(row["position"]), str(row["question_id"]))
    if prompt_key not in base_prompt_cache:
        base_prompt_cache[prompt_key] = _format_base_prompt(trace, tokenizer)
    reasoning = str(trace["reasoning_trace"])
    token_ids = tokenizer.encode(reasoning, add_special_tokens=False)
    prefix_count = int(row["prefix_token_count"])
    if prefix_count > len(token_ids):
        raise ValueError(
            f"Prefix length {prefix_count} exceeds trace length "
            f"{len(token_ids)} for {prompt_key}"
        )
    prefix = tokenizer.decode(
        token_ids[:prefix_count],
        skip_special_tokens=False,
    )
    return f"{base_prompt_cache[prompt_key]}{THOUGHT_START}{prefix}{THOUGHT_END}"


def _resolve_model_dtype(name: str):
    import torch

    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def resolve_layers(
    layers: Sequence[int] | None,
    num_hidden_layers: int,
) -> tuple[int, ...]:
    max_layer = int(num_hidden_layers)
    if layers is not None:
        resolved = tuple(int(layer) for layer in layers)
    else:
        step = max(1, max_layer // 12)
        resolved = tuple(sorted({0, *range(step, max_layer + 1, step), max_layer}))
    invalid = [layer for layer in resolved if layer < 0 or layer > max_layer]
    if invalid:
        raise ValueError(
            f"Layer indices must be between 0 and {max_layer}: {invalid}"
        )
    return resolved


def _model_text_config(config: Any) -> Any:
    return getattr(config, "text_config", config)


def _hidden_state_model(model: Any) -> Any:
    return getattr(model, "model", model)


def _activation_rows_filled(
    activations: np.ndarray,
    *,
    chunk_size: int = 128,
) -> np.ndarray:
    filled = np.zeros(activations.shape[0], dtype=bool)
    for start in range(0, activations.shape[0], chunk_size):
        end = min(start + chunk_size, activations.shape[0])
        chunk = np.asarray(activations[start:end])
        filled[start:end] = np.any(chunk != 0, axis=(1, 2))
    return filled


@dataclass(frozen=True)
class ChunkedActivationArray:
    manifest_path: Path
    shape: tuple[int, int, int]
    dtype: np.dtype
    shards: tuple[dict[str, object], ...]

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> "ChunkedActivationArray":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "activation_shards_v1":
            raise ValueError(f"Unknown activation shard format: {manifest_path}")
        shape = tuple(int(value) for value in manifest["shape"])
        if len(shape) != 3:
            raise ValueError(f"Expected 3D activation shape in {manifest_path}")
        return cls(
            manifest_path=manifest_path,
            shape=(shape[0], shape[1], shape[2]),
            dtype=np.dtype(str(manifest["dtype"])),
            shards=tuple(manifest["shards"]),
        )

    def layer(self, layer_offset: int, dtype: np.dtype = np.float32) -> np.ndarray:
        output = np.empty((self.shape[0], self.shape[2]), dtype=dtype)
        for shard in self.shards:
            start = int(shard["start"])
            end = int(shard["end"])
            path = self.manifest_path.parent / str(shard["path"])
            activations = np.load(path, mmap_mode="r")
            output[start:end] = np.asarray(
                activations[:, layer_offset, :],
                dtype=dtype,
            )
        return output


def _activation_shard_manifest_path(output_dir: Path) -> Path:
    return output_dir / "activations_shards" / "manifest.json"


def _load_activation_array(
    activations_path: Path,
    output_dir: Path,
) -> tuple[np.ndarray | ChunkedActivationArray, Path]:
    if activations_path.exists():
        return np.load(activations_path, mmap_mode="r"), activations_path
    shard_manifest_path = _activation_shard_manifest_path(output_dir)
    if shard_manifest_path.exists():
        return (
            ChunkedActivationArray.from_manifest(shard_manifest_path),
            shard_manifest_path,
        )
    raise FileNotFoundError(
        "Activations not found. Expected either "
        f"{activations_path} or {shard_manifest_path}"
    )


def _activation_layer_features(
    activations: np.ndarray | ChunkedActivationArray,
    layer_offset: int,
) -> np.ndarray:
    if isinstance(activations, ChunkedActivationArray):
        return activations.layer(layer_offset, dtype=np.float32)
    return np.asarray(activations[:, layer_offset, :], dtype=np.float32)


def extract_probe_activations(
    config: ActivationExtractionConfig,
) -> dict[str, object]:
    config.validate()
    started_at = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    examples_path = config.output_dir / "activation_probe_examples.parquet"
    if config.reuse_examples and examples_path.exists():
        examples = pd.read_parquet(examples_path)
    else:
        examples = build_activation_probe_examples(
            input_root=config.input_root,
            output_dir=config.output_dir,
            deciles=config.deciles,
            datasets=config.datasets,
            seeds=config.seeds,
            max_examples=config.max_examples,
            random_seed=config.random_seed,
        )
    examples_ready_at = time.perf_counter()

    activation_path = config.output_dir / "activations.npy"
    activation_manifest_path = config.output_dir / "activation_manifest.json"
    if (
        config.reuse_activations
        and activation_manifest_path.exists()
    ):
        shard_manifest_path = _activation_shard_manifest_path(config.output_dir)
        if activation_path.exists() or shard_manifest_path.exists():
            manifest = json.loads(
                activation_manifest_path.read_text(encoding="utf-8")
            )
            manifest["reused"] = True
            return manifest

    import torch
    from numpy.lib.format import open_memmap
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        torch_dtype=_resolve_model_dtype(config.model_dtype),
        device_map=config.device_map,
        trust_remote_code=True,
    )
    model.eval()
    text_config = _model_text_config(model.config)
    hidden_size = int(text_config.hidden_size)
    layers = resolve_layers(config.layers, int(text_config.num_hidden_layers))
    activation_dtype = (
        np.float16 if config.activation_dtype == "float16" else np.float32
    )
    activation_shape = (len(examples), len(layers), hidden_size)
    resumed_from_partial = False
    preexisting_filled_rows = 0
    missing_indices: np.ndarray | None = None
    if config.reuse_activations and activation_path.exists():
        existing = np.load(activation_path, mmap_mode="r+")
        if existing.shape == activation_shape and existing.dtype == activation_dtype:
            activations = existing
            filled_rows = _activation_rows_filled(activations)
            preexisting_filled_rows = int(filled_rows.sum())
            missing_indices = np.flatnonzero(~filled_rows).astype(int)
            resumed_from_partial = bool(missing_indices.size)
        else:
            activations = open_memmap(
                activation_path,
                mode="w+",
                dtype=activation_dtype,
                shape=activation_shape,
            )
    else:
        activations = open_memmap(
            activation_path,
            mode="w+",
            dtype=activation_dtype,
            shape=activation_shape,
        )
    model_ready_at = time.perf_counter()

    trace_cache = _trace_cache_for_examples(examples)
    base_prompt_cache: dict[tuple[str, int, str], str] = {}
    first_device = next(model.parameters()).device
    hidden_model = _hidden_state_model(model)
    extraction_order = (
        examples.sort_values(
            ["prefix_token_count", "trace_token_count", "example_index"]
        )
        .index.to_numpy()
        .astype(int)
    )
    if missing_indices is not None:
        if missing_indices.size == 0:
            extraction_order = np.asarray([], dtype=int)
        else:
            missing_set = set(missing_indices.tolist())
            extraction_order = np.asarray(
                [
                    index
                    for index in extraction_order
                    if int(index) in missing_set
                ],
                dtype=int,
            )
    try:
        from tqdm.auto import tqdm

        batch_starts: Iterable[int] = tqdm(
            range(0, len(extraction_order), config.batch_size),
            desc="Extracting activations",
        )
    except ImportError:
        batch_starts = range(0, len(extraction_order), config.batch_size)

    for batch_number, start in enumerate(batch_starts):
        batch_indices = extraction_order[start : start + config.batch_size]
        batch = examples.iloc[batch_indices]
        prompts = [
            _prompt_for_example(row, trace_cache, base_prompt_cache, tokenizer)
            for _, row in batch.iterrows()
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        encoded = {
            key: value.to(first_device)
            for key, value in encoded.items()
        }
        with torch.inference_mode():
            output = hidden_model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        lengths = encoded["attention_mask"].sum(dim=1) - 1
        torch_batch_indices = torch.arange(
            len(batch),
            device=lengths.device,
        )
        for layer_offset, layer in enumerate(layers):
            selected = output.hidden_states[layer][torch_batch_indices, lengths]
            activations[
                batch_indices,
                layer_offset,
                :,
            ] = selected.float().cpu().numpy()
        del output, encoded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if batch_number % 50 == 0:
            activations.flush()
    activations.flush()
    finished_at = time.perf_counter()

    manifest = {
        "experiment": "activation_probe_extraction",
        "model": config.model,
        "input_root": config.input_root,
        "output_dir": config.output_dir,
        "examples_path": examples_path,
        "activations_path": activation_path,
        "example_count": len(examples),
        "layers": layers,
        "hidden_size": hidden_size,
        "activation_dtype": config.activation_dtype,
        "model_dtype": config.model_dtype,
        "batch_size": config.batch_size,
        "batching": "sorted_by_prefix_token_count",
        "processed_rows": int(len(extraction_order)),
        "preexisting_filled_rows": preexisting_filled_rows,
        "resumed_from_partial": resumed_from_partial,
        "target_positive_counts": {
            target: int(examples[target].sum())
            for target in PROBE_TARGETS
        },
        "target_positive_rates": {
            target: float(examples[target].mean())
            for target in PROBE_TARGETS
        },
        "timings_seconds": {
            "examples": round(examples_ready_at - started_at, 3),
            "model_initialization": round(model_ready_at - examples_ready_at, 3),
            "activation_extraction": round(finished_at - model_ready_at, 3),
            "total": round(finished_at - started_at, 3),
        },
        "reused": False,
    }
    _write_json(activation_manifest_path, manifest)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return manifest


def _standardize(
    train: np.ndarray,
    test: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-6] = 1.0
    train_scaled = (train - mean) / scale
    test_scaled = None if test is None else (test - mean) / scale
    return train_scaled, test_scaled, mean, scale


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30, 30)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_predict_numpy(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from .matched_analysis import _fit_logistic

    train_scaled, test_scaled, mean, scale = _standardize(train_x, test_x)
    predictions = np.full((len(test_x), train_y.shape[1]), np.nan)
    coefficients = np.zeros((train_y.shape[1], train_x.shape[1]))
    intercepts = np.zeros(train_y.shape[1])
    if train_x.shape[1] > 512:
        raise ValueError(
            "The numpy backend is intended for tests and small feature sets; "
            "use the torch backend for full hidden-state probes."
        )
    for target_index in range(train_y.shape[1]):
        target = train_y[:, target_index].astype(int)
        if len(np.unique(target)) < 2:
            continue
        coefficient = _fit_logistic(train_scaled, target, ridge=ridge)
        intercepts[target_index] = coefficient[0]
        coefficients[target_index] = coefficient[1:]
        logits = (
            np.column_stack([np.ones(len(test_scaled)), test_scaled])
            @ coefficient
        )
        predictions[:, target_index] = _sigmoid(logits)
    return predictions, coefficients, intercepts, mean, scale


def _fit_predict_torch(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import torch

    rng = np.random.default_rng(seed)
    train_scaled, test_scaled, mean, scale = _standardize(train_x, test_x)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = train_x.shape[1]
    target_count = train_y.shape[1]
    model = torch.nn.Linear(input_dim, target_count).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    positives = train_y.sum(axis=0)
    negatives = len(train_y) - positives
    pos_weight = np.where(positives > 0, negatives / np.maximum(positives, 1), 1.0)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
    )
    train_y_tensor = torch.tensor(
        train_y.astype(np.float32),
        dtype=torch.float32,
        device=device,
    )
    for _ in range(epochs):
        order = rng.permutation(len(train_scaled))
        for start in range(0, len(order), batch_size):
            batch_indices = order[start : start + batch_size]
            batch_x = torch.tensor(
                train_scaled[batch_indices],
                dtype=torch.float32,
                device=device,
            )
            logits = model(batch_x)
            loss = criterion(logits, train_y_tensor[batch_indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(test_scaled), batch_size):
            batch_x = torch.tensor(
                test_scaled[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            predictions.append(torch.sigmoid(model(batch_x)).cpu().numpy())
    coefficient = model.weight.detach().cpu().numpy()
    intercept = model.bias.detach().cpu().numpy()
    return np.vstack(predictions), coefficient, intercept, mean, scale


def _available_backend(requested: str) -> str:
    if requested == "numpy":
        return "numpy"
    if requested == "torch":
        return "torch"
    try:
        import torch  # noqa: F401

        return "torch"
    except ImportError:
        return "numpy"


def _fold_ids(examples: pd.DataFrame, folds: int) -> np.ndarray:
    return np.asarray(
        [
            _fold_for_attempt(str(row.dataset), int(row.position), folds)
            for row in examples.itertuples(index=False)
        ],
        dtype=int,
    )


def _classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, object]:
    valid = np.isfinite(scores)
    y = labels[valid].astype(int)
    s = scores[valid].astype(float)
    if len(y) == 0:
        return {
            "row_count": 0,
            "positive_rate": float("nan"),
            "auc": float("nan"),
            "brier": float("nan"),
            "accuracy_at_0_5": float("nan"),
        }
    return {
        "row_count": int(len(y)),
        "positive_rate": float(y.mean()),
        "auc": auc_score(y, s),
        "brier": float(np.mean((s - y) ** 2)),
        "accuracy_at_0_5": float(((s >= 0.5).astype(int) == y).mean()),
    }


def _save_model(
    output_dir: Path,
    target_names: Sequence[str],
    layer: int,
    coefficients: np.ndarray,
    intercepts: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> Path:
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"layer_{int(layer):03d}.npz"
    np.savez_compressed(
        path,
        targets=np.asarray(list(target_names)),
        layer=np.asarray([int(layer)]),
        coefficients=coefficients.astype(np.float32),
        intercepts=intercepts.astype(np.float32),
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
    )
    return path


def train_activation_probes(
    config: ProbeTrainingConfig,
) -> dict[str, object]:
    config.validate()
    started_at = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    examples_path = (
        config.examples_path
        or config.output_dir / "activation_probe_examples.parquet"
    )
    activations_path = config.activations_path or config.output_dir / "activations.npy"
    if not examples_path.exists():
        raise FileNotFoundError(f"Examples not found: {examples_path}")

    examples = pd.read_parquet(examples_path).reset_index(drop=True)
    activations, resolved_activations_path = _load_activation_array(
        activations_path,
        config.output_dir,
    )
    if len(examples) != activations.shape[0]:
        raise ValueError(
            f"Example count {len(examples)} does not match activations "
            f"{activations.shape[0]}"
        )
    manifest_path = config.output_dir / "activation_manifest.json"
    activation_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    layer_numbers = tuple(
        int(layer)
        for layer in (
            config.layers
            or activation_manifest.get("layers")
            or range(activations.shape[1])
        )
    )
    if len(layer_numbers) != activations.shape[1]:
        raise ValueError(
            "Layer count does not match activation tensor shape: "
            f"{len(layer_numbers)} vs {activations.shape[1]}"
        )

    target_names = list(config.targets)
    labels = examples[target_names].astype(float).to_numpy()
    fold_ids = _fold_ids(examples, config.folds)
    backend = _available_backend(config.backend)
    loaded_at = time.perf_counter()

    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    model_paths: dict[str, str] = {}
    for layer_offset, layer in enumerate(layer_numbers):
        features = _activation_layer_features(activations, layer_offset)
        layer_scores = np.full((len(examples), len(target_names)), np.nan)
        for fold in range(config.folds):
            train = fold_ids != fold
            test = fold_ids == fold
            if not test.any():
                continue
            if backend == "torch":
                scores, _, _, _, _ = _fit_predict_torch(
                    features[train],
                    labels[train],
                    features[test],
                    epochs=config.epochs,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                    weight_decay=config.weight_decay,
                    seed=config.seed + int(layer) * 100 + fold,
                )
            else:
                scores, _, _, _, _ = _fit_predict_numpy(
                    features[train],
                    labels[train],
                    features[test],
                    ridge=config.ridge,
                )
            layer_scores[test] = scores

        if backend == "torch":
            _, coefficients, intercepts, mean, scale = _fit_predict_torch(
                features,
                labels,
                features[:1],
                epochs=config.epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                seed=config.seed + int(layer) * 1000,
            )
        else:
            _, coefficients, intercepts, mean, scale = _fit_predict_numpy(
                features,
                labels,
                features[:1],
                ridge=config.ridge,
            )
        model_paths[str(layer)] = str(
            _save_model(
                config.output_dir,
                target_names,
                int(layer),
                coefficients,
                intercepts,
                mean,
                scale,
            )
        )

        for target_index, target in enumerate(target_names):
            metrics = _classification_metrics(
                labels[:, target_index],
                layer_scores[:, target_index],
            )
            metric_rows.append(
                {
                    "layer": int(layer),
                    "target": target,
                    **metrics,
                }
            )
            for example_index, score in enumerate(layer_scores[:, target_index]):
                prediction_rows.append(
                    {
                        "example_index": int(examples.loc[example_index, "example_index"]),
                        "layer": int(layer),
                        "target": target,
                        "score": float(score) if np.isfinite(score) else np.nan,
                        "label": int(labels[example_index, target_index]),
                        "fold": int(fold_ids[example_index]),
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(metric_rows)
    predictions_path = config.output_dir / "probe_predictions.parquet"
    metrics_path = config.output_dir / "probe_metrics.parquet"
    predictions.to_parquet(predictions_path, index=False)
    metrics.to_parquet(metrics_path, index=False)
    halting = evaluate_probe_halting(
        examples,
        predictions,
        confidence_thresholds=config.confidence_thresholds,
        probe_thresholds=config.probe_thresholds,
    )
    halting_path = config.output_dir / "probe_halting_summary.parquet"
    halting.to_parquet(halting_path, index=False)
    finished_at = time.perf_counter()

    summary = {
        "experiment": "activation_probe_training",
        "output_dir": config.output_dir,
        "examples_path": examples_path,
        "activations_path": resolved_activations_path,
        "predictions_path": predictions_path,
        "metrics_path": metrics_path,
        "halting_summary_path": halting_path,
        "example_count": len(examples),
        "layers": layer_numbers,
        "targets": target_names,
        "backend": backend,
        "folds": config.folds,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "model_paths": model_paths,
        "best_auc_by_target": (
            metrics.sort_values("auc", ascending=False)
            .groupby("target")
            .head(1)
            .to_dict(orient="records")
        ),
        "best_halting_by_target": (
            halting[halting["policy_family"] == "probe_confidence"]
            .sort_values("delta_vs_final", ascending=False)
            .groupby("target")
            .head(1)
            .to_dict(orient="records")
            if not halting.empty
            else []
        ),
        "timings_seconds": {
            "loading": round(loaded_at - started_at, 3),
            "training_and_evaluation": round(finished_at - loaded_at, 3),
            "total": round(finished_at - started_at, 3),
        },
    }
    _write_json(config.output_dir / "activation_probe_summary.json", summary)
    write_activation_probe_report(
        config.output_dir / "activation_probe_report.md",
        summary,
        metrics,
        halting,
    )
    return summary


def evaluate_probe_halting(
    examples: pd.DataFrame,
    predictions: pd.DataFrame,
    confidence_thresholds: Sequence[float] = DEFAULT_CONFIDENCE_THRESHOLDS,
    probe_thresholds: Sequence[float] = DEFAULT_PROBE_THRESHOLDS,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    base = examples.set_index("example_index", drop=False)
    records: list[dict[str, object]] = []

    def summarize(
        rows: list[dict[str, object]],
        policy_family: str,
        target: str,
        layer: int | None,
        confidence_threshold: float,
        probe_threshold: float | None,
    ) -> None:
        if not rows:
            return
        frame = pd.DataFrame(rows)
        records.append(
            {
                "policy_family": policy_family,
                "target": target,
                "layer": layer,
                "confidence_threshold": confidence_threshold,
                "probe_threshold": probe_threshold,
                "attempt_count": len(frame),
                "accuracy": float(frame["selected_correct"].mean()),
                "final_accuracy": float(frame["final_correct"].mean()),
                "delta_vs_final": float(
                    frame["selected_correct"].mean()
                    - frame["final_correct"].mean()
                ),
                "stop_rate": float(frame["stopped_early"].mean()),
                "mean_stop_decile": float(frame["stop_decile"].mean()),
                "median_stop_decile": float(frame["stop_decile"].median()),
            }
        )

    attempt_columns = ["dataset", "seed", "position", "question_id"]
    grouped_examples = {
        key: group.sort_values("decile")
        for key, group in base.groupby(attempt_columns, sort=False)
    }
    for confidence_threshold in confidence_thresholds:
        rows: list[dict[str, object]] = []
        for group in grouped_examples.values():
            selected_rows = group[
                group["normalized_prediction_probability"].astype(float)
                >= float(confidence_threshold)
            ]
            selected = selected_rows.iloc[0] if not selected_rows.empty else None
            final_correct = bool(group["final_correct"].iloc[0])
            rows.append(
                {
                    "selected_correct": (
                        bool(selected["current_correct"])
                        if selected is not None
                        else final_correct
                    ),
                    "final_correct": final_correct,
                    "stopped_early": selected is not None,
                    "stop_decile": (
                        int(selected["decile"]) if selected is not None else 100
                    ),
                }
            )
        summarize(
            rows,
            policy_family="confidence_only",
            target="confidence_only",
            layer=None,
            confidence_threshold=float(confidence_threshold),
            probe_threshold=None,
        )

    for (target, layer), score_rows in predictions.groupby(["target", "layer"]):
        scores = score_rows.set_index("example_index")["score"]
        scored = base.join(scores.rename("probe_score"), how="inner")
        grouped_scored = {
            key: group.sort_values("decile")
            for key, group in scored.groupby(attempt_columns, sort=False)
        }
        for confidence_threshold in confidence_thresholds:
            for probe_threshold in probe_thresholds:
                rows = []
                for group in grouped_scored.values():
                    selected_rows = group[
                        (
                            group["normalized_prediction_probability"].astype(float)
                            >= float(confidence_threshold)
                        )
                        & (
                            group["probe_score"].astype(float)
                            >= float(probe_threshold)
                        )
                    ]
                    selected = (
                        selected_rows.iloc[0]
                        if not selected_rows.empty
                        else None
                    )
                    final_correct = bool(group["final_correct"].iloc[0])
                    rows.append(
                        {
                            "selected_correct": (
                                bool(selected["current_correct"])
                                if selected is not None
                                else final_correct
                            ),
                            "final_correct": final_correct,
                            "stopped_early": selected is not None,
                            "stop_decile": (
                                int(selected["decile"])
                                if selected is not None
                                else 100
                            ),
                        }
                    )
                summarize(
                    rows,
                    policy_family="probe_confidence",
                    target=str(target),
                    layer=int(layer),
                    confidence_threshold=float(confidence_threshold),
                    probe_threshold=float(probe_threshold),
                )
    return pd.DataFrame(records)


def write_activation_probe_report(
    path: Path,
    summary: dict[str, object],
    metrics: pd.DataFrame,
    halting: pd.DataFrame,
) -> None:
    lines = [
        "# Activation Probe Report",
        "",
        f"- Examples: {int(summary['example_count']):,}",
        f"- Layers: {', '.join(str(layer) for layer in summary['layers'])}",
        f"- Targets: {', '.join(summary['targets'])}",
        f"- Backend: {summary['backend']}",
        "",
        "## Targets",
        "",
    ]
    for target, description in PROBE_TARGETS.items():
        lines.append(f"- `{target}`: {description}")
    lines.extend(["", "## Best AUC", ""])
    if metrics.empty:
        lines.append("No metrics were produced.")
    else:
        best = (
            metrics.sort_values("auc", ascending=False)
            .groupby("target")
            .head(5)
        )
        lines.append("| Target | Layer | AUC | Brier | Positive rate |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in best.itertuples(index=False):
            lines.append(
                f"| `{row.target}` | {int(row.layer)} | "
                f"{float(row.auc):.3f} | {float(row.brier):.3f} | "
                f"{float(row.positive_rate):.1%} |"
            )
    lines.extend(["", "## Best Halting Deltas", ""])
    deployable = halting[halting["policy_family"] == "probe_confidence"]
    if deployable.empty:
        lines.append("No probe halting policies were evaluated.")
    else:
        best_halting = (
            deployable.sort_values("delta_vs_final", ascending=False)
            .groupby("target")
            .head(5)
        )
        lines.append(
            "| Target | Layer | Confidence | Probe | Accuracy | "
            "Delta vs final | Stop rate |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in best_halting.itertuples(index=False):
            lines.append(
                f"| `{row.target}` | {int(row.layer)} | "
                f"{float(row.confidence_threshold):.2f} | "
                f"{float(row.probe_threshold):.2f} | "
                f"{float(row.accuracy):.1%} | "
                f"{float(row.delta_vs_final):+.1%} | "
                f"{float(row.stop_rate):.1%} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_activation_probe_pipeline(
    extraction_config: ActivationExtractionConfig,
    training_config: ProbeTrainingConfig,
    train: bool = True,
) -> dict[str, object]:
    extraction_manifest = extract_probe_activations(extraction_config)
    if not train:
        return extraction_manifest
    return train_activation_probes(training_config)
