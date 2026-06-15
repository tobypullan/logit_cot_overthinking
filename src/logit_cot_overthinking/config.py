from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProbeConfig:
    model: str = "google/gemma-4-12B-it"
    dataset: str = "TIGER-Lab/MMLU-Pro"
    dataset_format: str = "auto"
    split: str = "test"
    selection: str = "contiguous"
    row_indices: tuple[int, ...] = ()
    start_row: int = 0
    num_rows: int = 3
    seed: int = 0
    trace_max_tokens: int = 4096
    max_model_len: int = 8192
    max_num_seqs: int = 64
    gpu_memory_utilization: float = 0.90
    resume_traces: bool = False
    output_dir: Path = Path("outputs/smoke")

    def validate(self) -> None:
        if self.dataset_format not in {
            "auto",
            "mmlu-pro",
            "gpqa-diamond",
        }:
            raise ValueError(
                "dataset_format must be 'auto', 'mmlu-pro', or "
                "'gpqa-diamond'"
            )
        if self.selection not in {
            "contiguous",
            "balanced-categories",
            "indices",
        }:
            raise ValueError(
                "selection must be 'contiguous', 'balanced-categories', "
                "or 'indices'"
            )
        if self.selection == "indices":
            if not self.row_indices:
                raise ValueError(
                    "row_indices must be provided when selection is 'indices'"
                )
            if len(set(self.row_indices)) != len(self.row_indices):
                raise ValueError("row_indices must not contain duplicates")
            if any(index < 0 for index in self.row_indices):
                raise ValueError("row_indices must be non-negative")
            if self.num_rows != len(self.row_indices):
                raise ValueError(
                    "num_rows must equal the number of row_indices"
                )
        elif self.row_indices:
            raise ValueError(
                "row_indices can only be used when selection is 'indices'"
            )
        if self.start_row < 0:
            raise ValueError("start_row must be non-negative")
        if self.num_rows < 1:
            raise ValueError("num_rows must be at least 1")
        if self.trace_max_tokens < 1:
            raise ValueError("trace_max_tokens must be at least 1")
        if self.max_model_len <= self.trace_max_tokens:
            raise ValueError("max_model_len must be greater than trace_max_tokens")
        if self.max_num_seqs < 1:
            raise ValueError("max_num_seqs must be at least 1")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["output_dir"] = str(self.output_dir)
        values["row_indices"] = list(self.row_indices)
        return values
