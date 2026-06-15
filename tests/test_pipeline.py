import json
from types import SimpleNamespace

import pytest

from logit_cot_overthinking.config import ProbeConfig
from logit_cot_overthinking.pipeline import _load_reusable_traces


def test_load_reusable_traces_skips_truncated_records(tmp_path) -> None:
    config = ProbeConfig(output_dir=tmp_path, num_rows=2, resume_traces=True)
    questions = [
        SimpleNamespace(position=0, question_id="q0"),
        SimpleNamespace(position=1, question_id="q1"),
    ]
    summary = {"config": config.to_dict()}
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    records = [
        {"position": 0, "question_id": "q0", "truncated": False},
        {"position": 1, "question_id": "q1", "truncated": True},
    ]
    (tmp_path / "traces.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    reusable = _load_reusable_traces(config, questions)

    assert list(reusable) == [(0, "q0")]


def test_load_reusable_traces_rejects_identity_mismatch(tmp_path) -> None:
    config = ProbeConfig(output_dir=tmp_path, num_rows=2, resume_traces=True)
    previous = config.to_dict()
    previous["seed"] = 99
    (tmp_path / "summary.json").write_text(
        json.dumps({"config": previous}),
        encoding="utf-8",
    )
    (tmp_path / "traces.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="different run configuration"):
        _load_reusable_traces(config, [])


def test_load_reusable_traces_accepts_pre_adapter_summary(tmp_path) -> None:
    config = ProbeConfig(output_dir=tmp_path, num_rows=1, resume_traces=True)
    previous = config.to_dict()
    previous.pop("dataset_format")
    (tmp_path / "summary.json").write_text(
        json.dumps({"config": previous}),
        encoding="utf-8",
    )
    (tmp_path / "traces.jsonl").write_text(
        json.dumps(
            {
                "position": 0,
                "question_id": "q0",
                "truncated": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    questions = [SimpleNamespace(position=0, question_id="q0")]

    reusable = _load_reusable_traces(config, questions)

    assert list(reusable) == [(0, "q0")]
