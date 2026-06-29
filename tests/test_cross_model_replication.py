from __future__ import annotations

import json
from pathlib import Path

from logit_cot_overthinking.cross_model_replication import (
    CrossModelReplicationConfig,
    build_cross_model_replication_plan,
    infer_adapter_status,
    slugify_model_id,
    write_cross_model_replication_plan,
)


def test_slugify_model_id_is_path_safe() -> None:
    assert slugify_model_id("google/gemma-4-12B-it") == "google_gemma_4_12b_it"
    assert slugify_model_id("Org/Model Name!") == "org_model_name"


def test_adapter_status_marks_non_gemma_as_blocked() -> None:
    assert infer_adapter_status("google/gemma-4-12B-it")["runnable"]
    qwen_status = infer_adapter_status("Qwen/Qwen3-14B")
    assert not qwen_status["runnable"]
    assert qwen_status["adapter"] == "adapter_required"


def test_build_cross_model_plan_separates_runnable_and_blocked_models(
    tmp_path: Path,
) -> None:
    config = CrossModelReplicationConfig(
        models=("google/gemma-4-12B-it", "Qwen/Qwen3-14B"),
        output_root=tmp_path / "replication",
        seeds=(0, 1),
        per_cohort=3,
    )

    plan = build_cross_model_replication_plan(config)
    by_model = {item["model"]: item for item in plan["models"]}

    gemma = by_model["google/gemma-4-12B-it"]
    assert gemma["runnable"]
    assert len(gemma["commands"]) == 6
    assert "trajectory-run-matched-controls" in gemma["commands"][0]["command"]
    assert "--seeds 0-1" in gemma["commands"][0]["command"]
    assert "trajectory-run-branching-intervention" in gemma["commands"][-1][
        "command"
    ]

    qwen = by_model["Qwen/Qwen3-14B"]
    assert not qwen["runnable"]
    assert qwen["commands"] == []
    assert len(qwen["blocked_commands"]) == 6


def test_write_cross_model_plan_outputs_manifest_and_commands(
    tmp_path: Path,
) -> None:
    config = CrossModelReplicationConfig(
        models=("google/gemma-4-12B-it",),
        output_root=tmp_path / "replication",
        seeds=(0,),
        per_cohort=2,
        include_branching_setup=False,
    )

    result = write_cross_model_replication_plan(config)

    manifest_path = Path(result["manifest_path"])
    commands_path = Path(result["commands_path"])
    assert manifest_path.exists()
    assert commands_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_count"] == 1
    assert manifest["runnable_model_count"] == 1
    commands = commands_path.read_text(encoding="utf-8")
    assert "trajectory-analyze-early-commitment" in commands
    assert "trajectory-run-branching-intervention" not in commands
