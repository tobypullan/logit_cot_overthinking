#!/usr/bin/env bash
set -euo pipefail

experiment_dir="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
  pwd
)"
repository_dir="$(dirname -- "$experiment_dir")"
cd "$repository_dir"

python_path=".venv/bin/python"
probe_path=".venv/bin/trajectory-probe"
planner_path="scripts/plan_mmlu_pro_disjoint.py"

if [[ ! -x "$python_path" || ! -x "$probe_path" ]]; then
  echo "Project environment not found. Run: python -m venv .venv && .venv/bin/python -m pip install -e '.[dev]'" >&2
  exit 1
fi

planner_args=(
  "$planner_path"
  --exclude-num-rows 1000
  --exclude-selection-seed 0
  --num-rows 2000
  --selection-seed 0
)

if [[ "${1:-}" == "--plan-only" ]]; then
  exec "$python_path" "${planner_args[@]}" --format summary
fi
if [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--plan-only]" >&2
  exit 2
fi

row_indices="$("$python_path" "${planner_args[@]}" --format csv)"

exec "$probe_path" \
  --dataset TIGER-Lab/MMLU-Pro \
  --dataset-format mmlu-pro \
  --split test \
  --selection indices \
  --row-indices "$row_indices" \
  --num-rows 2000 \
  --seed 0 \
  --trace-max-tokens 32768 \
  --max-model-len 49152 \
  --max-num-seqs 16 \
  --output-dir outputs/mmlu_pro_gemma4_12b_n2000_disjoint_seed0
