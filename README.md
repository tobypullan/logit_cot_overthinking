# Logit CoT Overthinking

This repository implements the three-stage logit trajectory probing protocol
from [Probing the Trajectories of Reasoning Traces in Large Language
Models](https://arxiv.org/abs/2601.23163):

1. Generate a complete reasoning trace.
2. Slice it at token deciles from 0% through 100%.
3. Reinject each prefix and measure the next-token distribution over the valid
   answer letters.

The implementation supports MMLU-Pro, GPQA Diamond, and
[`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it).
Gemma's thought-channel format is handled explicitly. MMLU-Pro's variable
number of choices and GPQA Diamond's nested choice mappings are preserved.

## Installation

The model requires a recent NVIDIA GPU and roughly 40 GB of GPU memory in
BF16. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Smoke Run

Run the first three rows of the MMLU-Pro test split:

```bash
trajectory-probe \
  --start-row 0 \
  --num-rows 3 \
  --trace-max-tokens 4096 \
  --max-model-len 8192 \
  --output-dir outputs/smoke
```

The command writes:

- `traces.jsonl`: raw generations and parsed reasoning traces.
- `trajectory.parquet`: one row per question and reasoning decile.
- `summary.json`: configuration, per-decile accuracy, validation checks, and
  per-stage runtime timings.

The probe uses Gemma 4's recommended trace sampling configuration
(`temperature=1.0`, `top_p=0.95`, `top_k=64`). Probe generations are greedy,
unfiltered one-token generations. Exact log-probabilities are requested for
each question's valid bare answer-letter tokens.

## CLI

```bash
trajectory-probe --help
```

Important options include:

- `--model`: Hugging Face model ID.
- `--dataset` and `--dataset-format`: dataset ID and supported schema adapter.
- `--seed`: generation seed.
- `--selection`: contiguous rows or deterministic category-balanced sampling.
- `--start-row` and `--num-rows`: contiguous test-split row range.
- `--trace-max-tokens`: maximum generated tokens per full trace.
- `--max-model-len`: vLLM context-window allocation.
- `--max-num-seqs`: vLLM scheduler concurrency, independent of sample count.
- `--output-dir`: output location.

Run unit tests with:

```bash
pytest
```

## Visualizations

Create the smoke-run figures with:

```bash
trajectory-visualize --input-dir outputs/smoke
```

Figures are written to `outputs/smoke/figures/`:

- `trajectory_overview.png`: aggregate accuracy, commitment, non-choice mass,
  and flip rate across reasoning deciles.
- `correct_answer_heatmap.png`: correct-answer probability and argmax answer
  for every question and decile.
- `choice_probability_trajectories.png`: per-question distributions over all
  valid answer letters.
- `runtime_and_trace_lengths.png`: trace lengths and latest-pass runtime by
  stage.

For runs above 50 questions, the per-question figures are replaced by scalable
aggregate views:

- `category_accuracy_heatmap.png`: accuracy by category and reasoning decile.
- `outcome_probability_trajectories.png`: correct-answer commitment grouped by
  stable/gained/lost/stable-wrong outcome.
- `outcomes_by_category.png`: outcome composition within each category.

## MMLU-Pro 1,000-Question Run

Step 2 uses a deterministic category-balanced sample. With 1,000 questions,
the 14 MMLU-Pro categories contribute either 71 or 72 questions each. Selected
rows retain their original test-split order.

```bash
trajectory-probe \
  --selection balanced-categories \
  --num-rows 1000 \
  --seed 0 \
  --trace-max-tokens 16384 \
  --max-model-len 20480 \
  --max-num-seqs 32 \
  --output-dir outputs/mmlu_pro_gemma4_12b_n1000_seed0
```

To extend a completed token-capped run, use the same run identity and add
`--resume-traces`. Matching complete traces are retained and only missing or
truncated traces are generated again.

Generate figures after the run:

```bash
trajectory-visualize \
  --input-dir outputs/mmlu_pro_gemma4_12b_n1000_seed0
```

Analyze cases that were correct at any probe decile but wrong at the final
decile:

```bash
trajectory-analyze-lost \
  --input-dir outputs/mmlu_pro_gemma4_12b_n1000_seed0
```

This writes `lost_cases.parquet`, `lost_summary.json`, `lost_report.md`, a
compact CSV of confidence-filtered losses, and five figures to
`<input-dir>/lost_analysis/`. The report distinguishes the endpoint `lost`
label from gained-then-lost trajectories. Its stricter `robust_loss` flag also
requires a high-confidence correct answer before the end, substantial final
probability mass on valid answer letters, and agreement between the final
probe and generated answer. Both confidence thresholds are configurable from
the CLI.

## GPQA Diamond Smoke Run

Run the first three questions from
[`fingertap/GPQA-Diamond`](https://huggingface.co/datasets/fingertap/GPQA-Diamond):

```bash
trajectory-probe \
  --dataset fingertap/GPQA-Diamond \
  --dataset-format gpqa-diamond \
  --split test \
  --start-row 0 \
  --num-rows 3 \
  --trace-max-tokens 16384 \
  --max-model-len 20480 \
  --max-num-seqs 16 \
  --output-dir outputs/gpqa_diamond_gemma4_12b_smoke
```

The adapter extracts GPQA's trailing four-choice block, preserves nested
choice mappings present in the source questions, and assigns stable IDs based
on test-split row positions. The output schema is identical to an MMLU-Pro
run, so the visualization and lost-case analysis commands work unchanged.

After validating the smoke run, the full dataset command is:

```bash
trajectory-probe \
  --dataset fingertap/GPQA-Diamond \
  --dataset-format gpqa-diamond \
  --split test \
  --start-row 0 \
  --num-rows 198 \
  --seed 0 \
  --trace-max-tokens 16384 \
  --max-model-len 20480 \
  --max-num-seqs 16 \
  --output-dir outputs/gpqa_diamond_gemma4_12b_seed0
```

## Candidate Seed Reruns

Rerun the 17 confidence-filtered MMLU-Pro candidates and 8 normalized GPQA
reversal candidates across seeds 0 through 9:

```bash
trajectory-rerun-candidates \
  --seeds 0-9 \
  --output-root outputs/candidate_reruns_gemma4_12b
```

The command loads Gemma once for all 250 traces. It writes normal
`traces.jsonl`, `trajectory.parquet`, and `summary.json` artifacts under
`<output-root>/<dataset>/seed_<seed>/`, plus a root `manifest.json`.

Continue every token-capped trace in the full GPQA, full MMLU-Pro, and
candidate rerun outputs from its exact stored response prefix:

```bash
trajectory-extend-capped \
  --extension-max-tokens 16384 \
  --max-extension-rounds 1 \
  --max-model-len 49152
```

The original outputs are preserved. Complete extended runs are written to
parallel `_extended` directories, and only changed trajectories are probed
again. A trace that still does not close after the additional 16,384-token
budget has its thought channel explicitly closed and its answer sampled. Such
runaway traces remain in the analysis with `forced_completion=true`.

Analyze recurrence across the ten completed seed reruns:

```bash
trajectory-analyze-seeds \
  --input-root outputs/candidate_reruns_gemma4_12b_extended
```

This writes per-attempt and per-candidate Parquet tables, a JSON summary, a
Markdown report, and four seed-stability figures under
`<input-root>/analysis/`.

Compare the original capped endpoints with their completed traces:

```bash
trajectory-analyze-extensions
```

This writes a per-trace table, summary, report, and correctness-transition
figures under `outputs/trace_extension_analysis_gemma4_12b/`.

Run the matched-control experiment with 25 loss, final-correct, and
stable-wrong questions per dataset across ten seeds:

```bash
trajectory-run-matched-controls \
  --seeds 0-9 \
  --per-cohort 25 \
  --output-root outputs/matched_controls_gemma4_12b
```

Controls are matched within dataset and category to the nearest seed-0 trace
length. The command writes a `cohort_selection.parquet`, a manifest, and
standard per-seed run artifacts.

Complete any token-capped matched-control attempts:

```bash
trajectory-extend-matched-controls
```

After extending capped traces, analyze recurrence and future-loss predictors:

```bash
trajectory-analyze-matched-controls \
  --input-root outputs/matched_controls_gemma4_12b_extended
```

Prediction is evaluated only at checkpoints where the current probe is
correct, using five-fold cross-validation grouped by question.

The implementation follows the paper's public
[reference repository](https://github.com/AndresAlgaba/probing_reasoning_traces)
where applicable, with Gemma-specific prompting and parsing added here.
