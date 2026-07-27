# Logit CoT Overthinking

This repository implements the three-stage logit trajectory probing protocol
from [Probing the Trajectories of Reasoning Traces in Large Language
Models](https://arxiv.org/abs/2601.23163):

1. Generate a complete reasoning trace.
2. Slice it at token deciles from 0% through 100%.
3. Reinject each prefix and measure the next-token distribution over the valid
   answer letters.

The implementation supports MMLU-Pro, GPQA Diamond, SWE-QA, and
[`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it).
Gemma's thought-channel format is handled explicitly. MMLU-Pro's variable
number of choices and GPQA Diamond's nested choice mappings are preserved.
SWE-QA's repository code context and four-choice mappings are preserved.

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

## SWE-QA Repository-Code Experiment

[`lailaelkoussy/swe-qa`](https://huggingface.co/datasets/lailaelkoussy/swe-qa)
contains four-choice code-comprehension questions derived from real Python
repositories in SWE-bench. Each question requires reasoning across multiple
code entities. The `noisy_oracle` split adds plausible distractor chunks and
is the primary setting for this experiment.

The adapter places the supplied code before the question, preserves the source
repository and question type as separate output columns, and uses their pair
as the analysis category. Consequently, `--selection balanced-categories`
balances across repository/question-type pairs rather than only repositories.

Run a three-question smoke test:

```bash
trajectory-probe \
  --dataset lailaelkoussy/swe-qa \
  --dataset-format swe-qa \
  --split noisy_oracle \
  --start-row 0 \
  --num-rows 3 \
  --seed 0 \
  --trace-max-tokens 8192 \
  --max-model-len 32768 \
  --max-num-seqs 2 \
  --output-dir outputs/swe_qa_gemma4_12b_smoke
```

After inspecting the smoke artifacts, run a deterministic 1,000-question
balanced sample:

```bash
trajectory-probe \
  --dataset lailaelkoussy/swe-qa \
  --dataset-format swe-qa \
  --split noisy_oracle \
  --selection balanced-categories \
  --num-rows 1000 \
  --seed 0 \
  --trace-max-tokens 16384 \
  --max-model-len 49152 \
  --max-num-seqs 8 \
  --output-dir outputs/swe_qa_gemma4_12b_n1000_seed0
```

Generate the standard trajectory figures and fragile-correctness analysis:

```bash
trajectory-visualize \
  --input-dir outputs/swe_qa_gemma4_12b_n1000_seed0

trajectory-analyze-lost \
  --input-dir outputs/swe_qa_gemma4_12b_n1000_seed0
```

The `oracle` and `noisy_oracle` splits contain the same questions with
different context construction, so they should not be treated as independent
evaluation sets.

## Disjoint 2,000-Question MMLU-Pro Expansion

The expansion launcher selects 2,000 MMLU-Pro test questions balanced across
the 14 categories after excluding the exact 1,000-question balanced seed-0
selection used above. The selection seed and generation seed are recorded
separately in the launcher, and the resulting probe summary stores all 2,000
explicit row indices.

Inspect the selection without starting the model:

```bash
experiments/run_mmlu_pro_n2000_disjoint.sh --plan-only
```

Start the run:

```bash
experiments/run_mmlu_pro_n2000_disjoint.sh
```

The output is written to
`outputs/mmlu_pro_gemma4_12b_n2000_disjoint_seed0/`. The run uses the completed
32,768-token trace budget and 49,152-token model context used by the extended
baseline, avoiding a separate trace-extension pass for most questions.

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

Two follow-up analyses probe the uncertainty-blip explanation more directly.
The confidence-threshold recurrence analysis reruns the matched-control
recurrence tables while requiring a high normalized correct-answer probability
at a pre-final checkpoint, and optionally a high normalized final wrong-answer
probability:

```bash
trajectory-analyze-confidence-recurrence \
  --input-root outputs/matched_controls_gemma4_12b_extended
```

The early-commitment analysis estimates recoverable accuracy if traces had
stopped at earlier checkpoints. Oracle policies use the true answer and are
upper bounds, while the proxy policy uses only prediction confidence and
stability:

```bash
trajectory-analyze-early-commitment \
  --input-root outputs/matched_controls_gemma4_12b_extended
```

The branching-intervention setup selects high-confidence currently-correct
checkpoints from traces that later go wrong and writes a branch request
manifest. By default this is a dry run that does not load the model; add
`--execute` to generate answer-only and continuation branches:

```bash
trajectory-run-branching-intervention \
  --input-root outputs/matched_controls_gemma4_12b_extended \
  --selection outputs/matched_controls_gemma4_12b/cohort_selection.parquet \
  --output-dir outputs/branching_intervention_gemma4_12b
```

For cross-model replication, first write a model-specific command plan. The
current runner is Gemma-specific, so non-Gemma models are listed as blocked
until a prompt/logit adapter is added or `--adapter gemma` is explicitly used
for a compatible checkpoint:

```bash
trajectory-plan-cross-model-replication \
  --models google/gemma-4-12B-it \
  --output-root outputs/cross_model_replication
```

Activation probes train layerwise linear classifiers for the deployable
correctness label `current_correct` and diagnostic future-risk labels:
`future_loss`, `future_change_to_wrong`, and `future_answer_flip`. The cheap
examples stage builds labels from existing matched-control trajectories; the
full stage also extracts hidden activations from Gemma and trains out-of-fold
probes for halting-policy evaluation:

```bash
trajectory-train-activation-probes \
  --input-root outputs/matched_controls_gemma4_12b_extended \
  --output-dir outputs/activation_probe_gemma4_12b
```

To train a diagnostic probe specifically on robust-loss versus no-loss
checkpoints from a single trajectory run, retain only checkpoints where the
model is currently correct and use decile-matched negative sampling:

```bash
trajectory-train-activation-probes \
  --input-root outputs/mmlu_pro_gemma4_12b_n2000_disjoint_seed0 \
  --output-dir outputs/activation_probe_mmlu_pro_n2000_robust_loss \
  --example-cohort robust_loss_vs_no_loss \
  --cohort-negative-ratio 3 \
  --targets robust_loss_case \
  --layers 0,4,8,12,16,20,24,28,32,36,40,44,48 \
  --epochs 16
```

This keeps every currently-correct checkpoint from a robust-loss trace, draws
three currently-correct no-loss checkpoints per positive within each decile,
and excludes broad losses that do not pass the robust-loss filters. Cross
validation remains grouped by question.

For an intervention evaluation, use one deployable candidate per question:
the earliest checkpoint whose normalized answer confidence is at least 0.9.
Unlike the diagnostic cohort, this retains currently-wrong checkpoints so the
reported stopping accuracy includes prevented self-corrections:

```bash
trajectory-train-activation-probes \
  --input-root outputs/mmlu_pro_gemma4_12b_n2000_disjoint_seed0 \
  --output-dir outputs/activation_probe_mmlu_pro_n2000_intervention \
  --example-cohort intervention_candidates \
  --intervention-confidence-threshold 0.9 \
  --targets robust_stop_candidate \
  --layers 0,4,8,12,16,20,24,28,32,36,40,44,48 \
  --epochs 16
```

The resulting `probe_halting_summary.parquet` reports question-grouped
out-of-fold accuracy, stop rate, beneficial stops, and harmful stops. Questions
without a qualifying checkpoint automatically continue to the final answer.

To separate "is the current answer correct?" from "will continued reasoning
lose it?", train the two-head intervention on the extracted candidates:

```bash
trajectory-two-head-intervention \
  --stage train \
  --source-dir outputs/activation_probe_mmlu_pro_n2000_intervention \
  --output-dir outputs/two_head_intervention_mmlu_pro_n2000 \
  --layers 0,4,8,12,16,20,24,28,32,36,40,44,48
```

Both thresholds must pass before stopping. Apply a policy to a separately
extracted candidate cache with:

```bash
trajectory-two-head-intervention \
  --stage apply \
  --output-dir outputs/two_head_intervention_mmlu_pro_n2000 \
  --evaluation-dir outputs/activation_probe_mmlu_pro_n1000_intervention_eval_layer16 \
  --output-path outputs/activation_probe_mmlu_pro_n1000_intervention_eval_layer16/external_two_head_safety_summary.json \
  --layer 16 \
  --correctness-threshold 0.7 \
  --loss-threshold 0.7
```

Large activation caches can be stored as `activations_shards/manifest.json`
plus row-wise `.npy` shards; the training loader accepts this layout whenever
the monolithic `activations.npy` is absent.

The implementation follows the paper's public
[reference repository](https://github.com/AndresAlgaba/probing_reasoning_traces)
where applicable, with Gemma-specific prompting and parsing added here.
