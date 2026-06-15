# Logit CoT Overthinking

This repository implements the three-stage logit trajectory probing protocol
from [Probing the Trajectories of Reasoning Traces in Large Language
Models](https://arxiv.org/abs/2601.23163):

1. Generate a complete reasoning trace.
2. Slice it at token deciles from 0% through 100%.
3. Reinject each prefix and measure the next-token distribution over the valid
   answer letters.

The initial implementation targets MMLU-Pro and
[`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it).
Gemma's thought-channel format is handled explicitly, and MMLU-Pro's variable
number of choices is preserved per question.

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
- `runtime_and_trace_lengths.png`: trace lengths and cached runtime by stage.

## MMLU-Pro 1,000-Question Run

Step 2 uses a deterministic category-balanced sample. With 1,000 questions,
the 14 MMLU-Pro categories contribute either 71 or 72 questions each. Selected
rows retain their original test-split order.

```bash
trajectory-probe \
  --selection balanced-categories \
  --num-rows 1000 \
  --seed 0 \
  --trace-max-tokens 4096 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --output-dir outputs/mmlu_pro_gemma4_12b_n1000_seed0
```

Generate figures after the run:

```bash
trajectory-visualize \
  --input-dir outputs/mmlu_pro_gemma4_12b_n1000_seed0
```

The implementation follows the paper's public
[reference repository](https://github.com/AndresAlgaba/probing_reasoning_traces)
where applicable, with Gemma-specific prompting and parsing added here.
