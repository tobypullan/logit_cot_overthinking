# Ten-seed candidate rerun analysis

All capped attempts were extended by up to 16,384 additional tokens. Runaway traces that still did not close were explicitly closed and are marked as forced completions.

## Dataset-level results

### MMLU-Pro

- 42 of 170 attempts reproduced the robust loss criterion (24.7%).
- Excluding forced completions: 42 of 170 (24.7%).
- 16 of 17 candidates reproduced at least once; 3 reproduced in at least half of seeds.
- 0 attempts required explicit thought-channel closure.
- Final probe accuracy across this selected set: 44.1%.

### GPQA Diamond

- 10 of 80 attempts reproduced the normalized reversal criterion (12.5%).
- Excluding forced completions: 10 of 80 (12.5%).
- 6 of 8 candidates reproduced at least once; 0 reproduced in at least half of seeds.
- 0 attempts required explicit thought-channel closure.
- Final probe accuracy across this selected set: 46.2%.

## Most recurrent candidates

| Dataset | Position | Question ID | Criterion seeds | Ever-correct/final-wrong | Final correct | Forced |
|---|---:|---|---:|---:|---:|---:|
| MMLU-Pro | 1049 | 1130 | 7/10 | 10/10 | 0/10 | 0/10 |
| MMLU-Pro | 4802 | 4941 | 5/10 | 6/10 | 4/10 | 0/10 |
| MMLU-Pro | 589 | 665 | 5/10 | 5/10 | 4/10 | 0/10 |
| GPQA Diamond | 107 | gpqa-diamond-107 | 4/10 | 8/10 | 2/10 | 0/10 |
| MMLU-Pro | 46 | 116 | 3/10 | 3/10 | 5/10 | 0/10 |
| MMLU-Pro | 2099 | 2199 | 3/10 | 3/10 | 4/10 | 0/10 |
| MMLU-Pro | 4671 | 4809 | 3/10 | 3/10 | 3/10 | 0/10 |
| MMLU-Pro | 11019 | 11240 | 3/10 | 3/10 | 7/10 | 0/10 |
| GPQA Diamond | 119 | gpqa-diamond-119 | 2/10 | 7/10 | 0/10 | 0/10 |
| MMLU-Pro | 1099 | 1181 | 2/10 | 7/10 | 0/10 | 0/10 |
| MMLU-Pro | 1395 | 1486 | 2/10 | 2/10 | 3/10 | 0/10 |
| MMLU-Pro | 2603 | 2714 | 2/10 | 2/10 | 7/10 | 0/10 |
| MMLU-Pro | 10953 | 11174 | 2/10 | 2/10 | 1/10 | 0/10 |
| GPQA Diamond | 70 | gpqa-diamond-070 | 1/10 | 7/10 | 0/10 | 0/10 |
| MMLU-Pro | 1806 | 1902 | 1/10 | 6/10 | 0/10 | 0/10 |

## Specific findings

- **MMLU-Pro:** the strongest candidate was `1130` at position 1049, reproducing in 7/10 seeds and ending wrong after being correct in 10/10.
- **MMLU-Pro:** 1 seed-0 candidate(s) never reproduced the target criterion: `6980`.
- **GPQA Diamond:** the strongest candidate was `gpqa-diamond-107` at position 107, reproducing in 4/10 seeds and ending wrong after being correct in 8/10.
- **GPQA Diamond:** 2 seed-0 candidate(s) never reproduced the target criterion: `gpqa-diamond-158`, `gpqa-diamond-095`.
- MMLU-Pro recurrence was associated much more strongly with prediction instability than trace length: target cases averaged about twice as many flips, while median trace length was similar.
- GPQA target reversals were concentrated in longer traces: their median reasoning length was roughly twice that of other candidate attempts.

## Seed-0 replay check

- **MMLU-Pro:** 0/17 traces matched the original seed-0 generation exactly; 8/17 retained the same final prediction.
- **GPQA Diamond:** 0/8 traces matched the original seed-0 generation exactly; 4/8 retained the same final prediction.
- Seed alone is therefore insufficient for bitwise replay under the changed batched scheduling. These runs are best treated as stochastic repeats rather than deterministic reproductions.

## Interpretation

- The seed-0 selection enriches for reversals, but most individual events are not deterministic across sampling seeds.
- Candidates with repeated reversals are stronger evidence of question-specific overthinking than one-off seed outcomes.
- Forced completions are retained for coverage but should be reported separately because their terminal answer was elicited by closing a runaway thought channel.
