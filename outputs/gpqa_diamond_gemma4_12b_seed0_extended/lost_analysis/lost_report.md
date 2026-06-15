# Lost-case analysis

Complete traces only.

## Main counts

- Ever correct, final wrong: 57
- Endpoint lost (correct at 0%, wrong at 100%): 11
- Gained then lost: 46
- Confidence-filtered robust losses: 2
- Normalized reversal candidates: 15
- Broad losses ending with a correct generated answer: 34
- Broad losses ending with an explicit wrong generated answer: 22
- Robust losses hidden by the endpoint label: 2

## Quantitative observations

- 30 cases were still correct at 80% or 90% of the trace before ending wrong.
- Lost cases had a median 8822 reasoning tokens versus 5588 for final-correct cases.
- Lost cases had a median 4 prediction flips versus 2 for final-correct cases.
- The median final valid-letter probability mass was only 0.23%; probe and generated answer agreed in 36.8% of broad losses.
- 2 of 2 robust traces used simulated retrieval language such as searching for a source, test bank, or exact answer.
- Highest broad loss rates: gpqa-diamond 28.8%.
- 2 of 2 robust losses came from gpqa-diamond.
- The leading broad-loss categories had no robust cases: . This indicates that low final answer-letter mass explains much of their apparent loss rate.

## Robust cases

| Question | Category | Change | Last correct | Peak correct p | Tokens |
|---:|---|---:|---:|---:|---:|
| gpqa-diamond-091 | gpqa-diamond | C->B | 80% | 0.560 | 25,023 |
| gpqa-diamond-018 | gpqa-diamond | D->A | 70% | 0.540 | 32,760 |

The robust filter requires high pre-final raw answer probability, substantial final probability mass on valid letters, and agreement between the final probe and the generated answer.

## Normalized reversal candidates

| Question | Change | First correct | Last correct | Peak raw p | Final letter mass |
|---:|---:|---:|---:|---:|---:|
| gpqa-diamond-091 | C->B | 20% | 80% | 0.560 | 0.575 |
| gpqa-diamond-095 | A->B | 0% | 90% | 0.979 | 0.040 |
| gpqa-diamond-158 | A->B | 0% | 90% | 0.809 | 0.014 |
| gpqa-diamond-116 | A->B | 0% | 90% | 0.265 | 0.022 |
| gpqa-diamond-101 | D->C | 10% | 80% | 0.989 | 0.050 |
| gpqa-diamond-065 | A->C | 0% | 70% | 0.021 | 0.851 |
| gpqa-diamond-119 | C->B | 10% | 60% | 0.881 | 0.008 |
| gpqa-diamond-096 | B->A | 60% | 60% | 0.156 | 0.517 |
| gpqa-diamond-100 | A->C | 0% | 60% | 0.040 | 0.977 |
| gpqa-diamond-109 | A->B | 0% | 50% | 0.036 | 0.988 |
| gpqa-diamond-097 | A->C | 0% | 50% | 0.006 | 0.903 |
| gpqa-diamond-107 | A->C | 0% | 20% | 0.273 | 0.901 |
| gpqa-diamond-070 | B->C | 10% | 10% | 0.858 | 0.008 |
| gpqa-diamond-087 | C->A | 10% | 10% | 0.799 | 0.003 |
| gpqa-diamond-133 | A->C | 0% | 0% | 0.081 | 0.046 |

Normalized candidates require at least 90% of valid-letter probability on the correct answer before the end, at least 90% on the final wrong answer at the end, and agreement with the generated answer. They can still have low absolute answer-letter probability.
