# Lost-case analysis

Complete traces only.

## Main counts

- Ever correct, final wrong: 45
- Endpoint lost (correct at 0%, wrong at 100%): 6
- Gained then lost: 39
- Confidence-filtered robust losses: 0
- Normalized reversal candidates: 8
- Broad losses ending with a correct generated answer: 32
- Broad losses ending with an explicit wrong generated answer: 12
- Robust losses hidden by the endpoint label: 0

## Quantitative observations

- 27 cases were still correct at 80% or 90% of the trace before ending wrong.
- Lost cases had a median 7845 reasoning tokens versus 3515 for final-correct cases.
- Lost cases had a median 4 prediction flips versus 2 for final-correct cases.
- The median final valid-letter probability mass was only 0.17%; probe and generated answer agreed in 26.7% of broad losses.
- Highest broad loss rates: gpqa-diamond 28.7%.
- The leading broad-loss categories had no robust cases: gpqa-diamond. This indicates that low final answer-letter mass explains much of their apparent loss rate.

## Robust cases

| Question | Category | Change | Last correct | Peak correct p | Tokens |
|---:|---|---:|---:|---:|---:|

The robust filter requires high pre-final raw answer probability, substantial final probability mass on valid letters, and agreement between the final probe and the generated answer.

## Normalized reversal candidates

| Question | Change | First correct | Last correct | Peak raw p | Final letter mass |
|---:|---:|---:|---:|---:|---:|
| gpqa-diamond-095 | A->B | 0% | 90% | 0.979 | 0.040 |
| gpqa-diamond-158 | A->B | 0% | 90% | 0.809 | 0.014 |
| gpqa-diamond-101 | D->C | 10% | 80% | 0.989 | 0.050 |
| gpqa-diamond-119 | C->B | 10% | 60% | 0.881 | 0.008 |
| gpqa-diamond-107 | A->C | 0% | 20% | 0.273 | 0.901 |
| gpqa-diamond-070 | B->C | 10% | 10% | 0.858 | 0.008 |
| gpqa-diamond-087 | C->A | 10% | 10% | 0.799 | 0.003 |
| gpqa-diamond-133 | A->C | 0% | 0% | 0.081 | 0.046 |

Normalized candidates require at least 90% of valid-letter probability on the correct answer before the end, at least 90% on the final wrong answer at the end, and agreement with the generated answer. They can still have low absolute answer-letter probability.
