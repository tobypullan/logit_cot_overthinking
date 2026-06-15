# Matched-control overthinking experiment

Each dataset contributes 25 seed-0 loss questions, 25 final-correct controls, and 25 never-correct controls, matched within category and by seed-0 trace length. Every question was rerun at seeds 0 through 9.

## Cohort recurrence

| Dataset | Seed-0 cohort | Attempts | Broad losses | Rate | Final accuracy | Mean flips |
|---|---|---:|---:|---:|---:|---:|
| GPQA Diamond | final correct | 250 | 44 | 17.6% | 78.8% | 2.34 |
| GPQA Diamond | loss | 250 | 117 | 46.8% | 43.6% | 3.37 |
| GPQA Diamond | stable wrong | 250 | 40 | 16.0% | 21.6% | 2.48 |
| MMLU-Pro | final correct | 250 | 13 | 5.2% | 86.4% | 2.00 |
| MMLU-Pro | loss | 250 | 117 | 46.8% | 45.6% | 2.92 |
| MMLU-Pro | stable wrong | 250 | 47 | 18.8% | 15.2% | 2.56 |

Matched risk differences use a 5,000-sample bootstrap over the 25 matched triplets:

- **Gpqa Diamond loss vs final correct:** +29.2% (95% CI +15.6% to +43.2%).
- **Gpqa Diamond loss vs stable wrong:** +30.8% (95% CI +16.8% to +44.8%).
- **Mmlu Pro loss vs final correct:** +41.6% (95% CI +31.2% to +52.4%).
- **Mmlu Pro loss vs stable wrong:** +28.0% (95% CI +15.2% to +40.8%).

## Prediction while currently correct

AUCs use five-fold cross-validation grouped by question, so seeds of the same question cannot appear in both train and test folds.

| Dataset | Checkpoint | Confidence AUC | Instability AUC | + length AUC | + prefix language AUC |
|---|---:|---:|---:|---:|---:|
| Combined | 50% | 0.732 | 0.731 | 0.764 | 0.755 |
| Combined | 80% | 0.718 | 0.817 | 0.810 | 0.812 |
| Mmlu Pro | 50% | 0.667 | 0.698 | 0.701 | 0.678 |
| Mmlu Pro | 80% | 0.697 | 0.853 | 0.857 | 0.851 |
| Gpqa Diamond | 50% | 0.670 | 0.731 | 0.732 | 0.731 |
| Gpqa Diamond | 80% | 0.585 | 0.755 | 0.748 | 0.762 |

## Strongest late-stage signals

- `normalized_confidence_decline`: standardized coefficient +0.334.
- `prefix_repetition_score`: standardized coefficient +0.308.
- `prefix_self_correction`: standardized coefficient +0.272.
- `flips_so_far`: standardized coefficient +0.203.

The coefficients describe association within the matched sample, not a causal effect. Trace length is also an oracle feature because the eventual total length is unknown at an online stopping point.
