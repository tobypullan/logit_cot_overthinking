# Early-commitment intervention analysis

This analysis asks whether an answer would improve if the trace were stopped at an earlier checkpoint.

`oracle_first_correct` and `threshold_first_*` use the true answer or correctness while choosing a checkpoint. They are upper bounds for recoverable accuracy, not deployable policies.

## Policy summary

| Dataset | Baseline cohort | Policy | Attempts | Accuracy | Delta vs final | Stop rate | Median stop decile |
|---|---|---|---:|---:|---:|---:|---:|
| gpqa_diamond | stable_wrong | `final` | 250 | 21.6% | +0.0% | 0.0% | 100 |
| gpqa_diamond | stable_wrong | `oracle_first_correct` | 250 | 37.6% | +16.0% | 37.6% | 100 |
| gpqa_diamond | stable_wrong | `threshold_first_0.5` | 250 | 37.2% | +15.6% | 37.2% | 100 |
| gpqa_diamond | stable_wrong | `threshold_first_0.7` | 250 | 33.6% | +12.0% | 32.8% | 100 |
| gpqa_diamond | stable_wrong | `threshold_first_0.9` | 250 | 30.8% | +9.2% | 28.8% | 100 |
| gpqa_diamond | stable_wrong | `proxy_confidence_streak_0.9_s2` | 250 | 5.2% | -16.4% | 99.2% | 10 |
| gpqa_diamond | loss | `final` | 250 | 43.6% | +0.0% | 0.0% | 100 |
| gpqa_diamond | loss | `oracle_first_correct` | 250 | 90.4% | +46.8% | 89.2% | 10 |
| gpqa_diamond | loss | `threshold_first_0.5` | 250 | 89.6% | +46.0% | 88.4% | 20 |
| gpqa_diamond | loss | `threshold_first_0.7` | 250 | 85.2% | +41.6% | 83.6% | 20 |
| gpqa_diamond | loss | `threshold_first_0.9` | 250 | 80.4% | +36.8% | 75.6% | 30 |
| gpqa_diamond | loss | `proxy_confidence_streak_0.9_s2` | 250 | 44.0% | +0.4% | 96.8% | 20 |
| gpqa_diamond | final_correct | `final` | 250 | 78.8% | +0.0% | 0.0% | 100 |
| gpqa_diamond | final_correct | `oracle_first_correct` | 250 | 96.4% | +17.6% | 96.4% | 0 |
| gpqa_diamond | final_correct | `threshold_first_0.5` | 250 | 96.4% | +17.6% | 96.4% | 0 |
| gpqa_diamond | final_correct | `threshold_first_0.7` | 250 | 95.6% | +16.8% | 95.2% | 0 |
| gpqa_diamond | final_correct | `threshold_first_0.9` | 250 | 93.2% | +14.4% | 90.8% | 10 |
| gpqa_diamond | final_correct | `proxy_confidence_streak_0.9_s2` | 250 | 76.0% | -2.8% | 96.8% | 20 |
| mmlu_pro | loss | `final` | 250 | 45.6% | +0.0% | 0.0% | 100 |
| mmlu_pro | loss | `oracle_first_correct` | 250 | 92.4% | +46.8% | 92.4% | 10 |
| mmlu_pro | loss | `threshold_first_0.5` | 250 | 92.0% | +46.4% | 92.0% | 10 |
| mmlu_pro | loss | `threshold_first_0.7` | 250 | 90.0% | +44.4% | 89.6% | 10 |
| mmlu_pro | loss | `threshold_first_0.9` | 250 | 82.4% | +36.8% | 81.6% | 20 |
| mmlu_pro | loss | `proxy_confidence_streak_0.9_s2` | 250 | 47.2% | +1.6% | 96.0% | 20 |
| mmlu_pro | stable_wrong | `final` | 250 | 15.2% | +0.0% | 0.0% | 100 |
| mmlu_pro | stable_wrong | `oracle_first_correct` | 250 | 34.0% | +18.8% | 33.6% | 100 |
| mmlu_pro | stable_wrong | `threshold_first_0.5` | 250 | 32.8% | +17.6% | 32.4% | 100 |
| mmlu_pro | stable_wrong | `threshold_first_0.7` | 250 | 29.6% | +14.4% | 28.8% | 100 |
| mmlu_pro | stable_wrong | `threshold_first_0.9` | 250 | 26.0% | +10.8% | 24.8% | 100 |
| mmlu_pro | stable_wrong | `proxy_confidence_streak_0.9_s2` | 250 | 14.0% | -1.2% | 94.4% | 20 |
| mmlu_pro | final_correct | `final` | 250 | 86.4% | +0.0% | 0.0% | 100 |
| mmlu_pro | final_correct | `oracle_first_correct` | 250 | 91.6% | +5.2% | 91.6% | 10 |
| mmlu_pro | final_correct | `threshold_first_0.5` | 250 | 91.2% | +4.8% | 91.2% | 10 |
| mmlu_pro | final_correct | `threshold_first_0.7` | 250 | 91.2% | +4.8% | 91.2% | 10 |
| mmlu_pro | final_correct | `threshold_first_0.9` | 250 | 90.8% | +4.4% | 90.4% | 10 |
| mmlu_pro | final_correct | `proxy_confidence_streak_0.9_s2` | 250 | 82.0% | -4.4% | 99.2% | 20 |

## Broad-loss slice

Broad-loss attempts are those that were correct at some checkpoint but wrong at the final checkpoint.

| Dataset | Baseline cohort | Broad loss | Policy | Attempts | Accuracy | Delta vs final | Stop rate | Median stop decile |
|---|---|---:|---|---:|---:|---:|---:|---:|
| gpqa_diamond | stable_wrong | True | `final` | 40 | 0.0% | +0.0% | 0.0% | 100 |
| gpqa_diamond | stable_wrong | True | `oracle_first_correct` | 40 | 100.0% | +100.0% | 100.0% | 40 |
| gpqa_diamond | stable_wrong | True | `threshold_first_0.5` | 40 | 97.5% | +97.5% | 97.5% | 40 |
| gpqa_diamond | stable_wrong | True | `threshold_first_0.7` | 40 | 75.0% | +75.0% | 75.0% | 65 |
| gpqa_diamond | stable_wrong | True | `threshold_first_0.9` | 40 | 57.5% | +57.5% | 57.5% | 80 |
| gpqa_diamond | stable_wrong | True | `proxy_confidence_streak_0.9_s2` | 40 | 10.0% | +10.0% | 97.5% | 20 |
| gpqa_diamond | loss | True | `final` | 117 | 0.0% | +0.0% | 0.0% | 100 |
| gpqa_diamond | loss | True | `oracle_first_correct` | 117 | 100.0% | +100.0% | 100.0% | 10 |
| gpqa_diamond | loss | True | `threshold_first_0.5` | 117 | 98.3% | +98.3% | 98.3% | 10 |
| gpqa_diamond | loss | True | `threshold_first_0.7` | 117 | 88.9% | +88.9% | 88.9% | 20 |
| gpqa_diamond | loss | True | `threshold_first_0.9` | 117 | 78.6% | +78.6% | 78.6% | 30 |
| gpqa_diamond | loss | True | `proxy_confidence_streak_0.9_s2` | 117 | 35.0% | +35.0% | 94.9% | 20 |
| gpqa_diamond | final_correct | False | `final` | 206 | 95.6% | +0.0% | 0.0% | 100 |
| gpqa_diamond | final_correct | False | `oracle_first_correct` | 206 | 95.6% | +0.0% | 95.6% | 0 |
| gpqa_diamond | final_correct | False | `threshold_first_0.5` | 206 | 95.6% | +0.0% | 95.6% | 0 |
| gpqa_diamond | final_correct | False | `threshold_first_0.7` | 206 | 95.6% | +0.0% | 95.1% | 0 |
| gpqa_diamond | final_correct | False | `threshold_first_0.9` | 206 | 95.6% | +0.0% | 92.7% | 0 |
| gpqa_diamond | final_correct | False | `proxy_confidence_streak_0.9_s2` | 206 | 82.0% | -13.6% | 98.1% | 20 |
| gpqa_diamond | loss | False | `final` | 133 | 82.0% | +0.0% | 0.0% | 100 |
| gpqa_diamond | loss | False | `oracle_first_correct` | 133 | 82.0% | +0.0% | 79.7% | 20 |
| gpqa_diamond | loss | False | `threshold_first_0.5` | 133 | 82.0% | +0.0% | 79.7% | 20 |
| gpqa_diamond | loss | False | `threshold_first_0.7` | 133 | 82.0% | +0.0% | 78.9% | 20 |
| gpqa_diamond | loss | False | `threshold_first_0.9` | 133 | 82.0% | +0.0% | 72.9% | 20 |
| gpqa_diamond | loss | False | `proxy_confidence_streak_0.9_s2` | 133 | 51.9% | -30.1% | 98.5% | 20 |
| gpqa_diamond | stable_wrong | False | `final` | 210 | 25.7% | +0.0% | 0.0% | 100 |
| gpqa_diamond | stable_wrong | False | `oracle_first_correct` | 210 | 25.7% | +0.0% | 25.7% | 100 |
| gpqa_diamond | stable_wrong | False | `threshold_first_0.5` | 210 | 25.7% | +0.0% | 25.7% | 100 |
| gpqa_diamond | stable_wrong | False | `threshold_first_0.7` | 210 | 25.7% | +0.0% | 24.8% | 100 |
| gpqa_diamond | stable_wrong | False | `threshold_first_0.9` | 210 | 25.7% | +0.0% | 23.3% | 100 |
| gpqa_diamond | stable_wrong | False | `proxy_confidence_streak_0.9_s2` | 210 | 4.3% | -21.4% | 99.5% | 10 |
| gpqa_diamond | final_correct | True | `final` | 44 | 0.0% | +0.0% | 0.0% | 100 |
| gpqa_diamond | final_correct | True | `oracle_first_correct` | 44 | 100.0% | +100.0% | 100.0% | 20 |
| gpqa_diamond | final_correct | True | `threshold_first_0.5` | 44 | 100.0% | +100.0% | 100.0% | 20 |
| gpqa_diamond | final_correct | True | `threshold_first_0.7` | 44 | 95.5% | +95.5% | 95.5% | 20 |
| gpqa_diamond | final_correct | True | `threshold_first_0.9` | 44 | 81.8% | +81.8% | 81.8% | 20 |
| gpqa_diamond | final_correct | True | `proxy_confidence_streak_0.9_s2` | 44 | 47.7% | +47.7% | 90.9% | 30 |
| mmlu_pro | loss | True | `final` | 117 | 0.0% | +0.0% | 0.0% | 100 |
| mmlu_pro | loss | True | `oracle_first_correct` | 117 | 100.0% | +100.0% | 100.0% | 10 |
| mmlu_pro | loss | True | `threshold_first_0.5` | 117 | 99.1% | +99.1% | 99.1% | 10 |
| mmlu_pro | loss | True | `threshold_first_0.7` | 117 | 94.9% | +94.9% | 94.9% | 10 |
| mmlu_pro | loss | True | `threshold_first_0.9` | 117 | 78.6% | +78.6% | 78.6% | 20 |
| mmlu_pro | loss | True | `proxy_confidence_streak_0.9_s2` | 117 | 29.1% | +29.1% | 93.2% | 20 |
| mmlu_pro | stable_wrong | False | `final` | 203 | 18.7% | +0.0% | 0.0% | 100 |
| mmlu_pro | stable_wrong | False | `oracle_first_correct` | 203 | 18.7% | +0.0% | 18.2% | 100 |
| mmlu_pro | stable_wrong | False | `threshold_first_0.5` | 203 | 18.7% | +0.0% | 18.2% | 100 |
| mmlu_pro | stable_wrong | False | `threshold_first_0.7` | 203 | 18.7% | +0.0% | 17.7% | 100 |
| mmlu_pro | stable_wrong | False | `threshold_first_0.9` | 203 | 18.7% | +0.0% | 17.2% | 100 |
| mmlu_pro | stable_wrong | False | `proxy_confidence_streak_0.9_s2` | 203 | 12.8% | -5.9% | 96.1% | 20 |
| mmlu_pro | final_correct | False | `final` | 237 | 91.1% | +0.0% | 0.0% | 100 |
| mmlu_pro | final_correct | False | `oracle_first_correct` | 237 | 91.1% | +0.0% | 91.1% | 10 |
| mmlu_pro | final_correct | False | `threshold_first_0.5` | 237 | 91.1% | +0.0% | 91.1% | 10 |
| mmlu_pro | final_correct | False | `threshold_first_0.7` | 237 | 91.1% | +0.0% | 91.1% | 10 |
| mmlu_pro | final_correct | False | `threshold_first_0.9` | 237 | 91.1% | +0.0% | 90.7% | 10 |
| mmlu_pro | final_correct | False | `proxy_confidence_streak_0.9_s2` | 237 | 84.0% | -7.2% | 99.2% | 20 |
| mmlu_pro | loss | False | `final` | 133 | 85.7% | +0.0% | 0.0% | 100 |
| mmlu_pro | loss | False | `oracle_first_correct` | 133 | 85.7% | +0.0% | 85.7% | 10 |
| mmlu_pro | loss | False | `threshold_first_0.5` | 133 | 85.7% | +0.0% | 85.7% | 10 |
| mmlu_pro | loss | False | `threshold_first_0.7` | 133 | 85.7% | +0.0% | 85.0% | 20 |
| mmlu_pro | loss | False | `threshold_first_0.9` | 133 | 85.7% | +0.0% | 84.2% | 20 |
| mmlu_pro | loss | False | `proxy_confidence_streak_0.9_s2` | 133 | 63.2% | -22.6% | 98.5% | 20 |
| mmlu_pro | stable_wrong | True | `final` | 47 | 0.0% | +0.0% | 0.0% | 100 |
| mmlu_pro | stable_wrong | True | `oracle_first_correct` | 47 | 100.0% | +100.0% | 100.0% | 20 |
| mmlu_pro | stable_wrong | True | `threshold_first_0.5` | 47 | 93.6% | +93.6% | 93.6% | 30 |
| mmlu_pro | stable_wrong | True | `threshold_first_0.7` | 47 | 76.6% | +76.6% | 76.6% | 30 |
| mmlu_pro | stable_wrong | True | `threshold_first_0.9` | 47 | 57.4% | +57.4% | 57.4% | 80 |
| mmlu_pro | stable_wrong | True | `proxy_confidence_streak_0.9_s2` | 47 | 19.1% | +19.1% | 87.2% | 30 |
| mmlu_pro | final_correct | True | `final` | 13 | 0.0% | +0.0% | 0.0% | 100 |
| mmlu_pro | final_correct | True | `oracle_first_correct` | 13 | 100.0% | +100.0% | 100.0% | 10 |
| mmlu_pro | final_correct | True | `threshold_first_0.5` | 13 | 92.3% | +92.3% | 92.3% | 10 |
| mmlu_pro | final_correct | True | `threshold_first_0.7` | 13 | 92.3% | +92.3% | 92.3% | 10 |
| mmlu_pro | final_correct | True | `threshold_first_0.9` | 13 | 84.6% | +84.6% | 84.6% | 10 |
| mmlu_pro | final_correct | True | `proxy_confidence_streak_0.9_s2` | 13 | 46.2% | +46.2% | 100.0% | 30 |

## Interpretation

- `final` is the observed baseline endpoint.
- `oracle_first_correct` estimates the maximum accuracy recoverable by stopping exactly when an attempt first becomes correct.
- `threshold_first_*` asks whether high normalized true-answer probability would have been enough to stop earlier; it still uses correctness labels and is therefore an upper bound.
- `proxy_confidence_streak_*` uses only the predicted answer's normalized probability and prediction stability, then evaluates whether the chosen prediction was correct.
