# Confidence-threshold recurrence analysis

Qualified losses require a pre-final correct probe with normalized correct-answer probability at or above the listed threshold, a wrong final answer, and the listed minimum final normalized prediction probability.

## Cohort summary

| Dataset | Seed-0 cohort | Correct threshold | Final threshold | Attempts | Qualified losses | Rate | Final accuracy | Forced completions |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GPQA Diamond | final correct | 0.5 | 0.0 | 250 | 44 | 17.6% | 78.8% | 15 |
| GPQA Diamond | final correct | 0.5 | 0.7 | 250 | 35 | 14.0% | 78.8% | 15 |
| GPQA Diamond | final correct | 0.5 | 0.9 | 250 | 16 | 6.4% | 78.8% | 15 |
| GPQA Diamond | final correct | 0.7 | 0.0 | 250 | 42 | 16.8% | 78.8% | 15 |
| GPQA Diamond | final correct | 0.7 | 0.7 | 250 | 33 | 13.2% | 78.8% | 15 |
| GPQA Diamond | final correct | 0.7 | 0.9 | 250 | 15 | 6.0% | 78.8% | 15 |
| GPQA Diamond | final correct | 0.9 | 0.0 | 250 | 36 | 14.4% | 78.8% | 15 |
| GPQA Diamond | final correct | 0.9 | 0.7 | 250 | 29 | 11.6% | 78.8% | 15 |
| GPQA Diamond | final correct | 0.9 | 0.9 | 250 | 15 | 6.0% | 78.8% | 15 |
| GPQA Diamond | loss | 0.5 | 0.0 | 250 | 115 | 46.0% | 43.6% | 0 |
| GPQA Diamond | loss | 0.5 | 0.7 | 250 | 102 | 40.8% | 43.6% | 0 |
| GPQA Diamond | loss | 0.5 | 0.9 | 250 | 80 | 32.0% | 43.6% | 0 |
| GPQA Diamond | loss | 0.7 | 0.0 | 250 | 104 | 41.6% | 43.6% | 0 |
| GPQA Diamond | loss | 0.7 | 0.7 | 250 | 92 | 36.8% | 43.6% | 0 |
| GPQA Diamond | loss | 0.7 | 0.9 | 250 | 71 | 28.4% | 43.6% | 0 |
| GPQA Diamond | loss | 0.9 | 0.0 | 250 | 92 | 36.8% | 43.6% | 0 |
| GPQA Diamond | loss | 0.9 | 0.7 | 250 | 81 | 32.4% | 43.6% | 0 |
| GPQA Diamond | loss | 0.9 | 0.9 | 250 | 60 | 24.0% | 43.6% | 0 |
| GPQA Diamond | stable wrong | 0.5 | 0.0 | 250 | 39 | 15.6% | 21.6% | 9 |
| GPQA Diamond | stable wrong | 0.5 | 0.7 | 250 | 32 | 12.8% | 21.6% | 9 |
| GPQA Diamond | stable wrong | 0.5 | 0.9 | 250 | 22 | 8.8% | 21.6% | 9 |
| GPQA Diamond | stable wrong | 0.7 | 0.0 | 250 | 30 | 12.0% | 21.6% | 9 |
| GPQA Diamond | stable wrong | 0.7 | 0.7 | 250 | 24 | 9.6% | 21.6% | 9 |
| GPQA Diamond | stable wrong | 0.7 | 0.9 | 250 | 17 | 6.8% | 21.6% | 9 |
| GPQA Diamond | stable wrong | 0.9 | 0.0 | 250 | 23 | 9.2% | 21.6% | 9 |
| GPQA Diamond | stable wrong | 0.9 | 0.7 | 250 | 19 | 7.6% | 21.6% | 9 |
| GPQA Diamond | stable wrong | 0.9 | 0.9 | 250 | 13 | 5.2% | 21.6% | 9 |
| MMLU-Pro | final correct | 0.5 | 0.0 | 250 | 12 | 4.8% | 86.4% | 0 |
| MMLU-Pro | final correct | 0.5 | 0.7 | 250 | 6 | 2.4% | 86.4% | 0 |
| MMLU-Pro | final correct | 0.5 | 0.9 | 250 | 5 | 2.0% | 86.4% | 0 |
| MMLU-Pro | final correct | 0.7 | 0.0 | 250 | 12 | 4.8% | 86.4% | 0 |
| MMLU-Pro | final correct | 0.7 | 0.7 | 250 | 6 | 2.4% | 86.4% | 0 |
| MMLU-Pro | final correct | 0.7 | 0.9 | 250 | 5 | 2.0% | 86.4% | 0 |
| MMLU-Pro | final correct | 0.9 | 0.0 | 250 | 11 | 4.4% | 86.4% | 0 |
| MMLU-Pro | final correct | 0.9 | 0.7 | 250 | 6 | 2.4% | 86.4% | 0 |
| MMLU-Pro | final correct | 0.9 | 0.9 | 250 | 5 | 2.0% | 86.4% | 0 |
| MMLU-Pro | loss | 0.5 | 0.0 | 250 | 116 | 46.4% | 45.6% | 0 |
| MMLU-Pro | loss | 0.5 | 0.7 | 250 | 102 | 40.8% | 45.6% | 0 |
| MMLU-Pro | loss | 0.5 | 0.9 | 250 | 82 | 32.8% | 45.6% | 0 |
| MMLU-Pro | loss | 0.7 | 0.0 | 250 | 111 | 44.4% | 45.6% | 0 |
| MMLU-Pro | loss | 0.7 | 0.7 | 250 | 98 | 39.2% | 45.6% | 0 |
| MMLU-Pro | loss | 0.7 | 0.9 | 250 | 78 | 31.2% | 45.6% | 0 |
| MMLU-Pro | loss | 0.9 | 0.0 | 250 | 92 | 36.8% | 45.6% | 0 |
| MMLU-Pro | loss | 0.9 | 0.7 | 250 | 80 | 32.0% | 45.6% | 0 |
| MMLU-Pro | loss | 0.9 | 0.9 | 250 | 63 | 25.2% | 45.6% | 0 |
| MMLU-Pro | stable wrong | 0.5 | 0.0 | 250 | 44 | 17.6% | 15.2% | 2 |
| MMLU-Pro | stable wrong | 0.5 | 0.7 | 250 | 36 | 14.4% | 15.2% | 2 |
| MMLU-Pro | stable wrong | 0.5 | 0.9 | 250 | 32 | 12.8% | 15.2% | 2 |
| MMLU-Pro | stable wrong | 0.7 | 0.0 | 250 | 36 | 14.4% | 15.2% | 2 |
| MMLU-Pro | stable wrong | 0.7 | 0.7 | 250 | 29 | 11.6% | 15.2% | 2 |
| MMLU-Pro | stable wrong | 0.7 | 0.9 | 250 | 26 | 10.4% | 15.2% | 2 |
| MMLU-Pro | stable wrong | 0.9 | 0.0 | 250 | 27 | 10.8% | 15.2% | 2 |
| MMLU-Pro | stable wrong | 0.9 | 0.7 | 250 | 20 | 8.0% | 15.2% | 2 |
| MMLU-Pro | stable wrong | 0.9 | 0.9 | 250 | 17 | 6.8% | 15.2% | 2 |

## Matched risk differences

Risk differences average within matched triplets and bootstrap over match IDs.

- **Gpqa Diamond correct >= 0.5, final >= 0.0, loss vs final correct:** +28.4% (95% CI +14.4% to +42.4%; n=25 matches).
- **Gpqa Diamond correct >= 0.5, final >= 0.0, loss vs stable wrong:** +30.4% (95% CI +16.0% to +44.8%; n=25 matches).
- **Gpqa Diamond correct >= 0.5, final >= 0.7, loss vs final correct:** +26.8% (95% CI +14.0% to +40.4%; n=25 matches).
- **Gpqa Diamond correct >= 0.5, final >= 0.7, loss vs stable wrong:** +28.0% (95% CI +13.2% to +42.4%; n=25 matches).
- **Gpqa Diamond correct >= 0.5, final >= 0.9, loss vs final correct:** +25.6% (95% CI +14.4% to +36.8%; n=25 matches).
- **Gpqa Diamond correct >= 0.5, final >= 0.9, loss vs stable wrong:** +23.2% (95% CI +10.4% to +37.2%; n=25 matches).
- **Gpqa Diamond correct >= 0.7, final >= 0.0, loss vs final correct:** +24.8% (95% CI +11.2% to +38.4%; n=25 matches).
- **Gpqa Diamond correct >= 0.7, final >= 0.0, loss vs stable wrong:** +29.6% (95% CI +16.4% to +43.2%; n=25 matches).
- **Gpqa Diamond correct >= 0.7, final >= 0.7, loss vs final correct:** +23.6% (95% CI +11.6% to +36.8%; n=25 matches).
- **Gpqa Diamond correct >= 0.7, final >= 0.7, loss vs stable wrong:** +27.2% (95% CI +15.2% to +40.0%; n=25 matches).
- **Gpqa Diamond correct >= 0.7, final >= 0.9, loss vs final correct:** +22.4% (95% CI +12.0% to +33.6%; n=25 matches).
- **Gpqa Diamond correct >= 0.7, final >= 0.9, loss vs stable wrong:** +21.6% (95% CI +10.4% to +33.6%; n=25 matches).
- **Gpqa Diamond correct >= 0.9, final >= 0.0, loss vs final correct:** +22.4% (95% CI +7.6% to +36.8%; n=25 matches).
- **Gpqa Diamond correct >= 0.9, final >= 0.0, loss vs stable wrong:** +27.6% (95% CI +14.8% to +40.8%; n=25 matches).
- **Gpqa Diamond correct >= 0.9, final >= 0.7, loss vs final correct:** +20.8% (95% CI +8.8% to +34.0%; n=25 matches).
- **Gpqa Diamond correct >= 0.9, final >= 0.7, loss vs stable wrong:** +24.8% (95% CI +13.2% to +37.6%; n=25 matches).
- **Gpqa Diamond correct >= 0.9, final >= 0.9, loss vs final correct:** +18.0% (95% CI +8.4% to +28.4%; n=25 matches).
- **Gpqa Diamond correct >= 0.9, final >= 0.9, loss vs stable wrong:** +18.8% (95% CI +8.4% to +29.6%; n=25 matches).
- **Mmlu Pro correct >= 0.5, final >= 0.0, loss vs final correct:** +41.6% (95% CI +30.8% to +52.4%; n=25 matches).
- **Mmlu Pro correct >= 0.5, final >= 0.0, loss vs stable wrong:** +28.8% (95% CI +16.8% to +41.6%; n=25 matches).
- **Mmlu Pro correct >= 0.5, final >= 0.7, loss vs final correct:** +38.4% (95% CI +26.0% to +51.6%; n=25 matches).
- **Mmlu Pro correct >= 0.5, final >= 0.7, loss vs stable wrong:** +26.4% (95% CI +13.2% to +39.6%; n=25 matches).
- **Mmlu Pro correct >= 0.5, final >= 0.9, loss vs final correct:** +30.8% (95% CI +19.6% to +42.8%; n=25 matches).
- **Mmlu Pro correct >= 0.5, final >= 0.9, loss vs stable wrong:** +20.0% (95% CI +7.2% to +32.8%; n=25 matches).
- **Mmlu Pro correct >= 0.7, final >= 0.0, loss vs final correct:** +39.6% (95% CI +28.0% to +51.2%; n=25 matches).
- **Mmlu Pro correct >= 0.7, final >= 0.0, loss vs stable wrong:** +30.0% (95% CI +18.4% to +42.0%; n=25 matches).
- **Mmlu Pro correct >= 0.7, final >= 0.7, loss vs final correct:** +36.8% (95% CI +24.4% to +49.6%; n=25 matches).
- **Mmlu Pro correct >= 0.7, final >= 0.7, loss vs stable wrong:** +27.6% (95% CI +15.2% to +41.2%; n=25 matches).
- **Mmlu Pro correct >= 0.7, final >= 0.9, loss vs final correct:** +29.2% (95% CI +17.6% to +41.2%; n=25 matches).
- **Mmlu Pro correct >= 0.7, final >= 0.9, loss vs stable wrong:** +20.8% (95% CI +8.8% to +33.2%; n=25 matches).
- **Mmlu Pro correct >= 0.9, final >= 0.0, loss vs final correct:** +32.4% (95% CI +21.2% to +44.0%; n=25 matches).
- **Mmlu Pro correct >= 0.9, final >= 0.0, loss vs stable wrong:** +26.0% (95% CI +15.2% to +37.6%; n=25 matches).
- **Mmlu Pro correct >= 0.9, final >= 0.7, loss vs final correct:** +29.6% (95% CI +17.2% to +42.8%; n=25 matches).
- **Mmlu Pro correct >= 0.9, final >= 0.7, loss vs stable wrong:** +24.0% (95% CI +12.4% to +36.0%; n=25 matches).
- **Mmlu Pro correct >= 0.9, final >= 0.9, loss vs final correct:** +23.2% (95% CI +12.0% to +35.6%; n=25 matches).
- **Mmlu Pro correct >= 0.9, final >= 0.9, loss vs stable wrong:** +18.4% (95% CI +7.6% to +30.4%; n=25 matches).
