# Activation Probe Report

- Examples: 1,984
- Layers: 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48
- Targets: robust_stop_candidate
- Backend: torch

## Targets

- `current_correct`: The current prediction matches the true answer.
- `future_loss`: Current prediction is correct and the final prediction is wrong.
- `future_change_to_wrong`: The final prediction is wrong and differs from the current prediction.
- `future_answer_flip`: The final prediction differs from the current prediction.
- `robust_loss_case`: The currently correct checkpoint belongs to a robust-loss trace, rather than a trace with no loss.
- `robust_stop_candidate`: Stopping at this observable-confidence candidate preserves a currently correct answer from a robust-loss trace.

## Best AUC

| Target | Layer | AUC | Brier | Positive rate |
|---|---:|---:|---:|---:|
| `robust_stop_candidate` | 32 | 0.760 | 0.259 | 1.7% |
| `robust_stop_candidate` | 28 | 0.735 | 0.221 | 1.7% |
| `robust_stop_candidate` | 16 | 0.729 | 0.257 | 1.7% |
| `robust_stop_candidate` | 20 | 0.723 | 0.251 | 1.7% |
| `robust_stop_candidate` | 8 | 0.717 | 0.245 | 1.7% |

## Best Halting Deltas

| Target | Policy | Layer | Confidence | Probe | Accuracy | Delta vs final | Stop rate |
|---|---|---:|---:|---:|---:|---:|---:|
| `robust_stop_candidate` | probe_candidate | 36 | 0.90 | 0.90 | 74.1% | +0.2% | 1.2% |
| `robust_stop_candidate` | probe_candidate | 44 | 0.90 | 0.90 | 74.1% | +0.2% | 1.3% |
| `robust_stop_candidate` | probe_candidate | 28 | 0.90 | 0.70 | 74.1% | +0.2% | 12.8% |
| `robust_stop_candidate` | probe_candidate | 4 | 0.90 | 0.90 | 74.1% | +0.1% | 1.3% |
| `robust_stop_candidate` | probe_candidate | 48 | 0.90 | 0.90 | 74.1% | +0.1% | 0.6% |

## Best Probe-Only Halting Deltas

No probe-only halting policies were evaluated.
