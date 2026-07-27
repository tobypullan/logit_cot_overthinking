# Activation Probe Report

- Examples: 408
- Layers: 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48
- Targets: robust_loss_case
- Backend: torch

## Targets

- `current_correct`: The current prediction matches the true answer.
- `future_loss`: Current prediction is correct and the final prediction is wrong.
- `future_change_to_wrong`: The final prediction is wrong and differs from the current prediction.
- `future_answer_flip`: The final prediction differs from the current prediction.
- `robust_loss_case`: The currently correct checkpoint belongs to a robust-loss trace, rather than a trace with no loss.

## Best AUC

| Target | Layer | AUC | Brier | Positive rate |
|---|---:|---:|---:|---:|
| `robust_loss_case` | 36 | 0.726 | 0.220 | 25.0% |
| `robust_loss_case` | 20 | 0.722 | 0.226 | 25.0% |
| `robust_loss_case` | 32 | 0.712 | 0.232 | 25.0% |
| `robust_loss_case` | 48 | 0.708 | 0.235 | 25.0% |
| `robust_loss_case` | 24 | 0.708 | 0.223 | 25.0% |

## Best Halting Deltas

No probe halting policies were evaluated.

## Best Probe-Only Halting Deltas

No probe-only halting policies were evaluated.
