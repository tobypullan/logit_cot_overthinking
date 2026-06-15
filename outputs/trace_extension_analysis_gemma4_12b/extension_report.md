# Capped-trace extension analysis

Extended 92 traces. 7 still failed to close within the additional 16,384-token budget and were explicitly closed before sampling an answer.

## GPQA Diamond

- Extended 41 traces; 37 stopped naturally and 4 required explicit closure.
- The final prediction changed for 15 traces (36.6%).
- Correctness at the cap versus after extension: 16 -> 20 (net +4); 6 changed wrong to correct and 2 changed correct to wrong.
- 12 extended traces were ever-correct/final-wrong; 7 met the dataset's target `normalized_reversal_candidate` criterion.
- Median completed trace length was 20,909 tokens; the maximum was 32,764.

## MMLU-Pro

- Extended 51 traces; 48 stopped naturally and 3 required explicit closure.
- The final prediction changed for 20 traces (39.2%).
- Correctness at the cap versus after extension: 18 -> 22 (net +4); 9 changed wrong to correct and 5 changed correct to wrong.
- 13 extended traces were ever-correct/final-wrong; 1 met the dataset's target `robust_loss` criterion.
- Median completed trace length was 20,928 tokens; the maximum was 32,765.

## Interpretation

- A 16K endpoint is not a reliable proxy for these long traces: many predictions changed after continuation.
- Continuation improved net accuracy among capped traces, but also exposed additional correct-to-wrong reversals. Excluding the capped subset therefore biased both accuracy and loss-case counts.
- Forced completions are a small, separately labeled tail of pathological repetitive or unfinished reasoning.
