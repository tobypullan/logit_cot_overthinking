# GPQA Diamond trajectory analysis

## Scope

The run contains all 198 GPQA Diamond test questions. Forty-one traces reached
the 16,384-token generation limit, leaving 157 complete traces for endpoint
analysis. The capped traces are excluded below unless stated otherwise.

## Aggregate trajectory

- Accuracy rises from 27.4% at 0% reasoning to 68.8% at 90%.
- Accuracy falls to 58.0% at the full-trace probe, a 10.8-point decline.
- The generated final response is correct for 70.7% of complete traces after
  correcting verbose-answer parsing. Eleven responses do not yield a
  parseable final letter.
- Median valid-letter probability mass rises to 15.5% at 90% but falls to
  1.0% at 100%. Thus, part of the apparent final-probe decline is caused by
  Gemma preferring prose or another non-letter token after a completed trace.
- Complete traces have a median length of 5,811 tokens. The 90th percentile is
  12,973 tokens.

## Lost cases

- 45 of 157 complete traces (28.7%) are correct at least once and wrong at the
  final probe.
- Only 6 are captured by the endpoint `lost` label; 39 first become correct
  after the 0% probe and are later lost.
- 27 of the 45 remain correct as late as 80% or 90%.
- Lost traces are longer than final-correct traces: median 7,845 versus 3,515
  tokens.
- Lost traces are less stable: median 4 prediction flips versus 2.
- Thirty-two broad probe losses nevertheless end with the correct generated
  answer. Twelve end with an explicit wrong generated answer, and one final
  response is not parseable as a letter.

The previous MMLU-oriented robust filter finds zero cases because it requires
at least 50% absolute probability on valid answer letters at the final probe.
That requirement is too strict for GPQA, where final letter compliance is much
lower.

## Reversal candidates

Eight cases remain after requiring a strong preference within A-D for the
correct answer before the end, a strong final preference for the wrong answer,
and agreement with the generated final answer.

The clearest late reversals are:

- `gpqa-diamond-095` (A to B): correct through 90%. The model initially
  identifies ethyl 4-aminobenzoate, then incorrectly counts its oxygen atoms
  and switches to the formamide.
- `gpqa-diamond-158` (A to B): correct through 90%. It oscillates between
  atmospheric cutoff assumptions and eventually trusts a claimed external
  source favoring redshift 2.4.
- `gpqa-diamond-101` (D to C): correct through 80%. It changes from the lithium
  carboxylate to the neutral acid based on an assumed workup and a claimed
  source.
- `gpqa-diamond-119` (C to B): correct through 60%. It replaces the keyed
  bisulfite reagent with a more familiar hydronium cyanohydrin mechanism.

Four weaker candidates are correct only at 0% to 20%:
`gpqa-diamond-107`, `gpqa-diamond-070`, `gpqa-diamond-087`, and
`gpqa-diamond-133`. These may be transient probe hits rather than knowledge
that was stably acquired and then lost.

Across the candidates, repeated rechecking, substitution of a familiar
textbook mechanism, ambiguous convention choices, and simulated source lookup
are more visible than simple arithmetic drift.

## Truncation caveat

The 41 capped traces are not a random subset. Their probe accuracy at the cap
is 39.0%, compared with 58.0% for complete traces, and they do not contain a
true generated final answer. The complete-trace accuracy therefore likely
overstates performance on the full 198-question set. Extending these traces is
important before reporting a definitive GPQA accuracy or loss rate.

## MMLU-Pro comparison

Compared with the prior MMLU-Pro run:

- Broad loss rate: 28.7% versus 11.5%.
- Median complete trace length: 5,811 versus 2,041 tokens.
- Completion rate at 16,384 tokens: 79.3% versus 94.9%.
- Peak-to-final probe decline: 10.8 versus 2.1 percentage points.

GPQA therefore produces substantially more long and unstable trajectories, but
the final answer-letter probe is also less reliable. The eight normalized
candidates, especially the four late reversals, are the most useful cases for
the next qualitative or repeated-seed analysis.
