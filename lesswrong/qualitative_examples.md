# Qualitative examples for the LessWrong post

These are candidate examples, not proof. I selected them from the robust lost-case
tables and sanity-checked the corresponding `traces.jsonl` records. Because
`pyarrow`, `fastparquet`, `duckdb`, and `polars` were not installed, I did not
recover the full decile-by-decile `prediction_path` from `trajectory.parquet` or
`lost_cases.parquet`. The stats below come from the CSV/Markdown lost-analysis
artifacts and the trace JSONL files.

For integration, I would probably use one MMLU-Pro example and one GPQA example
in the post body, then put the others in a footnote or appendix. These are meant
to show what the broad-loss pattern looks like qualitatively, while keeping the
null hypothesis in view.

## Candidate 1: MMLU-Pro business, late arithmetic flip

- Dataset: MMLU-Pro (`TIGER-Lab/MMLU-Pro`)
- Position/question_id/category: position 722, question_id 799, business
- Source: `stemez-Business`
- Question: used-car installment-plan interest rate. The options are annual
  rates from 10% to 20%; the correct answer is J = 14.4%.
- Correct answer: J
- Final probe prediction: D = 13.3%
- Generated final answer: D
- Coarse transition: J -> D
- Full prediction path across deciles: not recovered from available non-Parquet
  artifacts.
- First/last correct decile: first correct not recovered; last correct = 90%.
- Confidence/instability stats: peak correct-answer probability 0.724; final
  valid-letter probability mass 0.798; 5 adjacent-decile prediction flips;
  27,827 trace tokens.
- Robustness flag: robust lost case in the MMLU-Pro lost analysis.

Why it is a good example: this is a very legible case for readers. The model is
still probe-correct at 90% of the trace, then the final probe and final generated
answer both land on the wrong nearby interest-rate option. It is not just "the
model never had a clue"; there was a late prefix where the answer-letter probe
favored the ground-truth answer.

Blog-safe paraphrase from the trace: the trace considers simple-interest and
amortized-loan interpretations, repeatedly recalculates monthly payments, and
ends by saying that 13.3% is closest even though the 90% probe favored 14.4%.

Suggested use in post: this is the most accessible MMLU example. It is good for
the main text if you want something a non-specialist can follow quickly.

## Candidate 2: MMLU-Pro philosophy, short false-reframing case

- Dataset: MMLU-Pro (`TIGER-Lab/MMLU-Pro`)
- Position/question_id/category: position 10953, question_id 11174, philosophy
- Source: `ori_mmlu-logical_fallacies`
- Question: "Arguing that someone couldn't have done something good because she
  holds a particular position commits the fallacy of..."
- Correct answer: C = Reprehensible personality
- Final probe prediction: A = Guilt by association
- Generated final answer: A = Guilt by association
- Coarse transition: C -> A
- Full prediction path across deciles: not recovered from available non-Parquet
  artifacts.
- First/last correct decile: first correct = 10%; last correct = 60%.
- Confidence/instability stats: peak correct-answer probability 0.998; final
  valid-letter probability mass 0.994; 2 adjacent-decile prediction flips; 2,364
  trace tokens.
- Robustness flag: robust lost case and normalized reversal candidate in the
  MMLU-Pro lost analysis.

Why it is a good example: this one is compact, high-confidence, and easy to
summarize. The trace initially treats option C as the ad-hominem-like answer,
then later reframes "holds a particular position" as association with a group
and locks onto option A. The final valid-letter mass is very high, so this looks
less like a final near-zero-probability argmax artifact than many broad losses.

Blog-safe paraphrase from the trace: early reasoning maps the prompt to an
attack on the person; late reasoning changes the frame to "association with a
position/group" and chooses guilt by association.

Suggested use in post: good if you want a short example of a late false premise
or reframing, rather than a long arithmetic trace.

## Candidate 3: GPQA Diamond organic chemistry, "going in circles" case

- Dataset: GPQA Diamond (`fingertap/GPQA-Diamond`)
- Position/question_id/category: position 91, question_id `gpqa-diamond-091`,
  gpqa-diamond
- Source: `fingertap/GPQA-Diamond`
- Question: enamine/enaminium reaction; identify reagent sequence A and product
  B.
- Correct answer: C = `(i) LDA, DME; (ii) CH3CH2I; (iii) H3O+; B = heptan-4-one`
- Final probe prediction: B = same reagent sequence, but
  `B = pentan-2-one + N,N-dimethylethanamine`
- Generated final answer: B
- Coarse transition: C -> B
- Full prediction path across deciles: not recovered from available non-Parquet
  artifacts.
- First/last correct decile: first correct = 20%; last correct = 80%.
- Confidence/instability stats: peak correct-answer probability 0.560; final
  valid-letter probability mass 0.575; 6 adjacent-decile prediction flips;
  25,023 trace tokens.
- Robustness flag: robust lost case and normalized reversal candidate in the
  GPQA lost analysis.

Why it is a good example: this is the best GPQA candidate for the main text. It
has a late correct prefix, many flips, and an explicitly unstable trace. The
model spends a long time redrawing the structure and reconsidering which carbon
is alkylated; near the end it effectively gives up on the structure bookkeeping
and selects the wrong product.

Blog-safe paraphrase from the trace: the trace repeatedly redraws the
enaminium structure, loses track of carbon counting, and ends with a
"going in circles" style admission before choosing option B.

Suggested use in post: use as the GPQA example if you want a qualitatively vivid
case of instability rather than just a numeric table row.

## Candidate 4: GPQA Diamond symmetry, very high final mass

- Dataset: GPQA Diamond (`fingertap/GPQA-Diamond`)
- Position/question_id/category: position 18, question_id `gpqa-diamond-018`,
  gpqa-diamond
- Source: `fingertap/GPQA-Diamond`
- Question: identify species from a reaction chain involving gases, a bright red
  product, hydrolysis to two acids, and a hazardous product; answer asks for the
  molecular symmetry group of E.
- Correct answer: D = C2v
- Final probe prediction: A = D_infinity_h
- Generated final answer: A
- Coarse transition: D -> A
- Full prediction path across deciles: not recovered from available non-Parquet
  artifacts.
- First/last correct decile: first correct not recovered; last correct = 70%.
- Confidence/instability stats: peak correct-answer probability 0.540; final
  valid-letter probability mass 0.998; 4 adjacent-decile prediction flips;
  32,760 trace tokens.
- Robustness flag: robust lost case in the GPQA lost analysis.

Why it is a good example: the final valid-letter mass is almost all on answer
letters and the generated answer agrees with the final probe, so it is a useful
counterexample to the idea that every broad loss is just a tiny final answer
mass artifact. It is less blog-friendly than Candidate 3 because the chemistry
identification is harder to explain quickly.

Blog-safe paraphrase from the trace: the model searches through candidate gas
identifications and symmetry assignments for a long time, is correct at a 70%
prefix, then ends by committing to the linear-symmetry option A.

Suggested use in post: good appendix example or backup GPQA example; Candidate 3
is more vivid for the main narrative.

## Caveats for the post

- These examples are illustrative. They do not establish that the model "knew"
  the answer and then lost it.
- The full decile letter paths should be filled in later if someone runs the
  analysis in an environment with a Parquet reader. The lost-analysis code says
  `prediction_path` is stored in the lost-case table, but this pass could only
  use CSV/Markdown/JSONL artifacts.
- Broad losses are heterogeneous. Candidate 1 could be a close arithmetic
  interpretation issue; Candidate 2 looks like false reframing; Candidate 3
  looks like trace instability; Candidate 4 is mainly useful because the final
  valid-letter mass is high.
- Avoid quoting long trace spans. The paraphrases above should be enough for the
  LessWrong post, with at most one very short phrase from Candidate 3 if desired.
