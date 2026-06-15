# Lost-case analysis

Complete traces only.

## Main counts

- Ever correct, final wrong: 122
- Endpoint lost (correct at 0%, wrong at 100%): 27
- Gained then lost: 95
- Confidence-filtered robust losses: 18
- Normalized reversal candidates: 37
- Broad losses ending with a correct generated answer: 56
- Broad losses ending with an explicit wrong generated answer: 62
- Robust losses hidden by the endpoint label: 17

## Quantitative observations

- 48 cases were still correct at 80% or 90% of the trace before ending wrong.
- Lost cases had a median 5155 reasoning tokens versus 1694 for final-correct cases.
- Lost cases had a median 4 prediction flips versus 1 for final-correct cases.
- The median final valid-letter probability mass was only 0.59%; probe and generated answer agreed in 45.9% of broad losses.
- 14 of 18 robust traces used simulated retrieval language such as searching for a source, test bank, or exact answer.
- Highest broad loss rates: engineering 22.2%, physics 19.7%, chemistry 19.4%, health 18.3%, law 14.1%.
- 11 of 18 robust losses came from law, history, business.
- The leading broad-loss categories had no robust cases: engineering, physics, chemistry, health. This indicates that low final answer-letter mass explains much of their apparent loss rate.

## Robust cases

| Question | Category | Change | Last correct | Peak correct p | Tokens |
|---:|---|---:|---:|---:|---:|
| 799 | business | J->D | 90% | 0.724 | 27,827 |
| 665 | business | C->H | 80% | 0.946 | 12,023 |
| 6980 | economics | G->F | 60% | 1.000 | 7,463 |
| 11174 | philosophy | C->A | 60% | 0.998 | 2,364 |
| 2199 | psychology | D->H | 60% | 0.675 | 4,835 |
| 1130 | law | A->D | 40% | 0.999 | 646 |
| 1181 | law | H->C | 40% | 0.771 | 705 |
| 5820 | other | E->D | 30% | 0.999 | 4,381 |
| 4746 | history | C->A | 30% | 0.999 | 3,429 |
| 2714 | psychology | B->A | 20% | 0.999 | 3,674 |
| 116 | business | J->E | 20% | 0.999 | 9,524 |
| 4809 | history | C->D | 20% | 0.957 | 5,149 |
| 1902 | law | H->F | 20% | 0.900 | 2,944 |
| 2462 | psychology | F->B | 20% | 0.610 | 5,033 |
| 11240 | philosophy | B->C | 10% | 0.992 | 1,793 |
| 1486 | law | F->A | 10% | 0.935 | 3,712 |
| 4710 | history | B->H | 10% | 0.760 | 1,010 |
| 4941 | history | F->C | 10% | 0.696 | 3,979 |

The robust filter requires high pre-final raw answer probability, substantial final probability mass on valid letters, and agreement between the final probe and the generated answer.

## Normalized reversal candidates

| Question | Change | First correct | Last correct | Peak raw p | Final letter mass |
|---:|---:|---:|---:|---:|---:|
| 665 | C->H | 10% | 80% | 0.946 | 0.556 |
| 6980 | G->F | 10% | 60% | 1.000 | 0.578 |
| 11174 | C->A | 10% | 60% | 0.998 | 0.994 |
| 1130 | A->D | 0% | 40% | 0.999 | 1.000 |
| 1181 | H->C | 10% | 40% | 0.771 | 0.753 |
| 5820 | E->D | 10% | 30% | 0.999 | 0.963 |
| 4746 | C->A | 10% | 30% | 0.999 | 0.956 |
| 2714 | B->A | 10% | 20% | 0.999 | 0.965 |
| 116 | J->E | 10% | 20% | 0.999 | 0.990 |
| 4809 | C->D | 10% | 20% | 0.957 | 0.992 |
| 1902 | H->F | 10% | 20% | 0.900 | 0.789 |
| 11240 | B->C | 10% | 10% | 0.992 | 0.994 |
| 1486 | F->A | 10% | 10% | 0.935 | 1.000 |
| 4710 | B->H | 10% | 10% | 0.760 | 0.999 |
| 10050 | H->C | 10% | 90% | 0.103 | 0.961 |
| 7610 | A->J | 0% | 70% | 0.005 | 0.204 |
| 6038 | C->B | 10% | 60% | 1.000 | 0.182 |
| 1879 | I->C | 40% | 60% | 0.174 | 0.306 |
| 3253 | I->G | 10% | 50% | 0.998 | 0.158 |
| 474 | A->G | 0% | 50% | 0.380 | 0.340 |
| 694 | A->J | 0% | 30% | 0.036 | 0.993 |
| 12149 | E->G | 10% | 20% | 0.987 | 0.371 |
| 10615 | D->B | 10% | 10% | 0.896 | 0.278 |
| 5608 | A->C | 0% | 10% | 0.393 | 0.999 |
| 10884 | A->G | 0% | 0% | 0.109 | 0.919 |
| 6968 | A->E | 0% | 0% | 0.085 | 0.043 |
| 5171 | A->B | 0% | 0% | 0.005 | 1.000 |
| 10750 | A->D | 0% | 0% | 0.003 | 1.000 |
| 6067 | A->C | 0% | 0% | 0.002 | 0.906 |
| 4889 | A->C | 0% | 0% | 0.002 | 0.980 |
| 4824 | A->H | 0% | 0% | 0.002 | 0.994 |
| 1343 | A->B | 0% | 0% | 0.002 | 0.248 |
| 6660 | A->E | 0% | 0% | 0.001 | 0.036 |
| 10810 | A->H | 0% | 0% | 0.001 | 1.000 |
| 6216 | A->C | 0% | 0% | 0.001 | 0.998 |
| 10982 | A->C | 0% | 0% | 0.000 | 0.077 |
| 7437 | A->J | 0% | 0% | 0.000 | 0.304 |

Normalized candidates require at least 90% of valid-letter probability on the correct answer before the end, at least 90% on the final wrong answer at the end, and agreement with the generated answer. They can still have low absolute answer-letter probability.
