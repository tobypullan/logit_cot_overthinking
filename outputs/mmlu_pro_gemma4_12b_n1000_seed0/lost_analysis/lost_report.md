# Lost-case analysis

Complete traces only.

## Main counts

- Ever correct, final wrong: 109
- Endpoint lost (correct at 0%, wrong at 100%): 25
- Gained then lost: 84
- Confidence-filtered robust losses: 17
- Robust losses hidden by the endpoint label: 16

## Quantitative observations

- 43 cases were still correct at 80% or 90% of the trace before ending wrong.
- Lost cases had a median 4766 reasoning tokens versus 1614 for final-correct cases.
- Lost cases had a median 4 prediction flips versus 1 for final-correct cases.
- The median final valid-letter probability mass was only 0.38%; probe and generated answer agreed in 56.9% of broad losses.
- 14 of 17 robust traces used simulated retrieval language such as searching for a source, test bank, or exact answer.
- Highest broad loss rates: chemistry 21.2%, engineering 19.5%, health 18.3%, physics 18.2%, law 14.1%.
- 11 of 17 robust losses came from law, history, psychology.
- The leading broad-loss categories had no robust cases: chemistry, engineering, health, physics. This indicates that low final answer-letter mass explains much of their apparent loss rate.

## Robust cases

| Question | Category | Change | Last correct | Peak correct p | Tokens |
|---:|---|---:|---:|---:|---:|
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
