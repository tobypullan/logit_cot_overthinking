# overall plan
 - implement PROBING THE TRAJECTORIES OF REASONING TRACES IN LARGE LANGUAGE MODELS method https://arxiv.org/pdf/2601.23163
 - get reasoning trace trajectories for 1000 questions from GPQA Diamond and MMLU-Pro benchmarks
 - do on multiple models of multiple sizes and do repeats: gemma 4 12b, gemma 4 26b, qwen 3.6 27b
 - analyse the trajectories for cases where the model gets the question correct at some point but in the end gets it wrong
 - these are the cases when the CoT reduces performance
 - check for a pattern between these cases - does it generalise?
