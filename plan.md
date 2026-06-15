# overall plan
 - implement PROBING THE TRAJECTORIES OF REASONING TRACES IN LARGE LANGUAGE MODELS method https://arxiv.org/pdf/2601.23163
 - get reasoning trace trajectories for the full 198-question GPQA Diamond test split and around 1000 questions from [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
 - do on multiple models of multiple sizes and do repeats: gemma 4 12b, gemma 4 26b, qwen 3.6 27b
 - analyse the trajectories for cases where the model gets the question correct at some point but in the end gets it wrong
 - these are the cases when the CoT reduces performance
 - check for a pattern between these cases - does it generalise?

# step 1
look up the PROBING THE TRAJECTORIES OF REASONING TRACES IN LARGE LANGUAGE MODELS paper via the link provided. Implement the logit-based trajectory probing method into the repo. 

Test that it works by performing a smoke run on the first 3 questions of MMLU-Pro with the gemma 4 12b model.
Especially test whether it runs as quickly as you'd expect - we are doing a lot of runs on a lot of datasets so don't want the runs to be taking longer than necessary 


# step 2
do a run of around 1000 questions from a mixture of categories in MMLU-pro on gemma 4 12b

analyse the results from this run. Specifically, look for cases where the model gets the question correct at some point but ends by getting the question wrong.


# step 3
run the trajectory probe on GPQA Diamond using `fingertap/GPQA-Diamond`.

Start with a three-question smoke run on Gemma 4 12B, then run the full
198-question test split after validating the artifacts, trace completion, and
runtime.
