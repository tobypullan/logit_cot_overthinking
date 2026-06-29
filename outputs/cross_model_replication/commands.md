# Cross-Model Replication Commands

This plan repeats the matched-control, extension, analysis, confidence-recurrence, early-commitment, and optional branching setup pipeline for each runnable model.

Replication mode: same matched question positions; baseline cohorts come from the source matched-selection run

## google/gemma-4-12B-it

Adapter: gemma
Runnable now: True
Note: Detected as Gemma-compatible from the model name.

### matched_controls

```bash
trajectory-run-matched-controls --model google/gemma-4-12B-it --seeds 0-9 --per-cohort 25 --output-root outputs/cross_model_replication/google_gemma_4_12b_it/matched_controls --trace-max-tokens 16384 --max-model-len 20480 --max-num-seqs 32 --gpu-memory-utilization 0.9
```

### extend_matched_controls

```bash
trajectory-extend-matched-controls --model google/gemma-4-12B-it --seeds 0-9 --input-root outputs/cross_model_replication/google_gemma_4_12b_it/matched_controls --output-root outputs/cross_model_replication/google_gemma_4_12b_it/matched_controls_extended --extension-max-tokens 16384 --max-model-len 49152 --max-num-seqs 16 --gpu-memory-utilization 0.9
```

### matched_analysis

```bash
trajectory-analyze-matched-controls --input-root outputs/cross_model_replication/google_gemma_4_12b_it/matched_controls_extended --selection outputs/cross_model_replication/google_gemma_4_12b_it/matched_controls/cohort_selection.parquet --output-dir outputs/cross_model_replication/google_gemma_4_12b_it/matched_controls_extended/analysis
```

### confidence_recurrence

```bash
trajectory-analyze-confidence-recurrence --input-root outputs/cross_model_replication/google_gemma_4_12b_it/matched_controls_extended --selection outputs/cross_model_replication/google_gemma_4_12b_it/matched_controls/cohort_selection.parquet --output-dir outputs/cross_model_replication/google_gemma_4_12b_it/matched_controls_extended/analysis/confidence_recurrence
```

### early_commitment

```bash
trajectory-analyze-early-commitment --input-root outputs/cross_model_replication/google_gemma_4_12b_it/matched_controls_extended --selection outputs/cross_model_replication/google_gemma_4_12b_it/matched_controls/cohort_selection.parquet --output-dir outputs/cross_model_replication/google_gemma_4_12b_it/matched_controls_extended/analysis/early_commitment
```

### branching_setup

```bash
trajectory-run-branching-intervention --model google/gemma-4-12B-it --input-root outputs/cross_model_replication/google_gemma_4_12b_it/matched_controls_extended --selection outputs/cross_model_replication/google_gemma_4_12b_it/matched_controls/cohort_selection.parquet --output-dir outputs/cross_model_replication/google_gemma_4_12b_it/branching_intervention --min-current-normalized-correct-probability 0.9 --max-candidates-per-dataset 25 --branch-modes answer_only,normal,short_verification,preserve_unless_decisive --branch-seeds 0-3
```
