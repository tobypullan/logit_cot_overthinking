# Workshop Draft

This directory contains an initial LaTeX paper draft for the fragile-correctness
hypothesis and the matched-control results.

Build from this directory with:

```bash
latexmk -pdf main.tex
```

or, if `latexmk` is unavailable:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The figures are copied from:

- `outputs/matched_controls_gemma4_12b_extended/analysis/cohort_loss_rates.png`
- `outputs/matched_controls_gemma4_12b_extended/analysis/predictive_auc_by_decile.png`
- `outputs/matched_controls_gemma4_12b_extended/analysis/feature_coefficients.png`
