# HCM

Code and assets for Hypertrophic Cardiomyopathy (HCM) variant pathogenicity modeling using a zero-leakage pipeline.

## What is in this repository

- `data/` — labeled and full variant datasets plus cached ESM2 delta embeddings.
- `scripts/` — end-to-end pipeline scripts:
  - `execute_month1.py`: data cleaning, leakage-column removal, ESM2 embedding extraction, and baseline LOGO model training.
  - `execute_month2.py`: two-tower hybrid training, LOGO evaluation, calibration, ablation, and ISM figure generation.
  - `execute_month3.py`: final model training and VUS re-stratification.
- `models/` — saved baseline and final hybrid model artifacts.
- `results/` — evaluation metrics and VUS ranking outputs.
- `figures/` — generated calibration/ISM/metrics figures.
- `paper/` — manuscript and supporting writeups.

## Quick start

1. Create and activate a Python 3.10 environment.
2. Install dependencies:

```bash
pip install pandas numpy scikit-learn torch transformers matplotlib seaborn joblib tabulate
```

3. Run the full pipeline from the repository root:

```bash
python scripts/execute_month1.py
python scripts/execute_month2.py
python scripts/execute_month3.py
```

## Reproducibility and leakage policy

The training pipeline explicitly drops known leaky metadata columns before modeling (`pop_freq`, `disease`, `sources`, `genomic_loc`, `review_status`, `clin_sig`).

For additional implementation details, see `reproducibility_README.md`.
