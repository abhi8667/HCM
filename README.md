# Deciphering the Molecular Grammar of Hypertrophic Cardiomyopathy (HCM)

This repository contains the implementation and assets for the research project:
**"Deciphering the Molecular Grammar of Hypertrophic Cardiomyopathy: A Zero-Leakage Language Modeling Approach to Sarcomeric Pathogenicity."**

It provides an end-to-end, leakage-controlled pipeline for HCM missense variant pathogenicity modeling using differential ESM-2 embeddings, structural/physicochemical features, and a two-tower neural architecture under Leave-One-Gene-Out (LOGO) validation.

## Research Objective

Hypertrophic Cardiomyopathy (HCM) variant interpretation remains limited by high VUS burden and evaluation leakage in many prediction workflows. This project is designed to test whether pathogenicity can be inferred from intrinsic molecular constraints without relying on ACMG-overlapping or clinically circular evidence channels.

## Methodological Principles (Aligned with the Manuscript)

- **Zero-leakage feature policy:** leakage-prone clinical fields are removed before modeling (`pop_freq`, `disease`, `sources`, `genomic_loc`, `review_status`, `clin_sig`).
- **Homology-safe validation:** **LOGO** split design (held-out gene evaluation).
- **Mutation-centric representation:** differential ESM-2 embedding vectors (`mutant - wild-type`).
- **Hybrid modeling:** tabular structural/biophysical tower + ESM embedding tower.
- **Clinical usability focus:** calibration-aware scoring and VUS re-stratification support.

## Headline Results Reported in the Paper

Under strict TNNT2 LOGO hold-out evaluation, the manuscript reports:

- **AUPRC:** `0.9151` (95% bootstrap CI: `0.8642–0.9649`)
- **Brier score:** `0.1506`
- **VUS prioritization output:** `4,523` unresolved variants scored for downstream triage

For full context and interpretation, see:
- `paper/hcm_paper_manuscript.tex`

## Repository Structure

- `data/`
  - Core datasets (`HCM_labeled_final.csv`, `HCM_all_variants_v2.csv`)
  - Cached embeddings (`esm2_delta_embeddings.npy`)
- `scripts/`
  - `execute_month1.py`: cleaning, de-leaking, embedding extraction, baseline model
  - `execute_month2.py`: two-tower training, LOGO evaluation, calibration, ablations, ISM outputs
  - `execute_month3.py`: final training and VUS re-stratification
  - Additional comparison/benchmark scripts
- `models/`
  - Baseline and final model artifacts
- `results/`
  - LOGO metrics and VUS ranking tables
- `figures/`
  - Calibration and ISM visual outputs
- `paper/`
  - Manuscript sources and references

## Reproducibility Workflow

### 1) Environment

Use Python 3.10 and install required dependencies:

```bash
pip install pandas numpy scikit-learn torch transformers matplotlib seaborn joblib tabulate
```

### 2) Execute the Pipeline

From repository root (`/tmp/workspace/abhi8667/HCM`):

```bash
python scripts/execute_month1.py
python scripts/execute_month2.py
python scripts/execute_month3.py
```

### 3) Generated Outputs

- Metrics: `results/logo_metrics.csv`
- VUS ranking: `results/VUS_restratification_table.csv`
- Figures: `figures/`

## Notes on Model Configuration

The manuscript’s publication configuration emphasizes frozen **ESM-2 650M** mutation-centric representations with strict leakage controls and LOGO evaluation. Scripts in this repository include practical execution defaults and can be adjusted to match publication settings exactly where needed.

## Citation

If you use this repository, please cite the manuscript in `paper/`.

## License

MIT License.
