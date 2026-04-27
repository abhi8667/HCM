# HCM — Unbiased Machine Learning for HCM Variant Pathogenicity Classification

> **Experiential Learning Project | RVCE 2025–26**
> *Utilizing Standing Variation to Reclassify Variants of Uncertain Significance*

---

## Project Overview

This repository contains the dataset, trained model artefacts, interpretability scripts, and Phase II report for a hybrid machine-learning system that classifies missense variants in nine HCM-associated sarcomeric genes as **Pathogenic** or **Benign**.

The system combines a **Tuned XGBoost** tabular model, a **1D-CNN** sequence model, and a **Logistic Regression late-fusion ensemble** to maximise balanced performance under significant class imbalance.

---

## Repository Contents

| File | Description |
|------|-------------|
| `HCM_labeled_final.csv` | Primary dataset — 1,954 labelled variants (9 genes, 67 columns) |
| `HCM_all_variants_v2.csv` | Broader variant set — 8,137 variants including VUS (58 columns) |
| `hcm_final_super_model.joblib` | Serialised model bundle (XGBoost + CNN + meta-learner + scalers + threshold) |
| `generate_interpretability_plots.py` | Produces SHAP beeswarm and Integrated-Gradients heatmap |
| `performance_summary.csv` | Per-model ROC-AUC, PR-AUC, Accuracy, Balanced Accuracy, F1, Recall |
| `phase2_report.tex` | Full Phase II LaTeX report source |
| `Windows PowerShell.txt` | Training run log (ground-truth output from `main.py`) |
| `Latest model Information-.txt` | Architecture summary and MCC note |
| `*.png` | Confusion matrix, ROC, PR, calibration, SHAP, IG heatmap plots |

---

## Quick-Start

```bash
pip install pandas numpy scikit-learn xgboost tensorflow joblib shap matplotlib

# Reproduce interpretability plots
python generate_interpretability_plots.py
```

---

## Model Architecture Summary

### Branch 1 — Tuned XGBoost (Tabular)
- Input: 29 hand-crafted biochemical/structural/positional features
- `scale_pos_weight = 2.809` (class-imbalance correction)
- 500 trees, max_depth = 6

### Branch 2 — 1D-CNN (Sequence + Auxiliary)
| Branch | Layer | Configuration |
|--------|-------|---------------|
| Sequence | Conv1D | 32 filters, kernel=3, padding=same, ReLU |
| Sequence | BatchNormalization | — |
| Sequence | GlobalMaxPooling1D | — |
| Auxiliary | Dense | 64 units, ReLU |
| Auxiliary | Dropout | rate=0.3 |
| Auxiliary | Dense | 32 units, ReLU |
| Auxiliary | Dropout | rate=0.3 |
| Fusion | Concatenate | CNN (32) + MLP (32) |
| Fusion | Dense | 32 units, ReLU |
| Fusion | Dropout | rate=0.3 |
| Fusion | Dense | 16 units, ReLU |
| Fusion | Dense | 1 unit, Sigmoid |

### Branch 3 — Late-Fusion Hybrid Ensemble
- Logistic Regression meta-learner trained on OOF probabilities
- `class_weight='balanced'`
- Learned coefficients: XGBoost=**3.868**, CNN=**3.572** (intercept=−5.134)

---

## Performance (5-Fold Stratified Cross-Validation, threshold=0.5294)

| Model | ROC-AUC | PR-AUC | Accuracy | Balanced Acc | F1-Macro | Recall Benign | Recall Pathogenic |
|-------|---------|--------|----------|--------------|----------|---------------|-------------------|
| Tuned XGBoost | 0.7402 | 0.8639 | 0.7702 | 0.5818 | 0.5800 | 0.1852 | 0.9785 |
| 1D-CNN | 0.7934 | 0.9049 | 0.7288 | 0.7257 | 0.6906 | 0.7193 | 0.7321 |
| **Hybrid Ensemble** | **0.8222** | **0.9141** | 0.7421 | **0.7529** | **0.7095** | **0.7758** | 0.7300 |

**Hybrid Ensemble Confusion Matrix** (N=1,954):

|  | Predicted Benign | Predicted Pathogenic |
|--|-----------------|---------------------|
| **Actual Benign** | 398 (TN) | 115 (FP) |
| **Actual Pathogenic** | 389 (FN) | 1,052 (TP) |

MCC = (1052×398 − 115×389) / √(1167×1441×513×1441) = **0.4539**

---

## 51-Feature Schema

| Category | Features | Count |
|----------|----------|-------|
| Physicochemical | Grantham score, ref/alt size & charge, size_change, charge_change | 7 |
| Structural/Functional | in_domain, in_coiled, in_helix, in_strand, in_turn, in_secondary, in_region, in_disordered, in_compbias, in_functional_site, in_ptm_site | 11 |
| Evolutionary/Positional | rel_position, pop_freq | 2 |
| Gene Identity | is_MYH7, is_MYBPC3, is_TNNT2, is_TNNI3, is_TPM1, is_MYL2, is_MYL3, is_ACTC1, is_TNNC1 | 9 |
| Sequence Window (CNN) | 11 positions × (size, charge) — indices −5 to +5 | 22 |
| **Total** | | **51** |

---

---

# Forensic Compliance Audit — Phase II Report

> **Role:** Forensic Data Scientist & Technical Auditor
> **Audit Date:** 2026-04-27
> **Audited Artefacts:** `HCM_labeled_final.csv`, `hcm_final_super_model.joblib`, `generate_interpretability_plots.py`, `Windows PowerShell.txt`, `performance_summary.csv`, `phase2_report.tex`

---

## CONFIRMED CLAIMS

### Task 1 — Dataset Verification

| Claim (Report) | Verified Value | Status |
|----------------|----------------|--------|
| Total variants = **1,954** | `len(HCM_labeled_final.csv)` = **1,954** | ✅ CONFIRMED |
| Pathogenic (1) = **73.7 %** (1,441 variants) | 1,441 / 1,954 = **73.7 %** | ✅ CONFIRMED |
| Benign (0) = **26.3 %** (513 variants) | 513 / 1,954 = **26.3 %** | ✅ CONFIRMED |
| MYH7 n = **986** | `df['gene'].value_counts()['MYH7']` = **986** | ✅ CONFIRMED |
| MYBPC3 n = **429** | = **429** | ✅ CONFIRMED |
| TNNT2 n = **130** | = **130** | ✅ CONFIRMED |
| TNNI3 n = **118** | = **118** | ✅ CONFIRMED |
| TPM1 n = **100** | = **100** | ✅ CONFIRMED |
| ACTC1 n = **64** | = **64** | ✅ CONFIRMED |
| MYL2 n = **54** | = **54** | ✅ CONFIRMED |
| MYL3 n = **37** | = **37** | ✅ CONFIRMED |
| TNNC1 n = **36** | = **36** | ✅ CONFIRMED |

### Task 2 — Feature Engineering

| Claim (Report) | Verified Value | Status |
|----------------|----------------|--------|
| 51 total features (29 tabular + 22 window) | `bundle['feature_names_aux']` = 29, `bundle['feature_names_seq']` = 22 → **51** | ✅ CONFIRMED |
| Sliding window: 11 positions (−5 to +5) | `WINDOW_HALF = 5`, columns `win_-5_*` … `win_+5_*` (11 positions × 2 channels = 22) | ✅ CONFIRMED |
| Physicochemical features (7) | Grantham + ref/alt size/charge + size_change + charge_change = **7** in `feature_names_aux` | ✅ CONFIRMED |
| Structural/Functional indicators (11) | in_domain … in_ptm_site = **11** in `feature_names_aux` | ✅ CONFIRMED |
| Gene one-hot encoding (9 genes) | is_MYH7 … is_TNNC1 = **9** in `feature_names_aux` | ✅ CONFIRMED |

### Task 3 — Model Architecture

| Claim (Report) | Verified Value | Status |
|----------------|----------------|--------|
| Conv1D: **32 filters, kernel=3**, padding=same, ReLU | `layer.get_config()` → filters=32, kernel_size=(3,), padding=same, activation=relu | ✅ CONFIRMED |
| **BatchNormalization** after Conv1D | Layer order confirmed in CNN summary | ✅ CONFIRMED |
| GlobalMaxPooling1D | Confirmed in CNN layer list | ✅ CONFIRMED |
| Auxiliary Dense: **64 → Dropout → 32 → Dropout** | Confirmed in CNN layers | ✅ CONFIRMED |
| XGBoost **`scale_pos_weight`** implemented | `booster.save_config()` → `scale_pos_weight = 2.80896688` | ✅ CONFIRMED |
| **5-Fold Stratified** cross-validation | Training log shows "Base model fold 1/5 … fold 5/5"; `class distribution: benign=513 (26.3%), pathogenic=1441 (73.7%)` printed per run | ✅ CONFIRMED |
| Meta-learner = **Logistic Regression** with `class_weight='balanced'` | `type(bundle['meta'])` = LogisticRegression, `class_weight='balanced'` | ✅ CONFIRMED |
| Optimal threshold = **0.5294** | `bundle['threshold']` = **0.52944171…** | ✅ CONFIRMED |

### Task 4 — Results Reconciliation

| Claim (Report) | Verified Value | Status |
|----------------|----------------|--------|
| Confusion matrix: TN=**398**, FP=**115**, FN=**389**, TP=**1,052** | `Windows PowerShell.txt` line 60–61: `[[ 398 115] / [ 389 1052]]` | ✅ CONFIRMED |
| MCC ≈ **0.45** (table) | Calculated from CM: (1052×398 − 115×389) / √(1167×1441×513×1441) = **0.4539** ≈ 0.45 | ✅ CONFIRMED |
| Pathogenic Grantham mean = **79.78** | `df[df.label==1]['grantham_score'].mean()` = **79.7814** | ✅ CONFIRMED |
| Benign Grantham mean = **60.85** | `df[df.label==0]['grantham_score'].mean()` = **60.8538** | ✅ CONFIRMED |
| Ensemble ROC-AUC = **0.8222** | `performance_summary.csv` Ensemble = **0.822241** | ✅ CONFIRMED |
| Ensemble PR-AUC = **0.9141** | = **0.914109** | ✅ CONFIRMED |
| XGBoost Benign Recall = **18.5 %** | = **0.185185 (18.5 %)** | ✅ CONFIRMED |
| XGBoost Pathogenic Recall = **97.8 %** | = **0.978487 (97.8 %)** | ✅ CONFIRMED |
| CNN Benign Recall = **71.9 %** | = **0.719298 (71.9 %)** | ✅ CONFIRMED |
| Ensemble Benign Recall = **77.6 %** | = **0.775828 (77.6 %)** | ✅ CONFIRMED |

---

## UNVERIFIED / ANOMALOUS CLAIMS

### Anomaly 1 — `add_onehot.py` Script Does Not Exist in Repository

> **Report Section:** "Verify the add_onehot.py logic"

**Finding:** No file named `add_onehot.py` (or any equivalent preprocessing script) exists anywhere in the repository. The one-hot gene encoding (`is_MYH7`, `is_MYBPC3`, …, `is_TNNC1`) is pre-computed and stored directly in `HCM_labeled_final.csv`. The sliding-window columns (`win_-5_size`, …, `win_+5_charge`) are also pre-computed in the CSV. The upstream preprocessing code that generated these columns has **not been committed to the repository**. Similarly, `main.py` (the training script) is absent — only its PowerShell execution log remains.

**Impact:** Reproducibility is limited. The feature-engineering pipeline cannot be re-run from source without `main.py` or the preprocessing scripts. The window logic (−5 to +5) is verifiable from the column names, but the Python logic producing those columns cannot be audited directly.

---

### Anomaly 2 — MCC Value Inconsistency Across Sources

> **Report Table 2:** MCC ≈ **0.45**
> **`Latest model Information-.txt`:** "approx value is **0.4512**"
> **Calculated from CM (TP=1052, TN=398, FP=115, FN=389):** **0.4539**

**Finding:** Three different MCC values appear across project files: the report gives ≈0.45, the model-info text gives 0.4512, and direct calculation from the logged confusion matrix yields 0.4539. All three round to ≈0.45, so there is no fundamental error, but the inconsistency suggests MCC was never calculated explicitly in the training code (confirmed by the note "MCC was not explicitly derived in the code" in `Latest model Information-.txt`). The most accurate value is **0.4539**, derived directly from the confusion matrix that was logged in the training output.

---

### Anomaly 3 — SHAP Analysis in `generate_interpretability_plots.py` Uses Wrong Feature Set

> **File:** `generate_interpretability_plots.py`, lines 78–81
> **Report Section 4.1:** "SHAP analysis of the **XGBoost** model identifies…Grantham score and relative position dominate"

**Finding:** The SHAP plot is computed against `feature_names_seq` (the 22 **window** features), not `feature_names_aux` (the 29 **tabular** features that XGBoost was actually trained on):

```python
# Line 78–81 (ANOMALY — wrong feature matrix)
X_tabular = pd.DataFrame(X_seq, columns=feature_names_seq)   # ← uses SEQ features
explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer(X_tabular)                            # ← XGBoost was NOT trained on these
```

The XGBoost model (`bundle['xgb']`) was trained exclusively on `feature_names_aux` (which includes Grantham score, rel_position, structural flags, gene one-hot, etc.). Passing the 22-column window feature matrix to TreeExplainer produces incorrect SHAP values and a misleading beeswarm plot. **The `shap_beeswarm.png` in the repository is therefore unreliable.** The correct call should use `X_aux` with `feature_names_aux`.

**Specific line:** `generate_interpretability_plots.py`, line 78 — replace `X_seq` with `X_aux` and `feature_names_seq` with `feature_names_aux`.

---

### Anomaly 4 — Report Describes 51 Features but CSV Contains 55 Candidate Columns

> **Report Section 2.2:** "Fifty-one features were engineered…"

**Finding:** `HCM_labeled_final.csv` contains 55 numeric/categorical columns that could naively be counted as features (`position`, `protein_length`, `ref_aa`, `alt_aa`, plus the 51 confirmed model inputs). The four extra columns (`position`, `protein_length`, `ref_aa`, `alt_aa`) are **not** present in `bundle['feature_names_aux']` and are therefore not fed to any model. They are intermediate engineering columns retained in the CSV but excluded from model input. The report's "51" count is correct for the *model input*, but the CSV is slightly misleading as it includes these extra columns without labelling them as non-model columns.

---

### Anomaly 5 — Meta-Learner "Mathematical Weights" Not Stated in Report

> **Report Section 2.3:** "A Logistic Regression meta-learner…learns the **optimal linear combination**"

**Finding:** The report describes the meta-learner qualitatively but never states the actual learned coefficients. From the saved model:
- **XGBoost probability coefficient:** 3.8676
- **CNN probability coefficient:** 3.5717
- **Intercept:** −5.1338
- **Ratio:** XGBoost weight is ~8.4% higher than CNN weight (3.868 vs. 3.572), indicating the meta-learner assigns slightly more weight to the tabular model's probability.

This is not an error — the report does not claim specific values — but the trained weights are now documented here for completeness.

---

## Audit Summary

| Category | Confirmed | Anomalous/Unverified |
|----------|-----------|----------------------|
| Dataset statistics | 11/11 | 0 |
| Feature engineering | 5/5 | 1 (add_onehot.py absent) |
| Model architecture | 8/8 | 0 |
| Training protocol | 2/2 | 0 |
| Results & metrics | 10/10 | 1 (SHAP bug in plot script) |
| Numerical precision | All within rounding | MCC inconsistency (minor) |

**Overall verdict:** The core quantitative claims in the Phase II report — dataset size, class distribution, gene counts, Grantham means, confusion matrix, ROC/PR-AUC, recall values, and model architecture — are **fully confirmed** by the artefacts in the repository. Two actionable anomalies were identified: (1) the upstream preprocessing/training code is not committed (reproducibility gap), and (2) `generate_interpretability_plots.py` computes SHAP values on the wrong feature matrix, making the beeswarm plot unreliable.

---

*Audit conducted against commit on branch `copilot/audit-dataset-verification` — 2026-04-27.*