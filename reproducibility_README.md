# Deciphering the Molecular Grammar of HCM (Implementation)

This repository contains the code and data necessary to reproduce the findings in our paper: *"Deciphering the molecular grammar of Hypertrophic Cardiomyopathy: A zero-leakage language modeling approach to sarcomeric pathogenicity."*

## Repository Structure

*   **`data/`**: Contains the raw and processed variant files (`HCM_labeled_final.csv`, `HCM_all_variants_v2.csv`). **Note:** All ClinVar meta-predictors and ACMG-derived clinical features have been explicitly purged to prevent data leakage.
*   **`models/`**: 
    *   `hcm_logo_baseline_model.joblib`: The baseline Random Forest model.
    *   `hybrid_model.pt`: The PyTorch Two-Tower Hybrid model weights (ESM-2 + Structural).
*   **`scripts/`**:
    *   `execute_month1.py`: Dataset cleaning, Leave-One-Gene-Out (LOGO) splits, and ESM-2 embedding extraction.
    *   `execute_month2.py`: PyTorch hybrid model training, Bootstrapping evaluation, Calibration (Brier score), and In Silico Mutagenesis (ISM) landscape generation.
    *   `execute_month3.py`: VUS re-stratification and identification of highly suspicious variants.
*   **`figures/`**: Generated outputs such as `calibration_plot_m2.png` and `ism_landscape_m2.png`.

## Reproducing the Results

We have ensured strict reproducibility of our pipeline.

1.  **Environment Setup:**
    ```bash
    conda create -n hcm_env python=3.10
    conda activate hcm_env
    pip install -r requirements.txt
    ```
2.  **Run the Pipeline End-to-End:**
    ```bash
    python execute_month1.py
    python execute_month2.py
    python execute_month3.py
    ```

## Key Methodological Upgrades
Unlike previous classifiers, this repository strictly adheres to **zero-leakage** validation:
1.  **Leave-One-Gene-Out (LOGO) Splits:** Proves the model learns generic structural vulnerabilities across the sarcomere rather than overfitting to specific gene distributions.
2.  **Evolutionary Embeddings:** Utilizes unsupervised `facebook/esm2_t33_650M_UR50D` representations, completely orthogonal to clinical reporting biases.
3.  **In Silico Mutagenesis (ISM):** Replaces generic global feature importance with structurally mappable mutation sensitivity landscapes.

## License
MIT License. Please cite our paper if you use this codebase.
