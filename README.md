# Deciphering the Molecular Grammar of Hypertrophic Cardiomyopathy (HCM)
<<<<<<< HEAD

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
=======
### *A Zero-Leakage Protein Language Modeling Approach to Sarcomeric Pathogenicity*

This repository contains the end-to-end variant interpretation pipeline, dataset curation tools, model weights, and publication assets for our research on predicting the clinical pathogenicity of missense mutations in sarcomeric genes associated with Hypertrophic Cardiomyopathy (HCM).

> [!IMPORTANT]
> This project enforces a **strictly zero-leakage** philosophy. By purging all population allele frequencies and prior clinical consensus assertions from model inputs, and using out-of-gene Leave-One-Gene-Out (LOGO) cross-validation, the framework isolates true molecular and evolutionary signals from historical database circularity.

---

## 1. Executive Summary & Core Paper Findings

### What the Paper is About
Interpretation of genetic variants remains the primary translation bottleneck in cardiology: a vast fraction of observed missense mutations in sarcomeric genes are classified as **Variants of Uncertain Significance (VUS)**, stalling cascade screening in families. 
>>>>>>> 689af50 (docs & results: Overhaul manuscript and README, add external benchmarking metrics and plots)

This paper introduces a deep-learning two-tower late-fusion framework that evaluates variants solely on **intrinsic biophysical and evolutionary constraints** without relying on ACMG-derived clinical heuristics.

### Key Claims and Findings
1. **Biophysical Inductive Bias as a Regularizer (The Fusion Trade-off)**:
   * **ESM-only Ablation**: Achieves high raw discriminative power (TNNT2 AUPRC: **0.9404**) but exhibits poor probabilistic calibration (Brier Score: **0.1893**). High-dimensional, unconstrained latent manifolds tend to overfit and produce overconfident posteriors.
   * **Hybrid Two-Tower Fusion**: Sacrifices a marginal amount of raw AUPRC (TNNT2 AUPRC: **0.9151**) but significantly improves calibration (Brier Score: **0.1506**). The 29-feature tabular tower acts as a regularizing biophysical prior, constraining the decision boundary to remain consistent with known physicochemical laws (charge inversion magnitudes, steric mismatch via Grantham distance, domain occupancy). This trade-off is highly desirable for clinical triage where predicted probabilities determine clinical actions.
2. **The *TNNC1* Generalization Boundary**:
   * Evaluated under out-of-gene LOGO transfer, the supervised model collapses to near-inverse prediction on *TNNC1* (AUROC: **0.3500**, $N=36$).
   * *TNNC1* encodes cardiac troponin C (cTnC), a tiny regulatory subunit (161 aa) fundamentally distinct from the massive mechanochemical engines (*MYH7* and *MYBPC3*) that dominate training. Tabular priors learned on motor-protein patterns undergo negative transfer when forced to predict on calcium-sensing loops.
   * Conversely, zero-shot foundational models like EVE (AUROC: **0.5962**) and AlphaMissense (AUROC: **0.6269**) bypass local training cohort scarcity by leveraging global multi-species alignments.
3. **The Hybrid Clinical Deployment Strategy**:
   * We propose a pragmatic deployment paradigm: use the supervised, sarcomere-calibrated Two-Tower model for well-powered genes ($N \ge 50$), and defer to global zero-shot foundational models (EVE/AlphaMissense) for data-scarce regulatory subunits ($N < 40$).
4. **Clinical Triage Utility**:
   * The pipeline retrained on the full annotated corpus was applied to **4,523 ClinVar VUS** across 9 sarcomeric genes, providing an actionable, probabilistically calibrated prioritization heatmap (`results/VUS_restratification_table.csv`) to sequence downstream wet-lab assays.

---

## 2. Directory Structure

```
├── data/                       # Curation datasets
│   ├── HCM_labeled_final.csv   # Labeled variant corpus (N = 1,954)
│   ├── HCM_all_variants_v2.csv # Unlabeled/VUS clinical cohort (4,523 VUS)
│   └── esm2_delta_embeddings.npy # Cached 650M ESM-2 differential vectors
├── scripts/                    # Month-by-month execution pipeline
│   ├── execute_month1.py       # Data purging, feature de-leaking, baseline models
│   ├── execute_month2.py       # PyTorch two-tower training, LOGO evaluation, calibration, ISM
│   └── execute_month3.py       # VUS re-stratification inference & ranking
├── models/                     # Saved model checkpoints and baseline configurations
├── results/                    # Exported metrics and prioritizations
│   ├── VUS_restratification_table.csv # Prioritized triage guide
│   ├── external_model_comparison.csv  # Benchmark metrics vs. EVE & AlphaMissense
│   └── logo_metrics.csv               # Leave-One-Gene-Out model evaluation metrics
├── figures/                    # Publication-grade calibration & ISM landscapes
└── paper/                      # LaTeX manuscript draft and assets
    ├── hcm_paper_manuscript.tex # IEEEtran-formatted paper script
    └── references.bib          # Full bibliography data
```

---

## 3. What Has Been Done (Current Implementation)

Our team has fully executed the primary computational phases:

- [x] **Month 1: Data Purging and Feature De-Leaking**
  * Stripped direct and indirect clinical leakage proxies (`pop_freq`, `disease`, `sources`, `genomic_loc`, `review_status`, `clin_sig`).
  * Structured wild-type ($s_i^{WT}$) and mutant ($s_i^{MUT}$) 11-mer residue windows to extract **mutation-centric differential embeddings** using a frozen `facebook/esm2_t33_650M_UR50D` model: $\Delta \mathbf{e}_i = \mathbf{e}_i^{MUT} - \mathbf{e}_i^{WT}$.
  * Trained and optimized traditional tabular baselines (LOGO Random Forest, tuned XGBoost) establishing majority-class collapse patterns when trained without Focal Loss.

- [x] **Month 2: Two-Tower Deep Learning Framework**
  * Implemented the PyTorch `HybridHCMModel` late-fusion architecture mapping ESM-2 and 29 structural features into a shared 64-dimensional latent space.
  * Optimized training using **Focal Loss** ($\alpha=0.25, \gamma=2$) to address class imbalance.
  * Evaluated robustness using strict **Leave-One-Gene-Out (LOGO)** splits with **bootstrap 95% confidence intervals** and Expected Calibration Error (ECE) monitoring.
  * Generated high-resolution *in silico* mutagenesis (ISM) landscapes for *MYH7*, *MYBPC3*, *TNNT2*, and other sarcomeric targets.

- [x] **Month 3: Clinical VUS Re-stratification**
  * Retrained the two-tower model on the entire annotated corpus.
  * Extracted ESM-2 embeddings for **4,523 clinical VUS candidates** and computed calibrated pathogenicity predictions.
  * Exported the final prioritized target list to `results/VUS_restratification_table.csv`.

- [x] **Manuscript Overhaul**
  * Edited the paper's LaTeX file to systematically fix narrative contradictions, mathematically defend the *TNNC1* generalization boundary, specify complete hyperparameters, and reframe zero-shot comparisons.

---

## 4. What Is To Be Done (Future Roadmap)

To prepare for high-impact journal submission and prospective clinical translation, the team is aligned to execute the following phases:

### Phase A: Gated Fusion Mechanisms (Architectural Upgrade)
* **Objective**: Replace simple concatenation late fusion with a learned gating mechanism, such as **Gated Linear Units (GLUs)** or a **cross-attention module**.
* **Rationale**: Simple concatenation weights biophysical and language features uniformly. A gated tower will allow the network to dynamically suppress the tabular prior when language representation is highly confident and amplify it when ESM features are highly uncertain.
* **Math**: $\mathbf{z}_i^{fus} = \mathbf{z}_i^{tab} \odot \sigma(W_g \mathbf{z}_i^{esm} + \mathbf{b}_g)$.

### Phase B: 3D Coordinate-Level Supervision
* **Objective**: Fuse a third tower using **Graph Neural Networks (GNNs)** trained on physical 3D contact coordinates (e.g., predicted structures from ESMFold/AlphaFold2).
* **Rationale**: Cardiomyopathy missense mutations often disrupt mechanically coupled, allosteric domains that are spatially proximal in 3D but distant in the 1D primary sequence. Adding geometric deep learning will help resolve difficult cases, particularly on rare targets like *TNNC1* where sequence-level representation learning is statistically constrained.

### Phase C: Experimental Wet-Lab Validation
* **Objective**: Cross-reference the highest-ranked prioritized VUS candidates ($P > 0.90$) with targeted functional assays.
* **Rationale**: Validate model predictions using orthogonal, experimentally measured readouts (ATPase kinetics, motility assays, and cellular contractility in engineered cardiac tissues) to confirm the causal mechanical mechanisms predicted by *in silico* mutagenesis.

### Phase D: Prospective Cohort Benchmarking
* **Objective**: Evaluate model calibration and ranking quality on prospective, independent clinical registries (unseen during this development phase).
* **Rationale**: Establish that the zero-leakage model maintains calibration under true clinical distribution shifts, supporting future integration into ACMG-based diagnostic workflows as supportive computational evidence.

## 5. Quick Start (Reproducing Findings)

### 1. Environment Setup
Create a dedicated python environment and install the required dependencies:
```bash
pip install pandas numpy scikit-learn torch transformers matplotlib seaborn joblib tabulate
```

### 2. Running the Pipeline End-to-End
Run the month-by-month scripts from the root directory:
```bash
# Month 1: Data curation, embedding extraction, baseline modeling
python scripts/execute_month1.py

# Month 2: Two-tower PyTorch training, LOGO bootstrapping, calibration, and ISM
python scripts/execute_month2.py

# Month 3: Retraining and full VUS prioritization inference
python scripts/execute_month3.py
```

### 3. Reviewing Metrics & Manuscript
* Comparative metrics vs. EVE and AlphaMissense are stored in [results/external_model_comparison.csv](file:///c:/Users/a7080/.gemini/antigravity/scratch/HCM/results/external_model_comparison.csv).
* The manuscript draft is located at [paper/hcm_paper_manuscript.tex](file:///c:/Users/a7080/.gemini/antigravity/scratch/HCM/paper/hcm_paper_manuscript.tex).

---

## 6. How to Cite
Please cite our work if you utilize this pipeline, embeddings, or VUS prioritization guides:
```bibtex
@article{HCMZeroLeakage2026,
  title={Deciphering the Molecular Grammar of Hypertrophic Cardiomyopathy: A Zero-Leakage Language Modeling Approach to Sarcomeric Pathogenicity},
  author={Computational Genomics Team},
  journal={Bioinformatics (Under Review)},
  year={2026}
}
```
