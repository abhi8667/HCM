# HCM Variant Dataset & Feature Engineering Pipeline

This directory contains the curated datasets used to train and validate our HCM variant pathogenicity prediction models. One of the core novelties of this project is the construction of a strictly de-leaked, highly structured machine learning dataset that unifies structural, evolutionary, and clinical evidence.

---

## Data Integration & Curation Flowchart

To construct the final training and validation datasets, we designed and executed the following structured data integration pipeline:

![Data Integration Flowchart](../figures/data_pipeline_flowchart.png)

---

## Data Sources & Pipeline Architecture

Our curated data pipeline is built upon the following components:

### 1. The Core Foundation
We unify **UniProt Swiss-Prot** (sequences, domains, and 3D-structural annotations) and **ClinVar** (clinical variant classifications, review status, and assertions) via the **EBI Proteins API** using a unified query interface. This allows us to map each clinical variant directly to its precise physical and biochemical structural environment.

### 2. The gnomAD Filter (Rare vs. Common)
We pull population allele frequencies and candidate benign variants from **gnomAD** (representing population allele frequencies from **~730,000 individuals**). 
* **Rare variants** without established clinical significance are prioritized for investigation.
* **Common variants** (high population allele frequency) are utilized as high-confidence **benign proxies**, preventing the standard literature-derived model bias where benign variants are underrepresented.

### 3. Feature Engineering
Our pipeline converts the raw API and database JSON structures into **51 numerical and categorical features** fully encoded for machine learning. These features describe:
* Grantham physicochemical distance scores.
* Mutation-induced chemical shifts and properties.
* Domain-level context (e.g., motor domain vs. coiled-coil vs. regulatory zone).
* Critical site markers (binding sites, secondary structures, PTM sites).
* Relative residue position in the folded chain.

### 4. Final Output
The curation pipeline outputs a premium, high-confidence dataset of **1,954 labeled missense variants** across **9 sarcomeric HCM genes**, fully structured and ready for model training, alongside a larger dataset of **8,137 variants** (including unresolved VUS candidates) for downstream risk ranking and re-stratification.

---

## Dataset Structure & Features

The dataset is divided into two primary files in this directory:

### 1. Labeled Training Dataset (`HCM_labeled_final.csv`)
* **Size:** 1,954 rows, 67 columns (51 encoded ML features + annotations).
* **Class Distribution:** 1,441 Pathogenic (73.7%) and 513 Benign (26.3%).
* **Variant Labeling:** Labeled as Pathogenic or Benign based on high-confidence ClinVar assertions (2+ stars, no conflicting reports) or gnomAD benign proxies.
* **Zero-Leakage Guarantee:** Crucially, all ACMG clinical criteria, sub-features, population frequencies, and clinical meta-predictors were removed from the input feature set before training to avoid label circularity.

### 2. Full Variant & VUS Dataset (`HCM_all_variants_v2.csv`)
* **Size:** 8,137 rows.
* **Purpose:** Includes all labeled variants plus **4,523+ Variants of Uncertain Significance (VUS)**. This file is used in Phase III (`execute_month3.py`) to perform high-throughput in silico screening and generate the VUS re-stratification risk rankings.

### 3. ESM-2 Embeddings (`esm2_delta_embeddings.npy`)
* **Size:** 1,954 rows × 1,280 dimensions.
* **Format:** Numpy array binary.
* **Description:** Precomputed differential embeddings (`Δe = e_mut - e_wt`) extracted from the 33-layer, 650M-parameter `esm2_t33_650M_UR50D` protein language model. These capture the exact physical perturbation caused by each mutation in evolutionary latent space.

---

## Dataset Features Cheat-Sheet

Here is how the engineered features map to biological properties:
* **Structural Mapping:** We decompose proteins into their specific functional zones (e.g., myosin converter domain, actin-binding cleft, or troponin regulatory regions).
* **Critical Site Markers:** The dataset contains boolean flags indicating if a residue lies in a ligand-binding site, a metal-binding site, an active site, a phosphorylation/PTM site, or a specific secondary structure (alpha-helix, beta-strand).
* **Predictive Drivers:** Features like **Grantham score** (measuring physicochemical distance between wildtype and mutant amino acids) represent structural change, while the structural accessibility flags denote if a residue is buried or surface-exposed.
* **Contextual Learning:** By linking each variant to its exact structural position, the model is trained to learn disease-specific patterns rather than just general rules.

For execution instructions to recreate these datasets from raw queries, see [reproducibility_README.md](../reproducibility_README.md).
