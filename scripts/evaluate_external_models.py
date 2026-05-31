import os
import sys
import argparse
import gzip
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score
import urllib.request
import urllib.error

# Setup Directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
EVE_CACHE_DIR = os.path.join(DATA_DIR, "eve_scores")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(EVE_CACHE_DIR, exist_ok=True)

# Genes and Accessions
GENE_ACCESSION = {
    "MYH7":   "P12883",
    "MYBPC3": "Q14896",
    "TNNT2":  "P45379",
    "TNNI3":  "P19429",
    "TPM1":   "P09493",
    "ACTC1":  "P68032",
    "MYL2":   "P10916",
    "MYL3":   "P08590",
    "TNNC1":  "P63316",
}

# ----------------- EVE LOGIC -----------------
def fetch_eve_csv(gene, accession):
    cache_path = os.path.join(EVE_CACHE_DIR, f"{accession}_eve.csv")
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    
    url = f"https://evemodel.org/api/proteins/{accession}/download/"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            raw_text = r.read().decode("utf-8")
        with open(cache_path, "w", encoding="utf-8") as fh:
            fh.write(raw_text)
        return pd.read_csv(cache_path)
    except Exception as e:
        print(f"  [{gene}] Could not fetch EVE data: {e}")
        return None

def process_eve_df(df, accession):
    if df is None: return None
    # Assuming columns like 'position', 'wt_aa', 'mt_aa', 'EVE_scores_ASM' are present
    pos_col = next((c for c in ["position", "protein_position"] if c in df.columns), None)
    ref_col = next((c for c in ["wt_aa", "ref_aa"] if c in df.columns), None)
    alt_col = next((c for c in ["mt_aa", "alt_aa"] if c in df.columns), None)
    score_col = next((c for c in ["EVE_scores_ASM", "EVE_score", "EVE_scores"] if c in df.columns), None)
    
    if not all([pos_col, ref_col, alt_col, score_col]):
        return None
        
    out = pd.DataFrame({
        "accession": accession,
        "position": pd.to_numeric(df[pos_col], errors="coerce"),
        "ref_aa": df[ref_col].astype(str).str.upper(),
        "alt_aa": df[alt_col].astype(str).str.upper(),
        "eve_score": pd.to_numeric(df[score_col], errors="coerce")
    }).dropna()
    return out

# ----------------- ALPHAMISSENSE LOGIC -----------------
def process_alphamissense(am_file, accessions):
    # Extracts relevant accessions from the massive AM file
    results = []
    print("  Scanning AlphaMissense file... (This may take a while)")
    try:
        # Check if gzip or normal
        f = gzip.open(am_file, 'rt') if am_file.endswith('.gz') else open(am_file, 'r')
        for i, line in enumerate(f):
            if line.startswith('#') or line.startswith('uniprot_id'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 4 and parts[0] in accessions:
                # uniprot_id, protein_variant, am_pathogenicity, am_class
                uniprot_id = parts[0]
                variant = parts[1] # e.g., M1T
                am_score = float(parts[2])
                ref_aa = variant[0]
                alt_aa = variant[-1]
                pos = int(variant[1:-1])
                results.append([uniprot_id, pos, ref_aa, alt_aa, am_score])
        f.close()
    except Exception as e:
        print(f"  Error reading AlphaMissense file: {e}")
        return None
    
    return pd.DataFrame(results, columns=["accession", "position", "ref_aa", "alt_aa", "am_score"])

# ----------------- SYNTHETIC GENERATORS -----------------
def make_synthetic_scores(hcm_df, accession, col_name, noise_scale):
    rows = hcm_df[hcm_df["accession"] == accession].copy()
    if len(rows) == 0: return None
    rng = np.random.default_rng(seed=hash(col_name) % 2**32)
    noise = rng.normal(0, noise_scale, size=len(rows))
    rows[col_name] = (rows["grantham_score"] + noise).clip(0, 215) / 215.0
    return rows[["accession", "position", "ref_aa", "alt_aa", col_name]]

# ----------------- MAIN SCRIPT -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic scores")
    args = parser.parse_args()

    print("=================================================================")
    print("  Unified External Validation Benchmark (EVE & AlphaMissense)")
    print("=================================================================")

    # 1. Load HCM labeled data
    hcm_csv = os.path.join(DATA_DIR, "HCM_labeled_final.csv")
    hcm = pd.read_csv(hcm_csv)
    if hcm["label"].dtype == object:
        hcm["label"] = (hcm["label"].str.strip().isin(["Pathogenic", "1", "True"])).astype(int)
    
    # 2. Load Two-Tower Results
    # Try looking for tuned results first, then logo_metrics
    tt_results_path = os.path.join(RESULTS_DIR, "model_comparison_metrics_all_genes_tuned.csv")
    if not os.path.exists(tt_results_path):
        tt_results_path = os.path.join(RESULTS_DIR, "logo_metrics.csv")
        
    tt_auprc, tt_auroc = {}, {}
    if os.path.exists(tt_results_path):
        tt_df = pd.read_csv(tt_results_path)
        # Filter to our model
        our_model = tt_df[tt_df["Model"].str.contains("Two-Tower", case=False, na=False)] if "Model" in tt_df.columns else tt_df
        for _, row in our_model.iterrows():
            g = row["Gene"]
            if "AUPRC" in row: tt_auprc[g] = float(row["AUPRC"])
            if "AUROC" in row: tt_auroc[g] = float(row["AUROC"])
    
    # 3. Load AlphaMissense Data
    am_df = None
    if args.dry_run:
        print("[DRY-RUN] Will generate synthetic AlphaMissense scores.")
    else:
        am_file = os.path.join(DATA_DIR, "AlphaMissense_aa_substitutions.tsv.gz")
        if not os.path.exists(am_file):
            print(f"WARNING: AlphaMissense file not found at {am_file}.")
            print("Please download it from https://github.com/google-deepmind/alphamissense or run with --dry-run.")
        else:
            am_df = process_alphamissense(am_file, set(GENE_ACCESSION.values()))

    # 4. Process each gene
    results = []
    
    for gene, accession in GENE_ACCESSION.items():
        print(f"\n[{gene}] Processing...")
        hcm_gene = hcm[hcm["gene"] == gene].copy()
        if len(hcm_gene) == 0: continue
        
        # EVE
        if args.dry_run:
            eve_scores = make_synthetic_scores(hcm_gene, accession, "eve_score", 15)
        else:
            eve_raw = fetch_eve_csv(gene, accession)
            eve_scores = process_eve_df(eve_raw, accession)
            
        # AlphaMissense
        if args.dry_run:
            am_scores = make_synthetic_scores(hcm_gene, accession, "am_score", 10)
        else:
            if am_df is not None:
                am_scores = am_df[am_df["accession"] == accession]
            else:
                am_scores = None
        
        # Merge
        merged = hcm_gene
        if eve_scores is not None:
            merged = merged.merge(eve_scores, on=["accession", "position", "ref_aa", "alt_aa"], how="left")
        else:
            merged["eve_score"] = np.nan
            
        if am_scores is not None:
            merged = merged.merge(am_scores, on=["accession", "position", "ref_aa", "alt_aa"], how="left")
        else:
            merged["am_score"] = np.nan

        # Evaluate
        eval_df = merged.dropna(subset=["eve_score", "am_score"])
        if len(eval_df) < 5 or len(eval_df["label"].unique()) < 2:
            print(f"  [{gene}] Insufficient data for comparison.")
            continue
            
        y_true = eval_df["label"].values
        eve_pred = eval_df["eve_score"].values
        am_pred = eval_df["am_score"].values
        
        if np.corrcoef(y_true, eve_pred)[0, 1] < 0: eve_pred = -eve_pred
        if np.corrcoef(y_true, am_pred)[0, 1] < 0: am_pred = -am_pred
        
        eve_auprc_val = average_precision_score(y_true, eve_pred)
        eve_auroc_val = roc_auc_score(y_true, eve_pred)
        
        am_auprc_val = average_precision_score(y_true, am_pred)
        am_auroc_val = roc_auc_score(y_true, am_pred)
        
        res = {
            "Gene": gene,
            "Two_Tower_AUPRC": tt_auprc.get(gene, np.nan),
            "EVE_AUPRC": eve_auprc_val,
            "AlphaMissense_AUPRC": am_auprc_val,
            "Two_Tower_AUROC": tt_auroc.get(gene, np.nan),
            "EVE_AUROC": eve_auroc_val,
            "AlphaMissense_AUROC": am_auroc_val,
            "N_evaluated": len(eval_df)
        }
        results.append(res)
        print(f"  [{gene}] AUPRC: Two-Tower={res['Two_Tower_AUPRC']:.3f} | EVE={res['EVE_AUPRC']:.3f} | AM={res['AlphaMissense_AUPRC']:.3f}")

    # 5. Save and Plot
    if not results:
        print("No evaluation data generated.")
        return
        
    res_df = pd.DataFrame(results)
    csv_out = os.path.join(RESULTS_DIR, "external_model_comparison.csv")
    res_df.to_csv(csv_out, index=False)
    print(f"\nSaved comparison to {csv_out}")
    
    # Plotting
    genes = res_df["Gene"]
    x = np.arange(len(genes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, res_df["Two_Tower_AUPRC"], width, label="Two-Tower (Ours)", color="#2563EB")
    ax.bar(x, res_df["EVE_AUPRC"], width, label="EVE", color="#DC2626")
    ax.bar(x + width, res_df["AlphaMissense_AUPRC"], width, label="AlphaMissense", color="#059669")
    ax.set_xticks(x)
    ax.set_xticklabels(genes)
    ax.set_title("AUPRC Comparison: Two-Tower vs EVE vs AlphaMissense")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "external_pr_comparison.png"), dpi=150)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, res_df["Two_Tower_AUROC"], width, label="Two-Tower (Ours)", color="#2563EB")
    ax.bar(x, res_df["EVE_AUROC"], width, label="EVE", color="#DC2626")
    ax.bar(x + width, res_df["AlphaMissense_AUROC"], width, label="AlphaMissense", color="#059669")
    ax.set_xticks(x)
    ax.set_xticklabels(genes)
    ax.set_title("AUROC Comparison: Two-Tower vs EVE vs AlphaMissense")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "external_roc_comparison.png"), dpi=150)
    
    print(f"Saved charts to {FIGURES_DIR}")

if __name__ == "__main__":
    main()
