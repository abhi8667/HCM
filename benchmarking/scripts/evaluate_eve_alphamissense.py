"""
evaluate_eve_alphamissense.py
==============================
Benchmarks the HCM dual-model pipeline (Two-Tower BCE + Baseline RF)
against EVE and AlphaMissense.

Run from anywhere:
    python benchmarking/scripts/evaluate_eve_alphamissense.py

    # Offline/test mode (synthetic scores — no internet or local files needed):
    python benchmarking/scripts/evaluate_eve_alphamissense.py --dry-run

Reads:
    <HCM_ROOT>/data/HCM_labeled_final.csv
    <HCM_ROOT>/results/logo_metrics.csv  (Two-Tower + Baseline RF AUPRC scores)

    EVE scores:   fetched from evemodel.org API and cached in benchmarking/data/eve_scores/
    AlphaMissense: expects local file at:
                  <HCM_ROOT>/data/AlphaMissense_aa_substitutions.tsv.gz
                  Download from: https://github.com/google-deepmind/alphamissense

Writes:
    <HCM_ROOT>/benchmarking/results/external_model_comparison.csv
    <HCM_ROOT>/benchmarking/figures/external_pr_comparison.png
    <HCM_ROOT>/benchmarking/figures/external_roc_comparison.png

Bug Fix (vs original evaluate_external_models.py):
    AlphaMissense and EVE encode scores at the NUCLEOTIDE level, so one protein
    variant (gene, position, ref_aa, alt_aa) can appear multiple times due to
    codon degeneracy. Without deduplication, a left-merge inflates N_evaluated
    beyond the actual number of HCM variants and skews AUPRC/AUROC.
    FIX: both EVE and AM scores are deduplicated (mean per protein variant)
    before merging, ensuring a strict 1-to-1 protein-variant match.

Model Update (dual-model pipeline):
    Results now compare Two-Tower Hybrid BCE AND Baseline RF against EVE and
    AlphaMissense. Both model AUPRCs are loaded from results/logo_metrics.csv.
"""

import os
import sys
import argparse
import gzip
import warnings
import urllib.request
import urllib.error

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Script lives at: <HCM_ROOT>/benchmarking/scripts/
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR    = os.path.dirname(SCRIPT_DIR)           # benchmarking/
PROJECT_ROOT = os.path.dirname(BENCH_DIR)            # HCM/

# Inputs from main project
HCM_CSV     = os.path.join(PROJECT_ROOT, "data",    "HCM_labeled_final.csv")
AM_FILE     = os.path.join(PROJECT_ROOT, "data",    "AlphaMissense_aa_substitutions.tsv.gz")
# logo_metrics.csv contains AUPRC for both Two-Tower Hybrid and Baseline RF
LOGO_CSV    = os.path.join(PROJECT_ROOT, "results", "logo_metrics.csv")

# Outputs inside benchmarking/ only
EVE_CACHE   = os.path.join(BENCH_DIR, "data",    "eve_scores")
RESULTS_DIR = os.path.join(BENCH_DIR, "results")
FIGURES_DIR = os.path.join(BENCH_DIR, "figures")

for d in (EVE_CACHE, RESULTS_DIR, FIGURES_DIR):
    os.makedirs(d, exist_ok=True)

OUT_CSV = os.path.join(RESULTS_DIR, "external_model_comparison.csv")

# ---------------------------------------------------------------------------
# Gene → UniProt accession (9 sarcomeric HCM genes)
# ---------------------------------------------------------------------------
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

MERGE_KEYS = ["accession", "position", "ref_aa", "alt_aa"]


# ---------------------------------------------------------------------------
# EVE helpers
# ---------------------------------------------------------------------------
def fetch_eve_csv(gene: str, accession: str):
    """Download EVE scores for one protein from evemodel.org (with local cache)."""
    cache_path = os.path.join(EVE_CACHE, f"{accession}_eve.csv")

    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 100:
        print(f"  [{gene}] Loading EVE from cache: {cache_path}")
        return pd.read_csv(cache_path)

    url = f"https://evemodel.org/api/proteins/{accession}/download/"
    print(f"  [{gene}] Downloading EVE from: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read().decode("utf-8")
        with open(cache_path, "w", encoding="utf-8") as fh:
            fh.write(raw)
        return pd.read_csv(cache_path)
    except urllib.error.HTTPError as e:
        print(f"  [{gene}] HTTP {e.code} — EVE not available.")
    except Exception as e:
        print(f"  [{gene}] EVE download failed: {type(e).__name__}: {e}")
    return None


def process_eve(df, accession: str):
    """
    Normalise EVE dataframe → (accession, position, ref_aa, alt_aa, eve_score).

    BUG FIX: deduplicate on protein coordinates (mean score) so that codon-
    degenerate rows do not inflate N_evaluated when merged.
    """
    if df is None:
        return None

    pos_col   = next((c for c in ["position", "protein_position", "pos"]           if c in df.columns), None)
    ref_col   = next((c for c in ["wt_aa",  "ref_aa", "WT_aa",  "wild_type_aa"]   if c in df.columns), None)
    alt_col   = next((c for c in ["mt_aa",  "alt_aa", "MUT_aa", "mutant_aa"]      if c in df.columns), None)
    score_col = next((c for c in ["EVE_scores_ASM", "EVE_scores", "EVE_score", "evol_indices"] if c in df.columns), None)

    if not all([pos_col, ref_col, alt_col, score_col]):
        print(f"    WARNING: Could not identify required EVE columns. Found: {list(df.columns[:8])}")
        return None

    out = pd.DataFrame({
        "accession": accession,
        "position":  pd.to_numeric(df[pos_col],   errors="coerce"),
        "ref_aa":    df[ref_col].astype(str).str.strip().str.upper(),
        "alt_aa":    df[alt_col].astype(str).str.strip().str.upper(),
        "eve_score": pd.to_numeric(df[score_col], errors="coerce"),
    }).dropna(subset=["eve_score", "position"])
    out["position"] = out["position"].astype(int)

    # ── DEDUP FIX ──────────────────────────────────────────────────────────
    # Multiple nucleotide-level rows can map to the same protein variant.
    # Take the mean score per unique protein variant so 1 HCM row → 1 score.
    before = len(out)
    out = out.groupby(MERGE_KEYS, as_index=False)["eve_score"].mean()
    after = len(out)
    if before != after:
        print(f"    [EVE dedup] {before} rows → {after} unique protein variants")
    # ───────────────────────────────────────────────────────────────────────

    return out


# ---------------------------------------------------------------------------
# AlphaMissense helpers
# ---------------------------------------------------------------------------
def load_alphamissense(am_file: str, accessions: set):
    """
    Stream-parse the (possibly gzipped) AlphaMissense TSV.
    Returns DataFrame with (accession, position, ref_aa, alt_aa, am_score)
    or None if the file is unavailable.
    """
    if not os.path.exists(am_file):
        print(f"  WARNING: AlphaMissense file not found: {am_file}")
        print("  Download it from: https://github.com/google-deepmind/alphamissense")
        print("  Place at: " + am_file)
        return None

    print("  Loading AlphaMissense file (streaming)...")
    rows = []
    try:
        opener = gzip.open(am_file, "rt") if am_file.endswith(".gz") else open(am_file, "r")
        with opener as fh:
            for line in fh:
                if line.startswith("#") or line.startswith("uniprot_id"):
                    continue
                parts = line.rstrip().split("\t")
                if len(parts) < 3:
                    continue
                uniprot = parts[0]
                if uniprot not in accessions:
                    continue
                variant  = parts[1]           # e.g. "M1T"
                am_score = parts[2]
                try:
                    ref_aa = variant[0]
                    alt_aa = variant[-1]
                    pos    = int(variant[1:-1])
                    score  = float(am_score)
                except (ValueError, IndexError):
                    continue
                rows.append([uniprot, pos, ref_aa, alt_aa, score])
    except Exception as e:
        print(f"  ERROR reading AlphaMissense: {e}")
        return None

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["accession", "position", "ref_aa", "alt_aa", "am_score"])

    # ── DEDUP FIX ──────────────────────────────────────────────────────────
    # AlphaMissense is nucleotide-level: same protein change can appear for
    # multiple codons. Deduplicate to exactly 1 row per protein variant.
    before = len(df)
    df = df.groupby(MERGE_KEYS, as_index=False)["am_score"].mean()
    after = len(df)
    if before != after:
        print(f"  [AM dedup] {before} rows → {after} unique protein variants")
    # ───────────────────────────────────────────────────────────────────────

    print(f"  AlphaMissense: {after} unique protein variants across target genes.")
    return df


# ---------------------------------------------------------------------------
# Synthetic score generator (--dry-run only)
# ---------------------------------------------------------------------------
def make_synthetic(hcm_df, accession: str, col: str, noise: float):
    rows = hcm_df[hcm_df["accession"] == accession].copy()
    if len(rows) == 0:
        return None
    rng = np.random.default_rng(seed=hash(col) % 2**32)
    rows[col] = (rows["grantham_score"] + rng.normal(0, noise, len(rows))).clip(0, 215) / 215.0
    return rows[MERGE_KEYS + [col]]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def safe_auprc(y_true, scores):
    mask = ~np.isnan(scores)
    yt = y_true[mask]; ys = scores[mask]
    if len(yt) < 5 or len(np.unique(yt)) < 2:
        return float("nan")
    return float(average_precision_score(yt, ys))


def safe_auroc(y_true, scores):
    mask = ~np.isnan(scores)
    yt = y_true[mask]; ys = scores[mask]
    if len(yt) < 5 or len(np.unique(yt)) < 2:
        return float("nan")
    return float(roc_auc_score(yt, ys))


def orient_score(y_true, scores):
    """Flip sign so that higher score = more pathogenic (required by sklearn AUC)."""
    if np.corrcoef(y_true, scores)[0, 1] < 0:
        return -scores
    return scores


# ---------------------------------------------------------------------------
# Bar-chart helper  (4 bars: Two-Tower, Baseline RF, EVE, AlphaMissense)
# ---------------------------------------------------------------------------
def bar_chart(genes, vals_tt, vals_rf, vals_ext1, vals_ext2,
              label1, label2, label3, label4, ylabel, title, path):
    x = np.arange(len(genes))
    w = 0.20
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_facecolor("#f9f9f9")
    b1 = ax.bar(x - 1.5*w, vals_tt,   w, label=label1, color="#2563EB", alpha=0.88)
    b2 = ax.bar(x - 0.5*w, vals_rf,   w, label=label2, color="#7C3AED", alpha=0.88)
    b3 = ax.bar(x + 0.5*w, vals_ext1, w, label=label3, color="#DC2626", alpha=0.88)
    b4 = ax.bar(x + 1.5*w, vals_ext2, w, label=label4, color="#059669", alpha=0.88)
    for bars in (b1, b2, b3, b4):
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(genes, fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6, label="Random (0.5)")
    ax.set_xlabel("Gene (LOGO hold-out)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Use synthetic scores (offline / CI mode)")
    args = parser.parse_args()

    print("=" * 65)
    print("  EVE & AlphaMissense Benchmark (dedup-corrected)")
    print("=" * 65)

    # 1. Load HCM dataset
    print(f"\n[Step 1] Loading HCM data from {HCM_CSV}")
    hcm = pd.read_csv(HCM_CSV)
    if hcm["label"].dtype == object:
        hcm["label"] = hcm["label"].str.strip().isin(["Pathogenic", "1", "True"]).astype(int)
    hcm["ref_aa"]   = hcm["ref_aa"].astype(str).str.strip().str.upper()
    hcm["alt_aa"]   = hcm["alt_aa"].astype(str).str.strip().str.upper()
    hcm["position"] = pd.to_numeric(hcm["position"], errors="coerce")
    print(f"         {len(hcm)} variants across {sorted(hcm['gene'].unique())}")

    # 2. Load Two-Tower and Baseline RF scores from logo_metrics.csv
    print(f"\n[Step 2] Loading Two-Tower and Baseline RF scores from {os.path.basename(LOGO_CSV)}")
    tt_auprc, tt_auroc = {}, {}
    rf_auprc, rf_auroc = {}, {}
    if os.path.exists(LOGO_CSV):
        logo_df = pd.read_csv(LOGO_CSV)
        if "Model" in logo_df.columns:
            tt_df = logo_df[logo_df["Model"].str.contains("Two-Tower", case=False, na=False)]
            rf_df = logo_df[logo_df["Model"].str.contains("Baseline RF", case=False, na=False)]
        else:
            tt_df = logo_df
            rf_df = pd.DataFrame()
        for _, row in tt_df.iterrows():
            g = row["Gene"]
            if "AUPRC" in row and pd.notna(row["AUPRC"]):
                tt_auprc[g] = float(row["AUPRC"])
            if "AUROC" in row and pd.notna(row.get("AUROC", float("nan"))):
                tt_auroc[g] = float(row["AUROC"])
        for _, row in rf_df.iterrows():
            g = row["Gene"]
            if "AUPRC" in row and pd.notna(row["AUPRC"]):
                rf_auprc[g] = float(row["AUPRC"])
            if "AUROC" in row and pd.notna(row.get("AUROC", float("nan"))):
                rf_auroc[g] = float(row["AUROC"])
        print(f"         Two-Tower loaded for: {sorted(tt_auprc.keys())}")
        print(f"         Baseline RF loaded for: {sorted(rf_auprc.keys())}")
    else:
        print(f"         WARNING: {LOGO_CSV} not found. Model scores will be NaN.")

    # 3. Load AlphaMissense (once, for all genes)
    print("\n[Step 3] Loading AlphaMissense scores")
    if args.dry_run:
        print("  [DRY-RUN] Will generate synthetic AlphaMissense scores per gene.")
        am_full = None
    else:
        am_full = load_alphamissense(AM_FILE, set(GENE_ACCESSION.values()))

    # 4. Per-gene evaluation loop
    print("\n[Step 4] Running per-gene evaluation ...\n")
    results = []

    for gene in sorted(GENE_ACCESSION.keys()):
        accession = GENE_ACCESSION[gene]
        print(f"  {'─'*55}")
        print(f"  Gene: {gene}  |  UniProt: {accession}")

        hcm_gene = hcm[hcm["gene"] == gene].copy()
        if len(hcm_gene) == 0:
            print(f"  [{gene}] No HCM variants — skipping.")
            continue

        # --- EVE ---
        if args.dry_run:
            eve_scores = make_synthetic(hcm_gene, accession, "eve_score", 15)
            print(f"  [{gene}] [DRY-RUN] Synthetic EVE: {len(eve_scores) if eve_scores is not None else 0} rows")
        else:
            eve_raw    = fetch_eve_csv(gene, accession)
            eve_scores = process_eve(eve_raw, accession)

        # --- AlphaMissense ---
        if args.dry_run:
            am_scores = make_synthetic(hcm_gene, accession, "am_score", 10)
            print(f"  [{gene}] [DRY-RUN] Synthetic AM:  {len(am_scores) if am_scores is not None else 0} rows")
        else:
            am_scores = am_full[am_full["accession"] == accession] if am_full is not None else None

        # --- Merge (strict 1-to-1 guaranteed by dedup above) ---
        merged = hcm_gene.copy()
        if eve_scores is not None:
            merged = merged.merge(eve_scores[MERGE_KEYS + ["eve_score"]],
                                  on=MERGE_KEYS, how="left")
        else:
            merged["eve_score"] = np.nan

        if am_scores is not None and len(am_scores) > 0:
            merged = merged.merge(am_scores[MERGE_KEYS + ["am_score"]],
                                  on=MERGE_KEYS, how="left")
        else:
            merged["am_score"] = np.nan

        # Verify no row explosion happened
        if len(merged) != len(hcm_gene):
            print(f"  [{gene}] WARNING: Row count changed {len(hcm_gene)} → {len(merged)}. "
                  f"Dedup may be incomplete — check accession/position types.")

        n_eve = merged["eve_score"].notna().sum()
        n_am  = merged["am_score"].notna().sum()
        print(f"  [{gene}] EVE matched: {n_eve}/{len(merged)} | AM matched: {n_am}/{len(merged)}")

        # Need both scores to compare fairly
        eval_df = merged.dropna(subset=["eve_score", "am_score"])
        if len(eval_df) < 5 or len(eval_df["label"].unique()) < 2:
            print(f"  [{gene}] Insufficient overlap for comparison ({len(eval_df)} rows) — skipping.")
            continue

        y_true   = eval_df["label"].values.astype(float)
        eve_pred = orient_score(y_true, eval_df["eve_score"].values)
        am_pred  = orient_score(y_true, eval_df["am_score"].values)

        eve_auprc = safe_auprc(y_true, eve_pred)
        eve_auroc = safe_auroc(y_true, eve_pred)
        am_auprc  = safe_auprc(y_true, am_pred)
        am_auroc  = safe_auroc(y_true, am_pred)
        tt_ap     = tt_auprc.get(gene, float("nan"))
        tt_ar     = tt_auroc.get(gene, float("nan"))
        rf_ap     = rf_auprc.get(gene, float("nan"))
        rf_ar     = rf_auroc.get(gene, float("nan"))

        print(f"  [{gene}] AUPRC — Two-Tower: {tt_ap:.3f} | RF: {rf_ap:.3f} | EVE: {eve_auprc:.3f} | AM: {am_auprc:.3f}")

        results.append({
            "Gene":                  gene,
            "Two_Tower_AUPRC":       round(tt_ap,       4),
            "Baseline_RF_AUPRC":     round(rf_ap,       4),
            "EVE_AUPRC":             round(eve_auprc,   4),
            "AlphaMissense_AUPRC":   round(am_auprc,    4),
            "Two_Tower_AUROC":       round(tt_ar,       4),
            "Baseline_RF_AUROC":     round(rf_ar,       4),
            "EVE_AUROC":             round(eve_auroc,   4),
            "AlphaMissense_AUROC":   round(am_auroc,    4),
            "N_evaluated":           len(eval_df),
            "N_total":               len(hcm_gene),
        })

    if not results:
        print("\nNo results generated. Check EVE/AM data availability.")
        return

    # 5. Save results
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_CSV, index=False)
    print(f"\n[Step 5] Saved -> {OUT_CSV}")
    try:
        print(res_df.to_markdown(index=False))
    except ImportError:
        print(res_df.to_string(index=False))

    # Print averages
    print(f"\n  Mean AUPRC (across {len(res_df)} genes):")
    print(f"    Two-Tower Hybrid : {res_df['Two_Tower_AUPRC'].mean():.4f}")
    print(f"    Baseline RF      : {res_df['Baseline_RF_AUPRC'].mean():.4f}")
    print(f"    EVE              : {res_df['EVE_AUPRC'].mean():.4f}")
    print(f"    AlphaMissense    : {res_df['AlphaMissense_AUPRC'].mean():.4f}")

    # 6. Plots
    print("\n[Step 6] Generating comparison plots ...")
    genes   = res_df["Gene"].tolist()
    tt_ap   = res_df["Two_Tower_AUPRC"].tolist()
    rf_ap   = res_df["Baseline_RF_AUPRC"].tolist()
    eve_ap  = res_df["EVE_AUPRC"].tolist()
    am_ap   = res_df["AlphaMissense_AUPRC"].tolist()
    tt_ar   = res_df["Two_Tower_AUROC"].tolist()
    rf_ar   = res_df["Baseline_RF_AUROC"].tolist()
    eve_ar  = res_df["EVE_AUROC"].tolist()
    am_ar   = res_df["AlphaMissense_AUROC"].tolist()

    bar_chart(genes, tt_ap, rf_ap, eve_ap, am_ap,
              "Two-Tower (Ours)", "Baseline RF (Ours)", "EVE", "AlphaMissense",
              "AUPRC",
              "HCM Variant Pathogenicity: Two-Tower vs Baseline RF vs EVE vs AlphaMissense — PR-AUC",
              os.path.join(FIGURES_DIR, "external_pr_comparison.png"))

    bar_chart(genes, tt_ar, rf_ar, eve_ar, am_ar,
              "Two-Tower (Ours)", "Baseline RF (Ours)", "EVE", "AlphaMissense",
              "AUROC",
              "HCM Variant Pathogenicity: Two-Tower vs Baseline RF vs EVE vs AlphaMissense — ROC-AUC",
              os.path.join(FIGURES_DIR, "external_roc_comparison.png"))

    print(f"\n[Done] All outputs in: {BENCH_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
