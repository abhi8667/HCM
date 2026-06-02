"""
fetch_revel_metarnn.py
======================
Fetches REVEL and MetaRNN pathogenicity scores from the FREE MyVariant.info
public API. No 46 GB download needed!

Strategy: protein coordinates (dbnsfp.genename + dbnsfp.aa.pos/ref/alt)
  - Completely assembly-independent (no hg19 vs hg38 issues)
  - Paginate through ALL dbNSFP entries for each of the 9 HCM genes
  - Merge with HCM dataset on (gene, aa_position, ref_aa, alt_aa)

Run from anywhere:
    python benchmarking/scripts/fetch_revel_metarnn.py

Reads:
    <HCM_ROOT>/data/HCM_labeled_final.csv
    <HCM_ROOT>/results/logo_metrics.csv  (Two-Tower + Baseline RF AUPRC for comparison)

Writes:
    <HCM_ROOT>/benchmarking/data/HCM_labeled_with_dbnsfp.csv
    <HCM_ROOT>/benchmarking/results/revel_metarnn_comparison.csv

Model Update (dual-model pipeline):
    Results now compare Two-Tower Hybrid BCE AND Baseline RF against REVEL
    and MetaRNN. Both model AUPRCs are loaded from results/logo_metrics.csv.
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score

try:
    import myvariant
except ImportError:
    print("ERROR: myvariant package not found. Run: pip install myvariant")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Script lives at: <HCM_ROOT>/benchmarking/scripts/
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR    = os.path.dirname(SCRIPT_DIR)           # benchmarking/
PROJECT_ROOT = os.path.dirname(BENCH_DIR)            # HCM/

# Input: main project data
HCM_CSV  = os.path.join(PROJECT_ROOT, "data",    "HCM_labeled_final.csv")
LOGO_CSV = os.path.join(PROJECT_ROOT, "results", "logo_metrics.csv")

# Output: write inside benchmarking/ only
DATA_DIR    = os.path.join(BENCH_DIR, "data")
RESULTS_DIR = os.path.join(BENCH_DIR, "results")
os.makedirs(DATA_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

OUT_CSV   = os.path.join(DATA_DIR,    "HCM_labeled_with_dbnsfp.csv")
BENCH_CSV = os.path.join(RESULTS_DIR, "revel_metarnn_comparison.csv")

HCM_GENES = ["MYH7", "MYBPC3", "TNNT2", "TNNI3", "TPM1",
             "ACTC1", "MYL2", "MYL3", "TNNC1"]

PAGE_SIZE = 1000   # MyVariant.info max per page


# ---------------------------------------------------------------------------
# Score extractor
# ---------------------------------------------------------------------------
def _pick_score(value):
    """Return float (max across transcripts) or None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        nums = [_pick_score(v) for v in value]
        nums = [n for n in nums if n is not None]
        return max(nums) if nums else None
    if isinstance(value, dict):
        return _pick_score(value.get("score"))
    return None


# ---------------------------------------------------------------------------
# Paginated bulk fetch per gene
# ---------------------------------------------------------------------------
def fetch_gene_scores(mv, gene: str) -> dict:
    """
    Returns lookup dict:
        { (gene, aa_pos, ref_aa, alt_aa) : {"REVEL_score": float, "MetaRNN_score": float} }

    Uses manual pagination with 'from_' parameter because fetch_all=True
    is broken in myvariant 1.0.0.

    Confirmed field names:
        dbnsfp.aa.pos / dbnsfp.aa.ref / dbnsfp.aa.alt
        dbnsfp.revel.score / dbnsfp.metarnn.score
    """
    lookup = {}
    fields = "dbnsfp.aa,dbnsfp.revel.score,dbnsfp.metarnn.score,dbnsfp.genename"
    offset = 0

    while True:
        try:
            resp = mv.query(
                f"dbnsfp.genename:{gene}",
                fields=fields,
                size=PAGE_SIZE,
                from_=offset,
            )
        except Exception as exc:
            print(f"\n    [WARNING] Query failed at offset {offset}: {exc}")
            break

        hits = resp.get("hits", [])
        if not hits:
            break

        for hit in hits:
            dbnsfp = hit.get("dbnsfp", {})
            if not dbnsfp:
                continue

            aa_block = dbnsfp.get("aa", {})

            # aa block can be a single dict or a list of dicts (multi-transcript)
            if isinstance(aa_block, dict):
                aa_list = [aa_block]
            elif isinstance(aa_block, list):
                aa_list = aa_block
            else:
                continue

            revel_score   = _pick_score(dbnsfp.get("revel"))
            metarnn_score = _pick_score(dbnsfp.get("metarnn"))

            for aa in aa_list:
                pos_raw = aa.get("pos")
                ref_raw = aa.get("ref")
                alt_raw = aa.get("alt")
                if pos_raw is None or ref_raw is None or alt_raw is None:
                    continue

                # pos/ref/alt can be scalars OR lists (multi-transcript)
                # e.g. MYBPC3: pos=[1095,1095,1095], ref="T", alt="P"
                def _to_list(v):
                    return v if isinstance(v, list) else [v]

                pos_list = _to_list(pos_raw)
                ref_list = _to_list(ref_raw)
                alt_list = _to_list(alt_raw)

                seen = set()
                for pi, ri, ai in zip(pos_list, ref_list, alt_list):
                    try:
                        pos = int(pi)
                    except (ValueError, TypeError):
                        continue
                    ref = str(ri).strip().upper()
                    alt = str(ai).strip().upper()
                    combo = (pos, ref, alt)
                    if combo in seen:
                        continue
                    seen.add(combo)

                    key = (gene, pos, ref, alt)
                    existing = lookup.get(key, {})
                    lookup[key] = {
                        "REVEL_score":   revel_score   if existing.get("REVEL_score")   is None else existing["REVEL_score"],
                        "MetaRNN_score": metarnn_score if existing.get("MetaRNN_score") is None else existing["MetaRNN_score"],
                    }

        total = resp.get("total", 0)
        offset += len(hits)
        print(f"    page {offset // PAGE_SIZE}: fetched {offset}/{total} ...", end="\r")

        if offset >= total:
            break
        time.sleep(0.2)

    return lookup


# ---------------------------------------------------------------------------
# AUPRC helper
# ---------------------------------------------------------------------------
def safe_auprc(y_true, scores):
    mask = pd.notna(scores)
    yt = np.array(y_true)[mask]
    ys = np.array(scores)[mask]
    if len(yt) < 5 or len(np.unique(yt)) < 2:
        return None
    return float(average_precision_score(yt, ys))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("  REVEL & MetaRNN Benchmark  --  MyVariant.info  (no download!)")
    print("  Method: per-gene bulk fetch by protein coordinates")
    print("=" * 65)

    # 1. Load HCM dataset
    print(f"\n[Step 1] Loading {HCM_CSV}")
    df = pd.read_csv(HCM_CSV)
    if df["label"].dtype == object:
        df["label"] = df["label"].str.strip().isin(
            ["Pathogenic", "1", "True"]).astype(int)
    df["ref_aa"]   = df["ref_aa"].astype(str).str.strip().str.upper()
    df["alt_aa"]   = df["alt_aa"].astype(str).str.strip().str.upper()
    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    print(f"         {len(df)} variants across {sorted(df['gene'].unique())}")

    # 2. Fetch dbNSFP scores per gene
    print("\n[Step 2] Fetching dbNSFP scores per gene (paginated) ...\n")
    mv = myvariant.MyVariantInfo()
    global_lookup = {}

    for gene in HCM_GENES:
        print(f"  [{gene}] querying MyVariant.info ...", flush=True)
        gene_lookup = fetch_gene_scores(mv, gene)
        global_lookup.update(gene_lookup)
        print(f"  [{gene}] done -- {len(gene_lookup)} unique (pos, ref, alt) entries.    ")
        time.sleep(0.5)

    print(f"\n  Total lookup entries across all 9 genes: {len(global_lookup)}")

    # 3. Merge into dataframe
    print("\n[Step 3] Merging scores ...")

    def get_score(row, score_key):
        pos = row["position"]
        if pd.isna(pos):
            return None
        key = (row["gene"], int(pos), row["ref_aa"], row["alt_aa"])
        return global_lookup.get(key, {}).get(score_key)

    df["REVEL_score"]   = df.apply(lambda r: get_score(r, "REVEL_score"),   axis=1)
    df["MetaRNN_score"] = df.apply(lambda r: get_score(r, "MetaRNN_score"), axis=1)

    n_revel   = df["REVEL_score"].notna().sum()
    n_metarnn = df["MetaRNN_score"].notna().sum()
    print(f"  REVEL   matched: {n_revel}/{len(df)} ({100*n_revel/len(df):.1f}%)")
    print(f"  MetaRNN matched: {n_metarnn}/{len(df)} ({100*n_metarnn/len(df):.1f}%)")

    # 4. Save enriched dataset
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[Step 4] Saved -> {OUT_CSV}")

    # 5. Per-gene AUPRC benchmark
    print("\n[Step 5] Computing benchmark ...")
    two_tower = {}
    baseline_rf = {}
    if os.path.exists(LOGO_CSV):
        logo = pd.read_csv(LOGO_CSV)
        logo["AUPRC"] = pd.to_numeric(logo["AUPRC"], errors="coerce")
        if "Model" in logo.columns:
            tt_rows = logo[logo["Model"].str.contains("Two-Tower", case=False, na=False)]
            rf_rows = logo[logo["Model"].str.contains("Baseline RF", case=False, na=False)]
            two_tower   = dict(zip(tt_rows["Gene"], tt_rows["AUPRC"]))
            baseline_rf = dict(zip(rf_rows["Gene"], rf_rows["AUPRC"]))
        else:
            two_tower = dict(zip(logo["Gene"], logo["AUPRC"]))

    rows = []
    for gene in sorted(df["gene"].unique()):
        gdf = df[df["gene"] == gene]
        y   = gdf["label"].values
        revel_auprc   = safe_auprc(y, gdf["REVEL_score"].values)
        metarnn_auprc = safe_auprc(y, gdf["MetaRNN_score"].values)
        rows.append({
            "Gene":              gene,
            "Two_Tower_AUPRC":   round(two_tower.get(gene, float("nan")), 4),
            "Baseline_RF_AUPRC": round(baseline_rf.get(gene, float("nan")), 4),
            "REVEL_AUPRC":       round(revel_auprc,   4) if revel_auprc   is not None else None,
            "MetaRNN_AUPRC":     round(metarnn_auprc, 4) if metarnn_auprc is not None else None,
            "N_REVEL":    int(gdf["REVEL_score"].notna().sum()),
            "N_MetaRNN":  int(gdf["MetaRNN_score"].notna().sum()),
            "N_total":    len(gdf),
        })

    bench = pd.DataFrame(rows)
    bench.to_csv(BENCH_CSV, index=False)

    print("\n--- Per-Gene Benchmark ---")
    try:
        print(bench.to_markdown(index=False))
    except ImportError:
        print(bench.to_string(index=False))

    # Summary averages
    valid = bench[bench["REVEL_AUPRC"].notna() & bench["Two_Tower_AUPRC"].notna()]
    if len(valid):
        mt    = valid["Two_Tower_AUPRC"].mean()
        mrf   = valid["Baseline_RF_AUPRC"].mean() if "Baseline_RF_AUPRC" in valid.columns else float("nan")
        mr    = valid["REVEL_AUPRC"].mean()
        mmrnn = bench[bench["MetaRNN_AUPRC"].notna()]["MetaRNN_AUPRC"].mean()
        print(f"\n  Averages across {len(valid)} genes with full data:")
        print(f"    Two-Tower  AUPRC: {mt:.4f}")
        if not np.isnan(mrf):
            print(f"    Baseline RF AUPRC: {mrf:.4f}   (Two-Tower vs RF: {mt-mrf:+.4f})")
        print(f"    REVEL      AUPRC: {mr:.4f}   (Two-Tower wins by {mt-mr:+.4f})")
        if not np.isnan(mmrnn):
            print(f"    MetaRNN    AUPRC: {mmrnn:.4f}   (Two-Tower wins by {mt-mmrnn:+.4f})")

    print(f"\n[Done] -> {BENCH_CSV}")
    print("=" * 65)


if __name__ == "__main__":
    main()
