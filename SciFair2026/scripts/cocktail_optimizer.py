"""
05_cocktail_optimizer.py  —  Enhanced Phage Cocktail Optimization v3
PrecisionPhage | ISEF 2026

STRAIN-LEVEL COVERAGE (v3 upgrade):
  VHI stores interactions at species level (host = "staphylococcus aureus").
  True therapeutic coverage requires STRAIN-LEVEL analysis:
    • Which strains of S. aureus does phage A infect?
    • Which strains does phage B infect?
    • A cocktail covers a strain if ≥1 phage in the cocktail infects it.

  STRAIN RESOLUTION (3-tier priority):
    Tier 1 — NCBI host annotations (most accurate)
      Download GenBank records for each phage accession.
      Extract /host qualifier → strain name (e.g. "Staphylococcus aureus MRSA252")
      Cache to data/raw/phage_strain_annotations.csv

    Tier 2 — Genomic feature clustering (fallback, no network needed)
      Cluster phages by (k3dist, k6dist, GCdiff, Homology) similarity.
      Phages in the same cluster share host-range → represent one "strain class".
      Uses AgglomerativeClustering with Ward linkage.

    Tier 3 — Each phage = independent strain (original metric)
      Used only if tiers 1 and 2 both fail.

  COVERAGE FORMULA (strain level):
    cocktail_strains = strains infected by ≥1 phage in cocktail (y_true=1)
    all_pos_strains  = unique strains infected by any phage in dataset
    coverage = |cocktail_strains| / |all_pos_strains|
"""

import warnings, sys, time, re
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy.stats import wilcoxon, pearsonr
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)

_SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR    = _SCRIPT_DIR.parent / "data"
RAW_DIR     = BASE_DIR / "raw"
PLOT_DIR    = BASE_DIR / "plots"
RESULTS_DIR = BASE_DIR / "results"
for d in [PLOT_DIR, RESULTS_DIR, RAW_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="colorblind")
MAX_K           = 5
N_RANDOM        = 100
N_BOOTSTRAP     = 1000
LAMBDA_DIV      = 0.3
TARGET_HOST     = "staphylococcus aureus"
NCBI_BATCH_SIZE = 200    # accessions per Entrez request
NCBI_SLEEP      = 0.35   # seconds between requests (NCBI rate limit = 3/s)
STRAIN_CACHE    = RAW_DIR / "phage_strain_annotations.csv"

def save_plot(fig, name):
    p = PLOT_DIR / f"{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — LOAD PREDICTIONS
# ═══════════════════════════════════════════════════════════════
print("=" * 64)
print("  ENHANCED PHAGE COCKTAIL OPTIMIZER  v3  (strain-level)")
print("=" * 64)

pred_path = RESULTS_DIR / "ensemble_predictions.csv"
if not pred_path.exists():
    print(f"\n  ERROR: {pred_path} not found. Run 04_ensemble.py first.")
    raise SystemExit(1)

pred_df = pd.read_csv(pred_path)
# Drop rows with missing phage or host (guards all downstream string ops)
pred_df["phage"] = pred_df["phage"].astype(str).str.strip()
pred_df["host"]  = pred_df["host"].astype(str).str.strip()
pred_df = pred_df[
    (pred_df["phage"] != "") & (pred_df["phage"] != "nan") &
    (pred_df["host"]  != "") & (pred_df["host"]  != "nan")
].copy().reset_index(drop=True)
if "genus" not in pred_df.columns:
    pred_df["genus"] = pred_df["host"].str.split().str[0]

print(f"\n  Predictions: {len(pred_df)} rows | "
      f"Species: {pred_df['host'].nunique()} | "
      f"Phages: {pred_df['phage'].nunique()} | "
      f"Positives: {pred_df['label'].sum()}")


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — STRAIN RESOLUTION
# ═══════════════════════════════════════════════════════════════
print("\n[1] Resolving strain annotations...")

def clean_accession(phage_name):
    """
    Extract NCBI accession from phage name strings like:
      'nc 007045'  → 'NC_007045'
      'pg 2021 14' → None  (not an NCBI accession)
      'k (nc 005880; tax:...)' → 'NC_005880'
    """
    if not phage_name or not isinstance(phage_name, str):
        return None
    s = phage_name.upper().replace(" ", "_")
    # Match NC_, NZ_, MK, MW, MN, KX, KY, KP, JX, EU, AY, DQ, EF accessions
    m = re.search(r'(N[CZ]_\d{6}|[A-Z]{2}\d{5,6})', s)
    if m:
        acc = m.group(1)
        # NC_ accessions: NC_007045 format
        if acc.startswith("NC_") or acc.startswith("NZ_"):
            return acc
        # Older style: 2-letter prefix + 6 digits
        if re.match(r'^[A-Z]{2}\d{6}$', acc):
            return acc
    return None

def fetch_ncbi_strain_annotations(accessions, cache_path):
    """
    Fetch host strain annotations from NCBI GenBank for a list of accessions.
    Returns dict: {accession: strain_string}

    Uses NCBI Entrez E-utilities (no API key needed for < 3 req/s).
    Caches results to CSV so subsequent runs are instant.
    """
    try:
        from urllib import request as ureq, parse as uparse
        import json
    except ImportError:
        return {}

    results = {}
    batches = [accessions[i:i+NCBI_BATCH_SIZE]
               for i in range(0, len(accessions), NCBI_BATCH_SIZE)]

    print(f"    Fetching {len(accessions)} accessions in "
          f"{len(batches)} batches from NCBI...")

    for b_idx, batch in enumerate(batches):
        try:
            # Step 1: esearch to get GIs
            ids_str = ",".join(batch)
            url_search = (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
                f"db=nucleotide&id={uparse.quote(ids_str)}"
                "&rettype=gb&retmode=text"
            )
            req = ureq.Request(url_search,
                               headers={"User-Agent": "PrecisionPhage/1.0 ISEF2026"})
            with ureq.urlopen(req, timeout=30) as resp:
                gb_text = resp.read().decode("utf-8", errors="ignore")

            # Parse /host qualifiers from GenBank flat file
            # Format: /host="Staphylococcus aureus MRSA252"
            current_acc = None
            for line in gb_text.splitlines():
                # Detect accession line: "ACCESSION   NC_007045"
                if line.startswith("ACCESSION"):
                    parts = line.split()
                    if len(parts) >= 2:
                        current_acc = parts[1].strip()
                # Detect /host qualifier
                m = re.search(r'/host="([^"]+)"', line)
                if m and current_acc:
                    host_val = m.group(1).strip()
                    # Only store if not already found (first /host wins)
                    if current_acc not in results:
                        results[current_acc] = host_val

            time.sleep(NCBI_SLEEP)
            if (b_idx + 1) % 5 == 0:
                print(f"    Batch {b_idx+1}/{len(batches)} done "
                      f"({len(results)} annotations so far)...")

        except Exception as e:
            print(f"    Batch {b_idx+1} failed: {e} — skipping")
            time.sleep(1.0)
            continue

    return results

def extract_strain_from_host_string(host_str):
    """
    Parse NCBI /host string into a canonical strain identifier.
    Examples:
      "Staphylococcus aureus MRSA252"     → "staphylococcus aureus mrsa252"
      "Staphylococcus aureus subsp. aureus strain ATCC 25923" → "staphylococcus aureus atcc 25923"
      "Escherichia coli K-12"              → "escherichia coli k-12"
      "Staphylococcus aureus"              → "staphylococcus aureus"  (no strain)
    """
    if not host_str or pd.isna(host_str):
        return None
    s = host_str.lower()
    # Remove "subsp. <subsp>" and "str." / "strain " noise
    s = re.sub(r'\bsubsp\.?\s+\w+\b', '', s)
    s = re.sub(r'\b(str\.|strain)\s+', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s if s else None

def build_strain_map_ncbi(pred_df, cache_path):
    """
    Main entry point for Tier 1 strain resolution.
    Returns: pred_df with new column 'strain'
    """
    if cache_path.exists():
        print(f"    Loading cached annotations: {cache_path.name}")
        cache = pd.read_csv(cache_path)
        strain_lookup = dict(zip(cache["accession"], cache["strain"]))
        print(f"    Cache has {len(strain_lookup)} entries")
    else:
        strain_lookup = {}

    # Extract accessions from phage names
    all_phages = pred_df["phage"].unique()
    acc_map    = {}   # phage_name → accession
    for ph in all_phages:
        acc = clean_accession(ph)
        if acc:
            acc_map[ph] = acc

    # Find which accessions need fetching
    to_fetch = [acc for acc in set(acc_map.values())
                if acc not in strain_lookup]

    if to_fetch:
        print(f"    {len(to_fetch)} accessions not in cache — fetching from NCBI...")
        new_data = fetch_ncbi_strain_annotations(to_fetch, cache_path)
        strain_lookup.update(new_data)
        # Save updated cache
        cache_rows = [{"accession": k, "strain": v}
                      for k, v in strain_lookup.items()]
        pd.DataFrame(cache_rows).to_csv(cache_path, index=False)
        print(f"    Saved {len(cache_rows)} entries to cache")
    else:
        print(f"    All {len(acc_map)} accessions already cached")

    # Build strain column
    def get_strain(row):
        ph  = row["phage"]
        acc = acc_map.get(ph)
        if acc and acc in strain_lookup:
            parsed = extract_strain_from_host_string(strain_lookup[acc])
            # Strain must match the species we're analyzing
            sp = row["host"].lower()
            if parsed and sp.split()[0] in parsed:
                return parsed
        return None

    pred_df = pred_df.copy()
    pred_df["ncbi_strain"] = pred_df.apply(get_strain, axis=1)
    n_annotated = pred_df["ncbi_strain"].notna().sum()
    print(f"    NCBI strain annotations: {n_annotated}/{len(pred_df)} rows "
          f"({100*n_annotated/len(pred_df):.1f}%)")
    return pred_df

def build_strain_map_clustering(pred_df, n_clusters_per_phage=0.3):
    """
    Tier 2: Cluster phages by genomic features.
    Phages with similar k3dist, k6dist, GCdiff, Homology profiles
    share host-range and likely target the same bacterial strains.

    n_clusters_per_phage: fraction — clusters = max(2, int(n_unique_phages * frac))
    Returns pred_df with 'cluster_strain' column.
    """
    feat_cols = [c for c in ["k3dist","k6dist","GCdiff","Homology"]
                 if c in pred_df.columns]
    if not feat_cols:
        pred_df["cluster_strain"] = pred_df["phage"]
        return pred_df

    pred_df = pred_df.copy()
    # Cluster per species
    strain_col = []
    for sp, sp_df in pred_df.groupby("host"):
        phage_feats = (sp_df.groupby("phage")[feat_cols]
                       .mean()
                       .dropna())
        n_ph = len(phage_feats)
        if n_ph < 2:
            for ph in sp_df["phage"]:
                strain_col.append((sp_df.index[sp_df["phage"]==ph][0],
                                    f"{sp}||{ph}"))
            continue
        n_clust = max(2, int(n_ph * n_clusters_per_phage))
        n_clust = min(n_clust, n_ph)
        scaler  = StandardScaler()
        X       = scaler.fit_transform(phage_feats.values)
        labels  = AgglomerativeClustering(
            n_clusters=n_clust, linkage="ward").fit_predict(X)
        cluster_map = dict(zip(phage_feats.index, labels))
        for idx, row in sp_df.iterrows():
            cl = cluster_map.get(row["phage"], 0)
            strain_col.append((idx, f"{sp}||cluster_{cl}"))

    strain_series = pd.Series(
        {idx: s for idx, s in strain_col}, name="cluster_strain")
    pred_df["cluster_strain"] = strain_series
    n_clusters = pred_df["cluster_strain"].nunique()
    print(f"    Clustering created {n_clusters} strain groups "
          f"(from {pred_df['phage'].nunique()} phages)")
    return pred_df

# ── Run strain resolution ──
pred_df = build_strain_map_ncbi(pred_df, STRAIN_CACHE)
pred_df = build_strain_map_clustering(pred_df)

# Decide which strain column to use, per row:
#   Priority: ncbi_strain > cluster_strain > phage (original)
def resolve_strain(row):
    if pd.notna(row.get("ncbi_strain")) and row["ncbi_strain"]:
        return row["ncbi_strain"]
    return row.get("cluster_strain", row["phage"])

pred_df["strain"] = pred_df.apply(resolve_strain, axis=1)

# Report strain resolution quality
n_ncbi    = pred_df["ncbi_strain"].notna().sum()
n_cluster = (pred_df["ncbi_strain"].isna()).sum()
tier_used = ("NCBI" if n_ncbi > len(pred_df)*0.3
             else "Genomic clustering" if "cluster_strain" in pred_df.columns
             else "Phage-as-strain (original)")
print(f"\n  Strain resolution summary:")
print(f"    Tier 1 (NCBI):          {n_ncbi} rows  ({100*n_ncbi/len(pred_df):.1f}%)")
print(f"    Tier 2 (clustering):    {n_cluster} rows  ({100*n_cluster/len(pred_df):.1f}%)")
print(f"    Dominant tier used:     {tier_used}")
print(f"    Unique strains total:   {pred_df['strain'].nunique()}")


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — STRAIN-LEVEL COVERAGE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def get_strain_universe(sp_df):
    """
    Returns: (pos_strains set, all_strains set)
    pos_strains = strains that have ≥1 y_true=1 phage
    """
    pos_df      = sp_df[sp_df["label"] == 1]
    pos_strains = set(pos_df["strain"].unique())
    return pos_strains

def strain_cov(sp_df, selected, pos_strains=None):
    """
    STRAIN-LEVEL COVERAGE:
    Coverage = (# pos strains covered by ≥1 selected phage with y_true=1)
               / (# total pos strains)

    A strain is "covered" if the cocktail contains ≥1 phage that:
      (a) is in `selected`, AND
      (b) has label=1 for this strain
    """
    if pos_strains is None:
        pos_strains = get_strain_universe(sp_df)
    n = len(pos_strains)
    if n == 0:
        return 0.0
    pos_df     = sp_df[sp_df["label"] == 1]
    sel_pos_df = pos_df[pos_df["phage"].isin(selected)]
    covered    = set(sel_pos_df["strain"].unique()) & pos_strains
    return len(covered) / n

def interaction_cov(sp_df, selected, all_pos=None):
    """Original interaction-pair-level coverage (kept for comparison)."""
    if all_pos is None:
        all_pos = set(sp_df[sp_df["label"]==1].index)
    n = len(all_pos)
    if n == 0:
        return 0.0
    return sp_df.loc[list(all_pos), "phage"].isin(selected).sum() / n

def topk_sel(sp_df, k):
    return (sp_df.sort_values("ensemble_proba", ascending=False)
            ["phage"].head(k).tolist())

def greedy_sel_strain(sp_df, k, cov_fn=None):
    """
    Greedy selection maximizing STRAIN coverage.
    At each step adds the phage that covers the most new positive strains.
    """
    if cov_fn is None:
        cov_fn = strain_cov
    pos_df      = sp_df[sp_df["label"] == 1]
    pos_strains = get_strain_universe(sp_df)
    covered     = set()
    selected    = []
    remaining   = list(sp_df["phage"].unique())

    for _ in range(k):
        if not remaining or covered == pos_strains:
            break
        best, best_gain = None, -1
        for ph in remaining:
            # New strains this phage would cover
            ph_strains  = set(pos_df[pos_df["phage"]==ph]["strain"].unique())
            new_strains = len((ph_strains & pos_strains) - covered)
            prob        = sp_df[sp_df["phage"]==ph]["ensemble_proba"].max()
            if new_strains > best_gain or (
                    new_strains == best_gain and best is not None and
                    prob > sp_df[sp_df["phage"]==best]["ensemble_proba"].max()):
                best, best_gain = ph, new_strains
        if best is None:
            break
        ph_strains = set(pos_df[pos_df["phage"]==best]["strain"].unique())
        covered   |= ph_strains & pos_strains
        selected.append(best)
        remaining.remove(best)
    return selected

def div_greedy_sel_strain(sp_df, k, lam=LAMBDA_DIV):
    """Diversity-aware greedy with STRAIN-level coverage scoring."""
    pos_df      = sp_df[sp_df["label"] == 1]
    pos_strains = get_strain_universe(sp_df)
    covered     = set()
    selected    = []
    remaining   = list(sp_df["phage"].unique())

    for _ in range(k):
        if not remaining or covered == pos_strains:
            break
        best, best_score = None, -np.inf
        nc = len(covered) or 1
        for ph in remaining:
            ph_strains = set(pos_df[pos_df["phage"]==ph]["strain"].unique()) & pos_strains
            new_gain   = len(ph_strains - covered) / max(len(pos_strains), 1)
            overlap    = len(ph_strains & covered) / nc
            score      = new_gain - lam * overlap
            if score > best_score:
                best_score, best = score, ph
        if best is None:
            break
        ph_strains = set(pos_df[pos_df["phage"]==best]["strain"].unique())
        covered   |= ph_strains & pos_strains
        selected.append(best)
        remaining.remove(best)
    return selected

def rand_sel(sp_df, k, rng):
    phages = sp_df["phage"].unique()
    return list(rng.choice(phages, size=min(k, len(phages)), replace=False))

def robust_strain_cov(sp_df, selected):
    """Coverage after removing the highest-probability phage (resistance sim)."""
    if len(selected) < 2:
        return 0.0
    probs = {ph: sp_df[sp_df["phage"]==ph]["ensemble_proba"].max()
             for ph in selected}
    best  = max(probs, key=probs.get)
    return strain_cov(sp_df, [p for p in selected if p != best])


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — PER-SPECIES ANALYSIS (BOTH METRICS)
# ═══════════════════════════════════════════════════════════════
print("\n[2] Per-species analysis (strain-level + interaction-level)...")

rng          = np.random.default_rng(42)
species_list = sorted(pred_df["host"].unique())
all_res      = []
skipped      = 0

for sp in species_list:
    sp_df   = pred_df[pred_df["host"]==sp].copy().reset_index(drop=True)
    n_pos   = (sp_df["label"]==1).sum()
    if n_pos < 2:
        skipped += 1
        continue

    pos_strains = get_strain_universe(sp_df)
    n_strains   = len(pos_strains)
    all_pos_ix  = set(sp_df[sp_df["label"]==1].index)

    r = {
        "species":    sp,
        "genus":      sp_df["genus"].iloc[0],
        "n_pairs":    len(sp_df),
        "n_positive": n_pos,
        "n_strains":  n_strains,
        "pos_rate":   n_pos / len(sp_df),
    }

    for k in range(1, MAX_K+1):
        tk = topk_sel(sp_df, k)
        gr = greedy_sel_strain(sp_df, k)
        dg = div_greedy_sel_strain(sp_df, k)

        # ── Strain-level metrics ──
        scov_tk = strain_cov(sp_df, tk, pos_strains)
        scov_gr = strain_cov(sp_df, gr, pos_strains)
        scov_dg = strain_cov(sp_df, dg, pos_strains)

        r[f"s_topk_cov@{k}"]   = scov_tk
        r[f"s_greedy_cov@{k}"] = scov_gr
        r[f"s_divgr_cov@{k}"]  = scov_dg
        r[f"greedy_phages@{k}"] = "|".join(gr)

        rand_s = [strain_cov(sp_df, rand_sel(sp_df, k, rng), pos_strains)
                  for _ in range(N_RANDOM)]
        r[f"s_random_cov@{k}"]     = np.mean(rand_s)
        r[f"s_random_cov_std@{k}"] = np.std(rand_s)

        r[f"s_greedy_robust@{k}"]  = robust_strain_cov(sp_df, gr)
        r[f"s_greedy_covdrop@{k}"] = scov_gr - r[f"s_greedy_robust@{k}"]

        # ── Interaction-level metrics (for comparison) ──
        r[f"i_greedy_cov@{k}"] = interaction_cov(sp_df, gr, all_pos_ix)
        r[f"i_random_cov@{k}"] = np.mean([
            interaction_cov(sp_df, rand_sel(sp_df, k, rng), all_pos_ix)
            for _ in range(N_RANDOM)])

    for k in range(1, MAX_K+1):
        r[f"s_greedy_marginal@{k}"] = (
            r[f"s_greedy_cov@{k}"] - r[f"s_greedy_cov@{k-1}"]
            if k > 1 else r[f"s_greedy_cov@1"])

    all_res.append(r)

results_df = pd.DataFrame(all_res)
results_df.to_csv(RESULTS_DIR / "cocktail_per_species.csv", index=False)
print(f"  Analyzed: {len(results_df)} species (skipped {skipped})")
print(f"  Total unique strains across all species: "
      f"{results_df['n_strains'].sum():.0f}")


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — BOOTSTRAP CIs
# ═══════════════════════════════════════════════════════════════
print("\n[3] Bootstrap CIs (1000 resamples)...")
ci_res = {}
brng   = np.random.default_rng(123)

for k in range(1, MAX_K+1):
    for s in ["s_topk","s_greedy","s_divgr","s_random",
              "i_greedy","i_random"]:
        col  = f"{s}_cov@{k}"
        if col not in results_df.columns:
            continue
        vals = results_df[col].dropna().values
        bm   = [brng.choice(vals, size=len(vals), replace=True).mean()
                for _ in range(N_BOOTSTRAP)]
        ci_res[f"{s}_ci@{k}"] = (
            float(np.mean(vals)),
            float(np.percentile(bm, 2.5)),
            float(np.percentile(bm, 97.5)))

pd.DataFrame(ci_res, index=["mean","ci_lo","ci_hi"]).T.to_csv(
    RESULTS_DIR / "cocktail_bootstrap_ci.csv")

for k in [1, 3, 5]:
    sm, slo, shi = ci_res.get(f"s_greedy_ci@{k}", (0,0,0))
    im, ilo, ihi = ci_res.get(f"i_greedy_ci@{k}", (0,0,0))
    print(f"  k={k}:  strain={sm:.3f} (95% CI: {slo:.3f}–{shi:.3f})  "
          f"  interaction={im:.3f} (95% CI: {ilo:.3f}–{ihi:.3f})")


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — GLOBAL METRICS
# ═══════════════════════════════════════════════════════════════
print("\n[4] Global metrics...")
gm = {}

for k in range(1, MAX_K+1):
    for s in ["s_topk","s_greedy","s_divgr","s_random","i_greedy","i_random"]:
        col = f"{s}_cov@{k}"
        if col in results_df.columns:
            gm[f"mean_{s}@{k}"]  = results_df[col].mean()
            gm[f"pct75_{s}@{k}"] = (results_df[col] >= 0.75).mean() * 100
    gm[f"s_greedy_over_random@{k}"] = (
        gm.get(f"mean_s_greedy@{k}",0) - gm.get(f"mean_s_random@{k}",0))
    gm[f"s_divgr_over_greedy@{k}"]  = (
        gm.get(f"mean_s_divgr@{k}",0) - gm.get(f"mean_s_greedy@{k}",0))
    gm[f"strain_vs_interaction_delta@{k}"] = (
        gm.get(f"mean_s_greedy@{k}",0) - gm.get(f"mean_i_greedy@{k}",0))
    gm[f"mean_s_covdrop@{k}"] = results_df[f"s_greedy_covdrop@{k}"].mean()

pd.Series(gm).to_csv(RESULTS_DIR / "cocktail_global_metrics.csv", header=["value"])

print(f"\n  {'':28} {'Random':>8} {'Top-K':>8} {'Greedy':>8} {'DivGrdy':>9}")
print("  " + "-"*60)
for k in range(1, MAX_K+1):
    r  = gm.get(f"mean_s_random@{k}",0)
    t  = gm.get(f"mean_s_topk@{k}",0)
    g  = gm.get(f"mean_s_greedy@{k}",0)
    dg = gm.get(f"mean_s_divgr@{k}",0)
    print(f"  STRAIN  coverage @ k={k}:       "
          f"{r:>8.3f} {t:>8.3f} {g:>8.3f} {dg:>9.3f}")
print()
print(f"\n  {'':28} {'Interaction':>12} {'Strain':>8} {'Delta':>8}")
print("  " + "-"*52)
for k in range(1, MAX_K+1):
    ig = gm.get(f"mean_i_greedy@{k}",0)
    sg = gm.get(f"mean_s_greedy@{k}",0)
    d  = gm.get(f"strain_vs_interaction_delta@{k}",0)
    print(f"  Greedy coverage @ k={k}:         "
          f"{ig:>12.3f} {sg:>8.3f} {d:>+8.3f}")


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════
print("\n[5] Statistical tests (strain-level)...")

def wtest(a, b, label):
    d  = np.array(a) - np.array(b)
    nz = d[d != 0]
    if len(nz) < 5:
        return 0.0, 1.0
    stat, p = wilcoxon(nz)
    sig = "SIGNIFICANT ✓" if p < 0.05 else "not significant"
    print(f"  {label}: p={p:.4f}  → {sig}")
    return stat, p

_,p13  = wtest(results_df["s_greedy_cov@3"], results_df["s_greedy_cov@1"], "k=1 vs k=3 (strain)")
_,p15  = wtest(results_df["s_greedy_cov@5"], results_df["s_greedy_cov@1"], "k=1 vs k=5 (strain)")
_,pgr  = wtest(results_df["s_greedy_cov@3"], results_df["s_random_cov@3"], "Greedy vs Random@3")
_,pkr  = wtest(results_df["s_topk_cov@3"],   results_df["s_random_cov@3"], "Top-K vs Random@3")
_,pgk  = wtest(results_df["s_greedy_cov@3"], results_df["s_topk_cov@3"],   "Greedy vs Top-K@3")
_,pdg  = wtest(results_df["s_divgr_cov@3"],  results_df["s_greedy_cov@3"], "DivGreedy vs Greedy@3")
_,pra  = wtest(results_df["s_greedy_robust@3"],
               results_df["s_greedy_robust@1"], "Cocktail@3 vs Single (after resistance)")
# Strain vs interaction comparison
_,psvi = wtest(results_df["s_greedy_cov@3"],
               results_df["i_greedy_cov@3"], "Strain vs Interaction metric@3")

pd.Series({
    "k1_vs_k3_p":p13, "k1_vs_k5_p":p15,
    "greedy_vs_random_p":pgr, "topk_vs_random_p":pkr,
    "greedy_vs_topk_p":pgk, "divgreedy_vs_greedy_p":pdg,
    "resistance_p":pra, "strain_vs_interaction_p":psvi,
    "s_greedy_over_random@3": gm.get("s_greedy_over_random@3",0),
    "s_divgr_over_greedy@3":  gm.get("s_divgr_over_greedy@3",0),
}).to_csv(RESULTS_DIR / "cocktail_statistics.csv", header=["value"])


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — GENUS AGGREGATION
# ═══════════════════════════════════════════════════════════════
print("\n[6] Genus aggregation...")
gagg = results_df.groupby("genus").agg(
    n_species        = ("species","count"),
    total_strains    = ("n_strains","sum"),
    mean_s_greedy_k3 = ("s_greedy_cov@3","mean"),
    mean_s_random_k3 = ("s_random_cov@3","mean"),
    mean_robust_k3   = ("s_greedy_robust@3","mean"),
    pct75_at_k3      = ("s_greedy_cov@3", lambda x: (x>=0.75).mean()*100),
).round(3).sort_values("mean_s_greedy_k3", ascending=False)
gagg.to_csv(RESULTS_DIR / "cocktail_genus_summary.csv")
print(gagg.head(8).to_string())


# ═══════════════════════════════════════════════════════════════
# SECTION 9 — CORRELATION
# ═══════════════════════════════════════════════════════════════
corr_r = corr_p = None
merged = results_df.copy()
merged["species_auc"] = np.nan
sp_auc_path = RESULTS_DIR / "ensemble_per_species.csv"
if sp_auc_path.exists():
    sp_auc = pd.read_csv(sp_auc_path)
    sp_auc.columns = ["species"] + list(sp_auc.columns[1:])
    sp_auc = sp_auc.rename(columns={"roc_auc":"species_auc"})
    merged = results_df.merge(
        sp_auc[["species","species_auc"]], on="species", how="inner")
    if len(merged) > 5:
        corr_r, corr_p = pearsonr(
            merged["species_auc"], merged["s_greedy_cov@3"])
        print(f"\n  AUC vs strain greedy_cov@3: r={corr_r:.3f}, p={corr_p:.4f}")


# ═══════════════════════════════════════════════════════════════
# SECTION 10 — S. AUREUS DEEP DIVE
# ═══════════════════════════════════════════════════════════════
print("\n[7] S. aureus deep dive (strain-level)...")
sa_df = pred_df[pred_df["host"]==TARGET_HOST].copy().reset_index(drop=True)

if len(sa_df) >= 5:
    sa_pos_strains = get_strain_universe(sa_df)
    sa_rank        = sa_df.sort_values("ensemble_proba", ascending=False)
    n_sa_strains   = len(sa_pos_strains)

    # Show how many unique strains NCBI resolved
    sa_ncbi = sa_df["ncbi_strain"].notna().sum()
    sa_clust = sa_df["cluster_strain"].nunique()

    print(f"\n  S. aureus: {len(sa_df)} pairs | "
          f"{(sa_df['label']==1).sum()} positives | "
          f"{n_sa_strains} unique strains")
    print(f"  Strain sources: {sa_ncbi} NCBI-annotated, "
          f"{sa_clust} cluster groups")

    print(f"\n  Top 10 predicted phages:")
    print(f"  {'Rank':<5} {'Phage':<32} {'Prob':>6}  {'True+':>5}  {'Strain':<30}")
    print("  " + "-"*82)
    for i, (_, row) in enumerate(sa_rank.head(10).iterrows(), 1):
        strain_short = str(row["strain"])[:28] if pd.notna(row["strain"]) else "unknown"
        print(f"  {i:<5} {row['phage'][:30]:<32} "
              f"{row['ensemble_proba']:>6.4f}  "
              f"{'✓' if row['label']==1 else '✗':>5}  "
              f"{strain_short:<30}")

    print(f"\n  Strain-level coverage curves:")
    print(f"  {'k':<4} {'Interaction':>12} {'Strain':>8} {'Robust':>8}  Phages selected")
    print("  " + "-"*75)
    for k in range(1, MAX_K+1):
        sel   = greedy_sel_strain(sa_df, k)
        icov  = interaction_cov(sa_df, sel)
        scov  = strain_cov(sa_df, sel, sa_pos_strains)
        rob   = robust_strain_cov(sa_df, sel)
        print(f"  k={k}  {icov:>12.4f} {scov:>8.4f} {rob:>8.4f}  "
              f"{', '.join(p[:20] for p in sel)}")

    # Unique strains covered at k=3
    gr3 = greedy_sel_strain(sa_df, 3)
    pos_pos = sa_df[sa_df["label"]==1]
    covered_strains = set(pos_pos[pos_pos["phage"].isin(gr3)]["strain"].unique())
    print(f"\n  Strains covered by greedy@3: {len(covered_strains)} / {n_sa_strains}")
    for s in sorted(covered_strains)[:10]:
        print(f"    • {s}")
    if len(covered_strains) > 10:
        print(f"    ... ({len(covered_strains)-10} more)")


# ═══════════════════════════════════════════════════════════════
# SECTION 11 — PLOTS
# ═══════════════════════════════════════════════════════════════
print("\n[8] Generating plots...")
ks = list(range(1, MAX_K+1))

# ── Plot 22: 4-panel strain-level overview ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
for strat, color, ls, lbl in [
    ("s_random","#BDBDBD","o:","Random (strain)"),
    ("s_topk",  "#4285F4","s-","Top-K (strain)"),
    ("s_greedy","#FF6B35","D--","Greedy (strain)"),
    ("s_divgr", "#1a9850","^-.","Div-Greedy (strain)"),
]:
    means = [gm.get(f"mean_{strat}@{k}",0) for k in ks]
    los   = [ci_res.get(f"{strat}_ci@{k}",(0,0,0))[1] for k in ks]
    his   = [ci_res.get(f"{strat}_ci@{k}",(0,0,0))[2] for k in ks]
    ax.plot(ks, means, ls, color=color, lw=2.5,
             label=f"{lbl} ({means[2]:.3f}@3)")
    ax.fill_between(ks, los, his, alpha=0.12, color=color)

# Also overlay interaction-level greedy for comparison
i_means = [gm.get(f"mean_i_greedy@{k}",0) for k in ks]
ax.plot(ks, i_means, "x--", color="#9C27B0", lw=1.5, alpha=0.7,
         label=f"Greedy (interaction) ({i_means[2]:.3f}@3)")
ax.axhline(0.75, color="gold", linestyle=":", lw=2)
ax.set_xlabel("k", fontsize=11)
ax.set_ylabel("Mean coverage (95% CI)", fontsize=11)
ax.set_title("A  Strain-Level Coverage vs k", fontsize=12, fontweight="bold")
ax.set_xticks(ks); ax.set_ylim(0, 1.05)
ax.legend(fontsize=7, loc="upper left"); ax.grid(True, alpha=0.4)

# Panel B: strain vs interaction delta
ax = axes[0, 1]
deltas = [gm.get(f"strain_vs_interaction_delta@{k}",0) for k in ks]
colors_d = ["#1a9850" if d >= 0 else "#F44336" for d in deltas]
bars = ax.bar(ks, deltas, color=colors_d, alpha=0.85)
for bar, d in zip(bars, deltas):
    ax.text(bar.get_x()+bar.get_width()/2,
             d + (0.002 if d >= 0 else -0.006),
             f"{d:+.3f}", ha="center", fontsize=9, fontweight="bold")
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("k", fontsize=11)
ax.set_ylabel("Strain − Interaction coverage", fontsize=11)
ax.set_title("B  Strain vs Interaction Coverage Delta\n"
              "(positive = strain-level is higher)",
              fontsize=11, fontweight="bold")
ax.set_xticks(ks); ax.grid(axis="y", alpha=0.4)

# Panel C: Resistance robustness (strain-level)
ax = axes[1, 0]
labels_r = ["Single\nphage", "Cocktail k=3\nnormal",
            "Cocktail k=3\nresistant", "Cocktail k=5\nnormal",
            "Cocktail k=5\nresistant"]
vals_r = [
    gm.get("mean_s_greedy@1", 0),
    gm.get("mean_s_greedy@3", 0),
    results_df["s_greedy_robust@3"].mean(),
    gm.get("mean_s_greedy@5", 0),
    results_df["s_greedy_robust@5"].mean(),
]
colors_r = ["#F44336","#FF6B35","#FF9800","#1a9850","#4CAF50"]
bars_r   = ax.bar(labels_r, vals_r, color=colors_r, alpha=0.85)
for bar, v in zip(bars_r, vals_r):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.005, f"{v:.3f}",
             ha="center", fontsize=9, fontweight="bold")
ax.axhline(0.75, color="gold", linestyle="--", lw=1.5)
ax.set_ylabel("Mean strain coverage", fontsize=11)
ax.set_title("C  Resistance Robustness (strain-level)\n"
              "(Remove highest-prob phage)",
              fontsize=11, fontweight="bold")
ax.set_ylim(0, max(vals_r)*1.25); ax.grid(axis="y", alpha=0.4)

# Panel D: Bootstrap CI comparison (strain vs interaction)
ax = axes[1, 1]
s_means = [ci_res.get(f"s_greedy_ci@{k}",(0,0,0))[0] for k in ks]
s_los   = [ci_res.get(f"s_greedy_ci@{k}",(0,0,0))[1] for k in ks]
s_his   = [ci_res.get(f"s_greedy_ci@{k}",(0,0,0))[2] for k in ks]
i_means_ci = [ci_res.get(f"i_greedy_ci@{k}",(0,0,0))[0] for k in ks]
i_los   = [ci_res.get(f"i_greedy_ci@{k}",(0,0,0))[1] for k in ks]
i_his   = [ci_res.get(f"i_greedy_ci@{k}",(0,0,0))[2] for k in ks]
r_means = [ci_res.get(f"s_random_ci@{k}",(0,0,0))[0] for k in ks]
r_los   = [ci_res.get(f"s_random_ci@{k}",(0,0,0))[1] for k in ks]
r_his   = [ci_res.get(f"s_random_ci@{k}",(0,0,0))[2] for k in ks]

ax.errorbar(ks, s_means, yerr=[np.array(s_means)-np.array(s_los),
                                 np.array(s_his)-np.array(s_means)],
             fmt="D-", color="#FF6B35", lw=2.5, capsize=5, label="Greedy (strain)")
ax.errorbar(ks, i_means_ci, yerr=[np.array(i_means_ci)-np.array(i_los),
                                    np.array(i_his)-np.array(i_means_ci)],
             fmt="x--", color="#9C27B0", lw=1.5, capsize=4, label="Greedy (interaction)")
ax.errorbar(ks, r_means, yerr=[np.array(r_means)-np.array(r_los),
                                  np.array(r_his)-np.array(r_means)],
             fmt="o:", color="#BDBDBD", lw=1.5, capsize=4, label="Random (strain)")
ax.fill_between(ks, s_los, s_his, alpha=0.15, color="#FF6B35")
ax.axhline(0.75, color="gold", linestyle=":", lw=1.5)
ax.set_xlabel("k", fontsize=11); ax.set_ylabel("Mean coverage", fontsize=11)
ax.set_title("D  Bootstrap 95% CI: Strain vs Interaction",
              fontsize=11, fontweight="bold")
ax.set_xticks(ks); ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

plt.suptitle("PrecisionPhage — Strain-Level Cocktail Optimization (LOSO-CV)",
              fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
save_plot(fig, "22_cocktail_overview")

# ── Plot 23: per-species strain coverage ──
top_sp = results_df.nlargest(30, "n_positive")
fig, ax = plt.subplots(figsize=(12, max(7, len(top_sp)*0.45)))
yp = np.arange(len(top_sp)); w23 = 0.20
ax.barh(yp-1.5*w23, top_sp["s_random_cov@3"], w23, color="#BDBDBD", alpha=0.7,
         label="Random (strain)@3")
ax.barh(yp-0.5*w23, top_sp["s_topk_cov@3"],   w23, color="#4285F4", alpha=0.85,
         label="Top-K (strain)@3")
ax.barh(yp+0.5*w23, top_sp["s_greedy_cov@3"], w23, color="#FF6B35", alpha=0.85,
         label="Greedy (strain)@3")
ax.barh(yp+1.5*w23, top_sp["i_greedy_cov@3"], w23, color="#9C27B0", alpha=0.6,
         label="Greedy (interaction)@3")
ax.set_yticks(yp)
ax.set_yticklabels([f"{r['species']} "
                     f"(n_strains={r['n_strains']:.0f})"
                     for _,r in top_sp.iterrows()], fontsize=8)
ax.axvline(0.75, color="gold", linestyle="--", lw=1.5)
ax.set_xlabel("Coverage @ k=3", fontsize=12)
ax.set_title("Per-Species: Strain vs Interaction Coverage @ k=3\n"
              "(Top 30 species by n_positive)",
              fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.set_xlim(0, 1.15); ax.grid(axis="x", alpha=0.4)
plt.tight_layout()
save_plot(fig, "23_cocktail_per_species")

# ── Plot 24: strain vs interaction scatter ──
fig, axes24 = plt.subplots(1, 2, figsize=(13, 5))
ax = axes24[0]
ax.scatter(results_df["i_greedy_cov@3"], results_df["s_greedy_cov@3"],
            alpha=0.5, c=results_df["n_strains"], cmap="viridis", s=60)
sm = plt.cm.ScalarMappable(cmap="viridis")
sm.set_array(results_df["n_strains"])
plt.colorbar(sm, ax=ax, label="# unique strains")
sa_r = results_df[results_df["species"]==TARGET_HOST]
if len(sa_r):
    ax.scatter(sa_r["i_greedy_cov@3"], sa_r["s_greedy_cov@3"],
                color="red", s=150, zorder=5, label="S. aureus")
ax.plot([0,1],[0,1],"--",color="gray",alpha=0.4,label="Equal")
ax.set_xlabel("Interaction coverage@3 (old)", fontsize=11)
ax.set_ylabel("Strain coverage@3 (new)", fontsize=11)
ax.set_title("Strain vs Interaction Coverage\nColour = # unique strains",
              fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.4)
ax.set_xlim(-0.05,1.05); ax.set_ylim(-0.05,1.05)

ax = axes24[1]
for strat, color, ls, lbl in [
    ("s_random","#BDBDBD","o:","Random (strain)"),
    ("s_greedy","#FF6B35","D--","Greedy (strain)"),
    ("i_greedy","#9C27B0","x-","Greedy (interaction)"),
]:
    means = [gm.get(f"mean_{strat}@{k}",0) for k in ks]
    los   = [ci_res.get(f"{strat}_ci@{k}",(0,0,0))[1] for k in ks]
    his   = [ci_res.get(f"{strat}_ci@{k}",(0,0,0))[2] for k in ks]
    ax.plot(ks, means, ls, color=color, lw=2.5, label=f"{lbl}")
    ax.fill_between(ks, los, his, alpha=0.12, color=color)
ax.axhline(0.75, color="gold", linestyle=":", lw=1.5)
ax.set_xlabel("k", fontsize=11); ax.set_ylabel("Mean coverage (95% CI)", fontsize=11)
ax.set_title("Strain vs Interaction: Full Comparison\nWith 95% Bootstrap CIs",
              fontsize=11, fontweight="bold")
ax.set_xticks(ks); ax.legend(fontsize=9); ax.grid(True, alpha=0.4)
plt.tight_layout()
save_plot(fig, "24_strain_vs_interaction")

# ── Plot 25: genus-level strain coverage ──
genera_show = (results_df.groupby("genus")["n_positive"]
               .sum().nlargest(10).index.tolist())
gd_rows = []
for g in genera_show:
    gdf = results_df[results_df["genus"]==g]
    for k in [1,3,5]:
        for s in ["s_random","s_greedy"]:
            gd_rows.append({"genus":g,"k":f"k={k}","strategy":s,
                             "coverage":gdf[f"{s}_cov@{k}"].mean()})
gd_pd = pd.DataFrame(gd_rows)
fig, ax = plt.subplots(figsize=(12,5))
x25 = np.arange(len(genera_show)); w25 = 0.13
for i,(k,s,color,hatch) in enumerate([
    ("k=1","s_greedy","#d73027",""), ("k=3","s_greedy","#fc8d59",""),
    ("k=5","s_greedy","#1a9850",""), ("k=3","s_random","#BDBDBD","//")
]):
    vals25 = []
    for g in genera_show:
        sub = gd_pd[(gd_pd["genus"]==g)&(gd_pd["k"]==k)&(gd_pd["strategy"]==s)]
        vals25.append(sub["coverage"].values[0] if len(sub) else 0)
    ax.bar(x25+(i-1.5)*w25, vals25, w25, color=color, alpha=0.85, hatch=hatch,
            label=f"{'Greedy' if 'greedy' in s else 'Random'} {k} (strain)")
ax.set_xticks(x25); ax.set_xticklabels(genera_show, rotation=30, ha="right")
ax.axhline(0.75, color="gold", lw=1.5, linestyle="--")
ax.set_ylabel("Mean strain coverage", fontsize=12)
ax.set_title("Strain Coverage by Genus (Top 10)", fontsize=11, fontweight="bold")
ax.legend(fontsize=8, ncol=4); ax.grid(axis="y", alpha=0.4); ax.set_ylim(0,1.1)
plt.tight_layout()
save_plot(fig, "25_coverage_by_genus")

# ── Plot 26: statistics ──
fig, axes26 = plt.subplots(1, 2, figsize=(13, 5))
ax = axes26[0]
bp_data = pd.DataFrame({
    "Random\n(strain)k=3":  results_df["s_random_cov@3"],
    "Top-K\n(strain)k=3":   results_df["s_topk_cov@3"],
    "Greedy\n(strain)k=3":  results_df["s_greedy_cov@3"],
    "Greedy\n(intxn)k=3":   results_df["i_greedy_cov@3"],
    "Greedy\n(strain)k=5":  results_df["s_greedy_cov@5"],
})
bp = bp_data.boxplot(ax=ax, patch_artist=True, notch=True,
                      medianprops=dict(color="black", lw=2))
for patch, c in zip(ax.patches[:5],
                    ["#BDBDBD","#4285F4","#FF6B35","#9C27B0","#FF6B35"]):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax.axhline(0.75, color="gold", linestyle="--", lw=1.5)
sig_g = "***" if pgr<0.001 else "**" if pgr<0.01 else "*" if pgr<0.05 else "ns"
ax.set_title(f"Strain Coverage Distributions\n"
              f"Greedy vs Random p={pgr:.4f} {sig_g}",
              fontsize=11, fontweight="bold")
ax.set_ylabel("Strain Coverage", fontsize=11); ax.grid(axis="y", alpha=0.4)

ax = axes26[1]
drop3s = results_df["s_greedy_covdrop@3"].values * 100
drop5s = results_df["s_greedy_covdrop@5"].values * 100
ax.hist(drop3s, bins=15, alpha=0.7, color="#FF6B35",
         label=f"k=3 (mean={drop3s.mean():.1f}%)")
ax.hist(drop5s, bins=15, alpha=0.7, color="#1a9850",
         label=f"k=5 (mean={drop5s.mean():.1f}%)")
ax.axvline(drop3s.mean(), color="#FF6B35", lw=2, linestyle="--")
ax.axvline(drop5s.mean(), color="#1a9850", lw=2, linestyle="--")
ax.set_xlabel("Strain coverage drop after resistance (%)", fontsize=11)
ax.set_ylabel("Number of species", fontsize=11)
ax.set_title("Resistance Robustness (strain-level)",
              fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
save_plot(fig, "26_cocktail_statistics")

# ── Plot 27: S. aureus publication-style ──
if len(sa_df) >= 5:
    sa_rank  = sa_df.sort_values("ensemble_proba", ascending=False)
    sa_g5    = greedy_sel_strain(sa_df, 5)
    sa_pos_d = sa_df[sa_df["label"]==1]
    fig27    = plt.figure(figsize=(16, 11))
    gs27     = gridspec.GridSpec(2, 3, figure=fig27, hspace=0.45, wspace=0.38)

    ax = fig27.add_subplot(gs27[0, 0])
    t10 = sa_rank.head(10); y_t = np.arange(len(t10))
    colors_t = ["#1a9850" if l==1 else "#F44336" for l in t10["label"]]
    ax.barh(y_t, t10["ensemble_proba"], color=colors_t, alpha=0.85)
    ax.set_yticks(y_t)
    ax.set_yticklabels(t10["phage"].str[:28], fontsize=8)
    ax.set_xlabel("Ensemble probability", fontsize=10)
    ax.set_title("S. aureus: Top 10 Candidate Phages\nGreen=TP | Red=FP",
                  fontsize=9, fontweight="bold")
    ax.axvline(0.5, color="orange", lw=1, linestyle="--")
    ax.grid(axis="x", alpha=0.4)

    ax = fig27.add_subplot(gs27[0, 1])
    sa_icov = [interaction_cov(sa_df, greedy_sel_strain(sa_df, k)) for k in ks]
    sa_scov = [strain_cov(sa_df, greedy_sel_strain(sa_df, k), sa_pos_strains) for k in ks]
    sa_rcov = [np.mean([strain_cov(sa_df, rand_sel(sa_df, k, rng), sa_pos_strains)
                         for _ in range(50)]) for k in ks]
    ax.plot(ks, sa_icov, "x--", color="#9C27B0", lw=2, label="Greedy (interaction)")
    ax.plot(ks, sa_scov, "D-",  color="#FF6B35", lw=2, label="Greedy (strain)")
    ax.plot(ks, sa_rcov, "o:",  color="#BDBDBD", lw=1.5, label="Random (strain)")
    ax.axhline(0.75, color="gold", linestyle=":", lw=1.5)
    ax.set_xlabel("k", fontsize=10); ax.set_ylabel("Coverage", fontsize=10)
    ax.set_title(f"S. aureus Coverage Curves\n"
                  f"({n_sa_strains} unique strains)",
                  fontsize=9, fontweight="bold")
    ax.set_xticks(ks); ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
    ax.set_ylim(0, 1.05)

    ax = fig27.add_subplot(gs27[0, 2])
    r_norm = [strain_cov(sa_df, greedy_sel_strain(sa_df, k), sa_pos_strains)
               for k in [1,3,5]]
    r_rob  = [robust_strain_cov(sa_df, greedy_sel_strain(sa_df, k))
               for k in [1,3,5]]
    xr = np.arange(3); wr = 0.35
    ax.bar(xr-wr/2, r_norm, wr, color="#FF6B35", alpha=0.85, label="Normal")
    ax.bar(xr+wr/2, r_rob,  wr, color="#F44336", alpha=0.65, label="Post-resistance")
    for xi, (n,r) in enumerate(zip(r_norm, r_rob)):
        ax.text(xi-wr/2, n+0.01, f"{n:.3f}", ha="center", fontsize=8)
        ax.text(xi+wr/2, r+0.01, f"{r:.3f}", ha="center", fontsize=8)
    ax.set_xticks(xr); ax.set_xticklabels([f"k={k}" for k in [1,3,5]])
    ax.set_ylabel("Strain coverage", fontsize=10)
    ax.set_title("S. aureus Resistance Robustness\n(Strain-level)",
                  fontsize=9, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.4)

    # Overlap matrix (strain-level: Jaccard on covered strains)
    ax = fig27.add_subplot(gs27[1, :])
    if len(sa_g5) >= 2:
        om = np.zeros((len(sa_g5), len(sa_g5)))
        for i, p1 in enumerate(sa_g5):
            for j, p2 in enumerate(sa_g5):
                s1 = set(sa_pos_d[sa_pos_d["phage"]==p1]["strain"].unique())
                s2 = set(sa_pos_d[sa_pos_d["phage"]==p2]["strain"].unique())
                u  = s1 | s2
                om[i,j] = len(s1&s2) / max(len(u), 1)
        im = ax.imshow(om, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(sa_g5))); ax.set_yticks(range(len(sa_g5)))
        lbs = [p[:30] for p in sa_g5]
        ax.set_xticklabels(lbs, rotation=25, ha="right", fontsize=8)
        ax.set_yticklabels(lbs, fontsize=8)
        for i in range(len(sa_g5)):
            for j in range(len(sa_g5)):
                ax.text(j, i, f"{om[i,j]:.2f}", ha="center", va="center",
                         fontsize=9,
                         color="white" if om[i,j] > 0.6 else "black")
        plt.colorbar(im, ax=ax, fraction=0.015, label="Strain Jaccard overlap")
        ax.set_title("S. aureus Phage Overlap Matrix (Greedy@5) — Strain Level\n"
                      "Low off-diagonal = high strain diversity = better cocktail",
                      fontsize=10, fontweight="bold")
    fig27.suptitle("S. aureus Deep Dive — PrecisionPhage (Strain-Level)",
                    fontsize=14, fontweight="bold")
    save_plot(fig27, "27_sa_aureus_cocktail")


# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════
sm3, slo3, shi3 = ci_res.get("s_greedy_ci@3", (0,0,0))
rm3, rlo3, rhi3 = ci_res.get("s_random_ci@3", (0,0,0))
im3, ilo3, ihi3 = ci_res.get("i_greedy_ci@3", (0,0,0))
d3m = results_df["s_greedy_covdrop@3"].mean()*100
d5m = results_df["s_greedy_covdrop@5"].mean()*100
dga = gm.get("s_divgr_over_greedy@3",0)*100

# S. aureus strain numbers
sa_s3 = strain_cov(sa_df, greedy_sel_strain(sa_df, 3), sa_pos_strains) if len(sa_df)>=5 else 0
n_sa_s = len(get_strain_universe(sa_df)) if len(sa_df)>=5 else 0

print()
print("=" * 62)
print("  STRAIN-LEVEL COCKTAIL ANALYSIS  —  FINAL SUMMARY")
print("=" * 62)
print(f"""
  Old interaction-level coverage@3 (Greedy): {im3:.3f}
  New strain-level    coverage@3 (Greedy):   {sm3:.3f}
  Delta:                                     {sm3-im3:+.3f}

  ──────────────────────────────────────────────
  Mean strain coverage@3 (Random):   {rm3:.3f}  (95% CI: {rlo3:.3f}–{rhi3:.3f})
  Mean strain coverage@3 (Greedy):   {sm3:.3f}  (95% CI: {slo3:.3f}–{shi3:.3f})
  Improvement over random:           +{gm.get('s_greedy_over_random@3',0)*100:.1f}%
  (Wilcoxon p={pgr:.4f}{'  SIGNIFICANT ✓' if pgr<0.05 else ''})

  Resistance robustness (strain coverage drop):
    k=3: {d3m:.1f}%  |  k=5: {d5m:.1f}%
  (Cocktail@3 vs single after resistance p={pra:.4f}{'  SIGNIFICANT ✓' if pra<0.05 else ''})

  Diversity-aware vs standard greedy:  {dga:+.1f}%
  (p={pdg:.4f}{'  SIGNIFICANT ✓' if pdg<0.05 else '  not significant'})

  95% CI strain coverage@3 (Greedy): [{slo3:.3f} – {shi3:.3f}]

  % species ≥75% strain coverage:
    k=3 Greedy:  {gm.get('pct75_s_greedy@3',0):.1f}%
    k=5 Greedy:  {gm.get('pct75_s_greedy@5',0):.1f}%
    k=3 Random:  {gm.get('pct75_s_random@3',0):.1f}%

  S. aureus strain coverage@3:       {sa_s3:.3f}
    ({n_sa_s} unique strains identified)
  Strain tier used:                  {tier_used}
""")
print("  Results:", str(RESULTS_DIR.resolve()))
print("  Plots:  ", str(PLOT_DIR.resolve()))
print("\n  Done!")