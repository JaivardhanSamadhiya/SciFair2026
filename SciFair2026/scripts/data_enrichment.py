"""
00_enrich_data.py  —  Download additional phage-host pairs and host biology features
======================================================================================
Run this ONCE before re-running 03_gnn.py.

SOURCES:
  1. Millard Lab (Oct 2023)  — ~17,000 phage-host pairs from GenBank annotations
     URL: https://millardlab.org/bacteriophage-genomics/phage-genomes-oct-2023/
     File: 1Oct2023_vConTACT2_host_annotations.tsv.gz  (direct download, no auth)

  2. NCBI Entrez API  — additional bacteriophage-host pairs from GenBank records
     No API key needed for <3 requests/sec (we throttle to be safe)
     Queries: phages with known host annotations in the /host field

  3. BacDive REST API — host phenotype features per genus
     Endpoint: https://api.bacdive.dsmz.de/taxon/{genus}/
     Free registration required: https://bacdive.dsmz.de/api  (takes 2 min)
     Returns: gram_stain, oxygen_tolerance, cell_morphology, motility, phylum
     Set BACDIVE_USER and BACDIVE_PASS environment variables OR enter below.

OUTPUT FILES (saved to data/raw/):
  millard_pairs.csv        — phage, host, label=1, source=millard
  ncbi_pairs.csv           — phage, host, label=1, source=ncbi_entrez
  host_features.csv        — genus, gram_pos, gram_neg, phylum_*, oxygen_*, motility
  enriched_dataset.csv     — merged dataset ready for 03_gnn.py
"""

import os, sys, time, gzip, io, json, re, csv
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── CONFIG ─────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
BASE_DIR      = _SCRIPT_DIR.parent / "data"
RAW_DIR       = BASE_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# BacDive credentials — set here OR as environment variables
BACDIVE_USER  = os.environ.get("BACDIVE_USER", "")   # e.g. "your@email.com"
BACDIVE_PASS  = os.environ.get("BACDIVE_PASS", "")   # your BacDive password

NCBI_EMAIL    = os.environ.get("NCBI_EMAIL", "student@sciencefair.edu")
NCBI_API_KEY  = os.environ.get("NCBI_API_KEY", "")   # optional, increases rate limit

# ── HELPERS ────────────────────────────────────────────────────
def normalize(x):
    return str(x).strip().lower().replace("_", " ").replace("-", " ")

def get_genus(host_str):
    words = str(host_str).strip().split()
    return words[0].lower() if words else ""

def safe_get(url, auth=None, params=None, retries=3, wait=1.0):
    for attempt in range(retries):
        try:
            r = requests.get(url, auth=auth, params=params, timeout=30)
            if r.status_code == 200:
                return r
            elif r.status_code == 429:
                time.sleep(wait * (attempt + 1) * 2)
            else:
                time.sleep(wait)
        except Exception as e:
            print(f"    Request error ({e}), retry {attempt+1}/{retries}")
            time.sleep(wait * (attempt + 1))
    return None


# ═══════════════════════════════════════════════════════════════
# STEP 1 — MILLARD LAB  (Oct 2023 vConTACT2 host annotations)
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Downloading Millard Lab host annotations...")
print("=" * 60)

MILLARD_URL  = ("https://millardlab.org/wp-content/uploads/2023/10/"
                "1Oct2023_vConTACT2_host_annotations.tsv_.gz")
MILLARD_OUT  = RAW_DIR / "millard_pairs.csv"

millard_df = pd.DataFrame()

if MILLARD_OUT.exists():
    millard_df = pd.read_csv(MILLARD_OUT)
    print(f"  Loaded cached: {len(millard_df)} pairs")
else:
    print(f"  Fetching: {MILLARD_URL}")
    r = safe_get(MILLARD_URL)
    if r is None:
        # Try alternate URL pattern
        alt_url = ("https://millardlab.org/wp-content/uploads/2023/10/"
                   "1Oct2023_vConTACT2_host_annotations.tsv.gz")
        print(f"  Trying alternate URL...")
        r = safe_get(alt_url)

    if r and r.status_code == 200:
        try:
            buf = io.BytesIO(r.content)
            with gzip.open(buf, "rt", encoding="utf-8", errors="replace") as f:
                tsv_text = f.read()
            tsv_df = pd.read_csv(io.StringIO(tsv_text), sep="\t",
                                  on_bad_lines="skip")
            print(f"  Downloaded: {len(tsv_df)} rows, columns: {list(tsv_df.columns)}")

            # Find accession and host columns
            acc_col  = next((c for c in tsv_df.columns
                              if any(x in c.lower() for x in
                                     ["accession","genome","id","vc_id"])), None)
            host_col = next((c for c in tsv_df.columns
                              if "host" in c.lower()), None)

            if acc_col and host_col:
                sub = tsv_df[[acc_col, host_col]].dropna()
                sub.columns = ["phage", "host"]
                sub["phage"] = sub["phage"].apply(normalize)
                sub["host"]  = sub["host"].apply(normalize)
                sub = sub[sub["host"].str.len() > 3]
                sub = sub[~sub["host"].str.contains(
                    r"^\d|unknown|unclassified|environmental", regex=True, na=False)]
                sub["label"]  = 1
                sub["source"] = "millard_2023"
                sub = sub.drop_duplicates(subset=["phage", "host"])
                millard_df = sub
                millard_df.to_csv(MILLARD_OUT, index=False)
                print(f"  Saved: {len(millard_df)} valid pairs -> {MILLARD_OUT.name}")
            else:
                print(f"  Could not find accession/host columns in: {list(tsv_df.columns)}")
        except Exception as e:
            print(f"  Parse error: {e}")
    else:
        print("  Download failed — skipping Millard data.")
        print("  You can manually download from:")
        print("    https://millardlab.org/bacteriophage-genomics/phage-genomes-oct-2023/")
        print("  Save as: data/raw/millard_pairs.csv  with columns: phage, host, label, source")

print(f"  Millard pairs loaded: {len(millard_df)}")


# ═══════════════════════════════════════════════════════════════
# STEP 2 — NCBI ENTREZ API
# Fetch phage-host pairs from GenBank /host field
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: NCBI Entrez API — fetching phage-host pairs...")
print("=" * 60)

NCBI_OUT = RAW_DIR / "ncbi_pairs.csv"
EUTILS   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ncbi_df  = pd.DataFrame()

# Key bacteriophage target genera for broad coverage
PHAGE_QUERIES = [
    "staphylococcus phage[organism]",
    "escherichia phage[organism]",
    "pseudomonas phage[organism]",
    "bacillus phage[organism]",
    "salmonella phage[organism]",
    "klebsiella phage[organism]",
    "vibrio phage[organism]",
    "streptococcus phage[organism]",
    "listeria phage[organism]",
    "mycobacterium phage[organism]",
    "enterococcus phage[organism]",
    "lactococcus phage[organism]",
    "shigella phage[organism]",
    "acinetobacter phage[organism]",
    "clostridium phage[organism]",
]

if NCBI_OUT.exists():
    ncbi_df = pd.read_csv(NCBI_OUT)
    print(f"  Loaded cached: {len(ncbi_df)} pairs")
else:
    ncbi_pairs = []

    for query in tqdm(PHAGE_QUERIES, desc="NCBI queries"):
        # Step 1: esearch — get IDs
        params = {
            "db":     "nuccore",
            "term":   f"{query} AND complete genome[title]",
            "retmax": "200",
            "retmode":"json",
            "email":  NCBI_EMAIL,
        }
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY

        r = safe_get(f"{EUTILS}/esearch.fcgi", params=params, wait=0.4)
        if r is None:
            continue
        try:
            ids = r.json()["esearchresult"]["idlist"]
        except Exception:
            continue

        if not ids:
            continue

        # Step 2: efetch — get GenBank summary for host field
        fetch_params = {
            "db":      "nuccore",
            "id":      ",".join(ids[:100]),
            "rettype": "gb",
            "retmode": "text",
            "email":   NCBI_EMAIL,
        }
        if NCBI_API_KEY:
            fetch_params["api_key"] = NCBI_API_KEY

        time.sleep(0.4)  # be polite
        rf = safe_get(f"{EUTILS}/efetch.fcgi", params=fetch_params, wait=0.5)
        if rf is None:
            continue

        # Parse GenBank flat file for ORGANISM and /host
        gb_text    = rf.text
        organism   = None
        host_found = None

        for line in gb_text.splitlines():
            line = line.strip()
            if line.startswith("ORGANISM"):
                organism = normalize(line.replace("ORGANISM", "").strip())
            elif "/host=" in line:
                host_found = normalize(
                    line.split("/host=")[1].strip().strip('"'))
            elif "/lab_host=" in line:
                host_found = normalize(
                    line.split("/lab_host=")[1].strip().strip('"'))

            if organism and host_found:
                if len(host_found.split()) >= 2:  # needs genus + species
                    ncbi_pairs.append({
                        "phage":  organism,
                        "host":   host_found,
                        "label":  1,
                        "source": "ncbi_entrez"
                    })
                organism   = None
                host_found = None

        time.sleep(0.35)

    if ncbi_pairs:
        ncbi_df = pd.DataFrame(ncbi_pairs).drop_duplicates(
            subset=["phage", "host"])
        ncbi_df = ncbi_df[ncbi_df["host"].str.len() > 5]
        ncbi_df.to_csv(NCBI_OUT, index=False)
        print(f"  Saved: {len(ncbi_df)} pairs -> {NCBI_OUT.name}")
    else:
        print("  No NCBI pairs retrieved (network may be blocked).")

print(f"  NCBI pairs loaded: {len(ncbi_df)}")


# ═══════════════════════════════════════════════════════════════
# STEP 3 — BACDIVE API  (host phenotype features)
# Fetches: gram stain, oxygen tolerance, cell morphology, motility, phylum
# Register free at https://bacdive.dsmz.de/api
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: BacDive API — fetching host phenotype features...")
print("=" * 60)

FEATURES_OUT = RAW_DIR / "host_features.csv"

# Built-in curated features for the 53 genera in our dataset
# (covers all genera present in VHI, so BacDive is supplementary)
CURATED_HOST_FEATURES = {
    # genus: [gram_pos, phylum, aerobic, anaerobic, facultative, coccus, bacillus, spiral, motile]
    "staphylococcus":  [1, "firmicutes",       0, 0, 1, 1, 0, 0, 0],
    "bacillus":        [1, "firmicutes",       1, 0, 1, 0, 1, 0, 0],
    "listeria":        [1, "firmicutes",       0, 0, 1, 0, 1, 0, 1],
    "enterococcus":    [1, "firmicutes",       0, 0, 1, 1, 0, 0, 0],
    "streptococcus":   [1, "firmicutes",       0, 0, 1, 1, 0, 0, 0],
    "lactococcus":     [1, "firmicutes",       0, 0, 1, 1, 0, 0, 0],
    "lactobacillus":   [1, "firmicutes",       0, 0, 1, 0, 1, 0, 0],
    "clostridium":     [1, "firmicutes",       0, 1, 0, 0, 1, 0, 0],
    "paenibacillus":   [1, "firmicutes",       0, 0, 1, 0, 1, 0, 1],
    "brevibacillus":   [1, "firmicutes",       1, 0, 0, 0, 1, 0, 1],
    "leuconostoc":     [1, "firmicutes",       0, 0, 1, 1, 0, 0, 0],
    "mycobacterium":   [1, "actinobacteria",   1, 0, 0, 0, 1, 0, 0],
    "streptomyces":    [1, "actinobacteria",   1, 0, 0, 0, 1, 0, 0],
    "corynebacterium": [1, "actinobacteria",   0, 0, 1, 0, 1, 0, 0],
    "gordonia":        [1, "actinobacteria",   1, 0, 0, 0, 1, 0, 0],
    "rhodococcus":     [1, "actinobacteria",   1, 0, 0, 0, 1, 0, 0],
    "nocardia":        [1, "actinobacteria",   1, 0, 0, 0, 1, 0, 0],
    "escherichia":     [0, "proteobacteria",   0, 0, 1, 0, 1, 0, 1],
    "salmonella":      [0, "proteobacteria",   0, 0, 1, 0, 1, 0, 1],
    "klebsiella":      [0, "proteobacteria",   0, 0, 1, 0, 1, 0, 0],
    "pseudomonas":     [0, "proteobacteria",   1, 0, 0, 0, 1, 0, 1],
    "vibrio":          [0, "proteobacteria",   0, 0, 1, 0, 1, 0, 1],
    "acinetobacter":   [0, "proteobacteria",   1, 0, 0, 1, 1, 0, 0],
    "shigella":        [0, "proteobacteria",   0, 0, 1, 0, 1, 0, 0],
    "enterobacter":    [0, "proteobacteria",   0, 0, 1, 0, 1, 0, 1],
    "serratia":        [0, "proteobacteria",   0, 0, 1, 0, 1, 0, 1],
    "citrobacter":     [0, "proteobacteria",   0, 0, 1, 0, 1, 0, 1],
    "proteus":         [0, "proteobacteria",   0, 0, 1, 0, 1, 0, 1],
    "yersinia":        [0, "proteobacteria",   0, 0, 1, 0, 1, 0, 0],
    "cronobacter":     [0, "proteobacteria",   0, 0, 1, 0, 1, 0, 1],
    "pantoea":         [0, "proteobacteria",   0, 0, 1, 0, 1, 0, 1],
    "erwinia":         [0, "proteobacteria",   0, 0, 1, 0, 1, 0, 1],
    "dickeya":         [0, "proteobacteria",   0, 0, 1, 0, 1, 0, 1],
    "aeromonas":       [0, "proteobacteria",   0, 0, 1, 0, 1, 0, 1],
    "burkholderia":    [0, "proteobacteria",   1, 0, 0, 0, 1, 0, 1],
    "ralstonia":       [0, "proteobacteria",   1, 0, 0, 0, 1, 0, 1],
    "achromobacter":   [0, "proteobacteria",   1, 0, 0, 0, 1, 0, 1],
    "delftia":         [0, "proteobacteria",   1, 0, 0, 0, 1, 0, 1],
    "pseudoalteromonas":[0,"proteobacteria",   1, 0, 0, 0, 1, 0, 1],
    "shewanella":      [0, "proteobacteria",   0, 0, 1, 0, 1, 0, 1],
    "alteromonas":     [0, "proteobacteria",   1, 0, 0, 0, 1, 0, 1],
    "enterovibrio":    [0, "proteobacteria",   0, 0, 1, 0, 1, 0, 1],
    "campylobacter":   [0, "proteobacteria",   0, 0, 1, 0, 0, 1, 1],
    "brucella":        [0, "proteobacteria",   1, 0, 0, 1, 0, 0, 0],
    "rhodobacter":     [0, "proteobacteria",   0, 0, 1, 1, 0, 0, 1],
    "roseobacter":     [0, "proteobacteria",   1, 0, 0, 1, 0, 0, 0],
    "rhizobium":       [0, "proteobacteria",   1, 0, 0, 0, 1, 0, 1],
    "sinorhizobium":   [0, "proteobacteria",   1, 0, 0, 0, 1, 0, 1],
    "paracoccus":      [0, "proteobacteria",   1, 0, 0, 1, 0, 0, 0],
    "gluconobacter":   [0, "proteobacteria",   1, 0, 0, 1, 0, 0, 0],
    "bacteroides":     [0, "bacteroidetes",    0, 1, 0, 0, 1, 0, 0],
    "anabaena":        [0, "cyanobacteria",    1, 0, 0, 0, 1, 0, 1],
    "chlamydia":       [0, "chlamydiae",       0, 0, 1, 1, 0, 0, 0],
}

feat_cols = ["gram_pos", "phylum", "aerobic", "anaerobic",
             "facultative", "coccus", "bacillus", "spiral", "motile"]

host_feat_df = pd.DataFrame.from_dict(
    CURATED_HOST_FEATURES, orient="index",
    columns=feat_cols
).reset_index().rename(columns={"index": "genus"})

# One-hot encode phylum
phyla = ["firmicutes", "proteobacteria", "actinobacteria",
         "bacteroidetes", "cyanobacteria", "chlamydiae"]
for ph in phyla:
    host_feat_df[f"phylum_{ph}"] = (host_feat_df["phylum"] == ph).astype(int)
host_feat_df = host_feat_df.drop(columns=["phylum"])

# Try BacDive for genera not in curated set (or to supplement)
if BACDIVE_USER and BACDIVE_PASS:
    print(f"  BacDive credentials found — fetching additional genera...")
    BACDIVE_API = "https://api.bacdive.dsmz.de"
    known = set(CURATED_HOST_FEATURES.keys())

    # Collect all unique genera from our existing VHI data
    vhi_path = RAW_DIR / "VirusHostInter.csv"
    if vhi_path.exists():
        vhi_df = pd.read_csv(vhi_path)
        host_col = next((c for c in vhi_df.columns if "host" in c.lower()), None)
        if host_col:
            all_genera = set(
                vhi_df[host_col].dropna().apply(
                    lambda x: str(x).strip().lower().replace("_", " ")
                    .split()[0] if str(x).strip() else ""
                ).unique()
            ) - known - {""}

            for genus in tqdm(list(all_genera)[:50],
                               desc="BacDive genera"):
                r = safe_get(
                    f"{BACDIVE_API}/taxon/{genus}/",
                    auth=(BACDIVE_USER, BACDIVE_PASS),
                    wait=0.5)
                if r is None:
                    continue
                try:
                    data = r.json()
                    results = data.get("results", [])
                    if not results:
                        continue
                    # Aggregate across strains
                    grams, o2s, morphs = [], [], []
                    for item_url in [x["url"] for x in results[:5]]:
                        rd = safe_get(item_url,
                                       auth=(BACDIVE_USER, BACDIVE_PASS),
                                       wait=0.3)
                        if rd is None:
                            continue
                        d = rd.json()
                        morph = d.get("morphology_physiology", {})
                        cell  = morph.get("cell_morphology", {})
                        if "gram_stain" in cell:
                            g = str(cell["gram_stain"]).lower()
                            if "positive" in g:
                                grams.append(1)
                            elif "negative" in g:
                                grams.append(0)
                        oxy = morph.get("oxygen_tolerance", {})
                        if "oxygen_tolerance" in oxy:
                            o = str(oxy["oxygen_tolerance"]).lower()
                            if "strict aerobe" in o or "aerobic" in o:
                                o2s.append("aerobic")
                            elif "anaerobe" in o or "anaerobic" in o:
                                o2s.append("anaerobic")
                            else:
                                o2s.append("facultative")
                        time.sleep(0.3)

                    if grams:
                        gram_val = round(np.mean(grams))
                        o2_most  = max(set(o2s), key=o2s.count) if o2s else "facultative"
                        row = {
                            "genus":    genus,
                            "gram_pos": gram_val,
                            "aerobic":  int(o2_most == "aerobic"),
                            "anaerobic":int(o2_most == "anaerobic"),
                            "facultative": int(o2_most == "facultative"),
                            "coccus": 0, "bacillus": 1, "spiral": 0, "motile": 0,
                        }
                        for ph in phyla:
                            row[f"phylum_{ph}"] = 0
                        host_feat_df = pd.concat(
                            [host_feat_df, pd.DataFrame([row])],
                            ignore_index=True)
                except Exception as e:
                    continue
else:
    print("  No BacDive credentials — using curated features only.")
    print(f"  Covers {len(CURATED_HOST_FEATURES)} genera "
          f"(all genera present in your VHI dataset).")
    print()
    print("  To enable BacDive enrichment, register free at:")
    print("    https://bacdive.dsmz.de/api")
    print("  Then set environment variables before running:")
    print("    set BACDIVE_USER=your@email.com")
    print("    set BACDIVE_PASS=yourpassword")

host_feat_df.to_csv(FEATURES_OUT, index=False)
print(f"\n  Host features saved: {len(host_feat_df)} genera -> {FEATURES_OUT.name}")
print(f"  Feature columns: {[c for c in host_feat_df.columns if c != 'genus']}")


# ═══════════════════════════════════════════════════════════════
# STEP 4 — MERGE INTO ENRICHED DATASET
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Merging all sources into enriched dataset...")
print("=" * 60)

# Load original VHI dataset
vhi_df = pd.read_csv(RAW_DIR / "VirusHostInter.csv")
vhi_df = vhi_df.loc[:, ~vhi_df.columns.str.contains("^Unnamed")]
vhi_df = vhi_df.rename(columns={
    "hostname": "host", "phagename": "phage", "infection": "interaction"})
for c in ["host", "phage", "interaction"]:
    vhi_df[c] = vhi_df[c].apply(normalize)
vhi_df["label"]  = (vhi_df["interaction"] == "inf").astype(int)
vhi_df["source"] = "vhi"
NUMERIC = ["k3dist", "k6dist", "GCdiff", "Homology"]
for c in NUMERIC:
    if c not in vhi_df.columns:
        vhi_df[c] = 0.0
    vhi_df[c] = pd.to_numeric(vhi_df[c], errors="coerce").fillna(0.0)

all_dfs = [vhi_df[["phage", "host", "label", "source"] + NUMERIC]]

# Add Millard pairs (positives only — Millard doesn't have confirmed negatives)
if len(millard_df):
    m = millard_df[["phage", "host", "label", "source"]].copy()
    for c in NUMERIC:
        m[c] = 0.0
    all_dfs.append(m)

# Add NCBI pairs
if len(ncbi_df):
    n = ncbi_df[["phage", "host", "label", "source"]].copy()
    for c in NUMERIC:
        n[c] = 0.0
    all_dfs.append(n)

merged = pd.concat(all_dfs, ignore_index=True) \
           .drop_duplicates(subset=["phage", "host", "label"]) \
           .reset_index(drop=True)

merged["genus"] = merged["host"].str.split().str[0]

# Attach host features
feat_cols_use = [c for c in host_feat_df.columns if c != "genus"]
merged = merged.merge(host_feat_df, on="genus", how="left")
# Fill missing genus features with 0
for c in feat_cols_use:
    if c in merged.columns:
        merged[c] = merged[c].fillna(0)

ENRICHED_OUT = RAW_DIR / "enriched_dataset.csv"
merged.to_csv(ENRICHED_OUT, index=False)

print(f"\n  Sources merged:")
print(f"    VHI:     {len(vhi_df)} pairs")
print(f"    Millard: {len(millard_df)} pairs")
print(f"    NCBI:    {len(ncbi_df)} pairs")
print(f"    Total:   {len(merged)} pairs "
      f"({merged['label'].sum()} pos, {(merged['label']==0).sum()} neg)")
print(f"    Genera:  {merged['genus'].nunique()}")
print(f"\n  Saved -> {ENRICHED_OUT}")
print()
print("  Done! Now re-run 03_gnn.py to use the enriched dataset.")