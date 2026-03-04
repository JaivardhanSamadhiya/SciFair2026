"""
Run from D:\\SciFair2026:
    python diagnose2.py
"""
import pandas as pd
from pathlib import Path

_SCRIPT_DIR   = Path("d:/SciFair2026/SciFair2026/scripts")
BASE_DIR      = _SCRIPT_DIR.parent / "data"
RAW_DIR       = BASE_DIR / "raw"

def normalize(x):
    return str(x).strip().lower().replace("_", " ")

print("=" * 60)
print("1. VirusHostInter.csv")
print("=" * 60)
vhi = pd.read_csv(RAW_DIR / "VirusHostInter.csv")
print(f"   Shape: {vhi.shape}")
print(f"   Columns: {list(vhi.columns)}")
print(f"   First row: {vhi.iloc[0].to_dict()}")
print()

# Find the host column — whatever it's actually called
host_col  = next((c for c in vhi.columns if "host" in c.lower()), None)
phage_col = next((c for c in vhi.columns if "phage" in c.lower() or "virus" in c.lower()), None)
inf_col   = next((c for c in vhi.columns if "infect" in c.lower() or "interact" in c.lower()), None)
print(f"   Detected host col:      {host_col!r}")
print(f"   Detected phage col:     {phage_col!r}")
print(f"   Detected infection col: {inf_col!r}")

if host_col:
    hosts = vhi[host_col].dropna().apply(normalize).unique()
    staph = [h for h in hosts if "staphylococcus" in h]
    print(f"\n   Unique host values containing 'staphylococcus' ({len(staph)}):")
    for h in staph[:10]:
        print(f"     '{h}'")
    print(f"\n   First 5 host values (raw): {list(vhi[host_col].head(5))}")

if inf_col:
    print(f"\n   Unique infection values: {vhi[inf_col].unique()[:10]}")

print()
print("=" * 60)
print("2. phage-bacteria-pairs.txt")
print("=" * 60)
pbp = pd.read_csv(RAW_DIR / "phage-bacteria-pairs.txt", sep="\t")
print(f"   Shape: {pbp.shape}")
print(f"   Columns: {list(pbp.columns)}")
print(f"   First row: {pbp.iloc[0].to_dict()}")
host_col2 = next((c for c in pbp.columns if "host" in c.lower()), None)
if host_col2:
    hosts2 = pbp[host_col2].dropna().apply(normalize).unique()
    staph2 = [h for h in hosts2 if "staphylococcus" in h]
    print(f"   Staph hosts: {staph2[:5]}")

print()
print("=" * 60)
print("3. phagesdb_pairs.csv")
print("=" * 60)
pdb_path = RAW_DIR / "phagesdb_pairs.csv"
if pdb_path.exists():
    pdb = pd.read_csv(pdb_path)
    print(f"   Shape: {pdb.shape}")
    print(f"   Columns: {list(pdb.columns)}")
    print(f"   First row: {pdb.iloc[0].to_dict()}")
    if "host" in pdb.columns:
        staph3 = pdb[pdb["host"].str.contains("staphylococcus", na=False)]
        print(f"   Staph rows: {len(staph3)}")
        print(f"   Sample staph phage names: {list(staph3['phage'].head(5))}")
        # Check sequence coverage
        import json
        with open(BASE_DIR / "genomes" / "phage_sequences.json") as f:
            seqs = set(json.load(f).keys())
        match = staph3["phage"].isin(seqs).sum()
        print(f"   Of those, with sequence: {match}/{len(staph3)}")