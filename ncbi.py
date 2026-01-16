"""
NCBI Genome Downloader
Downloads:
- Phage genomes from virus_id
- Staphylococcus aureus host genomes from host_name

Input:
staph_vhrdb_interactions.csv

Output:
staph_phage_data/
  ├── phages/
  └── hosts/
"""

import os
import pandas as pd
from tqdm import tqdm
from Bio import Entrez

# -------------------------
# CONFIG
# -------------------------
Entrez.email = "jaisamadhiya@gmail.com"   # REQUIRED by NCBI
INPUT_CSV = "staph_vhrdb_interactions.csv"

BASE_DIR = "staph_phage_data"
PHAGE_DIR = os.path.join(BASE_DIR, "phages")
HOST_DIR = os.path.join(BASE_DIR, "hosts")

os.makedirs(PHAGE_DIR, exist_ok=True)
os.makedirs(HOST_DIR, exist_ok=True)

# -------------------------
# LOAD INTERACTIONS
# -------------------------
df = pd.read_csv(INPUT_CSV)

phage_ids = sorted(df["virus_id"].astype(str).unique())
host_names = sorted(df["host_name"].unique())

# -------------------------
# DOWNLOAD PHAGE GENOMES
# -------------------------
print(f"\nDownloading {len(phage_ids)} phage genomes from NCBI...")

for vid in tqdm(phage_ids, desc="Phages"):
    out_file = os.path.join(PHAGE_DIR, f"{vid}.fasta")
    if os.path.exists(out_file):
        continue

    try:
        handle = Entrez.efetch(
            db="nuccore",
            id=vid,
            rettype="fasta",
            retmode="text"
        )
        seq = handle.read()
        handle.close()

        if len(seq.strip()) < 100:
            continue

        with open(out_file, "w") as f:
            f.write(seq)

    except Exception as e:
        print(f"[WARN] Phage {vid} failed: {e}")

# -------------------------
# DOWNLOAD HOST GENOMES
# -------------------------
print(f"\nDownloading host genomes ({len(host_names)} strains)...")

for host in tqdm(host_names, desc="Hosts"):
    safe_name = host.replace(" ", "_").replace("/", "_")
    out_file = os.path.join(HOST_DIR, f"{safe_name}.fasta")

    if os.path.exists(out_file):
        continue

    try:
        search = Entrez.esearch(
            db="nuccore",
            term=f"{host}[Organism] AND srcdb_refseq[PROP]",
            retmax=1
        )
        record = Entrez.read(search)
        search.close()

        if not record["IdList"]:
            continue

        genome_id = record["IdList"][0]

        fetch = Entrez.efetch(
            db="nuccore",
            id=genome_id,
            rettype="fasta",
            retmode="text"
        )
        seq = fetch.read()
        fetch.close()

        with open(out_file, "w") as f:
            f.write(seq)

    except Exception as e:
        print(f"[WARN] Host {host} failed: {e}")

# -------------------------
# DONE
# -------------------------
print("\nGENOME DOWNLOAD COMPLETE ✅")
print(f"Phage genomes → {PHAGE_DIR}")
print(f"Host genomes  → {HOST_DIR}")
