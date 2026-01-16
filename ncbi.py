"""
FULL DATA COLLECTOR: Staph aureus + corresponding phage genomes + VHRdb interactions
Windows compatible. Requires: biopython, pandas, requests, tqdm
Usage: python full_data_collector.py
"""

import os
import requests
import pandas as pd
from tqdm import tqdm
from Bio import Entrez

# -------------------------
# USER CONFIG
# -------------------------
Entrez.email = "jaisamadhiya@gmail.com"  # replace with your email
OUTPUT_DIR = "staph_phage_data"
HOST_DIR = os.path.join(OUTPUT_DIR, "host_genomes")
PHAGE_DIR = os.path.join(OUTPUT_DIR, "phage_genomes")
os.makedirs(HOST_DIR, exist_ok=True)
os.makedirs(PHAGE_DIR, exist_ok=True)
INTERACTIONS_FILE = os.path.join(OUTPUT_DIR, "staph_phage_interactions.csv")

# -------------------------
# STEP 1: Fetch S. aureus interactions from VHRdb
# -------------------------
print("Step 1: Fetching S. aureus host-phage interactions from VHRdb...")

HOST_LIST_URL = "https://viralhostrangedb.pasteur.cloud/api/host/?format=json"
HOST_DETAIL_URL = "https://viralhostrangedb.pasteur.cloud/api/host/{host_id}/?format=json"

resp = requests.get(HOST_LIST_URL)
resp.raise_for_status()
hosts = resp.json()

# filter Staphylococcus aureus hosts
s_aureus_hosts = [h for h in hosts if "Staphylococcus aureus" in h["name"]]

all_interactions = []

for host in tqdm(s_aureus_hosts, desc="Fetching host interactions"):
    host_id = host["id"]
    host_name = host["name"]
    resp = requests.get(HOST_DETAIL_URL.format(host_id=host_id))
    resp.raise_for_status()
    data = resp.json()
    interactions = data.get("interactions", [])
    for inter in interactions:
        virus_id = inter.get("virus_id")
        response = inter.get("response")  # 0/1/2, etc.
        if virus_id:
            all_interactions.append({
                "host_id": host_id,
                "host_name": host_name,
                "phage_id": virus_id,
                "interaction": response
            })

# save interactions to CSV
df_interactions = pd.DataFrame(all_interactions)
df_interactions.to_csv(INTERACTIONS_FILE, index=False)
print(f"Saved {len(df_interactions)} interactions to {INTERACTIONS_FILE}\n")

# -------------------------
# STEP 2: Download S. aureus genomes from NCBI
# -------------------------
print("Step 2: Searching for Staphylococcus aureus genomes in RefSeq...")

handle = Entrez.esearch(
    db="nuccore",
    term="Staphylococcus aureus[Organism] AND srcdb_refseq[PROP]",
    retmax=1000000
)
record = Entrez.read(handle)
host_ids = record["IdList"]
print(f"Found {len(host_ids)} host sequences.\n")

for seq_id in tqdm(host_ids, desc="Downloading host genomes"):
    out_file = os.path.join(HOST_DIR, f"{seq_id}.fasta")
    if os.path.exists(out_file):
        continue
    try:
        with Entrez.efetch(db="nuccore", id=seq_id, rettype="fasta", retmode="text") as handle_f:
            seq_data = handle_f.read()
            with open(out_file, "w") as f:
                f.write(seq_data)
    except Exception as e:
        print(f"Failed to download host {seq_id}: {e}")

print("Finished downloading host genomes.\n")

# -------------------------
# STEP 3: Download phage genomes from NCBI
# -------------------------
print("Step 3: Downloading corresponding phage genomes...")

# unique phages from VHRdb
virus_ids = df_interactions["phage_id"].dropna().unique()
print(f"{len(virus_ids)} unique phages to download.\n")

for virus_id in tqdm(virus_ids, desc="Downloading phage genomes"):
    out_file = os.path.join(PHAGE_DIR, f"{virus_id}.fasta")
    if os.path.exists(out_file):
        continue
    try:
        with Entrez.efetch(db="nuccore", id=str(virus_id), rettype="fasta", retmode="text") as handle_f:
            seq_data = handle_f.read()
            with open(out_file, "w") as f:
                f.write(seq_data)
    except Exception as e:
        print(f"Failed to download phage {virus_id}: {e}")

print("\nDATA COLLECTION COMPLETE ✅")
print(f"Host genomes folder: {HOST_DIR}")
print(f"Phage genomes folder: {PHAGE_DIR}")
print(f"Interaction labels: {INTERACTIONS_FILE}")
