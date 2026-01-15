"""
NCBI Assembly FTP downloader for Staphylococcus aureus (RefSeq)
Correct genome-level data collection.

Requires:
  - biopython
  - pandas
  - requests

Usage:
  python assembly_downloader.py
"""

import os
import time
import requests
from Bio import Entrez

# -------------------------
# CONFIG
# -------------------------
Entrez.email = "jaisamadhiya@gmail.com"
OUTPUT_DIR = "staph_phage_data"
HOST_DIR = os.path.join(OUTPUT_DIR, "host_genomes")

os.makedirs(HOST_DIR, exist_ok=True)

# -------------------------
# STEP 1: SEARCH ASSEMBLY DB
# -------------------------
print("Searching NCBI Assembly database (RefSeq, S. aureus)...")

search = Entrez.esearch(
    db="assembly",
    term="Staphylococcus aureus[Organism] AND refseq[filter]",
    retmax=100000
)
search_results = Entrez.read(search)
assembly_ids = search_results["IdList"]

print(f"Found {len(assembly_ids)} RefSeq assemblies.")

# -------------------------
# STEP 2: DOWNLOAD GENOMES
# -------------------------
success = 0
skipped = 0
failed = 0

for asm_id in assembly_ids:
    try:
        summary = Entrez.esummary(db="assembly", id=asm_id, report="full")
        doc = Entrez.read(summary)["DocumentSummarySet"]["DocumentSummary"][0]

        ftp = doc["FtpPath_RefSeq"]
        if not ftp:
            continue

        asm_name = ftp.split("/")[-1]
        filename = f"{asm_name}_genomic.fna.gz"
        out_path = os.path.join(HOST_DIR, filename)

        if os.path.exists(out_path):
            skipped += 1
            continue

        url = f"{ftp}/{filename}"
        print(f"Downloading {filename}")

        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()

        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        success += 1
        time.sleep(0.2)  # NCBI-safe rate

    except Exception as e:
        failed += 1
        print(f"FAILED {asm_id}: {e}")

print("\nDOWNLOAD COMPLETE")
print(f"Downloaded: {success}")
print(f"Skipped:    {skipped}")
print(f"Failed:     {failed}")
print(f"Location:   {HOST_DIR}")
