"""
UPDATED DATA COLLECTOR
- Downloads bacteriophage genomes from PHIS / RefSeq metadata
- Correctly parses genome_id column
- Filters non-phage viruses
- Restart-safe
"""

import os
from Bio import Entrez
from tqdm import tqdm

# -------------------------
# CONFIG
# -------------------------
Entrez.email = "jaisamadhiya@gmail.com"

BASE_DIR = "staph_phage_data"
PHAGE_DIR = os.path.join(BASE_DIR, "phages")
INPUT_FILE = "SciFair2026/data/raw/phage_ncbi_refseq_def_info.txt"  # <-- your PHIS file

os.makedirs(PHAGE_DIR, exist_ok=True)

# -------------------------
# STEP 1: PARSE ACCESSIONS
# -------------------------
def parse_accessions():
    accessions = set()
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue

            genome_id = parts[1].strip()      # NC_XXXX
            description = parts[2].lower()    # description text

            # filter to bacteriophages only
            if "phage" not in description:
                continue

            if genome_id.startswith(("NC_", "NZ_", "CP_", "NW_")):
                accessions.add(genome_id)

    return sorted(accessions)

phage_accessions = parse_accessions()
print(f"Found {len(phage_accessions)} RefSeq bacteriophage genomes")

# -------------------------
# STEP 2: DOWNLOAD PHAGE GENOMES
# -------------------------
print("Downloading phage genomes...")

for acc in tqdm(phage_accessions, desc="Phages"):
    out_file = os.path.join(PHAGE_DIR, f"{acc}.fasta")
    if os.path.exists(out_file):
        continue

    try:
        with Entrez.efetch(
            db="nuccore",
            id=acc,
            rettype="fasta",
            retmode="text"
        ) as handle:
            seq = handle.read()

        with open(out_file, "w") as f:
            f.write(seq)

    except Exception as e:
        print(f"[WARN] {acc} failed: {e}")

print("\nPhage genome download complete ✅")
print(f"Saved to: {PHAGE_DIR}")
