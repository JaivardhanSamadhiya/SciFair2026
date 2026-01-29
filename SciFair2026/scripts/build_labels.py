"""
Build labels for phage → Staphylococcus aureus infection
Using PHIS literature-curated phage-host pairs
Handles NCBI accession versioning correctly
"""

import os
import csv

# -------------------------
# PATHS
# -------------------------
PHAGE_DIR = "staph_phage_data/phages"
PHIS_PAIRS = "SciFair2026/data/raw/phage-bacteria-pairs.txt"
OUTPUT_CSV = "staph_phage_data/labels.csv"

TARGET_HOST = "staphylococcus aureus"

# -------------------------
# STEP 1: LOAD POSITIVE PHAGES (normalized)
# -------------------------
positive_phages = set()

with open(PHIS_PAIRS, "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue

        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue

        phage_acc = parts[0].strip().split(".")[0]  # <-- strip version
        host_name = parts[1].lower()

        if TARGET_HOST in host_name:
            positive_phages.add(phage_acc)

print(f"PHIS positives found (normalized): {len(positive_phages)}")

# -------------------------
# STEP 2: BUILD LABEL TABLE
# -------------------------
rows = []
pos = 0
neg = 0

for fname in os.listdir(PHAGE_DIR):
    if not fname.endswith(".fasta"):
        continue

    acc_full = fname.replace(".fasta", "")
    acc_base = acc_full.split(".")[0]  # <-- strip version

    label = 1 if acc_base in positive_phages else 0

    if label == 1:
        pos += 1
    else:
        neg += 1

    rows.append([acc_full, label])

# -------------------------
# STEP 3: SAVE
# -------------------------
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["phage_accession", "infects_staph"])
    writer.writerows(rows)

print(f"Saved labels → {OUTPUT_CSV}")
print(f"Positives: {pos}")
print(f"Negatives: {neg}")
