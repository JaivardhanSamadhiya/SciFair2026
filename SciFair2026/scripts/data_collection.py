"""
01b_fetch_genomes_phagesdb_api.py
==================================
Fetches phage genome sequences using ONLY the PhagesDB REST API
(the same endpoint that already worked and returned 30,987 pairs).

Run this AFTER 01_collect_data.py has already produced:
    SciFair2026/data/raw/phagesdb_pairs.csv

This script:
  1. Re-hits the PhagesDB API page by page
  2. Extracts the 'fasta_file' URL from each phage record
  3. Downloads that FASTA (it's on the same domain — same network path that worked)
  4. Saves sequences to genomes/phage_sequences.json
  5. Also maps PhagesDB names to your VirusHostInter phage names

Requirements:
    pip install requests pandas tqdm

Usage:
    python 01b_fetch_genomes_phagesdb_api.py
"""

import json
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR   = Path("SciFair2026/data")
RAW_DIR    = BASE_DIR / "raw"
GENOME_DIR = BASE_DIR / "genomes"
GENOME_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED  = 42
MAX_SEQ_LEN  = 300_000
HTTP_TIMEOUT = 30
API_SLEEP    = 0.1   # seconds between API page requests
FASTA_SLEEP  = 0.05  # seconds between FASTA downloads

PHAGESDB_API = "https://phagesdb.org/api/phages/?page_size=500&format=json"

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def safe_get(d: dict, *keys, default="") -> str:
    """Safely traverse nested dict, returning default if any key is missing/None."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k) or default
    return str(cur).strip()


def parse_fasta(text: str) -> str | None:
    """Extract sequence string from FASTA text."""
    lines = text.strip().split("\n")
    if not lines or not lines[0].startswith(">"):
        return None
    seq = "".join(l.strip() for l in lines[1:]).upper()
    seq = "".join(c for c in seq if c in "ACGTN")
    if 100 < len(seq) <= MAX_SEQ_LEN:
        return seq
    return None


def download_fasta(url: str) -> str | None:
    """Download a FASTA file from a URL and return the sequence string."""
    if not url or not url.startswith("http"):
        return None
    try:
        resp = requests.get(url, timeout=HTTP_TIMEOUT)
        if resp.status_code == 200:
            return parse_fasta(resp.text)
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════
# STEP 1 — Walk the PhagesDB API and collect all phage records
# ═══════════════════════════════════════════════════════════════
def collect_phagesdb_records() -> list[dict]:
    """
    Re-fetches all PhagesDB API pages and returns the full list of phage dicts.
    Each dict has: name, host, accession, fasta_file_url, cluster, subcluster
    """
    cache_path = BASE_DIR / "phagesdb_records_cache.json"
    if cache_path.exists():
        print(f"  Loading cached PhagesDB records from {cache_path}...")
        with open(cache_path) as f:
            return json.load(f)

    print("  Fetching all PhagesDB phage records via API...")
    records = []
    url     = PHAGESDB_API
    page    = 0

    while url:
        try:
            resp = requests.get(url, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  API error on page {page}: {e}")
            break

        for phage in data.get("results", []):
            ih = phage.get("isolation_host") or {}
            records.append({
                "name":          safe_get(phage, "phage_name").lower(),
                "name_original": safe_get(phage, "phage_name"),   # preserve case for FASTA URL
                "host_genus":    safe_get(ih, "genus").lower(),
                "host_species":  safe_get(ih, "species").lower(),
                "accession":     safe_get(phage, "genbank_accession"),
                "fasta_url":     safe_get(phage, "fasta_file"),   # direct FASTA download URL
                "cluster":       safe_get(phage, "cluster"),
                "subcluster":    safe_get(phage, "subcluster"),
                "genome_length": phage.get("genome_length") or 0,
            })

        page += 1
        url   = data.get("next")
        time.sleep(API_SLEEP)
        if page % 10 == 0:
            print(f"    page {page} — {len(records)} records so far...")

    # Save cache so we don't have to re-fetch
    with open(cache_path, "w") as f:
        json.dump(records, f)

    print(f"  Collected {len(records)} phage records.")
    return records


# ═══════════════════════════════════════════════════════════════
# STEP 2 — Download FASTA sequences from fasta_file URLs
# ═══════════════════════════════════════════════════════════════
def download_genome_sequences(records: list[dict]) -> dict[str, str]:
    out_path = GENOME_DIR / "phage_sequences.json"

    if out_path.exists():
        with open(out_path) as f:
            cache: dict = json.load(f)
        print(f"  Loaded {len(cache)} cached sequences.")
    else:
        cache = {}

    # Only attempt records that have a fasta_url and aren't cached yet
    to_fetch = [r for r in records
                if r["fasta_url"]
                and r["name"] not in cache
                and int(r["genome_length"] or 0) <= MAX_SEQ_LEN]

    print(f"  Records with FASTA URL available: "
          f"{sum(1 for r in records if r['fasta_url'])}")
    print(f"  Sequences to download: {len(to_fetch)}")

    if not to_fetch:
        print("  Nothing to download — all already cached or no FASTA URLs.")
        return cache

    # Test the first URL to see if FASTA downloads work
    print("\n  Testing FASTA download with first available record...")
    test_rec = to_fetch[0]
    test_seq = download_fasta(test_rec["fasta_url"])
    if test_seq:
        print(f"  FASTA download works! ({test_rec['name']}: {len(test_seq)} bp)")
        cache[test_rec["name"]] = test_seq
        to_fetch = to_fetch[1:]  # skip the one we just tested
    else:
        print(f"  FASTA URL test failed for: {test_rec['fasta_url']}")
        print("  This means the fasta_file URLs are also blocked on your network.")
        print("  Skipping genome download — model will use name-based features.")
        print("  See MANUAL DOWNLOAD INSTRUCTIONS below.\n")
        print_manual_instructions(records[:5])
        return cache

    # Download the rest
    failed = 0
    for i, rec in enumerate(tqdm(to_fetch, desc="Downloading genomes")):
        seq = download_fasta(rec["fasta_url"])
        if seq:
            cache[rec["name"]] = seq
        else:
            failed += 1

        time.sleep(FASTA_SLEEP)

        # Save every 200
        if (i + 1) % 200 == 0:
            with open(out_path, "w") as f:
                json.dump(cache, f)

    with open(out_path, "w") as f:
        json.dump(cache, f)

    print(f"  Downloaded: {len(cache)} sequences | Failed: {failed}")
    return cache


# ═══════════════════════════════════════════════════════════════
# STEP 3 — Save enriched pairs CSV with cluster info
# ═══════════════════════════════════════════════════════════════
def save_enriched_pairs(records: list[dict]):
    """Save a richer version of phagesdb_pairs with cluster/subcluster."""
    out_path = RAW_DIR / "phagesdb_pairs_enriched.csv"
    if out_path.exists():
        print("  Enriched pairs already saved.")
        return

    rows = []
    for r in records:
        if r["host_genus"]:
            rows.append({
                "phage":      r["name"],
                "host":       f"{r['host_genus']} {r['host_species']}".strip(),
                "accession":  r["accession"],
                "cluster":    r["cluster"],
                "subcluster": r["subcluster"],
                "label":      1,
                "source":     "phagesdb",
            })

    df = pd.DataFrame(rows).drop_duplicates(subset=["phage", "host"])
    df.to_csv(out_path, index=False)
    print(f"  Saved enriched pairs: {len(df)} rows → {out_path}")


# ═══════════════════════════════════════════════════════════════
# MANUAL DOWNLOAD INSTRUCTIONS (if all network paths are blocked)
# ═══════════════════════════════════════════════════════════════
def print_manual_instructions(sample_records: list[dict]):
    print("=" * 60)
    print("  MANUAL GENOME DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("""
Your network blocks both NCBI and PhagesDB FASTA downloads.
Here are three options:

OPTION A — Download on a different network (recommended)
  1. Copy this script to a laptop on a home/mobile hotspot
  2. Run it there — it will populate phage_sequences.json
  3. Copy phage_sequences.json back to:
       SciFair2026/data/genomes/phage_sequences.json

OPTION B — PhagesDB bulk download (manual)
  1. Go to: https://phagesdb.org/databases/
  2. Download "All Sequenced Phages FASTA" (one big file)
  3. Place it at: SciFair2026/data/genomes/all_phages.fasta
  4. Run this script again — it will auto-parse the bulk file

OPTION C — NCBI Virus portal (manual)
  1. Go to: https://www.ncbi.nlm.nih.gov/labs/virus/vssi/
  2. Search: Caudovirales (bacteriophages)
  3. Download FASTA sequences
  4. Place at: SciFair2026/data/genomes/ncbi_phages.fasta
  5. Run this script again — it will auto-parse it

OPTION D — Use name-based features only (still competitive)
  Your current dataset has 30,987 PhagesDB pairs.
  The model will use phage NAME character n-grams as features.
  This already gave PR-AUC ~0.90 in your original script.
  You can still do all ablation, multi-host, and external
  validation experiments — you just won't have genome k-mers.
  For ISEF, document this as a limitation and future work.
""")
    print("  Sample FASTA URLs (test these manually in your browser):")
    for r in sample_records:
        if r["fasta_url"]:
            print(f"    {r['name']}: {r['fasta_url']}")


# ═══════════════════════════════════════════════════════════════
# STEP 4 — Parse bulk FASTA if manually downloaded
# ═══════════════════════════════════════════════════════════════
def parse_bulk_fasta_if_present() -> dict[str, str]:
    """
    If the user manually downloaded a bulk FASTA file, parse it here.
    Looks for:  SciFair2026/data/genomes/all_phages.fasta
                SciFair2026/data/genomes/ncbi_phages.fasta
    """
    bulk_paths = [
        GENOME_DIR / "all_phages.fasta",
        GENOME_DIR / "ncbi_phages.fasta",
    ]
    cache_path = GENOME_DIR / "phage_sequences.json"

    existing = {}
    if cache_path.exists():
        with open(cache_path) as f:
            existing = json.load(f)

    new_seqs = {}
    for bulk_path in bulk_paths:
        if not bulk_path.exists():
            continue

        print(f"\n  Parsing bulk FASTA: {bulk_path} ...")
        current_name = None
        current_seq  = []

        with open(bulk_path) as f:
            for line in tqdm(f, desc=f"Parsing {bulk_path.name}"):
                line = line.strip()
                if line.startswith(">"):
                    # Save previous
                    if current_name and current_seq:
                        seq = "".join(current_seq).upper()
                        if 100 < len(seq) <= MAX_SEQ_LEN:
                            # Normalize name: take first word after ">", lowercase
                            new_seqs[current_name] = seq
                    # Start new
                    header = line[1:].split()[0].lower()
                    # Strip common prefixes like "gb|" or accession numbers
                    header = header.replace("|", "_").strip("_")
                    current_name = header
                    current_seq  = []
                else:
                    current_seq.append(line)

        # Save last record
        if current_name and current_seq:
            seq = "".join(current_seq).upper()
            if 100 < len(seq) <= MAX_SEQ_LEN:
                new_seqs[current_name] = seq

        print(f"  Parsed {len(new_seqs)} sequences from {bulk_path.name}")

    if new_seqs:
        merged = {**existing, **new_seqs}
        with open(cache_path, "w") as f:
            json.dump(merged, f)
        print(f"  Total sequences after merge: {len(merged)}")
        return merged

    return existing


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  PHAGESDB GENOME COLLECTION")
    print("=" * 60)

    # Check for manually-placed bulk FASTA files first
    print("\n[0] Checking for manually downloaded bulk FASTA files...")
    existing = parse_bulk_fasta_if_present()
    if existing:
        print(f"  Found {len(existing)} sequences already in cache.")

    # Collect all PhagesDB records (re-uses cached JSON if available)
    print("\n[1] Collecting PhagesDB phage records via API...")
    records = collect_phagesdb_records()

    # Save enriched pairs with cluster info
    print("\n[2] Saving enriched PhagesDB pairs CSV...")
    save_enriched_pairs(records)

    # Download FASTA sequences from fasta_file URLs
    print("\n[3] Downloading genome sequences from PhagesDB FASTA URLs...")
    phage_seqs = download_genome_sequences(records)

    # Final summary
    print("\n" + "=" * 60)
    n_seq = sum(1 for v in phage_seqs.values() if len(v) > 100)
    print(f"  Genome sequences collected: {n_seq}")
    print(f"  PhagesDB records total:     {len(records)}")

    if n_seq > 0:
        print(f"\n  Ready — run 02_model.py next.")
    else:
        print(f"\n  No genomes collected. See manual download instructions above.")
        print(f"  You can still run 02_model.py — it will use name-based features.")


if __name__ == "__main__":
    main()