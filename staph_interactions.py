"""
VHRdb Staphylococcus aureus interaction collector
Uses aggregated-responses (ground-truth experimental data)

Output:
staph_vhrdb_interactions.csv

Requires:
pip install requests pandas
"""

import requests
import pandas as pd

BASE_URL = "https://viralhostrangedb.pasteur.cloud/api"
HOST_SEARCH_TERM = "Staphylococcus aureus"
OUTPUT_CSV = "staph_vhrdb_interactions.csv"

# -------------------------------------------------
# STEP 1: Fetch all hosts
# -------------------------------------------------
print("Fetching all hosts...")
hosts_res = requests.get(f"{BASE_URL}/host/?format=json")
hosts_res.raise_for_status()
hosts_data = hosts_res.json()

# -------------------------------------------------
# STEP 2: Collect Staph aureus host IDs
# -------------------------------------------------
sau_hosts = []

for host in hosts_data:
    name = host.get("name", "")
    if HOST_SEARCH_TERM.lower() in name.lower():
        sau_hosts.append({
            "host_id": str(host["id"]),
            "host_name": name
        })

if not sau_hosts:
    raise RuntimeError("No Staphylococcus aureus hosts found.")

sau_host_ids = {h["host_id"] for h in sau_hosts}

print(f"Found {len(sau_hosts)} Staphylococcus aureus hosts")

# -------------------------------------------------
# STEP 3: Fetch aggregated responses
# -------------------------------------------------
print("Fetching aggregated virus-host responses...")
agg_res = requests.get(
    f"{BASE_URL}/aggregated-responses/?allow_overflow=true&format=json"
)
agg_res.raise_for_status()
agg_data = agg_res.json()

# -------------------------------------------------
# STEP 4: Extract S. aureus interactions
# -------------------------------------------------
print("Extracting S. aureus interactions...")
rows = []

for virus_id, host_map in agg_data.items():
    for host in sau_hosts:
        hid = host["host_id"]
        if hid in host_map:
            rows.append({
                "virus_id": virus_id,
                "host_id": hid,
                "host_name": host["host_name"],
                "infection_value": host_map[hid]["val"],
                "evidence_count": host_map[hid]["diff"]
            })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

# -------------------------------------------------
# DONE
# -------------------------------------------------
print("\nDATA COLLECTION COMPLETE ✅")
print(f"Total interactions: {len(df)}")
print(f"Saved to: {OUTPUT_CSV}")
