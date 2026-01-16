import requests

BASE_URL = "https://viralhostrangedb.pasteur.cloud/api"
HOST_SEARCH_TERM = "Staphylococcus aureus"

# --- Step 1: Get all hosts ---
print("Fetching all hosts...")
hosts_res = requests.get(f"{BASE_URL}/host/?format=json")
hosts_res.raise_for_status()
hosts_data = hosts_res.json()  # <-- this is a LIST

# --- Step 2: Find Staphylococcus aureus host IDs ---
sau_ids = []

for host in hosts_data:
    name = host.get("name", "")
    if HOST_SEARCH_TERM.lower() in name.lower():
        sau_ids.append(str(host["id"]))  # IDs are strings in response maps

if not sau_ids:
    print(f"No hosts found matching '{HOST_SEARCH_TERM}'")
    exit(1)

print(f"Found Staphylococcus aureus host ID(s): {sau_ids}")

# --- Step 3: Fetch aggregated responses ---
print("Fetching aggregated responses...")
agg_res = requests.get(
    f"{BASE_URL}/aggregated-responses/?allow_overflow=true&format=json"
)
agg_res.raise_for_status()
agg_data = agg_res.json()

# --- Step 4: Filter by host ID ---
print("Filtering virus-host interactions...")
sau_results = {}

for virus_id, host_map in agg_data.items():
    for host_id in sau_ids:
        if host_id in host_map:
            sau_results[virus_id] = {
                "infection_value": host_map[host_id]["val"],
                "evidence_count": host_map[host_id]["diff"]
            }

# --- Step 5: Output ---
print("\n=== Staphylococcus aureus Virus Interactions ===")

if not sau_results:
    print("No interaction data found.")
else:
    for virus_id, info in sau_results.items():
        print(
            f"Virus {virus_id}: "
            f"value={info['infection_value']} "
            f"(sources={info['evidence_count']})"
        )

print("\nDone ✅")
