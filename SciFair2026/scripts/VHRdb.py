import requests, json
import os

DATA_DIR = r"D:\SciFair2026\SciFair2026\data\raw"
VHR_FILE = os.path.join(DATA_DIR, "VHRdb.json")

if not os.path.exists(VHR_FILE):
    print("Fetching VHRdb JSON from API...")
    BASE_URL = "https://viralhostrangedb.pasteur.cloud/api"
    resp = requests.get(f"{BASE_URL}/aggregated-responses/?allow_overflow=true&format=json")
    resp.raise_for_status()
    vhr_data = resp.json()

    with open(VHR_FILE, "w") as f:
        json.dump(vhr_data, f)
    print(f"Saved VHRdb JSON → {VHR_FILE}")
else:
    print(f"VHRdb JSON already exists → {VHR_FILE}")
