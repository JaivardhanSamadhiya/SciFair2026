import requests

# Base URL for host list
HOST_LIST_URL = "https://viralhostrangedb.pasteur.cloud/api/host/?format=json"

# Fetch all hosts
response = requests.get(HOST_LIST_URL)
response.raise_for_status()  # ensure we catch HTTP errors

hosts = response.json()

print(f"Total hosts retrieved: {len(hosts)}\n")

# Print host ID and name for each
for host in hosts:
    host_id = host.get("id")
    name = host.get("name")
    tax_id = host.get("tax_id")
    print(f"ID: {host_id}, TaxID: {tax_id}, Name: {name}")
