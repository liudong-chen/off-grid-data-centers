"""Download Southern Company (SOCO) hourly demand for 2015–2025 from EIA-930.

Obtain your own key at https://www.eia.gov/opendata/register.php and set
EIA_API_KEY before running this file from the repository root.
"""
import os

API_KEY = os.environ.get("EIA_API_KEY", "")

import csv
import time
import requests

BASE_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameter_setting", "GPC_EIA930_demand_2015_2025.csv")

if not API_KEY:
    raise SystemExit("Set EIA_API_KEY; obtain your key at https://www.eia.gov/opendata/register.php.")

PAGE_SIZE = 5000
offset = 0
all_rows = []

while True:
    params = [
        ("api_key", API_KEY),
        ("frequency", "hourly"),
        ("data[0]", "value"),
        ("facets[respondent][]", "SOCO"),
        ("facets[type][]", "D"),
        ("start", "2015-01-01T00"),
        ("end", "2025-12-31T23"),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "asc"),
        ("offset", str(offset)),
        ("length", str(PAGE_SIZE)),
    ]

    print(f"Downloading rows {offset:,}–{offset + PAGE_SIZE - 1:,}")

    response = requests.get(
        BASE_URL,
        params=params,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        },
        timeout=120,
    )

    if not response.ok:
        print("Request URL:", response.url)
        print("EIA response:", response.text)
        response.raise_for_status()

    result = response.json()
    rows = result["response"]["data"]
    total = int(result["response"]["total"])

    all_rows.extend(rows)
    offset += len(rows)

    print(f"Downloaded {offset:,} of {total:,} rows")

    if not rows or offset >= total:
        break

    time.sleep(0.2)

if not all_rows:
    raise RuntimeError("EIA returned no Georgia Power demand records.")

columns = list(all_rows[0].keys())

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(all_rows)

print(f"Saved {len(all_rows):,} rows to {OUTPUT_FILE}")