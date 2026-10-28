#!/usr/bin/env python3
"""Download NSRDB hourly data for the 8 ERCOT weather zones, plus an optional
HRRR wind calibration set.

Obtain an API key at https://developer.nlr.gov/signup/. Set NREL_API_KEY to
your key and NREL_EMAIL to your contact email before requesting data.
NSRDB documentation: https://developer.nlr.gov/docs/solar/nsrdb/.

    python3 download_vre.py --years 2020            # 1 year, smoke test
    python3 download_vre.py                         # 2018-2025 solar + wind calib
    python3 download_vre.py --solar-only            # solar only, no calibration
    python3 download_vre.py --wind-all              # real 100 m wind for every year

Output: ./1_source_data/vre_raw/{solar,wind}/{Zone}_{year}.csv  + manifest.csv
Run from the project root; output goes to ./1_source_data/vre_raw/.

Standard library only.
"""
from __future__ import annotations
import argparse, csv, os, sys, time, urllib.parse, urllib.request, urllib.error

API_KEY = os.environ.get("NREL_API_KEY", "")
EMAIL   = os.environ.get("NREL_EMAIL", "")

# Verified against the live API docs:
#   NSRDB GOES Aggregated v4.0.0  1998-2025, interval 30 or 60, 4 km
#   HRRR MET Toolkit              2015-2025, hourly, 2 km (successor to legacy WTK)
#   legacy wtk-download           2007-2014 only -> unusable for this window
#   ERCOT load archive            2002-2025
# 2018-2025 is covered by all of them, and includes Feb 2021 (Uri).
DEFAULT_YEARS = list(range(2018, 2026))
CALIB_YEAR    = 2021           # the wind calibration year: contains Uri

# Representative points. NOT zone centroids — these sit where the resource
# actually is. Solar and wind differ within a zone (NorthCentral load is DFW,
# but its wind is the Abilene belt 200 km west).
POINTS = {
    #  zone            solar (lat, lon)      wind (lat, lon)
    "Coast":         ((29.00, -95.50),      (27.80, -97.40)),
    "East":          ((31.80, -94.80),      (31.80, -94.80)),
    "FarWest":       ((31.40, -103.00),     (31.40, -103.00)),
    "North":         ((34.20, -101.70),     (35.20, -101.80)),
    "NorthCentral":  ((32.80, -97.30),      (32.40, -99.70)),
    "South":         ((27.50, -98.50),      (27.00, -98.20)),
    "SouthCentral":  ((29.40, -98.50),      (28.90, -97.90)),
    "West":          ((32.00, -100.50),     (32.40, -100.40)),
}

NSRDB     = "https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download.csv"
NSRDB_ALT = "https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-conus-v4-0-0-download.csv"
HRRR      = "https://developer.nlr.gov/api/wind-toolkit/v2/wind/wtk-hrrr-met-toolkit-v1-0-0-download.csv"

# EVERYTHING needed downstream, in one pass:
#   PV model      ghi dni dhi clearsky_* solar_zenith_angle surface_albedo
#                 air_temperature wind_speed  (module cooling)
#   wind model    wind_speed wind_direction surface_pressure air_temperature
#                 relative_humidity dew_point   (hub-height shear + air density)
#   diagnostics   cloud_type total_precipitable_water fill_flag
SOLAR_ATTRS = ("ghi,dni,dhi,clearsky_ghi,clearsky_dni,clearsky_dhi,"
               "solar_zenith_angle,surface_albedo,air_temperature,dew_point,"
               "relative_humidity,surface_pressure,wind_speed,wind_direction,"
               "cloud_type,total_precipitable_water,fill_flag")
SOLAR_ATTRS_MIN = "ghi,dni,dhi,air_temperature,wind_speed,surface_albedo"

WIND_ATTRS     = ("windspeed_10m,windspeed_40m,windspeed_60m,windspeed_80m,"
                  "windspeed_100m,windspeed_120m,windspeed_160m,"
                  "winddirection_100m,temperature_100m,pressure_100m")
WIND_ATTRS_MIN = "windspeed_100m"


def parse_years(spec: str | None) -> list[int]:
    """'2020' | '2018-2025' | '2018,2021,2024'. Text after '#' is dropped,
    because zsh does not strip inline comments at an interactive prompt."""
    if not spec:
        return DEFAULT_YEARS
    spec = spec.split("#")[0].strip()
    if not spec:
        return DEFAULT_YEARS
    out: list[int] = []
    for part in spec.replace(" ", ",").split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    bad = [y for y in out if not (1998 <= y <= 2025)]
    if bad:
        sys.exit(f"years outside dataset coverage: {bad}")
    return sorted(set(out))


def fetch(url: str, params: dict, dest: str, tries: int = 5) -> tuple[bool, str]:
    if os.path.exists(dest) and os.path.getsize(dest) > 50_000:
        return True, "cached"
    q = urllib.parse.urlencode(params)
    for k in range(tries):
        try:
            with urllib.request.urlopen(f"{url}?{q}", timeout=300) as r:
                body = r.read()
            if len(body) < 5_000 or body[:1] == b"{":
                return False, f"short/JSON body: {body[:180]!r}"
            with open(dest, "wb") as f:
                f.write(body)
            n = body.count(b"\n")
            return True, f"{len(body)/1e6:.1f} MB, {n:,} lines"
        except urllib.error.HTTPError as e:
            detail = e.read()[:250].decode("utf8", "replace")
            if e.code in (429, 500, 502, 503, 504):
                wait = 20 * (k + 1)
                print(f"\n      HTTP {e.code}, retry in {wait}s", end=" ", flush=True)
                time.sleep(wait); continue
            return False, f"HTTP {e.code}: {detail}"
        except Exception as e:
            if k == tries - 1:
                return False, f"{type(e).__name__}: {e}"
            time.sleep(10 * (k + 1))
    return False, "retries exhausted"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=str, default=None,
                    help='"2020", "2018-2025", "2018,2021,2024". Default 2018-2025.')
    ap.add_argument("--out", default="1_source_data/vre_raw")
    ap.add_argument("--solar-only", action="store_true",
                    help="skip the wind calibration download entirely")
    ap.add_argument("--wind-all", action="store_true",
                    help="download real 100 m wind for EVERY year, not just the calibration year")
    ap.add_argument("--sleep", type=float, default=1.1, help="seconds between calls (limit 1/s)")
    a = ap.parse_args()
    if not API_KEY or not EMAIL:
        ap.error("Set your own NREL_API_KEY and NREL_EMAIL; obtain a key at https://developer.nlr.gov/signup/.")
    years = parse_years(a.years)

    os.makedirs(f"{a.out}/solar", exist_ok=True)
    os.makedirs(f"{a.out}/wind", exist_ok=True)

    if a.solar_only:
        wind_years: list[int] = []
    elif a.wind_all:
        wind_years = years
    else:
        wind_years = [CALIB_YEAR] if CALIB_YEAR in years else years[:1]

    jobs = [("solar", z, y, POINTS[z][0]) for z in POINTS for y in years]
    jobs += [("wind", z, y, POINTS[z][1]) for z in POINTS for y in wind_years]

    print(f"solar: {len(POINTS)} zones x {len(years)} years = {len(POINTS)*len(years)} files")
    print(f"wind : {len(POINTS)} zones x {len(wind_years)} years = {len(POINTS)*len(wind_years)} files"
          + ("  (calibration set)" if wind_years and not a.wind_all else ""))
    print(f"years: {years}\n")

    rows = []
    for i, (kind, zone, yr, (lat, lon)) in enumerate(jobs, 1):
        dest = f"{a.out}/{kind}/{zone}_{yr}.csv"
        common = dict(api_key=API_KEY, wkt=f"POINT({lon} {lat})", names=str(yr),
                      interval="60", utc="false", email=EMAIL,
                      reason="research", mailing_list="false")
        if kind == "solar":
            url, params, minimal = NSRDB, {**common, "attributes": SOLAR_ATTRS,
                                           "leap_day": "false"}, SOLAR_ATTRS_MIN
        else:
            url, params, minimal = HRRR, {**common, "attributes": WIND_ATTRS}, WIND_ATTRS_MIN
        print(f"[{i:>3}/{len(jobs)}] {kind:<5} {zone:<13} {yr} ({lat},{lon}) ...", end=" ", flush=True)
        ok, msg = fetch(url, params, dest)
        if not ok:                                        # retry with fewer attributes
            ok, m2 = fetch(url, {**params, "attributes": minimal}, dest)
            msg = f"{m2} (reduced attrs)" if ok else f"{msg} | reduced also failed: {m2}"
        if not ok and kind == "solar":                    # try the CONUS product
            ok, m3 = fetch(NSRDB_ALT, params, dest)
            msg = f"{m3} (GOES-conus)" if ok else f"{msg} | conus also failed: {m3}"
        print(msg if ok else f"FAILED — {msg}", flush=True)
        rows.append(dict(kind=kind, zone=zone, year=yr, lat=lat, lon=lon,
                         file=dest, ok=int(ok), note=msg))
        time.sleep(a.sleep)

    with open(f"{a.out}/manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    bad = [r for r in rows if not r["ok"]]
    print(f"\ndone. {len(rows)-len(bad)}/{len(rows)} succeeded -> {a.out}/")
    for r in bad:
        print(f"  FAILED {r['kind']} {r['zone']} {r['year']}: {r['note']}")
    if bad:
        print("re-run the script; successful files are cached and skipped.")


if __name__ == "__main__":
    main()
