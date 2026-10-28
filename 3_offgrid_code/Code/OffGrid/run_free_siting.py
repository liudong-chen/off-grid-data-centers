"""Scenario 1, free siting, GAS-ONLY: each campus designed in Far West.

With no wind, solar or battery the only zone-dependent inputs are the delivered gas price
and the land price, and Far West is the cheapest ERCOT zone on BOTH ($3.953/MMBtu, $714/acre),
so it dominates every other zone for every campus: free siting = every campus in Far West.
Hence one MIP solve per campus (17), not 136.

    OFFGRID_CASE=in/case OFFGRID_LOAD=in python3 run_free_siting.py
    -> out/S1_free_siting_gasonly_sites.json, out/S1_free_siting_gasonly.json
"""
import os, sys, json, time
import numpy as np, pandas as pd
import offgrid_mip as om

CD = os.environ.get('OFFGRID_CASE', 'in/case'); R = os.environ.get('OFFGRID_LOAD', 'in')
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f'{HERE}/out'
om.anchor_capex()
GASZ = json.load(open(f'{HERE}/zonal_gas_2030.json'))
ZM = {'Far West': 'FarWest', 'North Central': 'NorthCentral', 'South Central': 'SouthCentral'}
d = pd.read_csv(f'{R}/dc_load_hourly_MW.csv', index_col=0, parse_dates=True)
d = d[~((d.index.month == 2) & (d.index.day == 29))].drop(columns=['TOTAL_MW'], errors='ignore')
meta = pd.read_csv(f'{R}/site_parameters_and_results.csv')
meta['zone'] = meta.ercot_weather_zone.map(lambda z: ZM.get(z, z))
HOME = {r['site']: r for r in json.load(open(f'{OUT}/offgrid_mip_gasonly.json'))}
BEST_Z = min(GASZ, key=lambda z: (GASZ[z], om.og.LAND[z]))
assert BEST_Z == 'FarWest' and om.og.LAND[BEST_Z] == min(om.og.LAND.values())
PART = f'{OUT}/S1_free_siting_gasonly_sites.json'
rows = json.load(open(PART)) if os.path.exists(PART) else []
done = {r['site'] for r in rows}
for r in meta.itertuples():
    if r.site_id in done:
        continue
    t0 = time.time(); load = d[r.site_id].values[:om.H]
    home = HOME[r.site_id]
    if r.zone == BEST_Z:
        best = home
    else:
        best = om.design(load, BEST_Z, None, None, gas=GASZ[BEST_Z], allow=(False, False, False),
                         credit_cap=0.15, vre_cut_frac=0.0, sens_cuts=False)
    rows.append(dict(site=r.site_id, home_zone=r.zone, best_zone=BEST_Z, peak_MW=home['peak_MW'], energy_MWh=home['energy_MWh'],
                     home_cost=home['cost'], best_cost=best['cost'], home_lcoe=home['lcoe'], best_lcoe=best['lcoe'],
                     home_installed_MW=home['installed_MW'], best_installed_MW=best['installed_MW'],
                     home_mix=home['mix'], best_mix=best['mix'], best_LOLE=best['LOLE'], method='mip gas-only'))
    print(f"{r.site_id:<6}{r.zone:<14}{home['peak_MW']:>8,.0f}  home ${home['lcoe']:.2f}  FarWest ${best['lcoe']:.2f}  "
          f"({100*(1-best['lcoe']/home['lcoe']):.1f}%)  [{time.time()-t0:.0f}s]", flush=True)
    json.dump(rows, open(PART, 'w'), indent=1, default=float)
df = pd.DataFrame(rows)
E = df.energy_MWh.sum(); C_home, C_best = df.home_cost.sum(), df.best_cost.sum()
json.dump(dict(energy_MWh=float(E), offgrid_home_cost=float(C_home), offgrid_home_lcoe=float(C_home/E),
               offgrid_free_cost=float(C_best), offgrid_free_lcoe=float(C_best/E), saving_per_MWh=float((C_home-C_best)/E),
               saving_Byr=float((C_home-C_best)/1e9), installed_MW_home=float(df.home_installed_MW.sum()),
               installed_MW_free=float(df.best_installed_MW.sum()), sites_moving=int((df.home_zone != df.best_zone).sum()),
               free_zone=BEST_Z, method='mip gas-only (allow=(F,F,F)), every campus re-designed in Far West, the zone that '
               'is cheapest on both delivered gas and land and therefore dominates for a gas-only fleet'),
          open(f'{OUT}/S1_free_siting_gasonly.json', 'w'), indent=1, default=float)
print(f"announced sites ${C_home/E:.2f}/MWh; free siting (all Far West) ${C_best/E:.2f}/MWh")
