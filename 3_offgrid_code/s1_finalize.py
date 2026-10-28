"""Merge the free-siting parts (main run + parallel subsets), take the home-zone design from the
final sites file, rebuild the S1 summary.  python3 s1_finalize.py"""
import os, sys, json, glob
import numpy as np, pandas as pd
import offgrid_mip as om
CD = os.environ['OFFGRID_CASE']; R = os.environ['OFFGRID_LOAD']
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f'{HERE}/out'
om.anchor_capex(); GAS = json.load(open(f'{HERE}/zonal_gas_2030.json')); K = ['Recip', 'Aero35', 'Aero48', 'FrameCT', 'CC']
ZM = {'Far West': 'FarWest', 'North Central': 'NorthCentral', 'South Central': 'SouthCentral'}
d = pd.read_csv(f'{R}/dc_load_hourly_MW.csv', index_col=0, parse_dates=True); d = d[~((d.index.month == 2) & (d.index.day == 29))]
meta = pd.read_csv(f'{R}/site_parameters_and_results.csv'); meta['zone'] = meta.ercot_weather_zone.map(lambda z: ZM.get(z, z))
CF_S = {z: om.og.zonal_solar_cf(CD, z) for z in GAS}; CF_W = {z: om.zonal_wind_cf(CD, z) for z in GAS}
rows = {}
for f in glob.glob(f'{OUT}/S1_free_siting_sites*.json'):
    for r in json.load(open(f)):
        rows[r['site']] = r
FINAL = {r['site']: r for r in json.load(open(f'{OUT}/offgrid_mip_final.json'))}
for site, r in rows.items():
    fr = FINAL[site]; z = r['home_zone']
    r['zones'][z] = dict(cost=fr['cost'], lcoe=fr['lcoe'], installed_MW=fr['installed_MW'], wind_MW=fr['wind_MW'],
                         solar_MW=fr['solar_MW'], batt_MW=fr['batt_MW'], mix=fr['mix'], LOLE=fr['LOLE'])
    h = r['zones'][z]; bz = min(r['zones'], key=lambda q: r['zones'][q]['cost']); b = r['zones'][bz]
    r.update(best_zone=bz, home_cost=h['cost'], best_cost=b['cost'], home_lcoe=h['lcoe'], best_lcoe=b['lcoe'],
             home_installed_MW=h['installed_MW'], best_installed_MW=b['installed_MW'], home_mix=h['mix'], best_mix=b['mix'],
             best_wind_MW=b['wind_MW'], best_solar_MW=b['solar_MW'], home_wind_MW=h['wind_MW'], home_solar_MW=h['solar_MW'])
    print(f"{site:<5} home {z:<13} ${h['lcoe']:.2f}  best {bz:<13} ${b['lcoe']:.2f}", flush=True)
rows = [rows[s] for s in meta.site_id if s in rows]
json.dump(rows, open(f'{OUT}/S1_free_siting_sites.json', 'w'), indent=1, default=float)
df = pd.DataFrame([{k: v for k, v in r.items() if k != 'zones'} for r in rows])
E = df.energy_MWh.sum(); C_home, C_best = df.home_cost.sum(), df.best_cost.sum()
ONG = json.load(open(f'{HERE}/ongrid_final.json')); POOLR = json.load(open(f'{OUT}/offgrid_mip_pool_final.json'))
best_pool = min(POOLR.values(), key=lambda x: x['cost']); ZONES = list(GAS)
one_zone = {z: sum(r['zones'][z]['cost'] for r in rows)/E for z in ZONES}
json.dump(dict(energy_MWh=float(E), offgrid_home_cost=float(C_home), offgrid_home_lcoe=float(C_home/E),
               offgrid_free_cost=float(C_best), offgrid_free_lcoe=float(C_best/E), saving_per_MWh=float((C_home-C_best)/E),
               saving_Byr=float((C_home-C_best)/1e9), gap_closed_pct=float(100*(C_home-C_best)/(C_home-ONG['cost'])),
               installed_MW_home=float(df.home_installed_MW.sum()), installed_MW_free=float(df.best_installed_MW.sum()),
               sites_moving=int((df.home_zone != df.best_zone).sum()),
               destinations={k: int(v) for k, v in df.best_zone.value_counts().items()}, all_to_one_zone_lcoe=one_zone,
               pooled_lcoe=best_pool['lcoe'], ongrid_lcoe=ONG['lcoe'], ongrid_cost=ONG['cost'],
               wind_MW_free=float(df.best_wind_MW.sum()), solar_MW_free=float(df.best_solar_MW.sum()),
               note='MIP designs (offgrid_mip.py: lazy LOLE cuts, exact neighbourhood improvement, 2,000-rep confirmation); '
                    '2030 inputs as corrected, EIA-923 zonal gas, capital anchored to the on-grid candidate. '
                    'One-sided freedom: the on-grid case already sites by zone.'),
          open(f'{OUT}/S1_free_siting.json', 'w'), indent=1, default=float)
print(f"home ${C_home/E:.2f}  free ${C_best/E:.2f}  moving {(df.home_zone != df.best_zone).sum()}  {dict(df.best_zone.value_counts())}")
