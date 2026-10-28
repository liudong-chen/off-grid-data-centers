"""Scenario 1, free siting, with the MIP: each ERCOT campus designed in all eight zones
(the frontier-search version is Figures/fig123_package/02_code/73_S1_free_siting.py).

    python3 run_offgrid_mip_S1.py [sites_json]     # home-zone designs reused from sites_json
    -> out/S1_free_siting_sites.json, out/S1_free_siting.json   (same shape as the package)

Env BYONG_CASE, BYONG_LOAD as run_offgrid_mip.py. Paper basis: anchored capex, zonal gas,
free credits, sensitivity cuts.
"""
import os, sys, json, time
import numpy as np, pandas as pd
import offgrid_mip as om

CD = os.environ.get('BYONG_CASE'); R = os.environ.get('BYONG_LOAD')
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f'{HERE}/out'
HOME_JSON = sys.argv[1] if len(sys.argv) > 1 else f'{OUT}/offgrid_mip_final.json'
ONLY = sys.argv[2].split(',') if len(sys.argv) > 2 else None
om.anchor_capex()
GASZ = json.load(open(f'{HERE}/zonal_gas_2030.json'))
ZM = {'Far West': 'FarWest', 'North Central': 'NorthCentral', 'South Central': 'SouthCentral'}
KW = dict(credit_cap=1.0, vre_cut_frac=0.10, sens_cuts=True)

d = pd.read_csv(f'{R}/dc_load_hourly_MW.csv', index_col=0, parse_dates=True)
d = d[~((d.index.month == 2) & (d.index.day == 29))].drop(columns=['TOTAL_MW'])
meta = pd.read_csv(f'{R}/site_parameters_and_results.csv')
meta['zone'] = meta.ercot_weather_zone.map(lambda z: ZM.get(z, z))
ZONES = list(GASZ)
CF_S = {z: om.og.zonal_solar_cf(CD, z) for z in ZONES}
CF_W = {z: om.zonal_wind_cf(CD, z) for z in ZONES}
HOME = {r['site']: r for r in json.load(open(HOME_JSON))} if os.path.exists(HOME_JSON) else {}

PART = f'{OUT}/S1_free_siting_sites{"_" + sys.argv[2].replace(",", "_") if ONLY else ""}.json'
rows = json.load(open(PART)) if os.path.exists(PART) else []
done = {r['site'] for r in rows}
print(f"{'site':<6}{'home':<14}{'best':<14}{'peak':>8}{'$/MWh home':>11}{'$/MWh best':>11}{'saving':>8}"
      f"{'wind':>7}{'solar':>7}   time", flush=True)
for r in meta.itertuples():
    if r.site_id in done or (ONLY and r.site_id not in ONLY):
        continue
    t0 = time.time(); load = d[r.site_id].values[:om.H]; zres = {}
    for z in ZONES:
        if z == r.zone and r.site_id in HOME:
            res = HOME[r.site_id]
        else:
            res = om.design(load, z, CF_S[z], CF_W[z], gas=GASZ[z], allow=(True, CF_W[z] is not None, True), **KW)
        if res:
            zres[z] = res
    home = zres[r.zone]; best_z = min(zres, key=lambda z: zres[z]['cost']); best = zres[best_z]
    rows.append(dict(site=r.site_id, home_zone=r.zone, best_zone=best_z,
                     peak_MW=home['peak_MW'], energy_MWh=home['energy_MWh'],
                     home_cost=home['cost'], best_cost=best['cost'],
                     home_lcoe=home['lcoe'], best_lcoe=best['lcoe'],
                     home_installed_MW=home['installed_MW'], best_installed_MW=best['installed_MW'],
                     home_mix=home['mix'], best_mix=best['mix'],
                     best_wind_MW=best['wind_MW'], best_solar_MW=best['solar_MW'],
                     home_wind_MW=home['wind_MW'], home_solar_MW=home['solar_MW'],
                     zones={z: dict(cost=zres[z]['cost'], lcoe=zres[z]['lcoe'],
                                    installed_MW=zres[z]['installed_MW'], wind_MW=zres[z]['wind_MW'],
                                    solar_MW=zres[z]['solar_MW'], batt_MW=zres[z]['batt_MW'],
                                    mix=zres[z]['mix'], LOLE=zres[z]['LOLE']) for z in zres},
                     method='mip'))
    print(f"{r.site_id:<6}{r.zone:<14}{best_z:<14}{home['peak_MW']:>8,.0f}{home['lcoe']:>11.2f}"
          f"{best['lcoe']:>11.2f}{100*(1-best['lcoe']/home['lcoe']):>7.1f}%{best['wind_MW']:>7,.0f}"
          f"{best['solar_MW']:>7,.0f}   [{time.time()-t0:.0f}s]", flush=True)
    json.dump(rows, open(PART, 'w'), indent=1, default=float)
if ONLY:
    sys.exit(0)

df = pd.DataFrame([{k: v for k, v in r.items() if k != 'zones'} for r in rows])
E = df.energy_MWh.sum(); C_home, C_best = df.home_cost.sum(), df.best_cost.sum()
ONG = json.load(open(f'{HERE}/ongrid_final.json'))
POOLR = json.load(open(f'{OUT}/offgrid_mip_pool_final.json'))
best_pool = min(POOLR.values(), key=lambda x: x['cost'])
one_zone = {z: sum(r['zones'][z]['cost'] for r in rows)/E for z in ZONES}
print(f"\nannounced sites ${C_home/E:.2f}/MWh; cheapest zone ${C_best/E:.2f}/MWh; "
      f"{(df.home_zone != df.best_zone).sum()} of {len(df)} move; {dict(df.best_zone.value_counts())}")
json.dump(dict(energy_MWh=float(E), offgrid_home_cost=float(C_home), offgrid_home_lcoe=float(C_home/E),
               offgrid_free_cost=float(C_best), offgrid_free_lcoe=float(C_best/E),
               saving_per_MWh=float((C_home-C_best)/E), saving_Byr=float((C_home-C_best)/1e9),
               gap_closed_pct=float(100*(C_home-C_best)/(C_home-ONG['cost'])),
               installed_MW_home=float(df.home_installed_MW.sum()), installed_MW_free=float(df.best_installed_MW.sum()),
               sites_moving=int((df.home_zone != df.best_zone).sum()),
               destinations={k: int(v) for k, v in df.best_zone.value_counts().items()},
               all_to_one_zone_lcoe=one_zone, pooled_lcoe=best_pool['lcoe'], ongrid_lcoe=ONG['lcoe'],
               ongrid_cost=ONG['cost'], wind_MW_free=float(df.best_wind_MW.sum()),
               solar_MW_free=float(df.best_solar_MW.sum()),
               note='MIP designs (offgrid_mip.py: lazy LOLE cuts, exact neighbourhood improvement, 2,000-rep confirmation): '
                    '2030 inputs as corrected, EIA-923 zonal gas, wind, solar and a 4-h battery in the option set with the '
                    'model choosing, capital anchored to the on-grid candidate. One-sided freedom: the on-grid case already sites by zone.'),
          open(f'{OUT}/S1_free_siting.json', 'w'), indent=1, default=float)
print(f'written {OUT}/S1_free_siting.json')
