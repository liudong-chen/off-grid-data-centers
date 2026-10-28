"""Design all 17 ERCOT campuses with the MIP formulation (offgrid_mip.design).

    OFFGRID_CASE=<HOPE Data_TX8z_GTEP_dc dir> OFFGRID_LOAD=<DC_Load/load_profiles dir> \
        python3 run_offgrid_mip.py [S01,S02,...] [tag]

Writes out/offgrid_mip_<tag>.json, one record per campus, in the same shape as
offgrid_sites.json so the figure scripts can read either.
"""
import os, sys, json, time
import numpy as np, pandas as pd
import offgrid_mip as om

CD = os.environ.get('OFFGRID_CASE', 'Code/HOPE/ModelCases/TX8z_GTEP_dc_vre_case/Data_TX8z_GTEP_dc_vre')
R = os.environ.get('OFFGRID_LOAD', 'DC_Load/load_profiles')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
ZM = {'Far West': 'FarWest', 'North Central': 'NorthCentral', 'South Central': 'SouthCentral'}

d = pd.read_csv(f'{R}/dc_load_hourly_MW.csv', index_col=0, parse_dates=True)
d = d[~((d.index.month == 2) & (d.index.day == 29))]
meta = pd.read_csv(f'{R}/site_parameters_and_results.csv')
meta['zone'] = meta.ercot_weather_zone.map(lambda z: ZM.get(z, z))
only = sys.argv[1].split(',') if len(sys.argv) > 1 and sys.argv[1] != 'all' else None
TAG = sys.argv[2] if len(sys.argv) > 2 else 'all'
POOL = len(sys.argv) > 1 and sys.argv[1] == 'pool'   # one shared plant for all 17 campuses
FREE = os.environ.get('OFFGRID_FREE', '0') == '1'      # let VRE + battery substitute for machines
SENS = os.environ.get('OFFGRID_SENS', '0') == '1'      # MC sensitivity (Benders-style) cuts
CF_S = {z: om.og.zonal_solar_cf(CD, z) for z in om.og.GAS}
CF_W = {z: om.zonal_wind_cf(CD, z) for z in om.og.GAS}
# zonal delivered gas, $/MMBtu, 2030 -- the table the published v4 results and the HOPE vre
# cases use (Figures/figures_1_to_3/01_raw/zonal_gas_2030.json); offgrid_model.GAS is the
# older ERCOT-wide $3.891 fallback and is NOT used here
if os.environ.get('OFFGRID_RAW', '0') != '1':                # paper basis: capex anchored to on-grid
    FAC = om.anchor_capex()
    print(f"capex anchored to on-grid candidates: CT-class x{FAC['CT']:.5f}, CC x{FAC['CC']:.5f}", flush=True)
GASZ = json.load(open(os.environ.get('OFFGRID_GAS', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                             'zonal_gas_2030.json'))))

rows = []
os.makedirs(OUT, exist_ok=True)

if POOL:
    # the pooled architecture: the 17 hourly loads summed and served by ONE plant, designed
    # in each of the three zones offgrid_pool.json reports (Far West is the paper's case)
    tot = d[meta.site_id].values[:om.H].sum(axis=1)
    pool = {}
    for zone in ('FarWest', 'SouthCentral', 'NorthCentral'):
        t = time.time()
        res = om.design(tot, zone, CF_S[zone], CF_W[zone], gas=GASZ[zone],
                        credit_cap=1.0 if FREE else 0.15, vre_cut_frac=0.10 if FREE else 0.0,
                        sens_cuts=SENS, verbose=True, time_limit=300.0)
        if res is None:
            print(f'POOLED {zone}: NO DESIGN', flush=True); continue
        pool[zone] = res
        mix = ' '.join(f'{k}x{v}' for k, v in res['mix'].items())
        print(f"POOLED {zone:<14}{res['peak_MW']:>8,.0f}  {mix:<40} solar {res['solar_MW']:,.0f} "
              f"wind {res['wind_MW']:,.0f} batt {res['batt_MW']:,.0f}  {res['installed_MW']:,.0f} MW "
              f"({res['ratio']:.2f}x)  LOLE {res['LOLE']:.3f}  ${res['cost']/1e9:.3f} B/yr  "
              f"${res['lcoe']:.2f}/MWh  cuts {res['n_cuts']}  {time.time()-t:.0f}s", flush=True)
        json.dump(pool, open(f'{OUT}/offgrid_mip_pool_{TAG}.json', 'w'), indent=1, default=float)
    sys.exit(0)
print(f"{'site':<5}{'zone':<14}{'peak':>7}  {'portfolio':<32}{'solar':>6}{'wind':>6}{'batt':>6}"
      f"{'MW':>8}{'xpk':>6}{'LOLE':>7}{'$/MWh':>8}{'cuts':>5}{'s':>6}", flush=True)
for r in meta.itertuples():
    if only and r.site_id not in only:
        continue
    load = d[r.site_id].values[:om.H]
    res = om.design(load, r.zone, CF_S[r.zone], CF_W[r.zone], gas=GASZ[r.zone],
                    credit_cap=1.0 if FREE else 0.15, vre_cut_frac=0.10 if FREE else 0.0,
                    sens_cuts=SENS)
    if res is None:
        print(f'{r.site_id:<5}{r.zone:<14}{load.max():>7,.0f}  NO DESIGN', flush=True); continue
    res.update(site=r.site_id, home_zone=r.zone)
    rows.append(res)
    mix = ' '.join(f'{k}x{v}' for k, v in res['mix'].items())
    print(f"{r.site_id:<5}{r.zone:<14}{res['peak_MW']:>7,.0f}  {mix:<32}{res['solar_MW']:>6,.0f}"
          f"{res['wind_MW']:>6,.0f}{res['batt_MW']:>6,.0f}{res['installed_MW']:>8,.0f}"
          f"{res['ratio']:>6.2f}{res['LOLE']:>7.3f}{res['lcoe']:>8.2f}{res['n_cuts']:>5}"
          f"{res['seconds']:>6.0f}", flush=True)
    json.dump(rows, open(f'{OUT}/offgrid_mip_{TAG}.json', 'w'), indent=1, default=float)
if rows:
    pk = sum(r['peak_MW'] for r in rows); mw = sum(r['installed_MW'] for r in rows)
    c = sum(r['cost'] for r in rows); e = sum(r['energy_MWh'] for r in rows)
    print(f'\nFLEET {pk:,.0f} MW peak  {mw:,.0f} MW built ({mw/pk:.2f}x)  '
          f'${c/1e9:.2f} B/yr  ${c/e:.2f}/MWh')
