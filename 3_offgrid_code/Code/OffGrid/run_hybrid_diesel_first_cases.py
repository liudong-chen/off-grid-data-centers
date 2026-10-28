"""Diesel-first hybrid (method B) on the paper's OTHER cases, at a few shares.

    python3 run_hybrid_diesel_first_cases.py georgia   [0.3,0.6,1.0]
    python3 run_hybrid_diesel_first_cases.py 2035 0.5  [0.3,0.6,1.0]      # 20 GW fleet
    python3 run_hybrid_diesel_first_cases.py 2035 3.0  [0.3,0.6,1.0]      # 121 GW fleet
    -> out/hybrid_diesel_<tag>/site_*.json, all_sites.json      (tag: georgia, 2035_k0p5, 2035_k3)

Same algorithm as run_hybrid_diesel_first.py (size gensets alone for x*load at LOLE<=0.1,
then least-cost gas around them for the full load), same diesel parameters; the campus set,
gas prices, land and capex anchor are EXACTLY those of run_offgrid_mip_cases_gasonly.py, and
x = 0 is that script's gas-only MIP design (out/offgrid_mip_<tag>_gasonly_sites.json).
Env: BYONG_CASE, BYONG_LOAD (as the other scripts), SCREEN, CONFIRM, WORKERS.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import hybrid_diesel_lib as L
import run_hybrid_diesel_first as F

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = sys.argv[1]
K = float(sys.argv[2]) if CASE == '2035' else None
TAG = 'georgia' if CASE == 'georgia' else f'2035_k{K:g}'.replace('.', 'p')
SHARES = [float(s) for s in (sys.argv[3] if CASE == '2035' else (sys.argv[2] if len(sys.argv) > 2 else '0.3,0.6,1.0')).split(',')] \
    if (CASE == '2035' and len(sys.argv) > 3) or (CASE != '2035' and len(sys.argv) > 2) else [0.3, 0.6, 1.0]
CD = os.environ.get('BYONG_CASE', f'{HERE}/in/case')
R = os.environ.get('BYONG_LOAD', f'{HERE}/in')
OUT = f'{HERE}/out/hybrid_diesel_{TAG}'; os.makedirs(OUT, exist_ok=True)
DESIGN = f'{HERE}/out/offgrid_mip_{TAG}_gasonly_sites.json'
ZM = {'Far West': 'FarWest', 'North Central': 'NorthCentral', 'South Central': 'SouthCentral'}
G35 = {'Coast': 5.0, 'East': 4.866, 'FarWest': 3.997, 'NorthCentral': 5.187, 'South': 5.348,
       'SouthCentral': 5.044, 'West': 4.501, 'North': 4.168}
GA_GAS, GA_LAND = 5.688, 4670.0
GA_FLEET_MW, GA_N, CAMELLIA_MW = 12400.0, 31, 3200.0


def setup():
    """Anchor capex for the case, append diesel, and build the campus list."""
    if CASE == '2035':
        L.om.anchor_capex({'CC': 113755.0, 'CT': 96576.0})
    L.prepare()
    d = pd.read_csv(f'{R}/dc_load_hourly_MW.csv', index_col=0, parse_dates=True)
    d = d[~((d.index.month == 2) & (d.index.day == 29))].drop(columns=['TOTAL_MW'], errors='ignore')
    meta = pd.read_csv(f'{R}/site_parameters_and_results.csv')
    meta['zone'] = meta.ercot_weather_zone.map(lambda z: ZM.get(z, z))
    des = {r['site']: r for r in json.load(open(DESIGN))}
    camp = {}
    if CASE == 'georgia':
        L.GASZ['Georgia'] = GA_GAS; L.og.LAND['Georgia'] = GA_LAND
        pk = d.max(); shape = (pk/pk.sum()).sort_values(ascending=False)
        rest_MW = GA_FLEET_MW - CAMELLIA_MW; src = list(shape.index)
        q_src = np.linspace(0, 1, len(src)); q_new = np.linspace(0, 1, GA_N - 1)
        w = np.interp(q_new, q_src, [shape[s] for s in src]); w = w/w.sum()
        FLOOR = 100.0/rest_MW
        for _ in range(50):
            lo = w < FLOOR
            if not lo.any():
                break
            w = np.where(lo, FLOOR, w*(1 - lo.sum()*FLOOR)/w[~lo].sum())
        arche = [src[min(int(round(q*(len(src)-1))), len(src)-1)] for q in q_new]
        sizes = [('GA01_Camellia', src[0], CAMELLIA_MW)] + \
                [(f'GA{i+2:02d}', arche[i], float(rest_MW*w[i])) for i in range(GA_N - 1)]
        for n, s, mw in sizes:
            load = d[s].values[:L.H]*(mw/d[s].values[:L.H].max())
            camp[n] = ('Georgia', load, mw, GA_GAS, GA_LAND)
    else:
        for _, r in meta.iterrows():
            z = r.zone
            camp[r.site_id] = (z, d[r.site_id].values[:L.H]*K, float(r.mw_published)*K, G35[z], L.og.LAND[z])
    return camp, des


def run_site(args):
    site_id, shares = args
    camp, des = setup()
    z, load, mw_pub, gas, land = camp[site_id]
    c = L.Campus(site_id, z, load, des[site_id], None, None, mw_pub)
    c.s = c.w = c.b = 0.0
    c.gas, c.land = gas, land
    c.VR = L.var_costs(gas)
    VR_DSL = float(c.VR[L.IDX_DSL])
    out = dict(site=site_id, case=TAG, rule='diesel-first', vre_kept=False, zone=z, peak_MW=c.peak, energy_MWh=c.energy,
               mw_published=c.mw_published, gas_price=c.gas, land_price=c.land, diesel=L.DIESEL, diesel_fuel=L.DSL_FUEL,
               var_cost=c.VR.tolist(), n0=c.n0.tolist(), solar_MW=0.0, wind_MW=0.0, batt_MW=0.0, cases={})
    n6_0 = np.append(c.n0, 0)
    t = time.time()
    lole0 = float(c.lole_many(n6_0[None, :], F.CONFIRM)[0])
    out['cases']['0.0'] = dict(share=0.0, n=n6_0.tolist(), removed_MW=0.0, diesel_units=0, LOLE=lole0,
                               installed_MW=float(n6_0 @ L.og.S_MW), **c.cost(n6_0), seconds=time.time() - t,
                               note='gas-only MIP design (run_offgrid_mip_cases_gasonly.py)',
                               phaseA=dict(load_fraction=0.0, diesel_units=0, LOLE=None, energy_MWh=0.0,
                                           fixed_diesel=0.0, fuel_diesel=0.0))
    print(f'{TAG} {site_id} x=0.00  {c.peak:,.0f} MW  LOLE {lole0:.3f}  ${out["cases"]["0.0"]["lcoe"]:.2f}/MWh '
          f'(MIP ${des[site_id]["lcoe"]:.2f})', flush=True)
    for x in shares:
        t = time.time(); log = []
        D, loleA = F.size_diesel_alone(c, x, log)
        n6, loleB = F.optimise_gas(c, D, log)
        rec = dict(share=x, n=n6.tolist(), removed_MW=float((c.n0 - n6[:5]) @ L.og.S_MW[:5]),
                   diesel_units=int(D), LOLE=loleB, installed_MW=float(n6 @ L.og.S_MW), search=log,
                   seconds=time.time() - t, **c.cost(n6))
        eA = x*c.energy
        rec['phaseA'] = dict(load_fraction=x, diesel_units=int(D), LOLE=loleA, energy_MWh=eA,
                             fixed_diesel=float(D*L.og.S_MW[L.IDX_DSL]*L.og.FX_MW[L.IDX_DSL]), fuel_diesel=eA*VR_DSL,
                             lcoe=(D*L.og.S_MW[L.IDX_DSL]*L.og.FX_MW[L.IDX_DSL] + eA*VR_DSL)/eA)
        rec['diesel_share_installed'] = rec['diesel_MW']/rec['installed_MW']
        out['cases'][f'{x:.1f}'] = rec
        mix = ' '.join(f'{L.og.CAT[i]["k"]}x{n6[i]}' for i in range(5) if n6[i])
        print(f'{TAG} {site_id} x={x:.2f}  A: {D:5d} gensets (LOLE {loleA:.3f})  B: {mix:<26} gas {rec["gas_MW"]:7,.0f} MW  '
              f'ratio {rec["installed_MW"]/c.peak:.2f}  LOLE {loleB:.3f}  ${rec["lcoe"]:.2f}/MWh  ({rec["seconds"]:.0f}s)', flush=True)
        json.dump(out, open(f'{OUT}/site_{site_id}.json', 'w'), indent=1, default=float)
    return site_id


if __name__ == '__main__':
    camp, des = setup()
    sites = [s for s in des if not os.path.exists(f'{OUT}/site_{s}.json')
             or len(json.load(open(f'{OUT}/site_{s}.json'))['cases']) < len(SHARES) + 1]
    sites.sort(key=lambda s: -camp[s][1].max())
    print(f'{TAG}: {len(des)} campuses, {len(sites)} to run, shares {SHARES}', flush=True)
    with Pool(int(os.environ.get('WORKERS', 2))) as p:
        for s in p.imap_unordered(run_site, [(s, SHARES) for s in sites]):
            print(f'== {s} done', flush=True)
    allr = [json.load(open(f'{OUT}/site_{s}.json')) for s in des if os.path.exists(f'{OUT}/site_{s}.json')]
    json.dump(allr, open(f'{OUT}/all_sites.json', 'w'), indent=1, default=float)
    print(f'written {OUT}/all_sites.json ({len(allr)} campuses)')
