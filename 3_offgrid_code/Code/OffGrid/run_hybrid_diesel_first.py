"""Diesel-FIRST hybrid: size the gensets for year one, then optimise the gas around them.

    python3 run_hybrid_diesel_first.py [S01,... | all] [shares]     default 0.1,...,1.0

For every campus and diesel share x (x = share of the campus LOAD the gensets carry
alone in year one; x = 1.0 is the all-diesel campus):

  PHASE A  (month T_DSL .. T_GAS, gensets only)
    D_x = smallest number of 3-MW gensets with LOLE(D gensets, x * load) <= 0.1
    -- the campus energises x of its racks in year one AT THE SAME RELIABILITY STANDARD.
    No wind/solar/battery yet (they arrive with the gas).

  PHASE B  (month T_GAS .., islanded, full load)
    With the D_x gensets fixed in the portfolio (and, only if DSL_VRE=1, the published
    wind/solar/battery -- the MAIN case is gas + diesel only),
    choose the least-cost gas set (combined cycles, frame CTs, 35-MW aeros; recips and
    48-MW aeros are dominated by the gensets as reserve and by the aeros as energy) such
    that LOLE(full load) <= 0.1.  Every (nCC, nCT, nA35) on a grid is costed exactly by
    merit order and screened in ONE Monte Carlo batch (shared random draws); the cheapest
    feasible designs are then confirmed at CONFIRM reps in cost order.

  x = 0 is the published all-gas design (unchanged).

Writes out/hybrid_diesel/site_<id>.json (same schema as run_hybrid_diesel.py plus a
'phaseA' block per share) and all_sites.json.  Resumable per campus.
"""
import json
import os
import sys
import time
from itertools import product
from multiprocessing import Pool

import numpy as np

import hybrid_diesel_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))            # Code/OffGrid
BYONG = os.path.abspath(os.path.join(HERE, '..', '..'))
LOAD_DIR = os.environ.get('BYONG_LOAD', f'{BYONG}/DC_Load/load_profiles')
CASE = os.environ.get('BYONG_CASE',
                      f'{BYONG}/Code/HOPE/ModelCases/TX8z_GTEP_dc_vre_case/Data_TX8z_GTEP_dc_vre')
DESIGN = os.environ.get('BYONG_DESIGN', f'{HERE}/out/offgrid_mip_final.json')
OUT = os.environ.get('BYONG_OUT', os.path.join(HERE, 'out', 'hybrid_diesel')); os.makedirs(OUT, exist_ok=True)
SCREEN = int(os.environ.get('SCREEN', 100))
CONFIRM = int(os.environ.get('CONFIRM', 400))
TARGET = 0.1
iCC, iCT, iA35, iDSL = 4, 3, 1, L.IDX_DSL
VRE = os.environ.get('DSL_VRE', '0') == '1'      # main case: gas + diesel only;
                                                 # DSL_VRE=1 keeps the published wind/solar/battery


def size_diesel_alone(c, scale, log):
    """Smallest D with LOLE(D gensets alone, scale*load) <= TARGET; no VRE, no battery."""
    S = L.og.S_MW[iDSL]
    load = c.load*scale
    pk = float(load.max())

    def lole(counts, reps):
        N = np.zeros((len(counts), 6), np.int64); N[:, iDSL] = counts
        return L.og.lole_batch(load, N, reps)

    lo, hi = int(np.floor(pk/S)), int(np.ceil(1.35*pk/S)) + 5
    while True:
        grid = np.unique(np.linspace(lo, hi, 9).round().astype(int))
        l = lole(grid, SCREEN)
        log.append(dict(stage='A', grid=grid.tolist(), lole=l.round(3).tolist(), reps=SCREEN))
        ok = np.where(l <= TARGET)[0]
        if not len(ok):
            lo, hi = hi, int(hi*1.3) + 10; continue
        k = int(ok[0])
        if k == 0:
            if grid[0] <= lo + 1:
                hi = int(grid[0]); break
            lo = max(0, int(grid[0]) - (int(grid[1]) - int(grid[0]))); hi = int(grid[0]); continue
        lo, hi = int(grid[k-1]), int(grid[k])
        if hi - lo <= max(2, int(0.01*hi)):
            break
    D = hi; step = max(1, int(0.01*D))
    while True:
        cand = np.array([D, D + step, D + 2*step, D + 3*step])
        l = lole(cand, CONFIRM)
        log.append(dict(stage='A', grid=cand.tolist(), lole=l.round(4).tolist(), reps=CONFIRM))
        ok = np.where(l <= TARGET)[0]
        if len(ok):
            return int(cand[ok[0]]), float(l[ok[0]])
        D = int(cand[-1]) + step


def gas_grid(c, D):
    """Candidate gas sets around the gensets: (nCC, nCT, nA35)."""
    pk = c.peak
    # generous: islanded fleets run 1.1-2.6x peak, and small campuses are all aeros
    nCC = range(0, int(np.ceil(1.3*pk/627.0)) + 2)
    nCT = range(0, int(np.ceil(1.9*pk/419.0)) + 2)
    nA = range(0, (min(int(np.ceil(2.2*pk/35.0)) + 3, 60) if pk < 1500 else 7))
    N = []
    for a, b, d in product(nCC, nCT, nA):
        n = np.zeros(6, np.int64); n[iCC], n[iCT], n[iA35], n[iDSL] = a, b, d, D
        N.append(n)
    return np.array(N)


def optimise_gas(c, D, log):
    """Least-cost gas set given D gensets, full load, published VRE; LOLE <= TARGET."""
    N = gas_grid(c, D)
    cost = np.array([c.cost(n)['total'] for n in N])
    # cheap pre-screen: expected available capacity must at least cover the peak net load
    avail = np.array([(1 - u['q'])*(1 - u['poh']/L.og.H) for u in L.og.CAT])
    firm = (N*L.og.S_MW*avail).sum(1)
    netpk = float(np.maximum(c.net(), 0).max())
    keep = firm >= 0.98*netpk
    N, cost = N[keep], cost[keep]
    order = np.argsort(cost)
    N, cost = N[order], cost[order]
    # screen in batches of the cheapest first; stop at the first batch with a feasible design
    B = 256
    feas = []
    for a in range(0, len(N), B):
        l = c.lole_many(N[a:a + B], SCREEN)
        ok = np.where(l <= TARGET)[0]
        log.append(dict(stage='B', batch=[a, min(a + B, len(N))], n_feasible=int(len(ok)), reps=SCREEN))
        if len(ok):
            feas = [(int(a + k), float(l[k])) for k in ok]
            break
    if not feas:
        raise RuntimeError(f'{c.site}: no feasible gas set at D={D}')
    # confirm in cost order, four at a time
    idx = [k for k, _ in feas]
    for a in range(0, len(idx), 4):
        cand = N[idx[a:a + 4]]
        l = c.lole_many(cand, CONFIRM)
        log.append(dict(stage='B', confirm=[N[i].tolist() for i in idx[a:a + 4]], lole=l.round(4).tolist(), reps=CONFIRM))
        ok = np.where(l <= TARGET)[0]
        if len(ok):
            return cand[ok[0]], float(l[ok[0]])
    # none of the screened-feasible confirmed: step up the CT count on the cheapest
    n = N[idx[0]].copy()
    while True:
        n[iCT] += 1
        l = float(c.lole_many(n[None, :], CONFIRM)[0])
        log.append(dict(stage='B', stepup=n.tolist(), lole=l, reps=CONFIRM))
        if l <= TARGET:
            return n, l


def run_site(args):
    site_id, shares = args
    L.prepare()
    d, meta = L.load_sites(f'{LOAD_DIR}/dc_load_hourly_MW.csv', f'{LOAD_DIR}/site_parameters_and_results.csv')
    des = {r['site']: r for r in json.load(open(DESIGN))}
    r = meta[meta.site_id == site_id].iloc[0]
    c = L.Campus(site_id, r.zone, d[site_id].values, des[site_id],
                 L.og.zonal_solar_cf(CASE, r.zone), L.om.zonal_wind_cf(CASE, r.zone), r.mw_published)
    if not VRE:                                   # strip the campus renewables and battery
        c.s = c.w = c.b = 0.0
    VR_DSL = float(c.VR[iDSL])
    out = dict(site=site_id, rule='diesel-first', vre_kept=VRE, zone=r.zone, peak_MW=c.peak, energy_MWh=c.energy,
               mw_published=c.mw_published, gas_price=c.gas, diesel=L.DIESEL, diesel_fuel=L.DSL_FUEL,
               var_cost=c.VR.tolist(), n0=c.n0.tolist(), solar_MW=c.s, wind_MW=c.w, batt_MW=c.b, cases={})
    n6_0 = np.append(c.n0, 0)
    t = time.time()
    lole0 = float(c.lole_many(n6_0[None, :], CONFIRM)[0])
    out['cases']['0.0'] = dict(share=0.0, n=n6_0.tolist(), removed_MW=0.0, diesel_units=0, LOLE=lole0,
                               installed_MW=float(n6_0 @ L.og.S_MW), **c.cost(n6_0), seconds=time.time() - t,
                               phaseA=dict(load_fraction=0.0, diesel_units=0, LOLE=None, energy_MWh=0.0,
                                           fixed_diesel=0.0, fuel_diesel=0.0))
    if not VRE:
        out['published_with_vre'] = dict(n=n6_0.tolist(), solar_MW=des[site_id]['solar_MW'],
                                         wind_MW=des[site_id]['wind_MW'], batt_MW=des[site_id]['batt_MW'],
                                         lcoe=des[site_id]['lcoe'], LOLE=des[site_id]['LOLE'])
        out['cases']['0.0']['note'] = 'published gas machines, renewables/battery removed (not re-sized)'
        t = time.time(); log = []
        n6, loleB = optimise_gas(c, 0, log)
        out['cases']['0.0_gasopt'] = dict(share=0.0, n=n6.tolist(), removed_MW=0.0, diesel_units=0, LOLE=loleB,
                                          installed_MW=float(n6 @ L.og.S_MW), search=log, seconds=time.time() - t,
                                          **c.cost(n6), note='gas-only, re-optimised on the CC/CT/A35 grid')
        mix = ' '.join(f'{L.og.CAT[i]["k"]}x{n6[i]}' for i in range(5) if n6[i])
        print(f'{site_id} x=0.00 gas-only re-optimised: {mix:<28} ratio {n6 @ L.og.S_MW/c.peak:.2f}  LOLE {loleB:.3f}  '
              f'${out["cases"]["0.0_gasopt"]["lcoe"]:.2f}/MWh (published with VRE ${des[site_id]["lcoe"]:.2f})', flush=True)
    else:
        print(f'{site_id} x=0.00  LOLE {lole0:.3f}  ${out["cases"]["0.0"]["lcoe"]:.2f}/MWh', flush=True)
    for x in shares:
        t = time.time(); log = []
        D, loleA = size_diesel_alone(c, x, log)
        n6, loleB = optimise_gas(c, D, log)          # also at x = 1: gas is still added for ENERGY
                                                     # (CC $35/MWh vs diesel $219) if it pays
        rec = dict(share=x, n=n6.tolist(), removed_MW=float((c.n0 - n6[:5]) @ L.og.S_MW[:5]),
                   diesel_units=int(D), LOLE=loleB, installed_MW=float(n6 @ L.og.S_MW), search=log,
                   seconds=time.time() - t, **c.cost(n6))
        eA = x*c.energy
        rec['phaseA'] = dict(load_fraction=x, diesel_units=int(D), LOLE=loleA, energy_MWh=eA,
                             fixed_diesel=float(D*L.og.S_MW[iDSL]*L.og.FX_MW[iDSL]), fuel_diesel=eA*VR_DSL,
                             lcoe=(D*L.og.S_MW[iDSL]*L.og.FX_MW[iDSL] + eA*VR_DSL)/eA)
        rec['diesel_share_installed'] = rec['diesel_MW']/rec['installed_MW']
        out['cases'][f'{x:.1f}'] = rec
        mix = ' '.join(f'{L.og.CAT[i]["k"]}x{n6[i]}' for i in range(5) if n6[i])
        print(f'{site_id} x={x:.2f}  A: {D:5d} gensets (LOLE {loleA:.3f}, ${rec["phaseA"]["lcoe"]:.0f}/MWh)  '
              f'B: {mix:<28} gas {rec["gas_MW"]:6,.0f} MW  ratio {rec["installed_MW"]/c.peak:.2f}  '
              f'LOLE {loleB:.3f}  ${rec["lcoe"]:.2f}/MWh  ({rec["seconds"]:.0f}s)', flush=True)
        json.dump(out, open(f'{OUT}/site_{site_id}.json', 'w'), indent=1, default=float)
    return site_id


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    shares = [float(s) for s in sys.argv[2].split(',')] if len(sys.argv) > 2 else \
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    des = json.load(open(DESIGN))
    sites = [r['site'] for r in des] if which == 'all' else which.split(',')
    sites = [s for s in sites if not os.path.exists(f'{OUT}/site_{s}.json')
             or len(json.load(open(f'{OUT}/site_{s}.json'))['cases']) < len(shares) + 1 + (0 if VRE else 1)]
    peak = {r['site']: r['peak_MW'] for r in des}
    sites.sort(key=lambda s: -peak[s])
    with Pool(int(os.environ.get('WORKERS', 2))) as p:
        for s in p.imap_unordered(run_site, [(s, shares) for s in sites]):
            print(f'== {s} done', flush=True)
    allr = [json.load(open(f'{OUT}/site_{r["site"]}.json')) for r in des
            if os.path.exists(f'{OUT}/site_{r["site"]}.json')]
    json.dump(allr, open(f'{OUT}/all_sites.json', 'w'), indent=1, default=float)
