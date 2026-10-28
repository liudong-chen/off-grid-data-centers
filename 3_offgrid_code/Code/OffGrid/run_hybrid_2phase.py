"""Connection-AWARE hybrid: same year-one diesel as the diesel-first hybrid, but the gas is
chosen knowing the data center connects to the grid at month T_CON.

    python3 run_hybrid_2phase.py [S01,... | all] [shares]      default 0.0,0.1,...,1.0

Off-grid (run_hybrid_diesel_first.py) picks the gas set that minimises the ANNUAL islanded
cost -- i.e. gas capital AND diesel/gas fuel are paid every year for 20 years.
Here the gas is used islanded only from T_GAS to T_CON; afterwards the grid takes over,
the diesel retires and the data center's gas plant earns only its value to the grid.
So each candidate gas set n is ranked by its present value over the 20-yr horizon:

    PV(n) = W_ALL * (gas capital + land)               # owned for the whole horizon
          + W_ISL * (fuel_gas + fuel_diesel)            # own fuel only until connection
          - W_POST * sum_k credit_k * MW_k(n)           # grid value of the gas after connection

W_ISL  = discounted years in months (T_GAS, T_CON]   (~2.0 at 8 %)
W_POST = discounted years in months (T_CON, END]
W_ALL  = W_ISL + W_POST
credit = avoided ERCOT system cost per MW-yr, fitted on the 11 HOPE hybrid runs
         (CC ~ $72k, peakers ~ $4k, diesel 0; env CREDIT_CC / CREDIT_PK to change).
Diesel D_x is taken unchanged from the diesel-first run (out/hybrid_diesel/all_sites.json),
so year one is identical and only the gas differs.  Same LOLE <= 0.1 screen + confirm.

Writes out/hybrid_2phase/site_<id>.json (same schema as the diesel-first run, plus
'pv_2phase' per share) and all_sites.json.  Resumable per site.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import hybrid_diesel_lib as L
import run_hybrid_diesel_first as F

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get('HYB_SRC', f'{HERE}/out/hybrid_diesel/all_sites.json')
OUT = os.environ.get('BYONG_OUT2', os.path.join(HERE, 'out', 'hybrid_2phase')); os.makedirs(OUT, exist_ok=True)
T_GAS, T_CON, END = 24, int(os.environ.get('T_CON', 50)), 24 + 240
r_m = 1.08**(1/12) - 1
mo = np.arange(END + 1); disc = (1 + r_m)**(-mo.astype(float))
W_ISL = disc[(mo > T_GAS) & (mo <= T_CON)].sum()/12
W_POST = disc[mo > T_CON].sum()/12
W_ALL = W_ISL + W_POST
CREDIT = np.array([float(os.environ.get('CREDIT_PK', 3861.0))]*4 + [float(os.environ.get('CREDIT_CC', 71947.0)), 0.0])
NG = 5


def pv2(c, n):
    k = c.cost(n)
    cap = k['fixed_gas'] + k['land']
    fuel = k['fuel_gas'] + k['fuel_diesel']
    cred = float((n*L.og.S_MW*CREDIT).sum())
    return W_ALL*cap + W_ISL*fuel - W_POST*cred, k


def optimise_gas_2p(c, D, log):
    """Least-PV gas set given D gensets (connection-aware ranking); LOLE <= TARGET islanded."""
    N = F.gas_grid(c, D)
    avail = np.array([(1 - u['q'])*(1 - u['poh']/L.og.H) for u in L.og.CAT])
    firm = (N*L.og.S_MW*avail).sum(1)
    netpk = float(np.maximum(c.net(), 0).max())
    N = N[firm >= 0.98*netpk]
    cost = np.array([pv2(c, n)[0] for n in N])
    order = np.argsort(cost); N, cost = N[order], cost[order]
    B = 256; feas = []
    for a in range(0, len(N), B):
        l = c.lole_many(N[a:a + B], F.SCREEN)
        ok = np.where(l <= F.TARGET)[0]
        log.append(dict(stage='B2', batch=[a, min(a + B, len(N))], n_feasible=int(len(ok)), reps=F.SCREEN))
        if len(ok):
            feas = [int(a + k) for k in ok]; break
    if not feas:
        raise RuntimeError(f'{c.site}: no feasible gas set at D={D}')
    for a in range(0, len(feas), 4):
        cand = N[feas[a:a + 4]]
        l = c.lole_many(cand, F.CONFIRM)
        log.append(dict(stage='B2', confirm=[x.tolist() for x in cand], lole=l.round(4).tolist(), reps=F.CONFIRM))
        ok = np.where(l <= F.TARGET)[0]
        if len(ok):
            return cand[ok[0]], float(l[ok[0]])
    n = N[feas[0]].copy()
    while True:
        n[F.iCT] += 1
        l = float(c.lole_many(n[None, :], F.CONFIRM)[0])
        log.append(dict(stage='B2', stepup=n.tolist(), lole=l, reps=F.CONFIRM))
        if l <= F.TARGET:
            return n, l


def run_site(args):
    site_id, shares = args
    L.prepare()
    src = {s['site']: s for s in json.load(open(SRC))}[site_id]
    d, meta = L.load_sites(f'{F.LOAD_DIR}/dc_load_hourly_MW.csv', f'{F.LOAD_DIR}/site_parameters_and_results.csv')
    des = {r['site']: r for r in json.load(open(F.DESIGN))}
    r = meta[meta.site_id == site_id].iloc[0]
    c = L.Campus(site_id, r.zone, d[site_id].values, des[site_id],
                 L.og.zonal_solar_cf(F.CASE, r.zone), L.om.zonal_wind_cf(F.CASE, r.zone), r.mw_published)
    c.s = c.w = c.b = 0.0
    fn = f'{OUT}/site_{site_id}.json'
    out = json.load(open(fn)) if os.path.exists(fn) else \
        {k: v for k, v in src.items() if k != 'cases'} | dict(rule='connection-aware (two-phase)', T_CON=T_CON,
                                                           W_ISL=W_ISL, W_POST=W_POST, credit=CREDIT.tolist(), cases={})
    for x in shares:
        key = f'{x:.1f}'
        if key in out['cases']:
            continue
        s0 = src['cases']['0.0_gasopt' if x == 0 else key]
        D = int(s0['diesel_units']); t = time.time(); log = []
        n6, lole = optimise_gas_2p(c, D, log)
        pv, k = pv2(c, n6)
        pv_off = pv2(c, np.array(s0['n']))[0]
        if pv_off < pv:     # screening noise dropped the off-grid fleet (feasible at CONFIRM): keep it
            n6 = np.array(s0['n']); lole = s0['LOLE']; pv, k = pv2(c, n6)
            log.append(dict(stage='B2', fallback='off-grid fleet has lower PV'))
        rec = dict(share=x, n=n6.tolist(), diesel_units=D, LOLE=lole, installed_MW=float(n6 @ L.og.S_MW),
                   search=log, seconds=time.time() - t, **k, pv_2phase=pv, pv_2phase_offgrid_fleet=pv_off,
                   offgrid_n=s0['n'], offgrid_gas_MW=s0['gas_MW'])
        if x > 0:
            rec['phaseA'] = s0['phaseA']
        out['cases'][key] = rec
        mix = ' '.join(f'{L.og.CAT[i]["k"]}x{n6[i]}' for i in range(5) if n6[i])
        print(f'{site_id} x={x:.2f}  D {D:5d}  2-phase: {mix:<28} gas {k["gas_MW"]:6,.0f} MW (off-grid {s0["gas_MW"]:6,.0f})  '
              f'LOLE {lole:.3f}  PV saving ${(pv_off - pv)/1e6:,.0f}M  ({rec["seconds"]:.0f}s)', flush=True)
        json.dump(out, open(fn, 'w'), indent=1, default=float)
    return site_id


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    shares = [float(s) for s in sys.argv[2].split(',')] if len(sys.argv) > 2 else [round(0.1*i, 1) for i in range(11)]
    src = json.load(open(SRC))
    sites = [s['site'] for s in src] if which == 'all' else which.split(',')
    sites.sort(key=lambda s: -{a['site']: a['peak_MW'] for a in src}[s])
    print(f'W_ISL {W_ISL:.2f} yr, W_POST {W_POST:.2f} yr, credit {CREDIT.tolist()}', flush=True)
    with Pool(int(os.environ.get('WORKERS', 2))) as p:
        for s in p.imap_unordered(run_site, [(s, shares) for s in sites]):
            print(f'== {s} done', flush=True)
    allr = [json.load(open(f'{OUT}/site_{s["site"]}.json')) for s in src if os.path.exists(f'{OUT}/site_{s["site"]}.json')]
    json.dump(allr, open(f'{OUT}/all_sites.json', 'w'), indent=1, default=float)
