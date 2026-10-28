"""Re-run the last two stages of the method (polish_confirm: single-unit removals/swaps,
solar/wind/battery steps incl. drop/halve; 2,000-replication confirmation) on every design in a
results file, in that case's cost context. Used so that files produced before the polish gained
the drop/halve moves carry exactly the same final stages as the rest.

    python3 repolish.py ercot|georgia|2035_k0p5|2035_k3 <sites.json> [<pool.json>]
"""
import os, sys, json, shutil, time
import numpy as np, pandas as pd
import offgrid_mip as om
CASE, SITES = sys.argv[1], sys.argv[2]; POOL = sys.argv[3] if len(sys.argv) > 3 else None
CD = os.environ['OFFGRID_CASE']; R = os.environ['OFFGRID_LOAD']; HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(f'{R}/dc_load_hourly_MW.csv', index_col=0, parse_dates=True)
d = d[~((d.index.month == 2) & (d.index.day == 29))].drop(columns=['TOTAL_MW'])
meta = pd.read_csv(f'{R}/site_parameters_and_results.csv')
ZM = {'Far West': 'FarWest', 'North Central': 'NorthCentral', 'South Central': 'SouthCentral'}
meta['zone'] = meta.ercot_weather_zone.map(lambda z: ZM.get(z, z))


def zonal_cf(zone, typ, up=1.0):
    g = pd.read_csv(f'{CD}/gendata.csv'); c = pd.read_csv(f'{CD}/gendata_candidate.csv')
    k = c.index[(c.Zone == zone) & (c.Type == typ)]
    return None if not len(k) else pd.read_csv(f'{CD}/gen_availability_timeseries.csv', usecols=[f'G{len(g)+int(k[0])+1}']).values[:om.H, 0]*up


LAND = {}
if CASE == 'ercot':
    om.anchor_capex(); GAS = json.load(open(f'{HERE}/zonal_gas_2030.json')); up = 1.0
    loads = {s: d[s].values[:om.H] for s in meta.site_id}
elif CASE.startswith('2035'):
    om.anchor_capex({'CC': 113755.0, 'CT': 96576.0}); up = 1.03
    om.WIND_FIXED, om.SOLAR_FIXED, om.BATT_FIXED = 149412.0, 108393.0, om.BATT_FIXED*108393.0/117995.0
    GAS = {'Coast': 5.0, 'East': 4.866, 'FarWest': 3.997, 'NorthCentral': 5.187, 'South': 5.348, 'SouthCentral': 5.044, 'West': 4.501, 'North': 4.168}
    Kf = 0.5 if CASE.endswith('k0p5') else 3.0; loads = {s: d[s].values[:om.H]*Kf for s in meta.site_id}
else:
    om.anchor_capex(); up = 1.0; GAS = {'Georgia': 5.688}; LAND = {'Georgia': 4670.0}
    pk = d.max(); shape = (pk/pk.sum()).sort_values(ascending=False); src = list(shape.index)
    q = np.linspace(0, 1, 30); w = np.interp(q, np.linspace(0, 1, len(src)), [shape[x] for x in src]); w = w/w.sum(); F = 100.0/9200
    for _ in range(50):
        lo = w < F
        if not lo.any():
            break
        w = np.where(lo, F, w*(1 - lo.sum()*F)/w[~lo].sum())
    ar = [src[min(int(round(x*(len(src)-1))), len(src)-1)] for x in q]
    sizes = [('GA01_Camellia', src[0], 3200.0)] + [(f'GA{i+2:02d}', ar[i], float(9200*w[i])) for i in range(30)]
    loads = {n: d[s].values[:om.H]*(mw/d[s].values[:om.H].max()) for n, s, mw in sizes}
CF_S = {z: zonal_cf(z, 'SolarPV', up) for z in GAS} if CASE != 'georgia' else {}
CF_W = {z: zonal_cf(z, 'WindOn', up) for z in GAS} if CASE != 'georgia' else {'Georgia': None}
if CASE == 'georgia':
    scf = zonal_cf('Coast', 'SolarPV'); CF_S['Georgia'] = scf*(0.265*0.88/scf.mean())
K = ['Recip', 'Aero35', 'Aero48', 'FrameCT', 'CC']


def redo(rec, load):
    z = rec['zone']; site = om.Site(load, z, CF_S.get(z), CF_W.get(z), gas=GAS[z], land=LAND.get(z))
    allow = (True, CF_W.get(z) is not None, True)
    n0 = np.array([rec['mix'].get(k, 0) for k in K]); t = time.time()
    n, s, w, b, Lc, C = om.polish_confirm(site, n0, rec['solar_MW'], rec['wind_MW'], rec['batt_MW'], allow=allow)
    inst = float(n @ om.S_MW)
    vre_fixed = s*om.SOLAR_FIXED + w*om.WIND_FIXED + b*om.BATT_FIXED
    vre_land = (s*om.ACRES['solar'] + w*om.ACRES['wind'] + b*om.ACRES['storage'])*site.land*om.og.LAND_CRF
    new = dict(rec, n=[int(v) for v in n], mix={K[i]: int(n[i]) for i in range(5) if n[i]}, unit_count=int(n.sum()),
               installed_MW=inst, ratio=inst/site.peak, solar_MW=s, wind_MW=w, batt_MW=b, LOLE=Lc, cost=C['total'],
               fixed_cost=C['fixed'] - vre_fixed, land_cost=C['land'] - vre_land, vre_cost=vre_fixed + vre_land,
               fuel_cost=C['fuel'], acres=C['acres'], lcoe=C['total']/site.energy, confirm_reps=2000, repolished=True)
    ch = '' if (new['mix'] == rec['mix'] and abs(s - rec['solar_MW']) + abs(w - rec['wind_MW']) + abs(b - rec['batt_MW']) < 1) else ' CHANGED'
    print(f"{rec.get('site', z):<14} {rec['mix']} s{rec['solar_MW']:.0f} w{rec['wind_MW']:.0f} b{rec['batt_MW']:.0f} ${rec['lcoe']:.2f} -> "
          f"{new['mix']} s{s:.0f} w{w:.0f} b{b:.0f} ${new['lcoe']:.2f} LOLE {Lc:.3f}{ch}  [{time.time()-t:.0f}s]", flush=True)
    return new


for path, is_pool in ((SITES, False), (POOL, True)):
    if path is None:
        continue
    data = json.load(open(path)); shutil.copy(path, path.replace('.json', '_prepolish.json'))
    tot = np.sum(list(loads.values()), axis=0)
    if is_pool:
        data = {z: redo(r, tot) for z, r in data.items()}
    else:
        data = [redo(r, loads[r['site']]) for r in data]
    json.dump(data, open(path, 'w'), indent=1, default=float); print(f'written {path}\n', flush=True)
