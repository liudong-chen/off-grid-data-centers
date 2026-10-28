"""Scenario: OFF-GRID, PURE RENEWABLE.

Each of the 17 ERCOT campuses is served by solar, onshore wind and Li-ion batteries only,
no gas and no diesel, in its own weather zone. Least-cost design by a linear program over all
8,760 hours of the 2025 weather year: every hour's load must be met in full (no unserved
energy), which is the deterministic equivalent of LOLE <= 0.1 d/yr for a single weather year.
Battery power (MW) and energy (MWh) are sized separately, so the duration is free.

Costs (2025$, same basis as the paper):
  solar  $117,995/MW-yr, wind $157,009/MW-yr  (HOPE TX8z candidate file)
  battery power $30,048/MW-yr + energy $34,404/MWh-yr, the linear split that reproduces the
          2-h ($98,856) and 6-h ($236,472) costs in 1_source_data/parameter_setting/New_parameter.xlsx
          (4-h: $167,664 vs $161,512 in the catalogue)
  charging efficiency 0.85, discharging 1.0 (as HOPE storagedata)
  land: solar 5.0, wind 1.0, storage 0.05, campus 0.75 acre/MW x zonal price x 8%
Profiles: zonal candidate solar and wind availability, HOPE TX8z_GTEP_dc_vre case.
Not modelled (all make the cost here a LOWER bound): other weather years, battery outages and
degradation, inverter/interconnection limits, land availability.

    OFFGRID_CASE=... OFFGRID_LOAD=... python3 run_renewable_only.py   -> renewable_only_sites.json, renewable_only_summary.json
    OFFGRID_BATT_HOURS=4 ... python3 run_renewable_only.py           -> renewable_only_4h_sites.json, renewable_only_4h_summary.json
        (: batteries fixed at 4 hours, E = 4P, at the catalogue's 4-hour cost of $161,512/MW-yr,
         the battery the grid builds; used for the renewable hybrid HOPE case)
"""
import os, json, time
import numpy as np, pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack
import offgrid_mip as om

CD, R = os.environ['OFFGRID_CASE'], os.environ['OFFGRID_LOAD']
H = 8760
SOLAR, WIND = 117995.0, 157009.0
BP, BE = 30048.0, 34404.0
ETA_C, ETA_D = 0.85, 1.0
AC = dict(solar=5.0, wind=1.0, storage=0.05, dc=0.0)   # data center land removed 
LCRF = 0.08
HOURS = float(os.environ.get('OFFGRID_BATT_HOURS', 0) or 0)     # 0 = free duration
BATT4H = 161512.0                                             # $/MW-yr, 4-h battery, unit catalogue / HOPE candidate
TAG = f'renewable_only_{int(HOURS)}h' if HOURS else 'renewable_only'
ZM = {'Far West': 'FarWest', 'North Central': 'NorthCentral', 'South Central': 'SouthCentral'}
d = pd.read_csv(f'{R}/dc_load_hourly_MW.csv', index_col=0, parse_dates=True)
d = d[~((d.index.month == 2) & (d.index.day == 29))]
meta = pd.read_csv(f'{R}/site_parameters_and_results.csv')
meta['zone'] = meta.ercot_weather_zone.map(lambda z: ZM.get(z, z))


def design(L, cfs, cfw, land):
    T = len(L)
    # variables: s, w, P, E, ch[T], dis[T], soc[T]
    iS, iW, iP, iE = 0, 1, 2, 3
    ich, idis, isoc = 4, 4 + T, 4 + 2*T
    nv = 4 + 3*T
    c = np.zeros(nv)
    c[iS] = SOLAR + AC['solar']*land*LCRF
    c[iW] = WIND + AC['wind']*land*LCRF
    c[iP] = (BATT4H if HOURS else BP) + AC['storage']*land*LCRF
    c[iE] = 0.0 if HOURS else BE
    t = np.arange(T)
    rows, cols, vals, b = [], [], [], []
    def blk(r0, entries):
        for (cc, vv) in entries:
            rows.append(r0); cols.append(cc); vals.append(vv)
    A_ub_parts, b_ub = [], []
    # (1) balance: -(s cfs + w cfw + dis - ch) <= -L
    r = np.repeat(t, 4); cl = np.column_stack([np.full(T, iS), np.full(T, iW), idis + t, ich + t]).ravel()
    vl = np.column_stack([-cfs, -cfw, -np.ones(T), np.ones(T)]).ravel()
    A1 = coo_matrix((vl, (r, cl)), shape=(T, nv)); b1 = -L
    # (2) ch_t - P <= 0, dis_t - P <= 0, soc_t - E <= 0
    def lim(i0, ib):
        return coo_matrix((np.r_[np.ones(T), -np.ones(T)], (np.r_[t, t], np.r_[i0 + t, np.full(T, ib)])), shape=(T, nv))
    A_ub = vstack([A1, lim(ich, iP), lim(idis, iP), lim(isoc, iE)]).tocsr()
    b_ub = np.r_[b1, np.zeros(3*T)]
    # (3) soc_t - soc_{t-1} - eta_c ch_t + dis_t/eta_d = 0, cyclic
    prev = (t - 1) % T
    A_eq = coo_matrix((np.r_[np.ones(T), -np.ones(T), -ETA_C*np.ones(T), np.ones(T)/ETA_D],
                       (np.r_[t, t, t, t], np.r_[isoc + t, isoc + prev, ich + t, idis + t])), shape=(T, nv)).tocsr()
    b_eq = np.zeros(T)
    if HOURS:                                     # fixed duration: E - HOURS*P = 0
        A_eq = vstack([A_eq, coo_matrix(([1.0, -HOURS], ([0, 0], [iE, iP])), shape=(1, nv))]).tocsr()
        b_eq = np.zeros(T + 1)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)]*nv, method='highs')
    assert res.status == 0, res.message
    x = res.x
    s, w, P, E = x[iS], x[iW], x[iP], x[iE]
    gen = s*cfs + w*cfw
    curt = np.maximum(gen + x[idis:idis+T] - x[ich:ich+T] - L, 0).sum()
    return dict(solar_MW=s, wind_MW=w, batt_MW=P, batt_MWh=E, batt_hours=E/max(P, 1e-9),
                cost_gen_storage=res.fun, curtailed_MWh=curt, generated_MWh=gen.sum(),
                discharged_MWh=x[idis:idis+T].sum())


rows = []
for r in meta.itertuples():
    z = r.zone; t0 = time.time()
    L = d[r.site_id].values[:H].astype(float)
    cfs, cfw = om.og.zonal_solar_cf(CD, z), om.zonal_wind_cf(CD, z)
    land = om.og.LAND[z]
    o = design(L, cfs, cfw, land)
    campus_land = AC['dc']*L.max()*land*LCRF
    total = o['cost_gen_storage'] + campus_land
    acres = o['solar_MW']*AC['solar'] + o['wind_MW']*AC['wind'] + o['batt_MW']*AC['storage'] + L.max()*AC['dc']
    o.update(site=r.site_id, zone=z, peak_MW=L.max(), energy_MWh=L.sum(), land_cost=campus_land + (acres - L.max()*AC['dc'])*land*LCRF,
             total_cost=total, lcoe=total/L.sum(), installed_MW=o['solar_MW'] + o['wind_MW'] + o['batt_MW'], acres=acres,
             curtailment_share=o['curtailed_MWh']/o['generated_MWh'])
    rows.append(o)
    print(f"{r.site_id:<5}{z:<13}{L.max():>7,.0f} MW  solar {o['solar_MW']:>8,.0f}  wind {o['wind_MW']:>8,.0f}  "
          f"batt {o['batt_MW']:>7,.0f} MW/{o['batt_MWh']:>9,.0f} MWh ({o['batt_hours']:.0f} h)  curtail {o['curtailment_share']:.0%}  "
          f"${o['lcoe']:.2f}/MWh  [{time.time()-t0:.0f}s]", flush=True)
    json.dump(rows, open(f'{TAG}_sites.json', 'w'), indent=1, default=float)
df = pd.DataFrame(rows)
summ = dict(campuses=len(df), peak_MW=df.peak_MW.sum(), energy_MWh=df.energy_MWh.sum(), cost_Byr=df.total_cost.sum()/1e9,
            lcoe=df.total_cost.sum()/df.energy_MWh.sum(), solar_MW=df.solar_MW.sum(), wind_MW=df.wind_MW.sum(),
            batt_MW=df.batt_MW.sum(), batt_MWh=df.batt_MWh.sum(), acres=df.acres.sum(),
            curtailment_share=df.curtailed_MWh.sum()/df.generated_MWh.sum())
summ['batt_hours_fixed'] = HOURS or None
json.dump(summ, open(f'{TAG}_summary.json', 'w'), indent=1, default=float)
print(json.dumps(summ, indent=1, default=float))
