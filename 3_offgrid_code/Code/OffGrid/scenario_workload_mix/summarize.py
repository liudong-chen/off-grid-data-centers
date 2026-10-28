"""Workload-mix scenario: summary table. Needs out/<mix>/offgrid_mip_gasonly.json,
out/<mix>/offgrid_mip_pool_gasonly.json (run_all.sh) and, for the grid columns, the solved HOPE cases
TX8z_GTEP_dc_wl_<mix>_case (build_hope_cases.py, then HOPE).  python3 summarize.py -> summary.json, summary.csv"""
import json, os
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); B = os.path.abspath(os.path.join(H, '..', '..', '..'))
MC = f'{B}/Code/HOPE/ModelCases'; WN = f'{B}/Figures'
P = json.load(open(f'{WN}/fig123_package/01_raw/hybrid_diesel_summary.json'))['params']
R, MWP, TG = P['R'], P['MW_pub'], P['T_GAS']; END = TG + 240
mo = np.arange(END + 1); disc = (1 + R/12)**(-mo.astype(float))


def hope(case):
    m = f'{MC}/{case}/output/postprocess_snapshot/metadata.yml'
    if not os.path.exists(m):
        return None, None
    obj = next(float(l.split(':')[1]) for l in open(m) if l.strip().startswith('objective_value'))
    c = pd.read_csv(f'{MC}/{case}/output/capacity.csv'); n = c[c.New_Build == 1]
    return obj, float((n['Capacity_FIN (MW)'] - n['Capacity_INI (MW)']).clip(lower=0).sum())


def bound(c_off, c_on, E, rate=400e3):
    """Queue (months) above which gas-only off-grid supply, kept for the horizon, beats waiting;
    linear interpolation between whole months, as in the paper's bound."""
    def net(T):
        e = np.zeros(END + 1); c = np.zeros(END + 1); ew = np.zeros(END + 1); cw = np.zeros(END + 1)
        e[mo > TG] = rate*MWP; c[mo > TG] = c_off*E/12
        ew[mo > T] = rate*MWP; cw[mo > T] = c_on*E/12
        return ((e - c) - (ew - cw)) @ disc
    prev = net(0)
    for T in range(1, 241):
        cur = net(T)
        if prev < 0 <= cur:
            return round(T - 1 + (-prev)/(cur - prev), 1)
        prev = cur
    return None


obj_base, _ = hope('TX8z_GTEP_base_vre_case')
rows = []
for mix, case, d in [('base (archetype mix)', 'TX8z_GTEP_dc_vre_case', None),
                     ('training', 'TX8z_GTEP_dc_wl_training_case', 'training'),
                     ('inference', 'TX8z_GTEP_dc_wl_inference_case', 'inference'),
                     ('colocation', 'TX8z_GTEP_dc_wl_colocation_case', 'colocation')]:
    if d is None:
        S = json.load(open(f'{WN}/fig123_package/01_raw/offgrid_v4_sites.json'))
        Pl = json.load(open(f'{WN}/fig123_package/01_raw/offgrid_v4_pool.json'))['FarWest']
        prof = pd.read_csv(f'{B}/DC_Load/load_profiles/dc_load_hourly_MW.csv', index_col=0)
    else:
        f = f'{H}/out/{d}/offgrid_mip_gasonly.json'
        if not os.path.exists(f):
            print('missing', f); continue
        S = json.load(open(f))
        pf = f'{H}/out/{d}/offgrid_mip_pool_gasonly.json'
        Pl = json.load(open(pf))['FarWest'] if os.path.exists(pf) else None
        prof = pd.read_csv(f'{H}/profiles/{d}/dc_load_hourly_MW.csv', index_col=0)
    E = sum(s['energy_MWh'] for s in S); A = sum(s['installed_MW'] for s in S)
    sumpk = sum(s['peak_MW'] for s in S); comb = float(prof['TOTAL_MW'].max())
    c_off = sum(s['cost'] for s in S)/E
    obj, C = hope(case)
    c_on = (obj - obj_base)/E if obj is not None else None
    rec = dict(mix=mix, energy_TWh=E/1e6, sum_peaks_MW=sumpk, combined_peak_MW=comb, load_factor=E/8760/comb,
               offgrid_MW=A, offgrid_ratio_to_peaks=A/sumpk, offgrid_cost=c_off, offgrid_util=E/(A*8760),
               pooled_MW=Pl['installed_MW'] if Pl else None, grid_new_MW=C, grid_cost=c_on)
    if C:
        rec.update(capacity_ratio=A/C, premium=c_off/c_on - 1, bound_months_400k=bound(c_off, c_on, E))
        if Pl:
            r = Pl['installed_MW']/comb
            rec.update(reliability_MW=A - r*sumpk, noncoincident_MW=r*(sumpk - comb), existing_grid_MW=Pl['installed_MW'] - C)
    rows.append(rec)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()})
json.dump(rows, open(f'{H}/summary.json', 'w'), indent=1)
pd.DataFrame(rows).to_csv(f'{H}/summary.csv', index=False)
