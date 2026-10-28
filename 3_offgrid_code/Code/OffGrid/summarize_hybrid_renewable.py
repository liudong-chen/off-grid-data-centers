"""Pure-renewable hybrid: value against waiting for the grid, after the HOPE run.

Needs Code/HOPE/ModelCases/TX8z_GTEP_hybrid_ren4h_vre_case/output (build_hybrid_renewable_case.py, then HOPE).
Accounting is that of Fig. 3 (Figures/fig123_package/02_code/73_fig123.py): monthly flows, 8% to day 0,
20 years after month 24.  The renewable plant is built by month 24 (same as gas), runs islanded
until the grid arrives at month T, then connects:
    islanded cost   = renewable_only_4h_summary.json cost_Byr                      (months 24 .. T)
    connected cost  = (HOPE obj, renewable hybrid case - HOPE obj, base case)/E  +  plant's own annual cost/E
                      (the plant enters HOPE as existing capacity; its sunk capital is added here)
    waiting         = on-grid cost from month T (c_on of hybrid_diesel_summary.json)
    python3 summarize_hybrid_renewable.py  ->  out/hybrid_renewable_summary.json
"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); BYONG = os.path.abspath(os.path.join(HERE, '..', '..'))
MC = f'{BYONG}/Code/HOPE/ModelCases'
HS = json.load(open(f'{BYONG}/Figures/fig123_package/01_raw/hybrid_diesel_summary.json')); P = HS['params']
REN = json.load(open(f'{HERE}/scenario_renewable_only/renewable_only_4h_summary.json'))
E, C_ON, MW, R, T_GAS = REN['energy_MWh'], P['c_on'], P['MW_pub'], P['R'], P['T_GAS']
END = T_GAS + 240; mo = np.arange(END + 1); disc = (1 + R/12)**(-mo.astype(float))


def hope_obj(case):
    for d in (f'{MC}/{case}/output/postprocess_snapshot/metadata.yml', f'{MC}/{case}/output/metadata.yml'):
        if os.path.exists(d):
            for line in open(d):
                if line.strip().startswith('objective_value'):
                    return float(line.split(':')[1])
    return None


o_ren, o_base = hope_obj('TX8z_GTEP_hybrid_ren4h_vre_case'), hope_obj('TX8z_GTEP_base_vre_case')
if o_ren is None or o_base is None:
    raise SystemExit('HOPE output missing: run TX8z_GTEP_hybrid_ren4h_vre_case (and base) first')
K = REN['cost_Byr']*1e9                                  # renewable plant: capital + FOM + land, no fuel
c_sys = (o_ren - o_base)/E
c_con = c_sys + K/E
c_isl = K/E


def net(rate, T, build):
    earn = np.zeros(END + 1); cost = np.zeros(END + 1)
    if build:
        a = (mo > T_GAS) & (mo <= T); earn[a] = rate*MW; cost[a] = c_isl*E/12
        b = mo > max(T, T_GAS); earn[b] = rate*MW; cost[b] = c_con*E/12
    else:
        b = mo > T; earn[b] = rate*MW; cost[b] = C_ON*E/12
    return float(((earn - cost)*disc).sum())/1e9/20


out = dict(hope_obj_ren=o_ren, hope_obj_base=o_base, c_on=C_ON, c_islanded=c_isl, c_system_connected=c_sys,
           c_connected_incl_sunk=c_con, by_rate={})
for rate in (200e3, 300e3, 400e3, 500e3, 600e3):
    d = {T: net(rate, T, True) - net(rate, T, False) for T in range(0, 241)}
    pos = [T for T, v in d.items() if v >= 0 and T >= T_GAS]
    out['by_rate'][int(rate)] = dict(bound_months=min(pos) if pos else None,
                                     value_vs_wait_B_per_yr={T: d[T] for T in (36, 50, 72)})
    print(int(rate), 'bound', out['by_rate'][int(rate)]['bound_months'], {T: round(d[T], 1) for T in (36, 50, 72)})
print(f'islanded ${c_isl:.1f}/MWh; connected: system {c_sys:.1f} + sunk {K/E:.1f} = {c_con:.1f} $/MWh (grid {C_ON:.1f})')
os.makedirs(f'{HERE}/out', exist_ok=True)
json.dump(out, open(f'{HERE}/out/hybrid_renewable_summary.json', 'w'), indent=1)
