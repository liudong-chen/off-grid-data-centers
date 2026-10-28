"""Utilization of the data centers' own gas units after grid connection (hybrid supply).

Reads the solved hybrid HOPE cases: the campus units enter HOPE as existing capacity, appended
after the grid's own fleet in gendata.csv, and power.csv lists generation per generator in the same
order. Gas units are the campus rows typed NGCC/NGCT at gas variable cost; diesel gensets are the
campus NGCT rows at the diesel cost ($197/MWh) and are retired at connection (they dispatch ~0).
util = annual MWh of the campus gas units / (campus gas MW x 8760).
Writes Figures/fig123_package/01_raw/hybrid_connected_util.json (ERCOT 2030, x = 0 ... 1.0) and
Figures/fig5_package/01_raw/hybrid_connected_util_{ercot,georgia}.json (Fig. 5b labels).
"""
import json, os, sys
import pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
MC = f'{ROOT}/Code/HOPE/ModelCases'; WN = f'{ROOT}/Figures'
DIESEL_COST = 150.0        # $/MWh; campus rows above this are diesel gensets

def campus_util(case, base):
    dd = [d for d in os.listdir(f'{MC}/{case}') if d.startswith('Data_')][0]
    g = pd.read_csv(f'{MC}/{case}/{dd}/gendata.csv')
    n0 = len(pd.read_csv(f'{MC}/{base}/' + [d for d in os.listdir(f'{MC}/{base}') if d.startswith('Data_')][0] + '/gendata.csv'))
    p = pd.read_csv(f'{MC}/{case}/output/power.csv', usecols=['Technology', 'Zone', 'EC_Category', 'AnnSum'])
    pe = p[p.EC_Category == 'Existing'].reset_index(drop=True)
    assert len(pe) == len(g), (case, len(pe), len(g))
    camp = g.iloc[n0:].copy(); camp['MWh'] = pe.iloc[n0:].AnnSum.values
    gas = camp[camp.Type.isin(['NGCC', 'NGCT']) & (camp['Cost ($/MWh)'] < DIESEL_COST)]
    dsl = camp[camp['Cost ($/MWh)'] >= DIESEL_COST]
    mw, mwh = float(gas['Pmax (MW)'].sum()), float(gas.MWh.sum())
    return dict(gas_MWh=mwh, gas_MW=mw, util=mwh/(mw*8760), diesel_MW=float(dsl['Pmax (MW)'].sum()), diesel_MWh=float(dsl.MWh.sum()))

def run(cases, base, note):
    return dict(note=note, cases=', '.join(c for _, c in cases), table={x: campus_util(c, base) for x, c in cases})

ercot = run([(f'{i/10:.1f}', f'TX8z_GTEP_hybrid_dsl{i*10:03d}_vre_case') for i in range(11)], 'TX8z_GTEP_base_vre_case',
            "HOPE dispatch of the data centers' own gas after grid connection (diesel retired); util = MWh/(MW*8760)")
georgia = run([(f'{i/10:.1f}', f'GA1z_GTEP_hyb_dsl{i*10:03d}_case') for i in (0, 3, 6, 10)], 'GA1z_GTEP_base_case',
              'HOPE dispatch of the Georgia data centers own gas after connection')
for path, d in [(f'{WN}/fig123_package/01_raw/hybrid_connected_util.json', ercot),
                (f'{WN}/fig5_package/01_raw/hybrid_connected_util_ercot.json', ercot),
                (f'{WN}/fig5_package/01_raw/hybrid_connected_util_georgia.json', georgia)]:
    json.dump(d, open(path, 'w'), indent=1); print('wrote', os.path.relpath(path, ROOT))
for name, d in (('ERCOT', ercot), ('Georgia', georgia)):
    print(name, {x: round(v['util'], 3) for x, v in d['table'].items()})
