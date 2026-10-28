"""CO2 emissions of off-grid vs grid-connected supply.

Off-grid: merit-order dispatch of each data center's gas units (same merit order as
offgrid_mip.py: variable cost = VOM + heat rate x zonal gas price), fuel burn = heat
rate x generation, CO2 = fuel burn x EPA factor for natural gas.
Grid: HOPE emissions_state.csv (EF per unit type = heat rate x same factor),
increment = data-center case minus the base case without data centers.

EPA GHG Emission Factors Hub (2025), Table 1: natural gas 53.06 kg CO2/MMBtu;
distillate fuel oil No. 2 (diesel) 73.96 kg CO2/MMBtu.
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import offgrid_model as og

EF_GAS = 53.06e-3    # t CO2 / MMBtu
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
LOAD = f'{ROOT}/DC_Load/load_profiles/dc_load_hourly_MW.csv'
HOPE = f'{ROOT}/Code/HOPE/ModelCases'
OUT = f'{ROOT}/Code/OffGrid/out/emissions.json'

d = pd.read_csv(LOAD, index_col=0, parse_dates=True)
d = d[~((d.index.month == 2) & (d.index.day == 29))].drop(columns=['TOTAL_MW'])
HR = np.array([u['hr'] for u in og.CAT]); VOM = np.array([u['vom'] for u in og.CAT])
NAMES = [u['name'] for u in og.CAT]

def dispatch(load, n, gas):
    """Merit-order dispatch, hour by hour. Returns MWh and MMBtu by unit type."""
    vr = VOM + HR*gas; order = np.argsort(vr)
    rem = load.astype(float).copy(); mwh = np.zeros(len(n))
    for i in order:
        g = np.minimum(rem, n[i]*og.S_MW[i]); mwh[i] = g.sum(); rem = rem - g
    return mwh, mwh*HR

def offgrid(sites_file, k):
    rows = json.load(open(sites_file)); tot = dict(MWh=0.0, MMBtu=0.0, by_type={})
    for r in rows:
        load = d[r['site']].values[:og.H]*k
        mwh, mmbtu = dispatch(load, np.array(r['n']), r['gas'])
        chk = float((mwh*(VOM + HR*r['gas'])).sum())
        assert abs(chk - r['fuel_cost'])/r['fuel_cost'] < 0.02, (r['site'], chk, r['fuel_cost'])
        tot['MWh'] += mwh.sum(); tot['MMBtu'] += mmbtu.sum()
        for i, nm in enumerate(NAMES):
            tot['by_type'][nm] = tot['by_type'].get(nm, 0.0) + mmbtu[i]*EF_GAS
    tot['CO2_t'] = tot['MMBtu']*EF_GAS; tot['t_per_MWh'] = tot['CO2_t']/tot['MWh']
    tot['heat_rate'] = tot['MMBtu']/tot['MWh']; return tot

def grid(dc_case, base_case):
    e = lambda c: float(pd.read_csv(f'{HOPE}/{c}/output/emissions_state.csv').Emissions_ton.sum())
    return dict(dc=e(dc_case), base=e(base_case), CO2_t=e(dc_case) - e(base_case))

CASES = {
 '2030_40GW':  dict(sites=f'{ROOT}/Code/OffGrid/out/offgrid_mip_gasonly.json', k=1.0,
                    dc='TX8z_GTEP_dc_vre_case', base='TX8z_GTEP_base_vre_case'),
 '2035_20GW':  dict(sites=f'{ROOT}/Code/OffGrid/out/offgrid_mip_2035_k0p5_gasonly_sites.json', k=0.5,
                    dc='TX8z_GTEP_dc20_case', base='TX8z_GTEP_base2035_case'),
 '2035_121GW': dict(sites=f'{ROOT}/Code/OffGrid/out/offgrid_mip_2035_k3_gasonly_sites.json', k=3.0,
                    dc='TX8z_GTEP_dc120_case', base='TX8z_GTEP_base2035_case'),
}
res = {}
for name, c in CASES.items():
    o = offgrid(c['sites'], c['k']); g = grid(c['dc'], c['base']); g['t_per_MWh'] = g['CO2_t']/o['MWh']
    res[name] = dict(energy_MWh=o['MWh'], offgrid=o, grid=g,
                     ratio=o['CO2_t']/g['CO2_t'], extra_Mt=(o['CO2_t'] - g['CO2_t'])/1e6)
    print(f"{name}: energy {o['MWh']/1e6:.1f} TWh | off-grid {o['CO2_t']/1e6:.1f} Mt ({o['t_per_MWh']:.3f} t/MWh, HR {o['heat_rate']:.2f}) "
          f"| grid +{g['CO2_t']/1e6:.1f} Mt ({g['t_per_MWh']:.3f} t/MWh) | ratio {res[name]['ratio']:.2f} | extra {res[name]['extra_Mt']:.1f} Mt/yr")
json.dump(dict(EF_gas_t_per_MMBtu=EF_GAS, source='EPA GHG Emission Factors Hub 2025, Table 1', cases=res),
          open(OUT, 'w'), indent=1, default=float)
print('wrote', OUT)
