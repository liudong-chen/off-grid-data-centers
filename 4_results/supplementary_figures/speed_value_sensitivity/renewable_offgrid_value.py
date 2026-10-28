"""Pure-renewable off-grid against waiting for the grid, on the Fig. 3 accounting.
Same flows as figures_1_to_3/02_code/plot_figures_1_to_3.py (monthly, 8% discounted to day 0, 20 years after
month 24), with the off-grid plant replaced by the pure-renewable design
(3_offgrid_code/scenario_renewable_only/renewable_only_summary.json), built in 24 months and never connected.
python3 renewable_offgrid_value.py -> renewable_offgrid_value.json"""
import json, os
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); WN = os.path.abspath(os.path.join(H, '..', '..', 'main_figures'))
ROOT = os.path.abspath(os.path.join(H, '..', '..', '..'))
HS = json.load(open(f'{WN}/figures_1_to_3/01_raw/hybrid_diesel_summary.json')); P = HS['params']
ONG = json.load(open(f'{WN}/figures_1_to_3/01_raw/ongrid_final.json'))
REN = json.load(open(f'{ROOT}/3_offgrid_code/scenario_renewable_only/renewable_only_summary.json'))
REN4 = json.load(open(f'{ROOT}/3_offgrid_code/scenario_renewable_only/renewable_only_4h_summary.json'))
HYB = json.load(open(f'{ROOT}/3_offgrid_code/out/hybrid_renewable_summary.json'))
GAS = json.load(open(f'{WN}/figures_1_to_3/01_raw/offgrid_summary.json'))['separate']
E, C_ON, MW, R, T_GAS = ONG['energy_MWh'], P['c_on'], P['MW_pub'], P['R'], P['T_GAS']
END = T_GAS + 240; mo = np.arange(END + 1); disc = (1 + R/12)**(-mo.astype(float))

def net(rate, annual_cost, T_grid, build=True):
    earn = np.zeros(END + 1); cost = np.zeros(END + 1)
    on = mo > (T_GAS if build else T_grid)
    earn[on] = rate*MW
    cost[on] = annual_cost/12 if build else C_ON*E/12
    return float(((earn - cost)*disc).sum())/1e9/20

out = {}
for rate in (200e3, 300e3, 400e3, 500e3, 600e3):
    rec = {}
    for name, ac in (('renewable', REN['cost_Byr']*1e9), ('renewable_4h', REN4['cost_Byr']*1e9), ('gas', GAS['lcoe']*E)):
        d = {m: net(rate, ac, m) - net(rate, None, m, build=False) for m in range(0, 241)}
        pos = [m for m, v in d.items() if v >= 0]
        rec[name] = dict(value_vs_wait_B_per_yr={m: d[m] for m in (36, 50, 72, 120)},
                         bound_months=min(pos) if pos else None)
    rec['earnings_per_MWh'] = rate*MW*12/E
    h = HYB['by_rate'][str(int(rate))]                    # renewable hybrid, 4-h batteries, connected via HOPE
    rec['renewable_4h_hybrid'] = dict(bound_months=h['bound_months'], value_vs_wait_B_per_yr=h['value_vs_wait_B_per_yr'])
    out[int(rate)] = rec
    print(int(rate), 'earn $/MWh %.0f' % rec['earnings_per_MWh'],
          '| renewable bound', rec['renewable']['bound_months'], {k: round(v, 1) for k, v in rec['renewable']['value_vs_wait_B_per_yr'].items()},
          '| 4h island', rec['renewable_4h']['bound_months'], '| 4h hybrid', rec['renewable_4h_hybrid']['bound_months'], '| gas bound', rec['gas']['bound_months'])
out['renewable_lcoe'] = REN['lcoe']; out['renewable_4h_lcoe'] = REN4['lcoe']; out['renewable_4h_hybrid_connected'] = HYB['c_connected_incl_sunk']; out['gas_lcoe'] = GAS['lcoe']; out['grid_lcoe'] = C_ON
json.dump(out, open(f'{H}/renewable_offgrid_value.json', 'w'), indent=1)
