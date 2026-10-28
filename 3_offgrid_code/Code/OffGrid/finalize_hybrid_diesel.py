"""Make x = 0 of the diesel-first sweep the gas-only MIP off-grid design.

    python3 finalize_hybrid_diesel.py     (after run_hybrid_diesel_first.py and
                                           run_offgrid_mip_gasonly.py)

For every campus, cases['0.0'] is replaced by the design in out/offgrid_mip_gasonly.json
(the paper's MIP with solar/wind/battery disallowed), costed with the same cost function
as the diesel cases; the sweep's own grid-search x = 0 stays as cases['0.0_gasopt'] for a
consistency check only.  Rewrites site_*.json and all_sites.json in place.
"""
import json
import os

import numpy as np

import hybrid_diesel_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
BYONG = os.path.abspath(os.path.join(HERE, '..', '..'))
LOAD_DIR = os.environ.get('BYONG_LOAD', f'{BYONG}/DC_Load/load_profiles')
OUT = os.environ.get('BYONG_OUT', os.path.join(HERE, 'out', 'hybrid_diesel'))
GASONLY = os.environ.get('BYONG_GASONLY', f'{HERE}/out/offgrid_mip_gasonly.json')

L.prepare()
d, meta = L.load_sites(f'{LOAD_DIR}/dc_load_hourly_MW.csv', f'{LOAD_DIR}/site_parameters_and_results.csv')
mip = {r['site']: r for r in json.load(open(GASONLY))}
allr = []
for r in meta.itertuples():
    p = f'{OUT}/site_{r.site_id}.json'
    if not os.path.exists(p):
        continue
    s = json.load(open(p))
    g = mip[r.site_id]
    c = L.Campus(r.site_id, r.zone, d[r.site_id].values, g, None, None, r.mw_published)
    c.s = c.w = c.b = 0.0
    n6 = np.append(np.array(g['n'], int), 0)
    cost = c.cost(n6)
    assert abs(cost['lcoe'] - g['lcoe']) < 0.05, (r.site_id, cost['lcoe'], g['lcoe'])
    s['cases']['0.0'] = dict(share=0.0, n=n6.tolist(), removed_MW=0.0, diesel_units=0, LOLE=g['LOLE'],
                             installed_MW=float(n6 @ L.og.S_MW), **cost,
                             note='gas-only off-grid design, offgrid_mip.design with allow=(False,False,False), '
                                  'confirmed at 2,000 reps (out/offgrid_mip_gasonly.json)',
                             phaseA=dict(load_fraction=0.0, diesel_units=0, LOLE=None, energy_MWh=0.0,
                                         fixed_diesel=0.0, fuel_diesel=0.0))
    s['n0_gasonly'] = g['n']
    json.dump(s, open(p, 'w'), indent=1, default=float)
    allr.append(s)
    chk = s['cases'].get('0.0_gasopt', {}).get('lcoe')
    print(f"{r.site_id:4s} x=0 MIP gas-only {g['mix']}  ${g['lcoe']:.2f}/MWh"
          + (f"   (sweep grid search ${chk:.2f})" if chk else ''))
json.dump(allr, open(f'{OUT}/all_sites.json', 'w'), indent=1, default=float)
print('fleet x=0:', sum(r['installed_MW'] for r in mip.values()), 'MW',
      round(sum(r['cost'] for r in mip.values())/sum(r['energy_MWh'] for r in mip.values()), 2), '$/MWh')
