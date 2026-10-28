"""Scenario: OFF-GRID, GAS ONLY, TWO UNIT SIZES.

Same model and method as the main gas-only off-grid case (offgrid_mip.design: MIP with N-k rule,
no-good and Monte Carlo sensitivity cuts, neighbourhood improvement, polish, 2,000-replication
confirmation of LOLE <= 0.1 d/yr), but the catalogue is restricted to the SMALLEST and the
LARGEST natural gas machines: 20 MW reciprocating engine and 627 MW combined cycle. The
35 MW and 48 MW aeroderivatives and the 419 MW frame CT are removed (upper bound 0 in the MIP
and a prohibitive fixed cost so that no improvement, polish or confirmation step can add them).

    OFFGRID_CASE=... OFFGRID_LOAD=... python3 run_two_unit_sizes.py [sites|pool]
    -> two_unit_sizes_sites.json, two_unit_sizes_pool.json
"""
import os, sys, json, time
import numpy as np, pandas as pd
import offgrid_mip as om

KEEP = (0, 4)                                    # Recip 20 MW, CC 627 MW
om.anchor_capex()
for j in range(om.NC):
    if j not in KEEP:
        om.FX_MW[j] = 1e9                        # in place (shared with offgrid_model)
_orig = om.build_and_solve
def _restricted(site, *a, nmax=None, **k):
    if nmax is None:
        nmax = np.ceil(3.5*site.peak/om.S_MW).astype(int)
    nmax = np.array(nmax, int); nmax[[j for j in range(om.NC) if j not in KEEP]] = 0
    return _orig(site, *a, nmax=nmax, **k)
om.build_and_solve = _restricted

CD, R = os.environ['OFFGRID_CASE'], os.environ['OFFGRID_LOAD']
ZM = {'Far West': 'FarWest', 'North Central': 'NorthCentral', 'South Central': 'SouthCentral'}
GASZ = json.load(open('zonal_gas_2030.json'))
d = pd.read_csv(f'{R}/dc_load_hourly_MW.csv', index_col=0, parse_dates=True)
d = d[~((d.index.month == 2) & (d.index.day == 29))]
meta = pd.read_csv(f'{R}/site_parameters_and_results.csv')
meta['zone'] = meta.ercot_weather_zone.map(lambda z: ZM.get(z, z))
KW = dict(allow=(False, False, False), credit_cap=0.15, vre_cut_frac=0.0, sens_cuts=True)
MODE = sys.argv[1] if len(sys.argv) > 1 else 'sites'

if MODE == 'pool':
    tot = d[meta.site_id].values[:om.H].sum(axis=1); t0 = time.time()
    r = om.design(tot, 'FarWest', None, None, gas=GASZ['FarWest'], time_limit=300.0, **KW)
    assert all(r['n'][j] == 0 for j in range(om.NC) if j not in KEEP)
    print(f"POOLED FarWest {r['peak_MW']:,.0f} MW  {r['mix']}  {r['installed_MW']:,.0f} MW ({r['ratio']:.2f}x)  "
          f"LOLE {r['LOLE']:.3f}  ${r['lcoe']:.2f}/MWh  [{time.time()-t0:.0f}s]", flush=True)
    json.dump({'FarWest': r}, open('two_unit_sizes_pool.json', 'w'), indent=1, default=float)
    sys.exit(0)

rows = []
for s in meta.itertuples():
    t0 = time.time(); load = d[s.site_id].values[:om.H]
    r = om.design(load, s.zone, None, None, gas=GASZ[s.zone], **KW)
    assert all(r['n'][j] == 0 for j in range(om.NC) if j not in KEEP), r['mix']
    r.update(site=s.site_id, home_zone=s.zone); rows.append(r)
    print(f"{s.site_id:<5}{s.zone:<13}{r['peak_MW']:>7,.0f}  {str(r['mix']):<30}{r['installed_MW']:>8,.0f} MW "
          f"({r['ratio']:.2f}x)  LOLE {r['LOLE']:.3f}  ${r['lcoe']:.2f}/MWh  [{time.time()-t0:.0f}s]", flush=True)
    json.dump(rows, open('two_unit_sizes_sites.json', 'w'), indent=1, default=float)
c = sum(r['cost'] for r in rows); e = sum(r['energy_MWh'] for r in rows)
mw = sum(r['installed_MW'] for r in rows); pk = sum(r['peak_MW'] for r in rows)
print(f"FLEET {mw:,.0f} MW ({mw/pk:.2f}x)  ${c/1e9:.3f} B/yr  ${c/e:.2f}/MWh")
