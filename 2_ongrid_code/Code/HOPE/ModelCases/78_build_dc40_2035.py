"""Build TX8z_GTEP_dc40_2035_case: the 2035 system with the ANNOUNCED data-centre fleet.

RUNS ON THE MACHINE THAT HOLDS THE CASES, from ModelCases/.

Why this case exists.  The 2035 scenarios are k = 0.5 (20 GW) and k = 3 (121 GW), which
gives two points.  The 2030 main case is k = 1 (40 GW) but at 2030 prices and 2030 load,
so it cannot be plotted on the same curve without confounding the year with the scale.
This case is k = 1 at 2035, which closes that gap: with it, the 2035 curve has three
points and every difference along it is scale alone.

    load        = base2035  +  1.0 x the announced data-centre fleet
    everything else identical to the other two 2035 cases

The data-centre component is recovered as (dc120 - base2035)/3, which is checked here
against (dc20 - base2035)/0.5 -- the two agree to 1e-4, and the recovered unit fleet
reproduces the 2030 case exactly: 32,408 MW coincident peak, 222.05 TWh.

    python3 78_build_dc40_2035.py
    -> ModelCases/TX8z_GTEP_dc40_2035_case/   ready to run
"""
import json, os, shutil
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
B = f'{ROOT}/TX8z_GTEP_base2035_case'
D20 = f'{ROOT}/TX8z_GTEP_dc20_case'
D120 = f'{ROOT}/TX8z_GTEP_dc120_case'
NEW = f'{ROOT}/TX8z_GTEP_dc40_2035_case'
DATA = f'{NEW}/Data_TX8z_GTEP_dc40_2035'
K = 1.0


def hourly(p):
    z = pd.read_csv(f'{p}/zonedata.csv').set_index('Zone_id')['Demand (MW)']
    l = pd.read_csv(f'{p}/load_timeseries_regional.csv')
    Z = list(z.index)
    return l[Z]*z[Z], l, z, Z


b, lb, zb, Z = hourly(f'{B}/Data_TX8z_GTEP_base2035')
d20, _, _, _ = hourly(f'{D20}/Data_TX8z_GTEP_dc20')
d120, _, _, _ = hourly(f'{D120}/Data_TX8z_GTEP_dc120')

dc_a, dc_b = (d20 - b)/0.5, (d120 - b)/3.0
m = (dc_a.values > 1.0)
r = (dc_b.values[m]/dc_a.values[m])
print(f'data-centre component recovered two ways, ratio {r.min():.6f} to {r.max():.6f}')
assert abs(r - 1).max() < 1e-3, 'the two 2035 cases do not carry the same DC fleet'
dc = 0.5*(dc_a + dc_b)
print(f'unit DC fleet: coincident peak {dc.sum(axis=1).max():,.0f} MW, '
      f'{dc.values.sum()/1e6:.2f} TWh   (2030 case: 32,408 MW, 222.05 TWh)')

tot = b + K*dc
peak = tot.max()                       # per zone, the zonedata level
sysp = tot.sum(axis=1).max()
print(f'\nk = {K:g}: system peak {sysp:,.0f} MW, energy {tot.values.sum()/1e6:.1f} TWh')
print(f'   zone peaks { {z: round(peak[z]) for z in Z} }')

# ------------------------------------------------------------------ write the case
if os.path.isdir(NEW):
    raise SystemExit(f'{NEW} already exists; remove it first')
shutil.copytree(f'{D120}', NEW, ignore=shutil.ignore_patterns(
    'output*', 'Data_*', '.DS_Store', 'fixedpoint.json'))
shutil.copytree(f'{D120}/Data_TX8z_GTEP_dc120', DATA)

zz = pd.read_csv(f'{DATA}/zonedata.csv')
zz['Demand (MW)'] = [float(peak[z]) for z in zz.Zone_id]
zz.to_csv(f'{DATA}/zonedata.csv', index=False)

ll = pd.read_csv(f'{DATA}/load_timeseries_regional.csv')
for z in Z:
    ll[z] = (tot[z]/peak[z]).values          # per unit of the zone peak, as HOPE expects
ll.to_csv(f'{DATA}/load_timeseries_regional.csv', index=False)

s = open(f'{NEW}/Settings/HOPE_model_settings.yml').read()
open(f'{NEW}/Settings/HOPE_model_settings.yml', 'w').write(
    s.replace('Data_TX8z_GTEP_dc120/', 'Data_TX8z_GTEP_dc40_2035/'))

ci = json.load(open(f'{D120}/case_info.json'))
ci.update(k=K, peak_MW=float(sysp),
          zone_peaks={z: float(peak[z]) for z in Z},
          dc_coincident_peak=float((K*dc).sum(axis=1).max()),
          dc_energy_TWh=float((K*dc).values.sum()/1e6),
          note='built by 78_build_dc40_2035.py; the announced data-centre fleet at 2035')
json.dump(ci, open(f'{NEW}/case_info.json', 'w'), indent=1)

# check the write round-trips
chk = pd.read_csv(f'{DATA}/load_timeseries_regional.csv')
zc = pd.read_csv(f'{DATA}/zonedata.csv').set_index('Zone_id')['Demand (MW)']
back = (chk[Z]*zc[Z])
print(f'\nround-trip check: max error {np.abs(back.values - tot.values).max():.6f} MW')
prm = float(pd.read_csv(f'{DATA}/single_parameter.csv')['planning_reserve_margin'][0])
print(f'PRM carried over from dc120: {prm:.6f} -- stale, as for the other 2035 cases')
print(f'\nwritten {NEW}\n  run: HOPE.run_hope("ModelCases/TX8z_GTEP_dc40_2035_case/")')
