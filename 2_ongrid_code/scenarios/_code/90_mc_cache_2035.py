"""The Monte Carlo caches the three 2035 cases need, derived from the 2030 pair.

WHY THEY ARE NEEDED.  `ReliabilityPRM.jl` scores a portfolio against
`mc_cache_<tag>_{fleet,series,meta}.csv`, and only "base" and "dc" exist -- the 2030 pair.
Running a 2035 case with `case_tag = "dc"` would solve the 2035 system and then score it
against 2030 load.  The engine would not complain; the answer would simply be wrong.  So
each 2035 case gets its own tag: base2035, dc20, dc120.

WHY THEY CAN BE DERIVED RATHER THAN REBUILT.  Three of the four ingredients do not change
between the two planning years, and that is a property of the case files, not an
approximation:

    fleet     the existing fleet is identical -- EIA-860 lists no retirements after 2029,
              so 75_build_2035_cases.py left gendata.csv's units untouched
    vre       the existing fleet's wind and solar output is the same machines on the same
              weather years
    storage   13,504 MW / 21,254 MWh, unchanged

Only the LOAD moves, and it moves by construction:

    base_2035(h)  = base_2030(h) x 1.019^5           organic growth, five more years
    dc20(h)       = base_2035(h) + 0.5 x dc_2030(h)  same 17 campuses, same shapes
    dc120(h)      = base_2035(h) + 3.0 x dc_2030(h)

where dc_2030(h) is recovered as the difference of the two existing caches.  That is
exactly what the HOPE case files do, so the script does not trust the arithmetic -- it
ASSERTS that the 2025 weather-year peak it produces equals the case's own PK_system, read
from the case's zonedata and load timeseries, for all three cases.  If a case is ever
rebuilt differently, this fails rather than producing a plausible wrong number.

    python 90_mc_cache_2035.py
    -> ga_hope/../mc2035/mc_cache_{base2035,dc20,dc120}_{fleet,series,meta}.csv

Copy the nine files beside ReliabilityPRM.jl in 2_ongrid_code/HOPE/.
"""
import os
import numpy as np
import pandas as pd

H = '2_ongrid_code/HOPE'
U = f'{H}/ModelCases'
OUT = 'build/mc2035'
TX = ['Coast', 'East', 'FarWest', 'North', 'NorthCentral', 'South', 'SouthCentral', 'West']

GROWTH = 1.019**5                 # 2030 -> 2035 organic growth on the 2025 shape
K = {'base2035': 0.0, 'dc20': 0.5, 'dc120': 3.0}
CASEDIR = {'base2035': f'{U}/TX8z_GTEP_base2035_case',
           'dc20': f'{U}/TX8z_GTEP_dc20_case',
           'dc120': f'{U}/TX8z_GTEP_dc120_case'}

# ------------------------------------------------------------------ the 2030 caches
fleet = pd.read_csv(f'{H}/mc_cache_base_fleet.csv')
fl_dc = pd.read_csv(f'{H}/mc_cache_dc_fleet.csv')
assert fleet.equals(fl_dc), 'the base and dc fleet caches differ; they should not'
sb = pd.read_csv(f'{H}/mc_cache_base_series.csv')
sd = pd.read_csv(f'{H}/mc_cache_dc_series.csv')
mb = pd.read_csv(f'{H}/mc_cache_base_meta.csv').iloc[0]

YEARS = sorted(int(c.split('_')[1]) for c in sb.columns if c.startswith('load_'))
print(f'weather years in the 2030 cache: {YEARS[0]}-{YEARS[-1]} ({len(YEARS)})')
for y in YEARS:
    assert np.allclose(sb[f'vre_{y}'], sd[f'vre_{y}']), f'existing VRE differs in {y}'


def case_peak(tag):
    """The case's own PK_system: max over hours of the summed zonal load."""
    d = [x for x in os.listdir(CASEDIR[tag])
         if x.startswith('Data_') and not x.endswith('_BEFORE_2035_UPDATE')][0]
    dd = f'{CASEDIR[tag]}/{d}'
    zd = pd.read_csv(f'{dd}/zonedata.csv').set_index('Zone_id')['Demand (MW)']
    ld = pd.read_csv(f'{dd}/load_timeseries_regional.csv')
    return float((ld[TX].values*zd[TX].values[None, :]).sum(1).max())


os.makedirs(OUT, exist_ok=True)
print(f'\n{"tag":10s}{"cache peak, all years":>24s}{"2025 column":>14s}'
      f'{"case PK_system":>16s}{"":>8s}')
for tag, k in K.items():
    cols = {}
    for y in YEARS:
        base35 = sb[f'load_{y}'].values*GROWTH
        dc30 = sd[f'load_{y}'].values - sb[f'load_{y}'].values      # the DC component
        cols[f'load_{y}'] = base35 + k*dc30
        cols[f'vre_{y}'] = sb[f'vre_{y}'].values                    # existing fleet
    S = pd.DataFrame(cols)
    peak_all = float(max(S[f'load_{y}'].max() for y in YEARS))
    peak_2025 = float(S['load_2025'].max())
    pk_case = case_peak(tag)

    # the check that makes this derivation safe rather than plausible
    rel = abs(peak_2025 - pk_case)/pk_case
    assert rel < 2e-3, (f'{tag}: cache 2025 peak {peak_2025:,.0f} MW against the case\'s '
                        f'own {pk_case:,.0f} MW ({100*rel:.2f}%) -- the case was not built '
                        f'the way this script assumes')

    fleet.to_csv(f'{OUT}/mc_cache_{tag}_fleet.csv', index=False)
    S.to_csv(f'{OUT}/mc_cache_{tag}_series.csv', index=False)
    pd.DataFrame([{'sto_p_MW': mb['sto_p_MW'], 'sto_e_MWh': mb['sto_e_MWh'],
                   'peak_MW': peak_all, 'n_years': len(YEARS), 'eta': mb['eta']}]
                 ).to_csv(f'{OUT}/mc_cache_{tag}_meta.csv', index=False)
    print(f'{tag:10s}{peak_all:>24,.0f}{peak_2025:>14,.0f}{pk_case:>16,.0f}'
          f'{"  ok":>8s}')

print(f'\nfirm fleet {fleet.mw.sum():,.0f} MW over {int(fleet.n_units.sum())} units, '
      f'identical in all three (no post-2029 retirements)')
print(f'storage {mb["sto_p_MW"]:,.0f} MW / {mb["sto_e_MWh"]:,.0f} MWh, unchanged')
print(f'\nwritten {OUT}/mc_cache_*_{{fleet,series,meta}}.csv')
print('copy all nine beside ReliabilityPRM.jl in 2_ongrid_code/HOPE/')
