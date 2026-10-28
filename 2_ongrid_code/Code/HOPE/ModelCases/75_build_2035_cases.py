"""Bring the three 2035 HOPE cases onto the corrected 2030 inputs, plus the 2035 ones.

RUNS ON THE MACHINE THAT HOLDS THE CASES.  It edits, in place and with a backup:

    ModelCases/TX8z_GTEP_base2035_case/Data_TX8z_GTEP_base2035
    ModelCases/TX8z_GTEP_dc20_case/Data_TX8z_GTEP_dc20
    ModelCases/TX8z_GTEP_dc120_case/Data_TX8z_GTEP_dc120

THE CONSTRUCTION.  A 2035 case is the corrected 2030 case (TX8z_GTEP_dc_vre_case) with
the year-dependent inputs replaced, and nothing else touched.  That is the whole design:
every difference between the 2030 result and a 2035 result should be traceable to a row
of the parameter table, not to a modelling choice that drifted between them.

WHAT THE 2035 CASES ALREADY HAD RIGHT and is left alone:
  - the load.  Base load is the 2025 shape at 1.019^10 = 1.2071 (checked here against the
    2030 file: the ratio must be 1.019^5 to 1e-6).  The data-centre load is the same 17
    hourly shapes at k = 0.5 or 3.
  - the existing fleet and its hourly availability, the network, the zone table.
  - the candidate capital costs, which are already the workbook's 2035 column.

WHAT IS FIXED HERE, and why:
  1. CANDIDATE VRE CAPACITY FACTOR.  The 2035 files carry the OLD existing-fleet derate
     (wind 0.7242, solar 0.7839), so a plant built in 2035 would have the capacity factor
     of ERCOT's 2025 fleet.  The 2030 case was corrected to a measured new-build derate of
     0.90 wind and 0.88 solar; the same correction belongs here.  The weather year is the
     same 2025, so the corrected columns are copied across rather than rescaled.
  2. ZONAL GAS.  The 2035 files carry the old flat-scaled table ($5.10 everywhere except
     three zones).  Replaced by the EIA-923 measured differentials carried forward
     additively to the 2035 Henry Hub, with the Waha narrowing -- the same construction
     the 2030 case now uses (74_gas_year.py).  This is the largest single change: Far West
     goes from $5.10 to $4.00/MMBtu.
  3. BUILD LIMITS.  The 20,000 / 60,000 MW per-zone ceiling was an assigned round number
     and it bound in the 2035 runs -- NGCC/SouthCentral sat exactly on 20,000 MW in the
     20 GW case, wind/North exactly on 60,000 MW in the 121 GW case, so the answer was
     the assumption.  Replaced by the 2030 case's land-derived VRE limits (wind at
     3 MW/km2 on 20% of zone land, solar at 35 MW/km2 on 2%), thermal at 100,000 MW
     (economics-limited; a gas plant needs ~0.5 km2/GW) and storage at 50,000 MW /
     200,000 MWh.  These are physical, so they do not depend on the planning year.

WHAT IS NOT DECIDED HERE.  The planning reserve margin is a fixed point at LOLE = 0.1 and
the values in the files were derived under the OLD capacity factors and the OLD gas.  They
are left in place as a starting value and flagged in the report; they have to be
re-derived against the corrected inputs before the 2035 numbers are quoted.

    python3 75_build_2035_cases.py            # from ModelCases/
    python3 75_build_2035_cases.py --dry-run
"""
import json, os, shutil, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = f'{ROOT}/TX8z_GTEP_dc_vre_case/Data_TX8z_GTEP_dc_vre'      # corrected 2030 case
CASES = {'base2035': f'{ROOT}/TX8z_GTEP_base2035_case/Data_TX8z_GTEP_base2035',
         'dc20': f'{ROOT}/TX8z_GTEP_dc20_case/Data_TX8z_GTEP_dc20',
         'dc120': f'{ROOT}/TX8z_GTEP_dc120_case/Data_TX8z_GTEP_dc120'}
DRY = '--dry-run' in sys.argv

# --------------------------------------------------------------- the 2035 parameters
# Delivered gas by zone, $/MMBtu, 2035.  EIA-923 2025 receipts assigned to weather zones
# by EIA-860 coordinates; differentials carried forward additively to Henry Hub $5.05 x
# the measured Texas basis 0.8912; Far West and North keep 30% of their measured discount
# (Permian takeaway).  Produced by 74_gas_year.py.
GAS35 = {'Coast': 5.000, 'East': 4.866, 'FarWest': 3.997, 'North': 4.169,
         'NorthCentral': 5.187, 'South': 5.348, 'SouthCentral': 5.044, 'West': 4.501}
HR = {'NGCC': 6.226, 'NGCT': 9.142, 'NGST': 10.5}     # MMBtu/MWh
VOM = {'NGCC': 3.57, 'NGCT': 4.17, 'NGST': 4.17}      # $/MWh

# Annualised fixed cost, $/MW-yr, real 2025$.  ERCOT_CEM_Technology_Cost_Assumptions
# workbook, sheet Recommended_CEM_Base, column N (2035 total fixed).
COST35 = {'NGCC': 113755, 'NGCT': 96576, 'WindOn': 149412, 'SolarPV': 108393,
          'NuC': 706973}
BATT35 = 142528
NUC_VOM = 8.0                                          # $/MWh, assigned, unchanged

STO_MW, STO_MWH = 50000.0, 200000.0                    # as the corrected 2030 case


def var_cost(tech, zone):
    if tech in HR:
        return round(HR[tech]*GAS35[zone] + VOM[tech], 2)
    return NUC_VOM if tech == 'NuC' else 0.0


# ------------------------------------------------------------------------- checks
src_c = pd.read_csv(f'{SRC}/gendata_candidate.csv')
src_g = pd.read_csv(f'{SRC}/gendata.csv')
src_av = pd.read_csv(f'{SRC}/gen_availability_timeseries.csv')
src_z = pd.read_csv(f'{SRC}/zonedata.csv').set_index('Zone_id')['Demand (MW)']
src_l = pd.read_csv(f'{SRC}/load_timeseries_regional.csv')
NE = len(src_g)
ZONES = list(src_z.index)

base30 = f'{ROOT}/TX8z_GTEP_base_vre_case/Data_TX8z_GTEP_base_vre'
b30_z = pd.read_csv(f'{base30}/zonedata.csv').set_index('Zone_id')['Demand (MW)']
b30_l = pd.read_csv(f'{base30}/load_timeseries_regional.csv')
e30 = float((b30_l[ZONES]*b30_z[ZONES]).sum().sum())

print('CHECKS BEFORE ANY EDIT')
for c, p in CASES.items():
    av = pd.read_csv(f'{p}/gen_availability_timeseries.csv')
    g = pd.read_csv(f'{p}/gendata.csv')
    ex = [f'G{i+1}' for i in range(NE)]
    d = float(np.abs(src_av[ex].values - av[ex].values).max())
    same = g.drop(columns=['Cost ($/MWh)']).equals(src_g.drop(columns=['Cost ($/MWh)']))
    assert d < 1e-9, f'{c}: existing-fleet availability differs from the 2030 case'
    assert same, f'{c}: existing fleet differs from the 2030 case'
    assert pd.read_csv(f'{p}/linedata.csv').equals(pd.read_csv(f'{SRC}/linedata.csv'))
    print(f'   {c:9s} existing fleet, its availability and the network match the 2030 case')

# the load ratio must be exactly five more years of 1.9%
z35 = pd.read_csv(f'{CASES["base2035"]}/zonedata.csv').set_index('Zone_id')['Demand (MW)']
l35 = pd.read_csv(f'{CASES["base2035"]}/load_timeseries_regional.csv')
e35 = float((l35[ZONES]*z35[ZONES]).sum().sum())
print(f'   base load 2030 {e30/1e6:.3f} TWh -> 2035 {e35/1e6:.3f} TWh, ratio '
      f'{e35/e30:.6f} against 1.019^5 = {1.019**5:.6f}')
assert abs(e35/e30 - 1.019**5) < 1e-5, 'the 2035 base load is not the 2030 one grown five years'

# ------------------------------------------------------------------------ the edits
LIMITS = src_c.set_index(['Zone', 'Type'])['Pmax (MW)'].to_dict()
UNITS = src_c.set_index(['Zone', 'Type'])['n_units'].to_dict()

for c, p in CASES.items():
    bak = f'{p}_BEFORE_2035_UPDATE'
    if not DRY and not os.path.isdir(bak):
        shutil.copytree(p, bak)

    # 1. existing fleet: gas-fired variable cost at the 2035 zonal price
    g = pd.read_csv(f'{p}/gendata.csv')
    old = g['Cost ($/MWh)'].copy()
    m = g.Type.isin(HR)
    g.loc[m, 'Cost ($/MWh)'] = [var_cost(t, z) for t, z in
                                zip(g.Type[m], g.Zone[m])]
    nchg = int((g['Cost ($/MWh)'] != old).sum())

    # 2. candidates: 2035 capital, 2035 gas, land-derived limits, corrected CF
    cd = src_c.copy()
    cd['Cost ($/MW/yr)'] = [COST35[t] for t in cd.Type]
    cd['Cost ($/MWh)'] = [var_cost(t, z) for t, z in zip(cd.Type, cd.Zone)]
    cd['Pmax (MW)'] = [LIMITS[(z, t)] for z, t in zip(cd.Zone, cd.Type)]
    cd['n_units'] = [UNITS[(z, t)] for z, t in zip(cd.Zone, cd.Type)]

    # 3. availability: the corrected new-build derate, same weather year
    av = src_av.copy()

    # 4. storage: 2035 capital, the 2030 case's economics-limited ceiling
    st = pd.read_csv(f'{p}/storagedata_candidate.csv')
    st['Cost ($/MW/yr)'] = BATT35
    st['Max Power (MW)'] = STO_MW
    st['Capacity (MWh)'] = STO_MWH

    if not DRY:
        g.to_csv(f'{p}/gendata.csv', index=False)
        cd.to_csv(f'{p}/gendata_candidate.csv', index=False)
        av.to_csv(f'{p}/gen_availability_timeseries.csv', index=False)
        st.to_csv(f'{p}/storagedata_candidate.csv', index=False)

    prm = float(pd.read_csv(f'{p}/single_parameter.csv')['planning_reserve_margin'][0])
    print(f'\n{c}: {nchg} existing gas units repriced; {len(cd)} candidates rewritten; '
          f'storage {STO_MW:,.0f} MW / {STO_MWH:,.0f} MWh at ${BATT35:,}/MW-yr')
    print(f'   NGCC $/MWh by zone: '
          f'{ {z: var_cost("NGCC", z) for z in ZONES} }')
    print(f'   PRM left at {prm:.6f} -- DERIVED UNDER THE OLD INPUTS, must be re-derived')

wv = {z: round(float(src_av[f'G{NE+i+1}'].mean()), 4)
      for i, (z, t) in enumerate(zip(src_c.Zone, src_c.Type)) if t == 'WindOn'}
sv = {z: round(float(src_av[f'G{NE+i+1}'].mean()), 4)
      for i, (z, t) in enumerate(zip(src_c.Zone, src_c.Type)) if t == 'SolarPV'}
print(f'\ncandidate wind CF now  {wv}')
print(f'candidate solar CF now {sv}')
print('\nDRY RUN, nothing written' if DRY else '\nwritten; backups in *_BEFORE_2035_UPDATE')
