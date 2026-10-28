"""The 2035 vintage capacity-factor uplift: +3% on NEW-BUILD wind and solar only.

WHY.  Until now the 2035 cases carried the same new-build capacity factors as 2030, on the
grounds that the derate is a property of the commissioning cohort and there is no 2035
weather.  That is right about the weather and wrong about the machine: a turbine or module
commissioned in 2035 is a better machine than one commissioned in 2030, at the same site.

THE NUMBER, AND WHERE IT COMES FROM.  NREL ATB 2025, Moderate scenario, pro-rated from the
2023 base year to 2035 (12 years) down to the 2030-to-2035 span (5 years):

  Solar   ATB states a 6.7-8.3% improvement in capacity factor from the 2023 base year by
          2035, from bifaciality 0.65 -> 0.85, DC losses 14.1% -> 10.4%, inverter
          efficiency 96% -> 98% and degradation 0.7% -> 0.5%/yr.
          5/12 x (6.7 to 8.3) = 2.8% to 3.5%, central 3.1%.

  Wind    ATB gives a 2023 class-4 market-average net capacity factor of 43% and a 2035
          Moderate T1 figure of 48% at the same wind speed class, i.e. +11.6%; 5/12 of that
          is 4.8%.  But that comparison swaps a market-average turbine for an optimised
          configuration (rotor 134 m -> 170-196 m, hub 103 m -> 100-140 m), which is partly
          a change of basis rather than a vintage effect, so it is a CEILING.  The floor is
          the loss reduction alone, 17.8% -> 12% between 2023 and 2050, which is +3.1% by
          2035 and 1.3% over 2030-35.  Bracket 1.3% to 4.8%, midpoint 3.1%.

Both land on 3.1%, so a common +3% is used and stated as a central value rather than a
measurement.  It is applied as a SCALAR on the hourly profile, which is exactly right for
an efficiency or loss gain and only approximate for a hub-height change, since a taller
turbine alters the shape of the profile as well as its mean.  That caveat is in the paper.

WHAT IS AND IS NOT TOUCHED.  Only the CANDIDATE WindOn and SolarPV columns move.  The
existing fleet keeps its own calibration (0.7242 wind, 0.7839 solar), because those are the
same machines in both years, and the script asserts every existing column is unchanged to
the last bit before it writes anything.  It also asserts that no hour is pushed above full
output -- the pre-uplift maxima are 0.792 wind and 0.880 solar, so 1.03x stays inside 1.0
and the realised uplift is exactly 3%, not a clipped approximation.

    python 91_cf_2035_uplift.py
    -> cf2035/<case>/gen_availability_timeseries.csv   (ship these three to the device)
"""
import os
import numpy as np
import pandas as pd

U = 'Code/HOPE/ModelCases'
OUT = 'build/cf2035'
UPLIFT = 1.03
VRE = ('WindOn', 'SolarPV')

CASES = {'TX8z_GTEP_base2035_case': 'Data_TX8z_GTEP_base2035',
         'TX8z_GTEP_dc20_case': 'Data_TX8z_GTEP_dc20',
         'TX8z_GTEP_dc120_case': 'Data_TX8z_GTEP_dc120'}

os.makedirs(OUT, exist_ok=True)
for case, dat in CASES.items():
    d = f'{U}/{case}/{dat}'
    g = pd.read_csv(f'{d}/gendata.csv')
    c = pd.read_csv(f'{d}/gendata_candidate.csv')
    A = pd.read_csv(f'{d}/gen_availability_timeseries.csv')
    cols = [x for x in A.columns if x.startswith('G')]
    assert len(cols) == len(g) + len(c), f'{case}: {len(cols)} columns for {len(g)+len(c)} units'
    B = A.copy()

    touched = []
    for k in range(len(c)):
        if c.Type[k] in VRE:
            col = cols[len(g) + k]
            v = A[col].values
            hi = v.max()*UPLIFT
            assert hi <= 1.0, (f'{case}/{col}: {UPLIFT}x pushes the maximum to {hi:.4f}, '
                              f'above full output; the uplift would be clipped')
            B[col] = v*UPLIFT
            touched.append((f'{c.Type[k]}/{c.Zone[k]}', col, v.mean(), v.mean()*UPLIFT))

    # the existing fleet must not have moved, at all
    for e in range(len(g)):
        col = cols[e]
        assert (A[col].values == B[col].values).all(), f'{case}: existing column {col} moved'
    # nor the time mapping
    for t in ('Time Period', 'Month', 'Day', 'Hours'):
        if t in A.columns:
            assert (A[t].values == B[t].values).all(), f'{case}: {t} moved'

    od = f'{OUT}/{case}'
    os.makedirs(od, exist_ok=True)
    B.to_csv(f'{od}/gen_availability_timeseries.csv', index=False, float_format='%.6f')

    # read back what was written and confirm the realised uplift is 3%, not 2.97%
    C = pd.read_csv(f'{od}/gen_availability_timeseries.csv')
    worst = max(abs(C[col].mean()/m0 - UPLIFT) for _n, col, m0, _m1 in touched)
    assert worst < 1e-5, f'{case}: realised uplift is off by {worst:.2e}'

    print(f'{case}')
    print(f'   {len(touched)} candidate wind and solar columns uplifted by '
          f'{100*(UPLIFT-1):.0f}%, {len(g)} existing columns untouched')
    for n, col, m0, m1 in touched[:4]:
        print(f'     {n:<22} {col:>5}  {m0:.4f} -> {m1:.4f}')
    print(f'     ... and {len(touched)-4} more, realised uplift exact to '
          f'{worst:.1e}\n')

print(f'written {OUT}/<case>/gen_availability_timeseries.csv')
print('ship the three files to ModelCases/<case>/Data_*/ and re-run the PRM fixed point')
