"""Zonal delivered gas for any planning year, from one construction.

The 2030 table was assembled in pieces; this rebuilds it as a single function of the
planning year so the 2035 table is the same object with one number changed, and so the
2030 table can be checked against what the cases actually carry.

    1  MEASURE the 2025 zonal differentials.  Every Texas gas plant in EIA-923 Page 5 is
       assigned to an ERCOT weather zone by its EIA-860 coordinates, and the delivered
       price is quantity-weighted per zone.  This is data.

    2  SET THE LEVEL from Henry Hub.  Texas power plants paid $3.137/MMBtu delivered in
       2025 against a $3.52 Henry Hub -- a Texas basis of 0.891.  The planning-year
       Henry Hub times that basis is the ERCOT-wide delivered price.

    3  CARRY THE DIFFERENTIALS FORWARD ADDITIVELY.  Gas basis is quoted as a difference
       in $/MMBtu, not a ratio: Waha trades at Henry Hub minus so many cents.  Scaling
       multiplicatively would shrink the Permian discount in absolute terms, which is
       not how the market works.

    4  NARROW THE PERMIAN AND PANHANDLE DISCOUNT.  2025 was the trough of a takeaway
       constraint -- Waha averaged $0.17/MMBtu across 2024 and was negative on 36% of
       days.  Carrying that forward assumes the constraint never clears; it does, with
       >10 Bcf/d of new pipeline landing 2026-29, and the forward basis has already gone
       from -$3.60 to -$1.07, about 30% of its former width.  Far West and North keep
       30% of their measured discount to the ERCOT-wide price.  The other six zones are
       left as measured: Coast and South carry a Gulf LNG demand-pull premium that is
       widening, not narrowing.

The only judgement is the Henry Hub level, and it is stated as one number per year.

    python 74_gas_year.py
    -> out/zonal_gas_2030.json (re-derived, checked against the shipped table)
       out/zonal_gas_2035.json
"""
import json
import numpy as np
import pandas as pd

OUT = 'build/out'

# ---------------------------------------------------------------- the two judgements
#  2030: AEO2026 puts Henry Hub at $5-6/MMBtu real 2025$ through the early 2030s; the
#        conservative low end is taken.
#  2035: AEO2026 Counterfactual Baseline Henry Hub for 2035, $5.05 real 2025$ (the same
#        release also gives Texas Reliability Entity electric-power delivered gas at
#        $5.099 and US electric-power at $4.98, both consistent with this construction).
HENRY_HUB = {2030: 5.00, 2035: 5.05}
NARROW_ZONES, KEEP_FRACTION = ('FarWest', 'North'), 0.30

ZONES = ['Coast', 'East', 'FarWest', 'North', 'NorthCentral', 'South',
         'SouthCentral', 'West']
HR = {'NGCC': 6.226, 'NGCT': 9.142, 'NGST': 10.5}
VOM = {'NGCC': 3.57, 'NGCT': 4.17, 'NGST': 4.17}

# --------------------------------------------- step 1, measured, from 62_zonal_gas.py
M = json.load(open(f'{OUT}/zonal_gas.json'))
HH_2025 = M['henry_hub_2025']                       # 3.52
ERCOT_2025 = M['ercot_delivered_2025']              # 3.137, quantity-weighted
BASIS = M['texas_basis']                            # 0.891
MEASURED = M['zonal_2025_measured']                 # per zone, 2025 $/MMBtu
VOL = {'Coast': 125.0, 'East': 18.2, 'FarWest': 83.5, 'North': 74.6,
       'NorthCentral': 70.5, 'South': 29.2, 'SouthCentral': 147.8, 'West': 0.0}

print(f'measured 2025: ERCOT-wide ${ERCOT_2025:.3f}/MMBtu against Henry Hub '
      f'${HH_2025:.2f} -> Texas basis {BASIS:.4f}')
print(f'zonal 2025 measured: '
      f'{ {z: round(MEASURED[z], 3) for z in ZONES if z in MEASURED} }\n')


def build(year):
    hh = HENRY_HUB[year]
    target = hh*BASIS                               # step 2
    shift = target - ERCOT_2025                     # step 3
    g = {}
    for z in ZONES:
        p = MEASURED[z] + shift if z in MEASURED else target
        if z in NARROW_ZONES:                       # step 4
            p = target - KEEP_FRACTION*(target - p)
        g[z] = round(float(p), 3)
    vw = sum(g[z]*VOL[z] for z in ZONES)/sum(VOL.values())
    return dict(year=year, henry_hub=hh, texas_basis=round(BASIS, 4),
                ercot_wide_before_narrowing=round(target, 3),
                additive_shift=round(shift, 4),
                volume_weighted_after_narrowing=round(vw, 3),
                narrowed_zones=list(NARROW_ZONES), keep_fraction=KEEP_FRACTION,
                gas_by_zone=g)


R = {y: build(y) for y in HENRY_HUB}

print(f'{"zone":14s} {"2025 meas":>10s} {"2030":>8s} {"2035":>8s} {"change":>8s}  '
      f'{"NGCC 30":>8s} {"NGCC 35":>8s}')
for z in ZONES:
    a, b = R[2030]['gas_by_zone'][z], R[2035]['gas_by_zone'][z]
    m = MEASURED.get(z, float('nan'))
    print(f'{z:14s} {m:10.3f} {a:8.3f} {b:8.3f} {b-a:+8.3f}  '
          f'{HR["NGCC"]*a+VOM["NGCC"]:8.2f} {HR["NGCC"]*b+VOM["NGCC"]:8.2f}')
for y in (2030, 2035):
    print(f'{y}: Henry Hub ${R[y]["henry_hub"]:.2f} x basis {BASIS:.4f} = '
          f'${R[y]["ercot_wide_before_narrowing"]:.3f} ERCOT-wide before narrowing; '
          f'shift {R[y]["additive_shift"]:+.3f}; volume-weighted after narrowing '
          f'${R[y]["volume_weighted_after_narrowing"]:.3f}')

# ------------------------------------------------------- check against what is shipped
shipped = json.load(open(f'{OUT}/zonal_gas_2030.json'))
bad = {z: (shipped[z], R[2030]['gas_by_zone'][z]) for z in ZONES
       if abs(shipped[z] - R[2030]['gas_by_zone'][z]) > 0.0015}
print(f'\n2030 reproduction check against out/zonal_gas_2030.json: '
      f'{"OK, all eight zones match" if not bad else bad}')
assert not bad, 'the 2030 construction does not reproduce the shipped table'

json.dump(R[2035], open(f'{OUT}/zonal_gas_2035.json', 'w'), indent=1)
json.dump(R[2030], open(f'{OUT}/zonal_gas_2030_derivation.json', 'w'), indent=1)
print(f'written {OUT}/zonal_gas_2035.json')
