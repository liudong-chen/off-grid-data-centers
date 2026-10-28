"""Zonal delivered gas price for ERCOT, recomputed from EIA-923 with real provenance.

The case carried $3.891/MMBtu ERCOT-wide, attributed to EIA-923 fuel receipts, but that
figure does not reproduce from the file. This rebuilds the zonal prices from scratch:

  EIA-860 plant file          plant coordinates
  tx_zones.json               the eight ERCOT weather-zone polygons
  EIA-923 Page 5              2025 gas receipts, quantity and delivered cost per plant

  -> point-in-polygon assigns each plant to a zone
  -> quantity-weighted delivered $/MMBtu per zone, which fixes the DIFFERENTIALS
  -> the LEVEL is then set to the 2030 planning basis:

       AEO2026 Henry Hub, early 2030s, real 2025$   $5.00  (low end of $5-6)
       x Texas basis measured in 2025               0.891  ($3.137 delivered / $3.52 HH)
       = ERCOT-wide delivered, 2030                 $4.46

Both steps are measured; only the choice of AEO's low end is a judgement, and it is the
conservative one.
"""
import json, re
import numpy as np, pandas as pd
from shapely.geometry import shape, Point

E = '1_source_data/parameter_setting/eia8602025ER'
G923 = ('1_source_data/parameter_setting/Gas/'
        'EIA923_Schedules_2_3_4_5_M_12_2025_Early_Release_30JUN2026.xlsx')
OUT = 'build/out'

HH_2025, DELIVERED_2025, HH_2030 = 3.52, None, 5.00
ZONES = ['Coast', 'East', 'FarWest', 'North', 'NorthCentral', 'South',
         'SouthCentral', 'West']
HR = {'NGCC': 6.226, 'NGCT': 9.142, 'NGST': 10.5}
VOM = {'NGCC': 3.57, 'NGCT': 4.17, 'NGST': 4.17}

# ------------------------------------------------------------------- plants -> zones
p = pd.read_excel(f'{E}/2___Plant_Y2025_Early_Release.xlsx', skiprows=2)
p.columns = [re.sub(r'\s+', ' ', str(c)).strip() for c in p.columns]
latc = [c for c in p.columns if c.lower() == 'latitude'][0]
lonc = [c for c in p.columns if c.lower() == 'longitude'][0]
p = p[p['State'] == 'TX'][['Plant Code', 'Plant Name', 'County', latc, lonc]].copy()
p = p.rename(columns={latc: 'lat', lonc: 'lon'})
p['lat'] = pd.to_numeric(p['lat'], errors='coerce')
p['lon'] = pd.to_numeric(p['lon'], errors='coerce')
p = p.dropna(subset=['lat', 'lon'])

Z = {z: shape(v) for z, v in json.load(open(f'{OUT}/tx_zones.json')).items()}


def zone_of(lat, lon):
    pt = Point(lon, lat)
    for z in ZONES:
        if Z[z].contains(pt):
            return z
    # nearest zone if the point falls in a boundary gap
    return min(ZONES, key=lambda z: Z[z].distance(pt))


p['zone'] = [zone_of(a, b) for a, b in zip(p.lat, p.lon)]

# ------------------------------------------------------------ EIA-923 gas receipts
d = pd.read_excel(G923, sheet_name='Page 5 Fuel Receipts and Costs', skiprows=5)
d.columns = [re.sub(r'\s+', ' ', str(c)).strip() for c in d.columns]
d = d[(d['Plant State'] == 'TX') &
      (d['FUEL_GROUP'].astype(str).str.contains('Natural Gas', case=False, na=False))].copy()
for c, n in (('QUANTITY', 'Q'), ('Average Heat Content', 'HC'), ('FUEL_COST', 'C')):
    d[n] = pd.to_numeric(d[c], errors='coerce')
d = d.dropna(subset=['Q', 'HC', 'C'])
d = d[(d.C > 50) & (d.C < 3000) & (d.Q > 0)]
d['MMBtu'] = d.Q*d.HC
d['dollars'] = d.MMBtu*d.C/100.0

d = d.merge(p[['Plant Code', 'zone']], left_on='Plant Id', right_on='Plant Code', how='left')
unmatched = d.MMBtu[d.zone.isna()].sum()/d.MMBtu.sum()
d = d.dropna(subset=['zone'])

zg = d.groupby('zone').apply(
    lambda s: pd.Series({'MMBtu': s.MMBtu.sum(), 'price': s.dollars.sum()/s.MMBtu.sum(),
                         'plants': s['Plant Id'].nunique()}))
ercot_2025 = float(d.dollars.sum()/d.MMBtu.sum())

print(f'plants located: {len(p)};  receipts volume unmatched to a zone: {unmatched:.1%}')
print(f'ERCOT-wide 2025 delivered (quantity-weighted): ${ercot_2025:.3f}/MMBtu')
print(f'Henry Hub 2025: ${HH_2025:.2f}  ->  Texas basis {ercot_2025/HH_2025:.4f}\n')

basis = ercot_2025/HH_2025
target = round(HH_2030*basis, 3)          # $4.456 ERCOT-wide, the agreed level
# Gas basis is a DIFFERENTIAL in $/MMBtu, not a ratio: Waha is quoted as Henry Hub minus
# so many cents. Scaling zones multiplicatively would shrink the Permian discount in
# absolute terms, which is not how the market works. The zonal differentials measured in
# 2025 are therefore carried forward ADDITIVELY, with the shift set so the
# volume-weighted ERCOT-wide price lands on the agreed level.
shift = target - ercot_2025
print(f'2030 planning basis: AEO Henry Hub ${HH_2030:.2f} x basis {basis:.4f} '
      f'= ${target:.3f}/MMBtu ERCOT-wide')
print(f'additive shift applied to every zone: {shift:+.3f} $/MMBtu\n')

gas = {}
for z in ZONES:
    if z in zg.index and zg.loc[z, 'MMBtu'] > 1e5:
        gas[z] = round(float(zg.loc[z, 'price']) + shift, 3)
    else:
        gas[z] = round(target, 3)
chk = sum(gas[z]*float(zg.loc[z,'MMBtu']) for z in ZONES if z in zg.index and zg.loc[z,'MMBtu']>1e5)
chk /= sum(float(zg.loc[z,'MMBtu']) for z in ZONES if z in zg.index and zg.loc[z,'MMBtu']>1e5)
print(f'check: volume-weighted 2030 price = ${chk:.3f}/MMBtu (target ${target:.3f})\n')

print(f'{"zone":14s} {"plants":>6s} {"volume M":>9s} {"2025 $":>8s} {"2030 $":>8s} '
      f'{"NGCC":>7s} {"NGCT":>7s} {"NGST":>7s}')
for z in ZONES:
    have = z in zg.index and zg.loc[z, 'MMBtu'] > 1e5
    n = int(zg.loc[z, 'plants']) if have else 0
    v = zg.loc[z, 'MMBtu']/1e6 if have else 0.0
    p25 = float(zg.loc[z, 'price']) if have else float('nan')
    print(f'{z:14s} {n:6d} {v:9,.1f} {p25:8.3f} {gas[z]:8.3f} '
          f'{HR["NGCC"]*gas[z]+VOM["NGCC"]:7.2f} {HR["NGCT"]*gas[z]+VOM["NGCT"]:7.2f} '
          f'{HR["NGST"]*gas[z]+VOM["NGST"]:7.2f}')

old = {'FarWest': 5.000, 'North': 5.000, 'West': 5.000, 'South': 5.604,
       'NorthCentral': 4.913, 'SouthCentral': 4.688, 'Coast': 5.242, 'East': 5.000}
print(f'\n{"zone":14s} {"old $":>8s} {"new $":>8s} {"change":>8s}')
for z in ZONES:
    print(f'{z:14s} {old[z]:8.3f} {gas[z]:8.3f} {100*(gas[z]/old[z]-1):7.1f}%')

json.dump(dict(method='EIA-923 Page 5 2025 receipts, plants assigned to ERCOT weather '
                      'zones by EIA-860 coordinates against tx_zones.json; zonal '
                      'differentials measured, level set to AEO2026 Henry Hub low end '
                      'times the measured Texas basis',
               henry_hub_2025=HH_2025, ercot_delivered_2025=round(ercot_2025, 4),
               texas_basis=round(basis, 4), henry_hub_2030_AEO=HH_2030,
               ercot_delivered_2030=round(target, 3), additive_shift=round(shift, 4),
               gas_by_zone=gas, heat_rate=HR, vom=VOM,
               unmatched_volume_share=round(float(unmatched), 4),
               zonal_2025_measured={z: round(float(zg.loc[z, 'price']), 4)
                                    for z in ZONES if z in zg.index}),
          open(f'{OUT}/zonal_gas.json', 'w'), indent=1)
print(f'\nwritten {OUT}/zonal_gas.json')
