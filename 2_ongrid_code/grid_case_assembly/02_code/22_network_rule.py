"""A defensible interface rule, and a candidate set sized against what ERCOT builds.

THE PROBLEM. Interfaces in the current case are the sum of the thermal rating of
every in-service branch crossing a zone boundary. That over-states transfer capability
three separate ways, and this script removes all three.

THE RULE, in the order the tightenings apply:

  1. BULK VOLTAGE ONLY. A 115 kV or 161 kV circuit crossing a zone boundary serves
     local load on both sides; it cannot be dispatched to move a gigawatt across
     Texas. Only the top voltage class counts. (In the Texas2k synthetic case that
     class is 500 kV; in the real ERCOT system it is 345 kV.)

  2. N-1. A planning transfer limit has to survive the loss of one circuit, so the
     largest single circuit on the cut is removed. This is the standard planning
     convention and it is what makes the number a *limit* rather than a nameplate.

  3. PUBLISHED STABILITY LIMITS. Where ERCOT publishes a generic transmission
     constraint for the corridor, it binds below thermal and is used instead.

Each step has a physical reason, none is a fitted fudge factor, and the effect of
each is reported separately so a referee can see where the number moved.

WHAT THIS DOES NOT FIX, and the analysis says so plainly: at the solved dispatch the
total congestion rent across all seventeen interfaces is about $1 M/yr against a
candidate set costing $1,170 M/yr. Zonal marginal costs differ by at most the gas
price spread, $4.44/MWh, so there is almost nothing to arbitrage. Tightening
interfaces is right on its own terms but it is not, by itself, what would make
transmission expansion economic here.
"""
import os
import numpy as np, pandas as pd

OUT = 'build/out'
CASE = 'build/hope_update'
RAWF = ('1_source_data/network/Texas2k/Texas2k_series2025/'
        'Texas2k_series25_case1_summerpeak/Texas2k_series25_case1_summerpeak.RAW')
BUS = '1_source_data/network/Texas2k/tx2k_s25_bus_coords.csv'
AREA2ZONE = {1: 'FarWest', 2: 'North', 3: 'West', 4: 'South',
             5: 'NorthCentral', 6: 'SouthCentral', 7: 'Coast', 8: 'East'}
BULK_KV = 345.0          # nominal bulk threshold; Texas2k's top class is 500 kV

# ERCOT 2025 RTP Economic Study Stability Interface Limits Update, 2030 values.
GTC = {('FarWest', 'West'): (16200.0, 'West Texas Export'),
       ('North', 'NorthCentral'): (4440.0, 'Panhandle'),
       ('Coast', 'East'): (5595.0, 'North to Houston, mapped to Coast-East (judgement)')}

bus = pd.read_csv(BUS)
bus['zone'] = bus['area'].map(AREA2ZONE)
B2Z, B2KV = dict(zip(bus.bus, bus.zone)), dict(zip(bus.bus, bus.baseKV))
lines = open(RAWF).read().split('\n')


def section(a_key, b_key):
    a = next(i for i, l in enumerate(lines) if a_key in l)
    b = next(i for i, l in enumerate(lines) if b_key in l and i > a)
    return [l for l in lines[a+1:b] if l.strip() and not l.strip().startswith('@')]


recs = []
for l in section('BEGIN BRANCH DATA', 'END OF BRANCH DATA'):
    p = [x.strip() for x in l.split(',')]
    try:
        recs.append((int(p[0]), int(p[1]), float(p[7]), int(float(p[23]))))
    except (ValueError, IndexError):
        pass
tx = section('BEGIN TRANSFORMER DATA', 'END OF TRANSFORMER DATA')
k = 0
while k < len(tx):
    p = [x.strip() for x in tx[k].split(',')]
    try:
        i, j, kk = int(p[0]), int(p[1]), int(p[2])
    except (ValueError, IndexError):
        k += 1; continue
    try:
        recs.append((i, j, float([x.strip() for x in tx[k+2].split(',')][3]),
                     int(p[11])))
    except (ValueError, IndexError):
        pass
    k += 5 if kk != 0 else 4

br = pd.DataFrame(recs, columns=['i', 'j', 'rate', 'stat'])
br['zi'] = br.i.map(B2Z); br['zj'] = br.j.map(B2Z)
br['kv'] = np.minimum(br.i.map(B2KV), br.j.map(B2KV))
br = br.dropna(subset=['zi', 'zj', 'kv'])
br = br[(br.stat == 1) & (br.zi != br.zj) & (br.rate > 0)].copy()
br['a'] = np.minimum(br.zi, br.zj); br['b'] = np.maximum(br.zi, br.zj)

rows = []
for (a, b), d in br.groupby(['a', 'b']):
    bulk = d[d.kv >= BULK_KV]
    s1 = d.rate.sum()                                   # step 0: everything
    s2 = bulk.rate.sum()                                # step 1: bulk voltage only
    s3 = max(s2 - (bulk.rate.max() if len(bulk) else 0), 0.0)   # step 2: N-1
    g, note = GTC.get((a, b), (np.nan, ''))
    s4 = min(s3, g) if g == g else s3                   # step 3: stability limit
    rows.append(dict(a=a, b=b, n_all=len(d), n_bulk=len(bulk),
                     all_MW=s1, bulk_MW=s2, n1_MW=s3, final_MW=s4,
                     gtc_MW=g, note=note,
                     binding='GTC' if (g == g and g < s3) else
                             ('N-1' if s3 < s2 else 'bulk voltage')))
T = pd.DataFrame(rows).sort_values(['a', 'b']).reset_index(drop=True)

print('=== interface capability, MW, as each tightening is applied ===')
print(f"{'corridor':<26}{'all':>9}{'bulk kV':>9}{'+N-1':>9}{'+GTC':>9}"
      f"{'vs all':>8}  what binds")
for r in T.itertuples():
    print(f'{r.a+"-"+r.b:<26}{r.all_MW:>9,.0f}{r.bulk_MW:>9,.0f}{r.n1_MW:>9,.0f}'
          f'{r.final_MW:>9,.0f}{r.final_MW/r.all_MW:>7.0%}  {r.binding}')
print(f'{"TOTAL":<26}{T.all_MW.sum():>9,.0f}{T.bulk_MW.sum():>9,.0f}'
      f'{T.n1_MW.sum():>9,.0f}{T.final_MW.sum():>9,.0f}'
      f'{T.final_MW.sum()/T.all_MW.sum():>7.0%}')

# ---- what it does to each zone's ability to import or export
print('\n=== total interface capability per zone ===')
zb = pd.read_csv(f'{CASE}/Data_TX8z_GTEP_dc/zonedata.csv').set_index('Zone_id')
old = pd.read_csv(f'{CASE}/Data_TX8z_GTEP_dc/linedata.csv')
print(f"{'zone':<15}{'2030 peak':>11}{'now':>10}{'revised':>10}{'change':>9}"
      f"{'revised/peak':>14}")
for z in sorted(zb.index):
    o = old[(old.From_zone == z) | (old.To_zone == z)]['Forward Capacity (MW)'].sum()
    n = T[(T.a == z) | (T.b == z)].final_MW.sum()
    pk = zb.loc[z, 'Demand (MW)']
    print(f'{z:<15}{pk:>11,.0f}{o:>10,.0f}{n:>10,.0f}{n/o-1:>8.0%}{n/pk:>13.2f}x')

# ---- write the revised files
d = T[['a', 'b', 'final_MW']].copy()
d.columns = ['From_zone', 'To_zone', 'Forward Capacity (MW)']
d['Reverse Capacity (MW)'] = d['Forward Capacity (MW)']
d['Loss (%)'] = 0.0
d['N_branches'] = T.n_bulk.values
d = d[d['Forward Capacity (MW)'] > 0]
d.to_csv(f'{OUT}/linedata_bulkN1.csv', index=False)

# ---- a candidate set that can express an ERCOT-scale build
# Unit cost checked against CREZ: $6.9 bn for 3,600 miles of 345 kV double circuit
# is $1.92 M/mile, and a 345 kV double circuit carries about 1,800 MW, so roughly
# $1,067 per MW-mile - which is where the $1,100 used here comes from.
UNIT, CRF = 1100.0, 0.0839
oldc = pd.read_csv(f'{CASE}/Data_TX8z_GTEP_dc/linedata_candidate.csv')
oldc['miles'] = oldc['Cost (M$)']*1e6/(UNIT*3000.0*CRF)
cand = []
for r in oldc.itertuples():
    mi = r.miles
    for tag, cap, mult in (('a', 3000.0, 1.00),      # 345 kV double circuit
                           ('b', 6000.0, 1.85)):     # a second circuit / higher voltage
        cand.append(dict(From_zone=r.From_zone, To_zone=r.To_zone,
                         **{'Forward Capacity (MW)': cap, 'Reverse Capacity (MW)': cap,
                            'Cost (M$)': round(UNIT*3000.0*mi*CRF*mult/1e6, 3),
                            'X': round(r.X*3000.0/cap, 5), 'Loss (%)': 0.0}))
C = pd.DataFrame(cand)
C.to_csv(f'{OUT}/linedata_candidate_expanded.csv', index=False)
print(f'\nrevised interfaces  -> {OUT}/linedata_bulkN1.csv  ({len(d)} corridors)')
print(f'expanded candidates -> {OUT}/linedata_candidate_expanded.csv '
      f'({len(C)} options, {C["Forward Capacity (MW)"].sum():,.0f} MW, '
      f'${C["Cost (M$)"].sum():,.0f} M/yr if all built)')
T.to_csv(f'{OUT}/interface_rule.csv', index=False)
