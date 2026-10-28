"""Inter-zone transfer capability for the 8-zone model.

Two sources, and they disagree by a lot:

  THERMAL SUM   sum of RATE1 over every Texas2k branch crossing a zone boundary.
                An upper bound: it ignores stability and voltage limits.
  ERCOT GTC     the stability interface limits ERCOT actually enforces, from the
                2025 RTP Economic Study Stability Interface Limits Update.

Where a GTC applies to a corridor we use the GTC. Elsewhere we use the thermal
sum. Both are written out so the difference is visible.
"""
import os, re, json
import numpy as np, pandas as pd

RAWF = ('1_source_data/network/Texas2k/Texas2k_series2025/'
        'Texas2k_series25_case1_summerpeak/Texas2k_series25_case1_summerpeak.RAW')
BUS  = '1_source_data/network/Texas2k/tx2k_s25_bus_coords.csv'
OUT  = 'build/out'; os.makedirs(OUT, exist_ok=True)

AREA2ZONE = {1:'FarWest', 2:'North', 3:'West', 4:'South',
             5:'NorthCentral', 6:'SouthCentral', 7:'Coast', 8:'East'}

bus = pd.read_csv(BUS)
bus['zone'] = bus['area'].map(AREA2ZONE)
B2Z = dict(zip(bus.bus, bus.zone))

lines = open(RAWF).read().split('\n')
def section(start_key, end_key):
    a = next(i for i, l in enumerate(lines) if start_key in l)
    b = next(i for i, l in enumerate(lines) if end_key in l and i > a)
    return [l for l in lines[a + 1:b] if l.strip() and not l.strip().startswith('@')]

# ---------- AC branches ----------
recs = []
for l in section('BEGIN BRANCH DATA', 'END OF BRANCH DATA'):
    p = [x.strip() for x in l.split(',')]
    try:
        # I,J,CKT,R,X,B,NAME,RATE1..RATE12,GI,BI,GJ,BJ,STAT,MET,LEN,...
        #  0 1  2  3 4 5   6    7  ..  18   19 20 21 22   23  24  25
        i, j, rate, stat = int(p[0]), int(p[1]), float(p[7]), int(float(p[23]))
    except (ValueError, IndexError):
        continue
    recs.append((i, j, rate, stat, 'line'))

# ---------- transformers: 4 (2-winding) or 5 (3-winding) physical lines each ----
tx = section('BEGIN TRANSFORMER DATA', 'END OF TRANSFORMER DATA')
k = 0
while k < len(tx):
    p = [x.strip() for x in tx[k].split(',')]
    try:
        i, j, kk = int(p[0]), int(p[1]), int(p[2])
    except (ValueError, IndexError):
        k += 1; continue
    nrec = 5 if kk != 0 else 4
    try:
        stat = int(p[11])
        w1 = [x.strip() for x in tx[k + 2].split(',')]
        rate = float(w1[3])
        recs.append((i, j, rate, stat, 'xfmr'))
    except (ValueError, IndexError):
        pass
    k += nrec

br = pd.DataFrame(recs, columns=['i', 'j', 'rate', 'stat', 'kind'])
br['zi'] = br.i.map(B2Z); br['zj'] = br.j.map(B2Z)
br = br.dropna(subset=['zi', 'zj'])
br = br[(br.stat == 1) & (br.zi != br.zj) & (br.rate > 0)]
br['a'] = np.minimum(br.zi, br.zj); br['b'] = np.maximum(br.zi, br.zj)
print(f'{len(br)} in-service branches cross a zone boundary '
      f'({(br.kind=="line").sum()} lines, {(br.kind=="xfmr").sum()} transformers)')

th = (br.groupby(['a', 'b'])
        .agg(thermal_MW=('rate', 'sum'), n_branches=('rate', 'size'))
        .reset_index())

# ---------- ERCOT stability interface limits, 2025 RTP (2030 values) ----------
# https://www.ercot.com/files/docs/2025/08/05/2025-Regional-Transmission-Plan-RTP-
#   Economic-Study-Stability-Interface-Limits-Update-8_04_25-.pdf
# 9999 in that table means "no limit needed", so only the real constraints appear.
GTC = {
    ('FarWest', 'West'):        (16200.0, 'West Texas Export, 2030 value'),
    ('North', 'NorthCentral'):  (4440.0,  'Panhandle'),
    # JUDGEMENT CALL: ERCOT's "North to Houston" interface has no single
    # counterpart in an 8-zone topology - Houston (Coast) is fed from East and
    # from SouthCentral. Applied to Coast-East as the northern path. Review.
    ('Coast', 'East'):          (5595.0,  'North to Houston, mapped to Coast-East (judgement)'),
}
gtc_rows = {tuple(sorted(k)): v for k, v in GTC.items()}

th['gtc_MW'] = [gtc_rows.get((a, b), (np.nan, ''))[0] for a, b in zip(th.a, th.b)]
th['gtc_note'] = [gtc_rows.get((a, b), (np.nan, ''))[1] for a, b in zip(th.a, th.b)]
th['limit_MW'] = np.where(th.gtc_MW.notna(), th.gtc_MW, th.thermal_MW).round(1)
th['basis'] = np.where(th.gtc_MW.notna(), 'ERCOT GTC', 'Texas2k thermal sum')

print('\n=== inter-zone corridors ===')
print(th[['a', 'b', 'n_branches', 'thermal_MW', 'gtc_MW', 'limit_MW', 'basis']]
      .round(0).to_string(index=False))
binding = th[th.gtc_MW.notna()]
if len(binding):
    print('\nwhere ERCOT publishes a stability limit, thermal sums overstate it by:')
    for r in binding.itertuples():
        print(f'  {r.a}-{r.b}: thermal {r.thermal_MW:,.0f} MW vs GTC {r.gtc_MW:,.0f} MW '
              f'({r.thermal_MW/r.gtc_MW:.1f}x)')

th.to_csv(f'{OUT}/interzone_capacity.csv', index=False)
print(f'\nwritten to {OUT}/interzone_capacity.csv')
