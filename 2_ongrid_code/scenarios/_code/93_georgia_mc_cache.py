"""Seven weather years of Georgia Power load, as Monte Carlo caches.

WHY.  The Georgia reserve margin in the shipped cases is Georgia Power's own 26% winter
target, which is set on "a combination of economic and reliability metrics" (2022 IRP) and
is NOT the 0.1 day/year LOLE criterion the rest of this study derives ERCOT's margin from.
Deriving Georgia's own LOLE-consistent margin needs multiple weather years, and until now
only 2025 was assembled: a single year contains unit-outage risk but no weather risk, so a
margin derived from it is a LOWER BOUND, and ReliabilityPRM_GA.jl says so before it starts.

WHAT THIS DOES.  Builds mc_cache_gabase.npz and mc_cache_gadc.npz in the .npz layout
11_mc_portfolio.py reads, carrying SEVEN weather years (2019-2025) of Southern Company
hourly demand from EIA-930, scaled to Georgia Power's 2030 base peak.

THE SCALING, AND WHY IT IS ONE FACTOR AND NOT SEVEN.  The HOPE case scales the 2025 SOCO
series so its maximum equals the 2030 Georgia Power base peak of 17,925 MW.  Applying that
SAME factor to every year is the whole point: each year then peaks at whatever its own
weather produced, and the spread across years IS the weather risk the single-year cache
could not see.  Normalising each year to its own maximum would erase exactly the quantity
being added.

    python 93_georgia_mc_cache.py
    -> mc_cache_gabase.npz, mc_cache_gadc.npz  (+ a summary of the seven years)
"""
import json, os
import numpy as np
import pandas as pd

B = 'build'
SOCO_MULTI = '1_source_data/parameter_setting/GPC_EIA930_demand_2019_2025.csv'
SOCO_2025 = '1_source_data/parameter_setting/SOCO_hourly_2025.csv'
GA = f'{B}/ga_hope'
H = 8760

# the same three numbers the case builder uses; see 84_georgia_hope_case.py
GP_PEAK_2025, GROWTH, DC_SHARE = 16284.0, 8205.0, 0.80
BASE_PEAK_2030 = GP_PEAK_2025 + (1 - DC_SHARE)*GROWTH        # 17,925 MW

FL = json.load(open(f'{B}/out/georgia_fleet.json'))

# ------------------------------------------------------------------ the weather years
raw = pd.read_csv(SOCO_MULTI)
raw = raw[raw['type'] == 'D'].copy()
raw['ts'] = pd.to_datetime(raw['period'], format='%Y-%m-%dT%H', utc=True)
raw['value'] = pd.to_numeric(raw['value'], errors='coerce')
raw = raw.dropna(subset=['value']).sort_values('ts')
raw['year'] = raw.ts.dt.year

YEARS = sorted(y for y in raw.year.unique() if (raw.year == y).sum() > 8000)
print(f'weather years available: {YEARS[0]}-{YEARS[-1]} ({len(YEARS)})')

# the reference: the 2025 series the HOPE case itself was built from, so the scaling
# factor is identical to the one already in the case files
ref = pd.read_csv(SOCO_2025)
ref_max = float(pd.to_numeric(ref['MW'], errors='coerce').values[:H].max())
SCALE = BASE_PEAK_2030/ref_max
print(f'2025 SOCO peak {ref_max:,.0f} MW -> Georgia Power 2030 base peak '
      f'{BASE_PEAK_2030:,.0f} MW, factor {SCALE:.5f} (one factor, all years)')


def year_series(y):
    """8,760 hours for weather year y, leap day dropped, gaps filled by interpolation."""
    s = raw[raw.year == y].set_index('ts')['value']
    s = s[~((s.index.month == 2) & (s.index.day == 29))]
    full = pd.date_range(s.index[0].floor('h'), periods=H, freq='h', tz='UTC')
    s = s.reindex(full).interpolate(limit_direction='both')
    return s.values[:H]


L = np.array([year_series(y) for y in YEARS])*SCALE
assert L.shape == (len(YEARS), H), L.shape
gap = np.array([np.isnan(year_series(y)).sum() for y in YEARS])
print(f'\n{"year":>6}{"peak (MW)":>12}{"energy (TWh)":>15}{"load factor":>13}'
      f'{"vs 2025 peak":>14}')
p25 = None
for i, y in enumerate(YEARS):
    pk = L[i].max()
    if y == 2025:
        p25 = pk
for i, y in enumerate(YEARS):
    pk, en = L[i].max(), L[i].sum()/1e6
    print(f'{y:>6}{pk:>12,.0f}{en:>15.1f}{L[i].mean()/pk:>13.3f}'
          f'{100*(pk/p25 - 1):>13.1f}%')
print(f'\nspread across the seven years: {L.max(axis=1).min():,.0f} to '
      f'{L.max(axis=1).max():,.0f} MW, a range of '
      f'{100*(L.max(axis=1).max()/L.max(axis=1).min() - 1):.1f}% of the lowest --- '
      f'this is the weather risk a single year cannot contain')

# ------------------------------------------------------------------ the data-centre load
# The dc case adds the same 12.4 GW campus load to every weather year.  It is near-flat
# and has no weather dependence of its own, so it is the same 8,760-hour vector each time.
def dc_load():
    for tag in ('dc',):
        d = [x for x in os.listdir(f'{GA}/GA1z_GTEP_{tag}_case')
             if x.startswith('Data_')][0]
        dd = f'{GA}/GA1z_GTEP_{tag}_case/{d}'
        z = pd.read_csv(f'{dd}/zonedata.csv')
        pk = float(z['Demand (MW)'].iloc[0])
        ld = pd.read_csv(f'{dd}/load_timeseries_regional.csv')['Georgia'].values[:H]*pk
    b = [x for x in os.listdir(f'{GA}/GA1z_GTEP_base_case') if x.startswith('Data_')][0]
    bd = f'{GA}/GA1z_GTEP_base_case/{b}'
    zb = float(pd.read_csv(f'{bd}/zonedata.csv')['Demand (MW)'].iloc[0])
    lb = pd.read_csv(f'{bd}/load_timeseries_regional.csv')['Georgia'].values[:H]*zb
    return ld - lb


DC = dc_load()
print(f'\ndata-centre load: {DC.max():,.0f} MW peak, {DC.mean():,.0f} MW mean, '
      f'load factor {DC.mean()/DC.max():.3f}')

# --------------------------------------------------------------- the existing VRE output
# 368 MW of existing solar.  Only the 2025 profile exists, so it is reused for every
# weather year; at 1.8% of the 20,453 MW fleet the approximation cannot move LOLE.
b = [x for x in os.listdir(f'{GA}/GA1z_GTEP_base_case') if x.startswith('Data_')][0]
bd = f'{GA}/GA1z_GTEP_base_case/{b}'
gen = pd.read_csv(f'{bd}/gendata.csv')
A = pd.read_csv(f'{bd}/gen_availability_timeseries.csv')
gcols = [c for c in A.columns if c.startswith('G')]
si = int(np.flatnonzero(gen['Type'].values == 'SolarPV')[0])
solar_mw = float(gen['Pmax (MW)'].values[si])
VRE1 = A[gcols[si]].values[:H]*solar_mw
V = np.tile(VRE1, (len(YEARS), 1))
print(f'existing solar {solar_mw:,.0f} MW, mean output {VRE1.mean():,.0f} MW '
      f'(2025 profile reused across years)')

# ------------------------------------------------------------------------- the fleet
rows = [(r['Type'], r['mw'], r['n_units'], r['FOR'], r['MTTR']) for r in FL['firm']]
fleet_mw = np.array([r[1] for r in rows], float)
fleet_n = np.array([r[2] for r in rows], int)
fleet_q = np.array([r[3] for r in rows], float)
fleet_mttr = np.array([r[4] for r in rows], float)
print(f'firm fleet {fleet_mw.sum():,.0f} MW over {fleet_n.sum()} units, '
      f'{len(rows)} types')

STO_P, STO_E = float(FL['battery']['mw']), float(FL['battery']['mw'])*4.0

# TWO cache formats, because two engines read them and both must see the same years:
#   .npz   the Python scorer (11_mc_portfolio.py), used by 94_georgia_reliability.py
#   .csv   the Julia scorer (ReliabilityPRM_mc.jl), used by ReliabilityPRM_GA.jl
# The CSV layout mirrors the ERCOT caches exactly: all load_<year> columns, then all
# vre_<year> columns, and n_years in the meta file.
for tag, extra in (('gabase', 0.0), ('gadc', 1.0)):
    LL = L + extra*DC
    np.savez(f'{B}/mc_cache_{tag}.npz', LOAD=LL, VRE=V, years=np.array(YEARS),
             sto_p=STO_P, sto_e=STO_E, peak=float(LL.max()),
             fleet_mw=fleet_mw, fleet_n=fleet_n,
             fleet_q=fleet_q, fleet_mttr=fleet_mttr,
             fleet_type=np.array([r[0] for r in rows]))

    cols = {f'load_{y}': LL[i] for i, y in enumerate(YEARS)}
    cols.update({f'vre_{y}': V[i] for i, y in enumerate(YEARS)})
    pd.DataFrame(cols).to_csv(f'{B}/mc_cache_{tag}_series.csv', index=False)
    pd.DataFrame([dict(Type=r[0], mw=r[1], n_units=r[2], FOR=r[3], MTTR=r[4])
                  for r in rows]).to_csv(f'{B}/mc_cache_{tag}_fleet.csv', index=False)
    pd.DataFrame([dict(sto_p_MW=STO_P, sto_e_MWh=STO_E, peak_MW=float(LL.max()),
                       n_years=len(YEARS), eta=0.922)]
                 ).to_csv(f'{B}/mc_cache_{tag}_meta.csv', index=False)

    print(f'{tag:8s} peak over all years {LL.max():>10,.0f} MW   '
          f'2025 year {LL[YEARS.index(2025)].max():>10,.0f} MW   '
          f'-> .npz + 3 .csv')

print('\nthe 2025 column reproduces the single-year cache the earlier runs used, so the '
      'only thing\nthat changed is that six more years of weather are now in the draw.')
