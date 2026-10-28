"""The three Monte Carlo cache files ReliabilityPRM_GA.jl reads, for Georgia.

`ReliabilityPRM_mc.jl` scores a portfolio from three small CSVs per case tag, sitting
beside the Julia file:

    mc_cache_<tag>_fleet.csv    Type, mw, n_units, FOR, MTTR   -- the firm fleet
    mc_cache_<tag>_series.csv   load_<year>, vre_<year>        -- one column pair per
                                                                  weather year
    mc_cache_<tag>_meta.csv     sto_p_MW, sto_e_MWh, peak_MW, n_years, eta

This writes them for tags `gabase` and `gadc`, in exactly the schema the engine expects,
so the Georgia case runs through the same scorer as ERCOT with nothing changed inside it.

    fleet   from 85_georgia_fleet.py: EIA-860 unit counts, Facts & Figures owned MW
    load    EIA-930 SOCO hourly, scaled to Georgia Power, plus the data-centre fleet
    vre     Georgia Power's 368 MW of existing solar on the Georgia profile
    storage 65 MW / 260 MWh of battery, eta 0.922

ONE REAL LIMITATION, STATED HERE RATHER THAN BURIED.  The ERCOT cache carries EIGHT
weather years, 2018-25, and the Monte Carlo rotates through them so the LOLE reflects
weather risk as well as machine risk.  For Georgia only 2025 was available (EIA-930 SOCO,
the file downloaded for this case), so every replication reuses the same year and the
scorer sees ONLY unit-outage risk.  That understates LOLE, and it understates it exactly
where Georgia is most exposed: the binding event Georgia Power plans for is an extreme
winter morning of the Elliott/January-2024 kind, which a single normal year does not
contain.  The derived PRM should therefore be read as a LOWER BOUND.

To fix it, drop more EIA-930 BALANCE files into parameter_setting and re-run: the script
picks up every year it finds and writes one column pair per year, and the Julia engine
rotates through however many there are.

    python 86_georgia_mc_cache.py
    -> mc_cache_gabase_*.csv, mc_cache_gadc_*.csv  (in ga_hope/, to copy beside the .jl)
"""
import glob, json, os
import numpy as np
import pandas as pd

B = 'build'
OUT = f'{B}/ga_hope'
PS = 'parameter_setting'
CD = f'{B}/TX8z_GTEP_cases_vre/TX8z_GTEP_dc_case/Data_TX8z_GTEP_dc'
R = 'DC_Load/load_profiles'
H = 8760

GP_PEAK_2025, GROWTH, DC_SHARE = 16284.0, 8205.0, 0.80
SOLAR_CF = 0.265*0.88
FLEET_MW, CAMELLIA = 12400.0, 3200.0
ETA = 0.922

F = json.load(open(f'{B}/out/georgia_fleet.json'))

# ------------------------------------------------------------------- weather years
# every EIA-930 BALANCE pair in parameter_setting becomes one weather year
years = {}
for f in sorted(glob.glob(f'{PS}/EIA930_BALANCE_*.csv')):
    y = int(os.path.basename(f).split('_')[2])
    years.setdefault(y, []).append(f)
SERIES = {}
for y, fs in sorted(years.items()):
    parts = []
    for f in fs:
        for ch in pd.read_csv(f, usecols=['Balancing Authority',
                                          'UTC Time at End of Hour',
                                          'Demand (MW) (Adjusted)'], chunksize=300000):
            parts.append(ch[ch['Balancing Authority'] == 'SOCO'])
    d = pd.concat(parts)
    d['t'] = pd.to_datetime(d['UTC Time at End of Hour'], format='mixed', utc=True)
    d = d.sort_values('t').drop_duplicates('t')
    v = pd.to_numeric(d['Demand (MW) (Adjusted)'], errors='coerce').values
    v = v[~np.isnan(v)][:H]
    if len(v) < H:                       # a leap year or a short file
        v = np.pad(v, (0, H - len(v)), mode='edge')
    SERIES[y] = v
    print(f'weather year {y}: {len(v)} hours, SOCO peak {v.max():,.0f} MW, '
          f'{v.sum()/1e6:.1f} TWh')
assert SERIES, 'no EIA930_BALANCE_*.csv found'
if len(SERIES) == 1:
    print('\n*** ONE WEATHER YEAR ONLY. The Monte Carlo will see unit-outage risk but '
          'not\n    weather risk, so the derived PRM is a LOWER BOUND. See the header.\n')

# ------------------------------------------------- Georgia Power load, per weather year
base_peak_2030 = GP_PEAK_2025 + (1 - DC_SHARE)*GROWTH
GPB = {y: v*(base_peak_2030/v.max()) for y, v in SERIES.items()}

d = pd.read_csv(f'{R}/dc_load_hourly_MW.csv', index_col=0, parse_dates=True)
d = d[~((d.index.month == 2) & (d.index.day == 29))].drop(columns=['TOTAL_MW'])
pk = d.max(); shape = (pk/pk.sum()).sort_values(ascending=False); src = list(shape.index)
rest = FLEET_MW - CAMELLIA
q = np.linspace(0, 1, 30)
w = np.interp(q, np.linspace(0, 1, len(src)), [shape[x] for x in src]); w = w/w.sum()
FL = 100.0/rest
for _ in range(50):
    lo = w < FL
    if not lo.any():
        break
    w = np.where(lo, FL, w*(1 - lo.sum()*FL)/w[~lo].sum())
ar = [src[min(int(round(x*(len(src)-1))), len(src)-1)] for x in q]
sizes = [(src[0], CAMELLIA)] + [(ar[i], float(rest*w[i])) for i in range(30)]
dc = np.zeros(H)
for site, mw in sizes:
    v = d[site].values[:H]
    dc += v*(mw/v.max())

# ----------------------------------------------------------------- existing VRE
g0 = pd.read_csv(f'{CD}/gendata.csv'); c0 = pd.read_csv(f'{CD}/gendata_candidate.csv')
kk = c0.index[(c0.Zone == 'Coast') & (c0.Type == 'SolarPV')][0]
scf = pd.read_csv(f'{CD}/gen_availability_timeseries.csv',
                  usecols=[f'G{len(g0)+int(kk)+1}']).values[:H, 0]
scf = scf*(SOLAR_CF/scf.mean())
vre = F['solar']['mw']*scf

os.makedirs(OUT, exist_ok=True)
fleet = pd.DataFrame(F['firm'])[['Type', 'mw', 'n_units', 'FOR', 'MTTR']]

for tag, add in (('gabase', np.zeros(H)), ('gadc', dc)):
    fleet.to_csv(f'{OUT}/mc_cache_{tag}_fleet.csv', index=False)
    cols = {}
    for y in sorted(SERIES):
        cols[f'load_{y}'] = GPB[y] + add
        cols[f'vre_{y}'] = vre
    pd.DataFrame(cols).to_csv(f'{OUT}/mc_cache_{tag}_series.csv', index=False)
    peak = float(max(c.max() for k, c in cols.items() if k.startswith('load_')))
    pd.DataFrame([{'sto_p_MW': F['battery']['mw'],
                   'sto_e_MWh': F['battery']['mw']*4.0,
                   'peak_MW': peak, 'n_years': len(SERIES), 'eta': ETA}]
                 ).to_csv(f'{OUT}/mc_cache_{tag}_meta.csv', index=False)
    print(f'{tag:7s} peak {peak:9,.0f} MW  firm {fleet.mw.sum():9,.0f} MW '
          f'({fleet.mw.sum()/peak:.3f}x)  {len(SERIES)} weather year(s)')

print(f'\nwritten {OUT}/mc_cache_ga*_{{fleet,series,meta}}.csv')
print('copy them beside ReliabilityPRM_GA.jl in Code/HOPE/')
