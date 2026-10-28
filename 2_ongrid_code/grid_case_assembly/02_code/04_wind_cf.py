"""Wind capacity factor by ERCOT weather zone, from HRRR MET Toolkit 100 m wind.

Turbine: IEC class 2/3 composite, 100 m hub, ~3.4 MW class - the NREL ATB
reference machine for land-based wind. Power curve below is normalised, so CF
is output / nameplate.

Air density correction: the power curve is defined at 1.225 kg/m3, so the wind
speed is scaled by (rho/1.225)^(1/3) before the lookup. Panhandle sites sit near
1,100 m where density is ~10% low, worth ~3% of speed.
"""
import os, glob, json
import numpy as np, pandas as pd

RAW = '1_source_data/vre_raw/wind'
OUT = 'build/out'; os.makedirs(OUT, exist_ok=True)

# NREL ATB reference land-based turbine, normalised power curve
V_PC  = np.array([0,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,
                  12,12.5,13,25,25.01,40])
P_PC  = np.array([0,0,0.006,0.026,0.058,0.100,0.156,0.226,0.312,0.415,0.534,0.665,
                  0.797,0.899,0.962,0.991,0.999,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0,0])

WAKE_AVAIL = 0.88   # wake losses ~8%, availability ~3%, electrical ~1.5%
R_AIR = 287.05      # J/(kg K)


def wind_cf(df):
    v = df['wind speed at 100m (m/s)'].values.astype(float)
    if 'air temperature at 100m (C)' in df and 'air pressure at 100m (hPa)' in df:
        T = df['air temperature at 100m (C)'].values.astype(float) + 273.15
        P = df['air pressure at 100m (hPa)'].values.astype(float) * 100.0
        rho = P / (R_AIR * T)
        v = v * (rho / 1.225) ** (1 / 3)
    return np.interp(v, V_PC, P_PC) * WAKE_AVAIL


rows, series = [], {}
for f in sorted(glob.glob(f'{RAW}/*.csv')):
    zone, yr = os.path.basename(f)[:-4].rsplit('_', 1)
    d = pd.read_csv(f, skiprows=1)
    idx = pd.to_datetime(dict(year=d.Year, month=d.Month, day=d.Day,
                              hour=d.Hour, minute=d.Minute))
    cf = pd.Series(wind_cf(d), index=pd.DatetimeIndex(idx))
    # drop Feb 29 so every year is exactly 8760
    cf = cf[~((cf.index.month == 2) & (cf.index.day == 29))]
    series[(zone, int(yr))] = cf
    rows.append(dict(zone=zone, year=int(yr), n=len(cf), cf_mean=cf.mean(),
                     v_mean=d['wind speed at 100m (m/s)'].mean()))

s = pd.DataFrame(rows).sort_values(['zone', 'year'])
print(s.pivot(index='zone', columns='year', values='cf_mean').round(4).to_string())
print('\nmean 100 m wind speed and CF by zone:')
print(s.groupby('zone').agg(v_ms=('v_mean', 'mean'), cf=('cf_mean', 'mean'))
       .round(3).sort_values('cf', ascending=False).to_string())

out = []
for (zone, yr), cf in series.items():
    out.append(pd.DataFrame({'zone': zone, 'year': yr, 'month': cf.index.month,
                             'day': cf.index.day, 'hour': cf.index.hour + 1,
                             'cf': cf.values}))
pd.concat(out).to_csv(f'{OUT}/wind_cf_hourly.csv', index=False)
s.to_csv(f'{OUT}/wind_cf_summary.csv', index=False)
json.dump(dict(turbine='NREL ATB reference land-based, 100 m hub, ~3.4 MW class',
               wake_availability_electrical=WAKE_AVAIL,
               density_correction='v * (rho/1.225)^(1/3), rho from HRRR T and P at 100 m',
               source='HRRR MET Toolkit v1.0.0, hourly, 2 km'),
          open(f'{OUT}/wind_model_params.json', 'w'), indent=1)
print(f'\nyears present: {sorted(s.year.unique())}')
print(f'written to {OUT}')
