"""Solar capacity factor by ERCOT weather zone, 2018-2025, from NSRDB.

Plant model: single-axis horizontal N-S tracker with backtracking, which is
what essentially all new ERCOT utility PV is. Fixed tilt is computed too, for
comparison only.

CF is AC output / AC nameplate, so it already includes inverter clipping at the
1.3 DC/AC ratio typical of recent ERCOT projects.
"""
import os, glob, json
import numpy as np, pandas as pd
import pvlib

RAW = 'vre_raw/solar'
OUT = 'build/out'; os.makedirs(OUT, exist_ok=True)

# ---- plant design (documented assumptions) ----
GCR        = 0.35     # ground cover ratio, typical single-axis ERCOT plant
MAX_ANGLE  = 60.0     # tracker rotation limit, degrees
ILR        = 1.30     # DC/AC ratio
ETA_INV    = 0.96     # inverter efficiency at nameplate
GAMMA_PDC  = -0.0034  # module power temperature coefficient, /degC
# PVWatts default loss stack is 14.08%. Temperature is modelled explicitly above
# and row-to-row shading is handled by backtracking, so those two come out:
#   soiling 2, mismatch 2, wiring 2, connections 0.5, LID 1.5, nameplate 1,
#   availability 3  ->  11.4% -> multiplier 0.886
DC_LOSS    = 0.886
TZ         = -6       # NSRDB returns local standard time, CST, no DST

# tracker point per zone, from download_vre.py POINTS
LATLON = {'Coast':(29.00,-95.50),'East':(31.80,-94.80),'FarWest':(31.40,-103.00),
          'North':(34.20,-101.70),'NorthCentral':(32.80,-97.30),'South':(27.50,-98.50),
          'SouthCentral':(29.40,-98.50),'West':(32.00,-100.50)}


def read_nsrdb(path):
    """NSRDB csv: 2 metadata rows, then the header. Returns df indexed by local time."""
    meta = pd.read_csv(path, nrows=1)
    elev = float(meta['Elevation'].iloc[0]) if 'Elevation' in meta else 300.0
    d = pd.read_csv(path, skiprows=2)
    idx = pd.to_datetime(dict(year=d.Year, month=d.Month, day=d.Day,
                              hour=d.Hour, minute=d.Minute))
    # NSRDB was requested with utc=false, so these stamps are LOCAL STANDARD
    # time (CST, UTC-6, no DST). pvlib treats a naive index as UTC, so the
    # timezone must be attached or solar noon lands six hours out.
    # POSIX sign convention: Etc/GMT+6 IS UTC-6.
    d.index = pd.DatetimeIndex(idx).tz_localize('Etc/GMT+6')
    return d, elev


def solar_cf(d, lat, lon, elev):
    # NSRDB hourly stamps are ALREADY hour-centred (Minute == 30), so the index
    # is the correct instant for the solar position. Do not shift it again.
    assert set(d.Minute.unique()) == {30}, "expected hour-centred NSRDB stamps"
    solpos = pvlib.solarposition.get_solarposition(
        d.index, lat, lon, altitude=elev, temperature=d['Temperature'].values)

    dni_extra = pvlib.irradiance.get_extra_radiation(d.index)
    albedo = d['Surface Albedo'].clip(0, 1) if 'Surface Albedo' in d else 0.2

    # ---- single-axis tracker, N-S horizontal axis, with backtracking ----
    tr = pvlib.tracking.singleaxis(solpos['apparent_zenith'], solpos['azimuth'],
                                   axis_tilt=0, axis_azimuth=180,
                                   max_angle=MAX_ANGLE, backtrack=True, gcr=GCR)
    tilt = tr['surface_tilt'].fillna(0)
    azim = tr['surface_azimuth'].fillna(180)

    poa = pvlib.irradiance.get_total_irradiance(
        tilt, azim, solpos['apparent_zenith'], solpos['azimuth'],
        dni=d['DNI'], ghi=d['GHI'], dhi=d['DHI'],
        dni_extra=dni_extra, albedo=albedo, model='haydavies')
    g = poa['poa_global'].fillna(0).clip(lower=0)

    # ---- cell temperature, open-rack glass/glass ----
    tcell = pvlib.temperature.sapm_cell(
        g, d['Temperature'], d['Wind Speed'],
        **pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass'])

    # ---- PVWatts DC, then clip to AC nameplate ----
    pdc = pvlib.pvsystem.pvwatts_dc(g, tcell, pdc0=ILR, gamma_pdc=GAMMA_PDC) * DC_LOSS  # per 1 MW AC
    pac = np.minimum(pdc * ETA_INV, 1.0).clip(lower=0)
    return pd.Series(pac.values, index=d.index)


rows, series = [], {}
for f in sorted(glob.glob(f'{RAW}/*.csv')):
    zone, yr = os.path.basename(f)[:-4].rsplit('_', 1)
    d, elev = read_nsrdb(f)
    lat, lon = LATLON[zone]
    cf = solar_cf(d, lat, lon, elev)
    series[(zone, int(yr))] = cf
    rows.append(dict(zone=zone, year=int(yr), n=len(cf), cf_mean=cf.mean(),
                     cf_max=cf.max(), elev_m=elev,
                     hours_above_0=int((cf > 0.001).sum())))

s = pd.DataFrame(rows).sort_values(['zone', 'year'])
print(s.pivot(index='zone', columns='year', values='cf_mean').round(4).to_string())
print('\nannual mean CF by zone, averaged over 2018-2025:')
print(s.groupby('zone').cf_mean.mean().round(4).sort_values(ascending=False).to_string())

# one tidy long file: zone x year x 8760
out = []
for (zone, yr), cf in series.items():
    t = pd.DataFrame({'zone': zone, 'year': yr,
                      'month': cf.index.month, 'day': cf.index.day,
                      'hour': cf.index.hour + 1, 'cf': cf.values})
    out.append(t)
pd.concat(out).to_csv(f'{OUT}/solar_cf_hourly.csv', index=False)
s.to_csv(f'{OUT}/solar_cf_summary.csv', index=False)
json.dump(dict(GCR=GCR, MAX_ANGLE=MAX_ANGLE, ILR=ILR, ETA_INV=ETA_INV,
               GAMMA_PDC=GAMMA_PDC, transposition='haydavies',
               tracking='single-axis NS horizontal, backtracking',
               cell_temp='sapm open_rack_glass_glass', tz='CST fixed (UTC-6)'),
          open(f'{OUT}/solar_model_params.json', 'w'), indent=1)
print(f'\nwritten to {OUT}')
