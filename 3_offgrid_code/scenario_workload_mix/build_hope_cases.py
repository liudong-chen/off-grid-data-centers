"""Grid cases for the workload-mix scenario.

For each mix, copies the inputs of 2_ongrid_code/HOPE/ModelCases/TX8z_GTEP_dc_vre_case (the ERCOT 2030 case
with data centers) and replaces the data center load: zonal load = base-case load (TX8z_GTEP_base_vre_case)
+ the mix's zonal data center profile (profiles/<mix>/dc_load_hourly_MW_by_weather_zone.csv), which is
exactly how the main case is built (checked: dc - base equals the main profile to 0.04 MW).
Everything else, including the reserve margin of the main case (13.17%), is unchanged.
    python3 build_hope_cases.py  ->  2_ongrid_code/HOPE/ModelCases/TX8z_GTEP_dc_wl_<mix>_case
Then, from 2_ongrid_code/HOPE:  julia --project=. -e 'using HOPE; HOPE.run_hope("ModelCases/TX8z_GTEP_dc_wl_<mix>_case/")'
"""
import os, shutil
import pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); B = os.path.abspath(os.path.join(H, '..', '..'))
MC = f'{B}/2_ongrid_code/HOPE/ModelCases'
SRC, SRCD = f'{MC}/TX8z_GTEP_dc_vre_case', 'Data_TX8z_GTEP_dc_vre'
BASE, BASED = f'{MC}/TX8z_GTEP_base_vre_case', 'Data_TX8z_GTEP_base_vre'
ZM = {'Far West': 'FarWest', 'North Central': 'NorthCentral', 'South Central': 'SouthCentral'}
SKIP = shutil.ignore_patterns('*_ORIGINAL*', '*_u351*', '*_3000MWcap*', '*_derate085*', 'linedata_n1_backup.csv', '__pycache__')

zb = pd.read_csv(f'{BASE}/{BASED}/zonedata.csv').set_index('Zone_id')['Demand (MW)']
lb = pd.read_csv(f'{BASE}/{BASED}/load_timeseries_regional.csv')
base_mw = pd.DataFrame({z: lb[z]*zb[z] for z in zb.index})

for mix in ('training', 'inference', 'colocation'):
    dst, dd = f'{MC}/TX8z_GTEP_dc_wl_{mix}_case', f'Data_TX8z_GTEP_dc_wl_{mix}'
    if os.path.exists(dst):
        print('exists, skipped:', dst); continue
    shutil.copytree(f'{SRC}/{SRCD}', f'{dst}/{dd}', ignore=SKIP)
    shutil.copytree(f'{SRC}/Settings', f'{dst}/Settings')
    zf = pd.read_csv(f'{H}/profiles/{mix}/dc_load_hourly_MW_by_weather_zone.csv', index_col=0, parse_dates=True)
    zf = zf[~((zf.index.month == 2) & (zf.index.day == 29))].iloc[:len(base_mw)]
    zf.columns = [ZM.get(c, c) for c in zf.columns]
    tot = base_mw.copy()
    for z in zf.columns:
        tot[z] = tot[z].values + zf[z].values
    zd = pd.read_csv(f'{dst}/{dd}/zonedata.csv')
    zd['Demand (MW)'] = [round(float(tot[z].max()), 4) for z in zd.Zone_id]
    zd.to_csv(f'{dst}/{dd}/zonedata.csv', index=False)
    l = lb.copy()
    for z in zb.index:
        l[z] = (tot[z]/tot[z].max()).round(6)
    l.to_csv(f'{dst}/{dd}/load_timeseries_regional.csv', index=False)
    sp = f'{dst}/Settings/HOPE_model_settings.yml'
    txt = open(sp).read(); assert f'DataCase: {SRCD}/' in txt
    open(sp, 'w').write(txt.replace(f'DataCase: {SRCD}/', f'DataCase: {dd}/'))
    print(f'{mix}: system peak {tot.sum(axis=1).max():,.0f} MW, data center energy {zf.sum().sum()/1e6:.1f} TWh -> {dst}')
