"""Build the 2030 ERCOT HOPE cases from the assembled case and the shipped inputs.

    python3 2_ongrid_code/HOPE/ModelCases/build_main_cases.py [--check]

Writes  2_ongrid_code/HOPE/ModelCases/TX8z_GTEP_dc_vre_case/Data_TX8z_GTEP_dc_vre     (with the data centers)
        2_ongrid_code/HOPE/ModelCases/TX8z_GTEP_base_vre_case/Data_TX8z_GTEP_base_vre (without them)
and the Settings folder of each case.  With --check nothing is written; the result is compared with the
case folders already present.

Inputs
  2_ongrid_code/grid_case_assembly/01_data/case_gasA_SOLVED   assembled case (05_assemble.py, 33_case_variants.py)
  2_ongrid_code/scenarios/_code/linedata_solved.csv                              N-1 corridor limits of the existing network
  3_offgrid_code/zonal_gas_2030.json                                 delivered gas price by zone, 2030 ($/MMBtu)
  1_source_data/DC_Load/load_profiles                                                  data center load (build_dc_load.py)

Transformations applied to the assembled case
  1. Gas units (existing and candidate): Cost ($/MWh) = VOM + heat rate x zonal gas price.
  2. New-build wind and solar capacity factors: the candidate availability columns carry the derate of
     new plants (0.90 wind, 0.88 solar) instead of the derate calibrated on the existing fleet
     (0.7242, 0.7839); existing units unchanged.
  3. Candidate limits: thermal and nuclear 100,000 MW per zone (economics-limited); wind and solar from
     developable land (3.0 MW/km2 on 20% of the zone for wind, 35 MW/km2 on 2% for solar); storage
     50,000 MW / 200,000 MWh per zone.
  4. Existing network: the N-1 corridor limits; candidate corridors 30,000 MW at the same cost per MW
     (capacity and cost x10, reactance /10 as ten parallel circuits).
  5. Planning reserve margin: starting value 0.1317 (data center case) and 0.1344 (base case); the
     reserve-margin fixed point (ReliabilityPRM.jl) rewrites it.
  6. Base case: the zonal load with the data center load removed, as per-unit profiles and zonal peaks.
"""
import json, os, shutil, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
SRC = f'{ROOT}/2_ongrid_code/grid_case_assembly/01_data/case_gasA_SOLVED'
LINES = f'{ROOT}/2_ongrid_code/scenarios/_code/linedata_solved.csv'
GAS = json.load(open(f'{ROOT}/3_offgrid_code/zonal_gas_2030.json'))
LOAD = f'{ROOT}/1_source_data/DC_Load/load_profiles'
CHECK = '--check' in sys.argv

HR = {'NGCC': 6.226, 'NGCT': 9.142, 'NGST': 10.5}          # MMBtu/MWh
VOM = {'NGCC': 3.57, 'NGCT': 4.17, 'NGST': 4.17}           # $/MWh
DERATE_OLD = {'WindOn': 0.7242, 'SolarPV': 0.7839}         # calibrated on the 2025 fleet
DERATE_NEW = {'WindOn': 0.90, 'SolarPV': 0.88}             # new plants (2023-24 cohort, EIA-860/923)
THERMAL_CAP = 100000.0
UNIT_MW = {'NGCC': 627.0, 'NGCT': 419.0, 'NuC': 1111.1}   # n_units = cap / unit size
VRE_CAP = {  # MW, from developable land
    'WindOn':  {'Coast': 32100, 'East': 42800, 'FarWest': 73800, 'North': 63600, 'NorthCentral': 42600,
                'South': 42300, 'SouthCentral': 57100, 'West': 49100},
    'SolarPV': {'Coast': 37500, 'East': 49900, 'FarWest': 86100, 'North': 74200, 'NorthCentral': 49700,
                'South': 49300, 'SouthCentral': 66600, 'West': 57300}}
STORAGE_MW, STORAGE_MWH = 50000.0, 200000.0
PRM = {'dc': 0.13169, 'base': 0.1344}
ZONE = {'Far West': 'FarWest', 'South Central': 'SouthCentral', 'North Central': 'NorthCentral'}


def gas_costs(df):
    d = df.copy()
    for i, r in d.iterrows():
        if r['Type'] in HR:
            d.at[i, 'Cost ($/MWh)'] = round(VOM[r['Type']] + HR[r['Type']] * GAS[r['Zone']], 2)
    return d


def candidates(gc):
    d = gc.copy()
    for i, r in d.iterrows():
        if r['Type'] in VRE_CAP:
            d.at[i, 'Pmax (MW)'] = float(VRE_CAP[r['Type']][r['Zone']])
        else:
            d.at[i, 'Pmax (MW)'] = THERMAL_CAP
            d.at[i, 'n_units'] = int(round(THERMAL_CAP / UNIT_MW[r['Type']]))
    return d


def availability(av, g, gc):
    d = av.copy()
    cols = [c for c in av.columns if c.startswith('G')]
    names = [('exist', t) for t in g.Type] + [('cand', t) for t in gc.Type]
    for c, (kind, typ) in zip(cols, names):
        if kind == 'cand' and typ in DERATE_NEW:
            d[c] = np.minimum(d[c].values.astype(float) * DERATE_NEW[typ] / DERATE_OLD[typ], 1.0)
    return d


def line_candidates(lc):
    d = lc.copy()
    for c in ('Forward Capacity (MW)', 'Reverse Capacity (MW)', 'Cost (M$)'):
        d[c] = (d[c] * 10).round(4)
    d['X'] = (d['X'] / 10).round(6)
    return d


def base_load(load_pu, zonedata):
    """Zonal load without the data centers: MW = per-unit profile x zonal peak, minus the data center load."""
    sites = pd.read_csv(f'{LOAD}/site_parameters_and_results.csv').set_index('site_id')
    dc = pd.read_csv(f'{LOAD}/dc_load_hourly_MW.csv', index_col=0)
    peak = zonedata.set_index('Zone_id')['Demand (MW)']
    mw = load_pu.copy(); zd = zonedata.copy()
    for z in peak.index:
        series = load_pu[z].values * peak[z]
        cols = [s for s in dc.columns if s in sites.index and ZONE.get(sites.loc[s, 'ercot_weather_zone'], sites.loc[s, 'ercot_weather_zone']) == z]
        if cols:
            series = series - dc[cols].sum(axis=1).values
        pk = float(series.max())
        mw[z] = (series / pk).round(6)
        zd.loc[zd.Zone_id == z, 'Demand (MW)'] = round(pk, 4)
    return mw, zd


def settings(case, data_dir):
    s = f'{HERE}/{case}/Settings'
    os.makedirs(s, exist_ok=True)
    for f in os.listdir(f'{SRC}/../../Settings') if os.path.isdir(f'{SRC}/../../Settings') else []:
        shutil.copy(f'{SRC}/../../Settings/{f}', s)
    open(f'{s}/HOPE_model_settings.yml', 'w').write(
        f"DataCase: {data_dir}/\nmodel_mode: GTEP\nresource_aggregation: 0\nendogenous_rep_day: 0\nexternal_rep_day: 0\n"
        "flexible_demand: 0\ninv_dcs_bin: 0\ncarbon_policy: 0\nclean_energy_policy: 0\nplanning_reserve_mode: 1\n"
        "operation_reserve_mode: 1\ntransmission_loss: 0\ntransmission_expansion: 1\nstorage_ld_duration_hours: 12\n"
        "write_shadow_prices: 0\nsummary_table: 1\nsave_postprocess_snapshot: 1\nsolver: gurobi\ndebug: 0\n")


def build():
    rd = lambda f: pd.read_csv(f'{SRC}/{f}')
    g, gc = gas_costs(rd('gendata.csv')), candidates(gas_costs(rd('gendata_candidate.csv')))
    av = availability(rd('gen_availability_timeseries.csv'), g, gc)
    sc = rd('storagedata_candidate.csv'); sc['Max Power (MW)'] = STORAGE_MW; sc['Capacity (MWh)'] = STORAGE_MWH
    lines, lc = pd.read_csv(LINES), line_candidates(rd('linedata_candidate.csv'))
    load, zd = rd('load_timeseries_regional.csv'), rd('zonedata.csv')
    load_b, zd_b = base_load(load, zd)
    common = dict(gendata=g, gendata_candidate=gc, gen_availability_timeseries=av, storagedata_candidate=sc,
                  linedata=lines, linedata_candidate=lc, carbonpolicies=rd('carbonpolicies.csv'),
                  rpspolicies=rd('rpspolicies.csv'), storagedata=rd('storagedata.csv'))
    out = {}
    for case, tag, ld, z in (('TX8z_GTEP_dc_vre_case', 'dc', load, zd), ('TX8z_GTEP_base_vre_case', 'base', load_b, zd_b)):
        sp = rd('single_parameter.csv'); sp['planning_reserve_margin'] = PRM[tag]
        out[case] = dict(common, load_timeseries_regional=ld, zonedata=z, single_parameter=sp)
    return out


def main():
    cases = build()
    for case, tables in cases.items():
        data_dir = 'Data_TX8z_GTEP_' + ('dc_vre' if 'dc' in case else 'base_vre')
        d = f'{HERE}/{case}/{data_dir}'
        if CHECK:
            worst = 0.0
            for name, df in tables.items():
                have = pd.read_csv(f'{d}/{name}.csv')
                num = df.select_dtypes('number').columns
                diff = float((df[num].values.astype(float) - have[num].values.astype(float)).__abs__().max()) if len(num) else 0.0
                same_text = all((df[c].astype(str).values == have[c].astype(str).values).all() for c in df.columns if c not in num)
                worst = max(worst, diff)
                print(f'{case:28s} {name:28s} max|diff| = {diff:.3g}  text {"ok" if same_text else "DIFFERS"}')
            print(f'{case}: largest numeric difference {worst:.3g}')
        else:
            os.makedirs(d, exist_ok=True)
            for name, df in tables.items():
                df.to_csv(f'{d}/{name}.csv', index=False)
            settings(case, data_dir)
            print('wrote', os.path.relpath(d, ROOT))


if __name__ == '__main__':
    main()
