"""Write a HOPE-format case directory for a (planning year, DC multiplier) pair.

Everything time-dependent is a function of YEAR below; the 2030 branch reproduces
the solved case (case_gasA_SOLVED) byte-for-byte in every parameter, which is the
regression test run at the bottom of this file.

Usage: python build_case.py YEAR K OUTDIR [--ceiling MW] [--sto-ceiling MW] [--prm X]
   K = data-centre multiplier applied to every campus (0 = base case, no data centres)
"""
import os, sys, json, argparse
import numpy as np, pandas as pd

U = '.'
PKG = '2_ongrid_code/grid_case_assembly'
HERE = os.path.dirname(os.path.abspath(__file__))
ZONES = ['Coast', 'East', 'FarWest', 'North', 'NorthCentral', 'South', 'SouthCentral', 'West']
ZMAP = {'COAST': 'Coast', 'EAST': 'East', 'FWEST': 'FarWest', 'NORTH': 'North',
        'NCENT': 'NorthCentral', 'SOUTH': 'South', 'SCENT': 'SouthCentral', 'WEST': 'West'}
DCMAP = {'North Central': 'NorthCentral', 'South Central': 'SouthCentral', 'Far West': 'FarWest'}
BASE_YEAR = 2025
LOAD_GROWTH_PCT = 1.9
WIND_VINTAGE_DERATE, SOLAR_VINTAGE_DERATE = 0.7242, 0.7839
# The solved 2030 case applies the existing-fleet calibration derate to new build as well.
# Kept here so 2035 is comparable with the reported 2030 result; flip to test.
DERATE_NEW_BUILD = True
# 'B' = the package's derateB variant (new build 0.85 wind / 0.90 solar, 33_case_variants.py); overrides DERATE_NEW_BUILD
NEW_BUILD_DERATE_MODE = os.environ.get('NEW_BUILD_DERATE', 'solved')

CC_SOLAR = {'West': 0.2721, 'FarWest': 0.2721, 'Coast': 0.3614, 'East': 0.3614,
            'North': 0.3614, 'NorthCentral': 0.3614, 'South': 0.3614, 'SouthCentral': 0.3614}
CC_WIND = {'Coast': 0.3083, 'South': 0.3083, 'SouthCentral': 0.3083, 'North': 0.3358,
           'West': 0.1568, 'FarWest': 0.1568, 'NorthCentral': 0.1568, 'East': 0.1568}
CC_BATT_4H, CC_THERMAL = 0.9329, 1.0
FOR_RATE = {'NGCC': 0.0357, 'NGCT': 0.0476, 'NGST': 0.0595, 'Coal': 0.0714, 'Nuclear': 0.0476,
            'Oil': 0.0595, 'Biomass': 0.0892, 'Hydro': 0.60, 'WindOn': 0.0, 'SolarPV': 0.0, 'NuC': 0.08}
GAS_2025 = {'FarWest': 3.891, 'North': 3.891, 'West': 3.891, 'South': 4.361, 'NorthCentral': 3.823,
            'SouthCentral': 3.648, 'Coast': 4.079, 'East': 3.891}       # EIA-923 receipts, 2025
NUC_FUEL = 8.00
HEAT_RATE = {'NGCC': 6.226, 'NGCT': 9.142, 'NGST': 10.5, 'Coal': 10.0}
VOM = {'NGCC': 3.57, 'NGCT': 4.17, 'NGST': 4.17, 'Coal': 5.00, 'Nuclear': NUC_FUEL,
       'Oil': 15.0, 'Biomass': 5.0, 'Hydro': 0.0, 'WindOn': 0.0, 'SolarPV': 0.0}
CO2_GAS, CO2_COAL = 53.07, 95.35
EF = {'NGCC': HEAT_RATE['NGCC']*CO2_GAS/1000, 'NGCT': HEAT_RATE['NGCT']*CO2_GAS/1000,
      'NGST': HEAT_RATE['NGST']*CO2_GAS/1000, 'Coal': HEAT_RATE['Coal']*CO2_COAL/1000,
      'Oil': 0.75, 'Biomass': 0.0, 'Nuclear': 0.0, 'Hydro': 0.0, 'WindOn': 0.0, 'SolarPV': 0.0}
THERMAL = {'NGCC', 'NGCT', 'NGST', 'Coal', 'Oil', 'Biomass'}

# ---------------------------------------------------------------- year-dependent inputs
# Candidate fixed cost $/MW-yr: 1_source_data/parameter_setting workbook, Recommended_CEM_Base,
# columns H (2030) and N (2035); CAPEX x CRF(7%, life) + FOM.  Verified to the cent.
YEAR_PARAMS = {
    2030: dict(
        gas_ercot=5.00,       # AEO2026, low end of the $5-6 early-2030s band (33_case_variants.py)
        cand={'NGCC': 118650, 'NGCT': 103382, 'WindOn': 157009, 'SolarPV': 117995, 'NuC': 740795},
        batt=161512,
        note='reproduces case_gasA_SOLVED'),
    2035: dict(
        gas_ercot=5.10,       # AEO2026 Counterfactual Baseline, Table 54 ERCOT electric-power
                              # delivered gas 2035 = $5.099/MMBtu real 2025$ (Henry Hub 5.05)
        cand={'NGCC': 113755, 'NGCT': 96576, 'WindOn': 149412, 'SolarPV': 108393, 'NuC': 706973},
        batt=142528,
        note='workbook 2035 columns; gas AEO2026 2035'),
}
UNIT_MW = {'NGCC': 350.877, 'NGCT': 60.06, 'WindOn': 200, 'SolarPV': 200, 'NuC': 1111.1}   # n_units only; not read by HOPE


def load_year(y):
    d = pd.read_excel(f'{U}/1_source_data/parameter_setting/ERCOT_load/Native_Load_{y}.xlsx')
    d = d.rename(columns={c: ZMAP.get(str(c).strip().upper(), str(c).strip()) for c in d.columns})
    t = pd.to_datetime(d[d.columns[0]].astype(str).str.replace(' 24:00', ' 00:00'),
                       format='%m/%d/%Y %H:%M', errors='coerce')
    d = d[ZONES].copy(); d.index = range(len(d))
    d['month'] = t.dt.month.values; d['day'] = t.dt.day.values
    d = d[~((d.month == 2) & (d.day == 29))].reset_index(drop=True)
    assert len(d) == 8760
    cal = pd.date_range('2025-01-01', periods=8760, freq='h')   # calendar columns only
    d['month'] = cal.month.values; d['day'] = cal.day.values
    return d


def dc_zone_load(k):
    dc = pd.read_csv(f'{U}/1_source_data/DC_Load/load_profiles/dc_load_hourly_MW_by_weather_zone.csv', index_col=0, parse_dates=True)
    dc = dc.rename(columns={c: DCMAP.get(c, c) for c in dc.columns})
    dc = dc[~((dc.index.month == 2) & (dc.index.day == 29))].reset_index(drop=True)
    return {z: (dc[z].values[:8760]*k if z in dc.columns else np.zeros(8760)) for z in ZONES}


def build(year, k, dd, ceiling=20000.0, sto_ceiling=10000.0, prm=0.13, line_cand='open'):
    YP = YEAR_PARAMS[year]
    growth = (1 + LOAD_GROWTH_PCT/100) ** (year - BASE_YEAR)
    scale = YP['gas_ercot']/3.891
    GAS = {z: round(v*scale, 3) for z, v in GAS_2025.items()}
    os.makedirs(dd, exist_ok=True)

    ld = load_year(BASE_YEAR)
    load = ld[ZONES].copy()*growth
    dcz = dc_zone_load(k) if k else {z: np.zeros(8760) for z in ZONES}
    for z in ZONES:
        load[z] = load[z].values + dcz[z]
    peak = load.max(); pu = load/peak
    pd.DataFrame({'Zone_id': ZONES, 'Demand (MW)': [round(peak[z], 4) for z in ZONES], 'State': 'TX'}
                 ).to_csv(f'{dd}/zonedata.csv', index=False)
    lt = pd.DataFrame({'Time Period': 1, 'Month': ld.month.values, 'Day': ld.day.values, 'Hours': np.arange(1, 8761)})
    for z in ZONES:
        lt[z] = pu[z].round(6).values
    lt['NI'] = 0.0
    lt.to_csv(f'{dd}/load_timeseries_regional.csv', index=False)

    fl = pd.read_csv(f'{HERE}/fleet/fleet_{year}.csv')
    g = pd.DataFrame({
        'Pmax (MW)': fl['Pmax (MW)'].round(2), 'Pmin (MW)': 0.0, 'Zone': fl.zone, 'Type': fl.Type,
        'Flag_RET': 0, 'Flag_RPS': 0,
        'Flag_thermal': fl.Type.isin(THERMAL).astype(int),
        'Flag_VRE': fl.Type.isin(['WindOn', 'SolarPV']).astype(int), 'Flag_mustrun': 0,
        'Cost ($/MWh)': [round(VOM.get(t, 0) + HEAT_RATE.get(t, 0)*GAS[z] if t in ('NGCC', 'NGCT', 'NGST')
                               else VOM.get(t, 0) + HEAT_RATE.get(t, 0)*2.20 if t == 'Coal'
                               else VOM.get(t, 0), 2) for t, z in zip(fl.Type, fl.zone)],
        'EF': [EF.get(t, 0.0) for t in fl.Type],
        'CC': [CC_THERMAL if t not in ('WindOn', 'SolarPV') else (CC_WIND[z] if t == 'WindOn' else CC_SOLAR[z])
               for t, z in zip(fl.Type, fl.zone)],
        'AF': 1.0, 'FOR': [FOR_RATE.get(t, 0.0) for t in fl.Type], 'n_units': fl.n_units,
    }).sort_values(['Zone', 'Type']).reset_index(drop=True)
    g.to_csv(f'{dd}/gendata.csv', index=False)

    rows = []
    for t, inv in YP['cand'].items():
        for z in ZONES:
            rows.append({'Pmax (MW)': ceiling, 'Pmin (MW)': 0.0, 'Zone': z, 'Type': t, 'Cost ($/MW/yr)': inv,
                         'Cost ($/MWh)': round(VOM[t] + HEAT_RATE[t]*GAS[z] if t in ('NGCC', 'NGCT') else (NUC_FUEL if t == 'NuC' else 0.0), 2),
                         'Flag_RPS': 0, 'Flag_thermal': int(t in ('NGCC', 'NGCT')), 'Flag_VRE': int(t in ('WindOn', 'SolarPV')),
                         'Flag_mustrun': 0, 'EF': EF.get(t, 0.0),
                         'CC': (CC_WIND[z] if t == 'WindOn' else CC_SOLAR[z] if t == 'SolarPV' else CC_THERMAL),
                         'AF': 1.0, 'FOR': FOR_RATE[t], 'n_units': max(1, int(round(ceiling/UNIT_MW[t])))})
    gc = pd.DataFrame(rows); gc.to_csv(f'{dd}/gendata_candidate.csv', index=False)

    st = pd.read_csv(f'{HERE}/fleet/storage_{year}.csv').set_index('zone').reindex(ZONES).reset_index()
    pd.DataFrame({'Zone': ZONES, 'Type': 'Battery', 'Capacity (MWh)': st['Capacity (MWh)'].round(1),
                  'Max Power (MW)': st['Max Power (MW)'].round(1), 'n_units': st['n_units'].astype(int),
                  'Charging efficiency': 0.922, 'Discharging efficiency': 0.922, 'Cost ($/MWh)': 0.0, 'EF': 0.0,
                  'CC': CC_BATT_4H, 'Charging Rate': 1.0, 'Discharging Rate': 1.0}).to_csv(f'{dd}/storagedata.csv', index=False)
    pd.DataFrame({'Zone': ZONES, 'Type': 'Battery', 'Capacity (MWh)': sto_ceiling*4.0, 'Max Power (MW)': sto_ceiling,
                  'Charging efficiency': 0.922, 'Discharging efficiency': 0.922, 'Cost ($/MW/yr)': YP['batt'],
                  'Cost ($/MWh)': 0.0, 'EF': 0.0, 'CC': CC_BATT_4H, 'Charging Rate': 1.0, 'Discharging Rate': 1.0}
                 ).to_csv(f'{dd}/storagedata_candidate.csv', index=False)

    # network: identical to the solved case
    # network actually solved by HOPE (2_ongrid_code/HOPE/ModelCases/TX8z_GTEP_dc_case/Data_TX8z_GTEP_dc/linedata.csv):
    # N-1-style de-rated corridor limits; the package copy in case_gasA_SOLVED is the as-is thermal sum
    pd.read_csv(f'{HERE}/linedata_solved.csv').to_csv(f'{dd}/linedata.csv', index=False)
    if line_cand == 'open':
        pd.read_csv(f'{PKG}/03_results/linedata_candidate_open.csv').to_csv(f'{dd}/linedata_candidate.csv', index=False)
    else:
        pd.read_csv(f'{PKG}/01_data/case_gasA_SOLVED/linedata_candidate.csv').to_csv(f'{dd}/linedata_candidate.csv', index=False)

    sol = pd.read_csv(f'{PKG}/03_results/_hourly_vre/solar_cf_hourly.csv')
    win = pd.read_csv(f'{PKG}/03_results/_hourly_vre/wind_cf_hourly.csv')
    S = {z: sol[(sol.zone == z) & (sol.year == BASE_YEAR)].cf.values[:8760] for z in ZONES}
    W = {z: win[(win.zone == z) & (win.year == BASE_YEAR)].cf.values[:8760] for z in ZONES}
    af = pd.DataFrame({'Time Period': 1, 'Month': ld.month.values, 'Day': ld.day.values, 'Hours': np.arange(1, 8761)})
    cols = {}
    for i, r in g.iterrows():
        cols[f'G{i+1}'] = (np.round(S[r.Zone]*SOLAR_VINTAGE_DERATE, 5) if r.Type == 'SolarPV' else
                           np.round(W[r.Zone]*WIND_VINTAGE_DERATE, 5) if r.Type == 'WindOn' else np.ones(8760))
    ng = len(g)
    for j, r in gc.iterrows():
        ds = {'solved': SOLAR_VINTAGE_DERATE, 'B': 0.90, 'none': 1.0}[NEW_BUILD_DERATE_MODE]
        dw = {'solved': WIND_VINTAGE_DERATE, 'B': 0.85, 'none': 1.0}[NEW_BUILD_DERATE_MODE]
        cols[f'G{ng+j+1}'] = (np.round(S[r.Zone]*ds, 5) if r.Type == 'SolarPV' else
                              np.round(W[r.Zone]*dw, 5) if r.Type == 'WindOn' else np.ones(8760))
    af = pd.concat([af, pd.DataFrame(cols)], axis=1)
    af.to_csv(f'{dd}/gen_availability_timeseries.csv', index=False)

    pd.DataFrame([{'State': 'TX', 'Time Period': 1, 'Allowance (tons)': 1e12}]).to_csv(f'{dd}/carbonpolicies.csv', index=False)
    pd.DataFrame([{'From_state': 'TX', 'To_state': 'TX', 'RPS': 0.0}]).to_csv(f'{dd}/rpspolicies.csv', index=False)
    pd.DataFrame([{'VOLL': 35000.0, 'planning_reserve_margin': round(prm, 6), 'BigM': 1e6, 'PT_RPS': 1e9, 'PT_emis': 1e9,
                   'spin_requirement': 0.03, 'delta_spin': 10/60, 'Inv_bugt_gen': 3.37e11, 'Inv_bugt_line': 3.3e10,
                   'Inv_bugt_storage': 5.5e10}]).to_csv(f'{dd}/single_parameter.csv', index=False)
    sd = os.path.join(os.path.dirname(dd.rstrip('/')), 'Settings'); os.makedirs(sd, exist_ok=True)
    open(f'{sd}/HOPE_model_settings.yml', 'w').write(f"DataCase: {os.path.basename(dd.rstrip('/'))}/\nmodel_mode: GTEP\ninv_dcs_bin: 0\nplanning_reserve_mode: 1\noperation_reserve_mode: 1\ncarbon_policy: 0\nclean_energy_policy: 0\nflexible_demand: 0\nendogenous_rep_day: 0\nexternal_rep_day: 0\ntransmission_loss: 0\ntransmission_expansion: 1\nstorage_ld_duration_hours: 12\nresource_aggregation: 0\nsolver: gurobi\n")
    info = dict(year=year, k=k, growth=growth, gas=GAS, peak_MW=float(load.sum(1).max()),
                zone_peaks=peak.round(1).to_dict(), dc_peak_by_zone={z: float(dcz[z].max()) for z in ZONES},
                dc_coincident_peak=float(sum(dcz.values()).max()), dc_energy_TWh=float(sum(dcz.values()).sum()/1e6),
                ceiling=ceiling, sto_ceiling=sto_ceiling, cand_cost=YP['cand'], batt_cost=YP['batt'], prm=prm,
                derate_new_build=NEW_BUILD_DERATE_MODE, line_candidates=line_cand)
    json.dump(info, open(f'{dd}/../case_info.json', 'w'), indent=1)
    return info


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('year', type=int); ap.add_argument('k', type=float); ap.add_argument('outdir')
    ap.add_argument('--ceiling', type=float, default=20000.0); ap.add_argument('--sto-ceiling', type=float, default=10000.0)
    ap.add_argument('--prm', type=float, default=0.13); ap.add_argument('--lines', default='open')
    a = ap.parse_args()
    info = build(a.year, a.k, a.outdir, a.ceiling, a.sto_ceiling, a.prm, a.lines)
    print(json.dumps({k: v for k, v in info.items() if k not in ('zone_peaks',)}, indent=1))
