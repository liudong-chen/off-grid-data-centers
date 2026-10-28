"""Assemble every HOPE GTEP input file for the ERCOT 8-zone case.

Writes  Data_TX8z_GTEP_base/   (no data centres)
        Data_TX8z_GTEP_dc/     (+ the 32.7 GW data-centre fleet)
        Settings/

Column names and semantics follow HOPE's read_input_data.jl and GTEP.jl exactly.
"""
import os, json, shutil
import numpy as np, pandas as pd

U    = '.'
OUTD = 'build/out'
CASE = 'Code/HOPE/ModelCases'

ZONES = ['Coast','East','FarWest','North','NorthCentral','South','SouthCentral','West']
ZMAP  = {'COAST':'Coast','EAST':'East','FWEST':'FarWest','NORTH':'North',
         'NCENT':'NorthCentral','SOUTH':'South','SCENT':'SouthCentral','WEST':'West'}
# DC profile uses spaced names
DCMAP = {'North Central':'NorthCentral','South Central':'SouthCentral','Far West':'FarWest'}

# ============================ decisions ============================
BASE_YEAR   = 2025    # weather year: load + solar + wind all present 2018-2025
TARGET_YEAR = 2030    # PLANNING year: fleet and load level are 2030
# Organic (non-large-load) peak growth applied to the 2025 shape. ERCOT's own
# realised peak grew 1.9%/yr over 2018-2025 (73,308 -> 83,679 MW). ERCOT's
# official long-term forecast is far steeper, but it is dominated by large-load
# interconnection requests - i.e. data centres - and this study adds its OWN
# 32.7 GW data-centre fleet, so using ERCOT's headline growth would double
# count. Replace with ERCOT's baseline-only zonal forecast when available.
LOAD_GROWTH_PCT = 1.9                       # %/yr, ASSIGNED
GROWTH = (1 + LOAD_GROWTH_PCT/100) ** (TARGET_YEAR - BASE_YEAR)
# Calibrated against ERCOT's OBSERVED 2025 fleet output (wind_solar2025.xlsx):
#   wind  observed CF 0.3282 vs modelled 0.4532 -> 0.7242
#   solar observed CF 0.2336 vs modelled 0.2981 -> 0.7839
# These embed CURTAILMENT as well as fleet vintage - see the documentation.
WIND_VINTAGE_DERATE  = 0.7242
SOLAR_VINTAGE_DERATE = 0.7839

# ---- ERCOT 2024 ELCC study, marginal ELCC, SUMMER AFTERNOON (RA_con binds at
#      the load peak). Summer-evening values are the sensitivity.
CC_SOLAR = {'West':0.2721,'FarWest':0.2721,'Coast':0.3614,'East':0.3614,
            'North':0.3614,'NorthCentral':0.3614,'South':0.3614,'SouthCentral':0.3614}
CC_WIND  = {'Coast':0.3083,'South':0.3083,'SouthCentral':0.3083,      # Wind Coastal
            'North':0.3358,                                            # Wind Panhandle
            'West':0.1568,'FarWest':0.1568,'NorthCentral':0.1568,'East':0.1568}
CC_BATT_4H = 0.9329
CC_THERMAL = 1.0      # ERCOT convention: outages covered BY the margin

# ---- NERC GADS class-average forced outage rates
# NERC GADS class averages, RESCALED so the capacity-weighted fleet value
# matches NERC's measured TRE (ERCOT) summer EFORd of 4.70%. The raw NERC-wide
# averages give 7.90%, which is 68% too high for ERCOT's younger gas-heavy fleet.
FOR_RATE = {'NGCC':0.0357,'NGCT':0.0476,'NGST':0.0595,'Coal':0.0714,'Nuclear':0.0476,
            'Oil':0.0595,'Biomass':0.0892,'Hydro':0.60,'WindOn':0.0,'SolarPV':0.0}

# ---- fuel: EIA-923 Page 5 "Fuel Receipts and Costs", 2025, BA = ERCO ------
# Quantity-weighted delivered price from ACTUAL plant fuel receipts.
# ERCOT-wide = $3.891/MMBtu from 26 reporting plants, 270 M MMBtu.
# Zonal values used where >=3 plants report; ERCOT-wide elsewhere.
# NOTE: FarWest / West / North have NO reporting plants, so the Permian (Waha)
# discount CANNOT be measured from this source and is not applied.
GAS = {'FarWest': 3.891, 'North': 3.891, 'West': 3.891, 'South': 4.361, 'NorthCentral': 3.823, 'SouthCentral': 3.648, 'Coast': 4.079, 'East': 3.891}
GAS_SOURCE = {'FarWest': 'ERCOT-wide fallback', 'North': 'ERCOT-wide fallback', 'West': 'ERCOT-wide fallback', 'South': 'EIA-923 receipts', 'NorthCentral': 'EIA-923 receipts', 'SouthCentral': 'EIA-923 receipts', 'Coast': 'EIA-923 receipts', 'East': 'ERCOT-wide fallback'}
GAS_PRICE  = 3.891            # ERCOT-wide, used for candidate defaults
NUC_FUEL   = 8.00     # $/MWh   <-- still ASSIGNED
HEAT_RATE  = {'NGCC':6.226,'NGCT':9.142,'NGST':10.5,'Coal':10.0}
VOM        = {'NGCC':3.57,'NGCT':4.17,'NGST':4.17,'Coal':5.00,'Nuclear':NUC_FUEL,
              'Oil':15.0,'Biomass':5.0,'Hydro':0.0,'WindOn':0.0,'SolarPV':0.0}
CO2_GAS    = 53.07    # kg CO2 / MMBtu, EPA
CO2_COAL   = 95.35
EF = {'NGCC':HEAT_RATE['NGCC']*CO2_GAS/1000, 'NGCT':HEAT_RATE['NGCT']*CO2_GAS/1000,
      'NGST':HEAT_RATE['NGST']*CO2_GAS/1000, 'Coal':HEAT_RATE['Coal']*CO2_COAL/1000,
      'Oil':0.75,'Biomass':0.0,'Nuclear':0.0,'Hydro':0.0,'WindOn':0.0,'SolarPV':0.0}

# ---- candidate costs: parameter_setting workbook, Recommended_CEM_Base, 2030
CAND = {   # Type    $/MW/yr   $/MWh                       unit MW (for the MC)
 'NGCC':   dict(inv=118650, op=VOM['NGCC']+HEAT_RATE['NGCC']*GAS_PRICE, unit=500),
 'NGCT':   dict(inv=103382, op=VOM['NGCT']+HEAT_RATE['NGCT']*GAS_PRICE, unit=200),
 'WindOn': dict(inv=157009, op=0.0,   unit=200),
 'SolarPV':dict(inv=117995, op=0.0,   unit=200),
 'NuC':    dict(inv=740795, op=NUC_FUEL, unit=300),
}
BATT = dict(inv=161512, op=0.0, dur=4.0)
CAND_BLOCK_MW = 20000.0     # per zone per technology build limit
BATT_BLOCK_MW = 10000.0

THERMAL = {'NGCC','NGCT','NGST','Coal','Oil','Biomass'}   # Flag_thermal = 1


# ============================ load ============================
def load_year(y):
    f = f'{U}/parameter_setting/ERCOT_load/Native_Load_{y}.xlsx'
    d = pd.read_excel(f)
    d = d.rename(columns={c: ZMAP.get(str(c).strip().upper(), str(c).strip())
                          for c in d.columns})
    t = pd.to_datetime(d[d.columns[0]].astype(str).str.replace(' 24:00', ' 00:00'),
                       format='%m/%d/%Y %H:%M', errors='coerce')
    d = d[ZONES].copy()
    d.index = range(len(d))
    d['month'] = t.dt.month.values; d['day'] = t.dt.day.values
    d = d[~((d.month == 2) & (d.day == 29))].reset_index(drop=True)   # -> 8760
    assert len(d) == 8760, f'{y}: {len(d)} rows'
    return d


# ============================ build ============================
def build(name, add_dc):
    dd = f'{CASE}/TX8z_GTEP_{name}_case/Data_TX8z_GTEP_{name}'
    os.makedirs(dd, exist_ok=True)

    ld = load_year(BASE_YEAR)
    load = ld[ZONES].copy() * GROWTH   # organic growth to TARGET_YEAR

    dc_added = {z: 0.0 for z in ZONES}
    if add_dc:
        dc = pd.read_csv('DC_Load/load_profiles/dc_load_hourly_MW_by_weather_zone.csv',
                         index_col=0, parse_dates=True)
        dc = dc.rename(columns={c: DCMAP.get(c, c) for c in dc.columns})
        dc = dc[~((dc.index.month == 2) & (dc.index.day == 29))].reset_index(drop=True)
        for z in ZONES:
            if z in dc.columns:
                load[z] = load[z].values + dc[z].values[:8760]
                dc_added[z] = float(dc[z].max())

    peak = load.max()                       # zonedata Demand (MW)
    pu   = load / peak                      # per-unit multiplier

    # ---------------- zonedata ----------------
    pd.DataFrame({'Zone_id': ZONES,
                  'Demand (MW)': [round(peak[z], 4) for z in ZONES],
                  'State': 'TX'}).to_csv(f'{dd}/zonedata.csv', index=False)

    # ---------------- load_timeseries_regional ----------------
    lt = pd.DataFrame({'Time Period': 1, 'Month': ld.month.values,
                       'Day': ld.day.values, 'Hours': np.arange(1, 8761)})
    for z in ZONES:
        lt[z] = pu[z].round(6).values
    lt['NI'] = 0.0
    lt.to_csv(f'{dd}/load_timeseries_regional.csv', index=False)

    # ---------------- gendata ----------------
    fl = pd.read_csv(f'{OUTD}/fleet_{TARGET_YEAR}.csv')
    g = pd.DataFrame({
        'Pmax (MW)': fl['Pmax (MW)'].round(2),
        'Pmin (MW)': 0.0,
        'Zone': fl.zone, 'Type': fl.Type,
        'Flag_RET': 0,
        'Flag_RPS': 0,
        'Flag_thermal': fl.Type.isin(THERMAL).astype(int),
        'Flag_VRE': fl.Type.isin(['WindOn', 'SolarPV']).astype(int),
        'Flag_mustrun': 0,
        'Cost ($/MWh)': [round(
            VOM.get(t, 0) + HEAT_RATE.get(t, 0) * GAS[z] if t in ('NGCC','NGCT','NGST')
            else VOM.get(t, 0) + HEAT_RATE.get(t, 0) * 2.20 if t == 'Coal'
            else VOM.get(t, 0), 2) for t, z in zip(fl.Type, fl.zone)],
        'EF': [EF.get(t, 0.0) for t in fl.Type],
        'CC': [CC_THERMAL if t not in ('WindOn', 'SolarPV') else
               (CC_WIND[z] if t == 'WindOn' else CC_SOLAR[z])
               for t, z in zip(fl.Type, fl.zone)],
        'AF': 1.0,
        'FOR': [FOR_RATE.get(t, 0.0) for t in fl.Type],
        'n_units': fl.n_units,
    })
    g = g.sort_values(['Zone', 'Type']).reset_index(drop=True)
    g.to_csv(f'{dd}/gendata.csv', index=False)

    # ---------------- gendata_candidate ----------------
    rows = []
    for t, c in CAND.items():
        for z in ZONES:
            rows.append({
                'Pmax (MW)': CAND_BLOCK_MW, 'Pmin (MW)': 0.0, 'Zone': z, 'Type': t,
                'Cost ($/MW/yr)': c['inv'],
                'Cost ($/MWh)': round(
                    VOM[t] + HEAT_RATE[t] * GAS[z] if t in ('NGCC','NGCT')
                    else c['op'], 2),
                'Flag_RPS': 0,
                'Flag_thermal': int(t in ('NGCC', 'NGCT')),
                'Flag_VRE': int(t in ('WindOn', 'SolarPV')),
                'Flag_mustrun': 0,
                'EF': EF.get(t, 0.0),
                'CC': (CC_WIND[z] if t == 'WindOn' else CC_SOLAR[z] if t == 'SolarPV'
                       else CC_THERMAL),
                'AF': 1.0,
                'FOR': FOR_RATE.get(t, 0.08 if t == 'NuC' else 0.0),
                'n_units': max(1, int(round(CAND_BLOCK_MW / c['unit']))),
            })
    gc = pd.DataFrame(rows)
    gc.to_csv(f'{dd}/gendata_candidate.csv', index=False)

    # ---------------- storage ----------------
    st = pd.read_csv(f'{OUTD}/storage_{TARGET_YEAR}.csv')
    st = st.set_index('zone').reindex(ZONES).reset_index()
    pd.DataFrame({
        'Zone': ZONES, 'Type': 'Battery',
        'Capacity (MWh)': st['Capacity (MWh)'].round(1),
        'Max Power (MW)': st['Max Power (MW)'].round(1),
        'n_units': st['n_units'].astype(int),
        'Charging efficiency': 0.922, 'Discharging efficiency': 0.922,
        'Cost ($/MWh)': 0.0, 'EF': 0.0, 'CC': CC_BATT_4H,
        'Charging Rate': 1.0, 'Discharging Rate': 1.0,
    }).to_csv(f'{dd}/storagedata.csv', index=False)

    pd.DataFrame({
        'Zone': ZONES, 'Type': 'Battery',
        'Capacity (MWh)': BATT_BLOCK_MW * BATT['dur'],
        'Max Power (MW)': BATT_BLOCK_MW,
        'Charging efficiency': 0.922, 'Discharging efficiency': 0.922,
        'Cost ($/MW/yr)': BATT['inv'], 'Cost ($/MWh)': BATT['op'],
        'EF': 0.0, 'CC': CC_BATT_4H,
        'Charging Rate': 1.0, 'Discharging Rate': 1.0,
    }).to_csv(f'{dd}/storagedata_candidate.csv', index=False)

    # ---------------- linedata ----------------
    iz = pd.read_csv(f'{OUTD}/interzone_capacity.csv')
    pd.DataFrame({'From_zone': iz.a, 'To_zone': iz.b,
                  'Forward Capacity (MW)': iz.limit_MW.round(1),
                  'Reverse Capacity (MW)': iz.limit_MW.round(1),
                  'Loss (%)': 0.0, 'N_branches': iz.n_branches
                  }).to_csv(f'{dd}/linedata.csv', index=False)
    pd.DataFrame(columns=['From_zone', 'To_zone', 'Forward Capacity (MW)',
                          'Reverse Capacity (MW)', 'Cost (M$)', 'X', 'Loss (%)']
                 ).to_csv(f'{dd}/linedata_candidate.csv', index=False)

    # ---------------- gen_availability_timeseries ----------------
    sol = pd.read_csv(f'{OUTD}/solar_cf_hourly.csv')
    win = pd.read_csv(f'{OUTD}/wind_cf_hourly.csv')
    wyr = BASE_YEAR if BASE_YEAR in set(win.year) else sorted(win.year.unique())[0]
    S = {z: sol[(sol.zone == z) & (sol.year == BASE_YEAR)].cf.values for z in ZONES}
    W = {z: win[(win.zone == z) & (win.year == wyr)].cf.values for z in ZONES}
    for z in ZONES:
        S[z] = S[z][:8760]; W[z] = W[z][:8760]

    af = pd.DataFrame({'Time Period': 1, 'Month': ld.month.values,
                       'Day': ld.day.values, 'Hours': np.arange(1, 8761)})
    for i, r in g.iterrows():                      # existing units are G1..Ng
        col = f'G{i+1}'
        if r.Type == 'SolarPV':
            af[col] = np.round(S[r.Zone] * SOLAR_VINTAGE_DERATE, 5)
        elif r.Type == 'WindOn':
            af[col] = np.round(W[r.Zone] * WIND_VINTAGE_DERATE, 5)
        else:
            af[col] = 1.0
    ng = len(g)
    for j, r in gc.iterrows():                     # candidates continue the numbering
        col = f'G{ng+j+1}'
        if r.Type == 'SolarPV':
            af[col] = np.round(S[r.Zone], 5)
        elif r.Type == 'WindOn':
            af[col] = np.round(W[r.Zone], 5)       # new turbines, no vintage derate
        else:
            af[col] = 1.0
    af.to_csv(f'{dd}/gen_availability_timeseries.csv', index=False)

    # ---------------- policy + single parameter ----------------
    pd.DataFrame([{'State': 'TX', 'Time Period': 1, 'Allowance (tons)': 1e12}]
                 ).to_csv(f'{dd}/carbonpolicies.csv', index=False)
    pd.DataFrame([{'From_state': 'TX', 'To_state': 'TX', 'RPS': 0.0}]
                 ).to_csv(f'{dd}/rpspolicies.csv', index=False)
    prm = json.load(open(f'{OUTD}/prm_star.json'))[name]['PRM_star']
    pd.DataFrame([{'VOLL': 35000.0, 'planning_reserve_margin': round(prm, 4),
                   'BigM': 1e6, 'PT_RPS': 1e9, 'PT_emis': 1e9,
                   'spin_requirement': 0.03, 'delta_spin': 10/60,
                   'Inv_bugt_gen': 1e12, 'Inv_bugt_line': 1e12,
                   'Inv_bugt_storage': 1e12}]
                 ).to_csv(f'{dd}/single_parameter.csv', index=False)

    # ---------------- settings ----------------
    sdir = f'{CASE}/TX8z_GTEP_{name}_case/Settings'; os.makedirs(sdir, exist_ok=True)
    open(f'{sdir}/HOPE_model_settings.yml', 'w').write(f"""\
DataCase: Data_TX8z_GTEP_{name}/
model_mode: GTEP
inv_dcs_bin: 0
planning_reserve_mode: 1
operation_reserve_mode: 1
carbon_policy: 0
clean_energy_policy: 0
flexible_demand: 0
endogenous_rep_day: 0
external_rep_day: 0
transmission_loss: 0
transmission_expansion: 1
storage_ld_duration_hours: 12
resource_aggregation: 0
solver: gurobi
""")
    return dict(case=name, peak_MW=float(load.sum(axis=1).max()),
                zone_peaks=peak.round(1).to_dict(), dc_added=dc_added,
                n_gen=len(g), n_cand=len(gc), wind_year=int(wyr))


os.makedirs(CASE, exist_ok=True)
info = [build('base', False), build('dc', True)]
json.dump(info, open(f'{OUTD}/case_summary.json', 'w'), indent=1)
for i in info:
    print(f"{i['case']:>5}: system peak {i['peak_MW']:>9,.0f} MW   "
          f"{i['n_gen']} existing rows, {i['n_cand']} candidate rows   wind year {i['wind_year']}")
print(f'\nwritten to {CASE}')
