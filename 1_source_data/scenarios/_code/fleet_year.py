"""Roll the ERCOT fleet forward from the EIA-860 2025 snapshot to a 2030 fleet.

  2030 fleet = Operable
             - units with Planned Retirement Year <= 2030
             + Proposed units that are genuinely COMMITTED and in service by 2030

"Committed" = EIA-860 status V (under construction >50%), U (under construction
<50%) or TS (construction complete, testing). Statuses T (permitted, not
started), L (approvals pending) and P (planned) are SPECULATIVE and are left
out - those are the builds the capacity-expansion model is supposed to decide.
"""
import os
import numpy as np, pandas as pd

E   = 'parameter_setting/eia8602025ER'
BUS = 'network/Texas2k/tx2k_s25_bus_coords.csv'
OUT = 'build/scen/fleet'
import sys
TARGET_YEAR = int(sys.argv[1])
# Framing: the 2030 EXISTING fleet is today's fleet minus announced
# retirements. Capacity already under construction is NOT pre-loaded - the model
# is asked to decide all new build, including replacing what retires.
INCLUDE_COMMITTED = False
COMMITTED   = ['V', 'U', 'TS']
# ERCOT is summer-peaking and accredits thermal at SEASONAL CAPABILITY, not
# nameplate. Gas turbines lose ~20% of nameplate in Texas summer heat.
USE_SUMMER_CAPACITY = True

AREA2ZONE = {1:'FarWest',2:'North',3:'West',4:'South',
             5:'NorthCentral',6:'SouthCentral',7:'Coast',8:'East'}
TECH = {
 'Natural Gas Fired Combined Cycle':'NGCC','Natural Gas Fired Combustion Turbine':'NGCT',
 'Natural Gas Steam Turbine':'NGST','Natural Gas Internal Combustion Engine':'NGCT',
 'Other Natural Gas':'NGCT','Other Gases':'NGCT','Conventional Steam Coal':'Coal',
 'Petroleum Coke':'Coal','Nuclear':'Nuclear','Onshore Wind Turbine':'WindOn',
 'Offshore Wind Turbine':'WindOn','Solar Photovoltaic':'SolarPV',
 'Solar Thermal with Energy Storage':'SolarPV','Solar Thermal without Energy Storage':'SolarPV',
 'Conventional Hydroelectric':'Hydro','Hydroelectric Pumped Storage':'Hydro',
 'Petroleum Liquids':'Oil','Wood/Wood Waste Biomass':'Biomass','Landfill Gas':'Biomass',
 'Municipal Solid Waste':'Biomass','Other Waste Biomass':'Biomass','All Other':'Oil',
}
MW  = 'Summer Capacity (MW)' if USE_SUMMER_CAPACITY else 'Nameplate Capacity (MW)'
NMW = 'Nameplate Capacity (MW)'

bus = pd.read_csv(BUS); bus['zone'] = bus['area'].map(AREA2ZONE)
bus = bus.dropna(subset=['latitude','longitude','zone'])
BLAT, BLON, BZ = np.radians(bus.latitude.values), np.radians(bus.longitude.values), bus.zone.values
def nz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    d = np.sin((BLAT-la)/2)**2 + np.cos(la)*np.cos(BLAT)*np.sin((BLON-lo)/2)**2
    return BZ[np.argmin(d)]

plant = pd.read_excel(f'{E}/2___Plant_Y2025_Early_Release.xlsx', skiprows=2)
er = plant[plant['Balancing Authority Code'].astype(str).str.strip()=='ERCO'].copy()
er['zone'] = [nz(r.Latitude, r.Longitude) for r in er.itertuples()]
p2z = dict(zip(er['Plant Code'], er['zone']))

xl = pd.ExcelFile(f'{E}/3_1_Generator_Y2025_Early_Release.xlsx')
op = xl.parse('Operable', skiprows=2); pr = xl.parse('Proposed', skiprows=2)

def prep(d):
    d = d[d['Plant Code'].isin(p2z)].copy()
    d['zone'] = d['Plant Code'].map(p2z)
    d['Type'] = d['Technology'].map(TECH)
    d[NMW] = pd.to_numeric(d[NMW], errors='coerce')
    if MW != NMW:
        d[MW] = pd.to_numeric(d.get(MW), errors='coerce').fillna(d[NMW])
    return d.dropna(subset=[MW]).query(f'`{MW}` > 0')

op = prep(op); pr = prep(pr)

# ---- existing, minus retirements ----
op['ret'] = pd.to_numeric(op['Planned Retirement Year'], errors='coerce')
alive = op[op['Status'].astype(str).str.strip().isin(['OP','SB'])]
retire = alive[alive.ret <= TARGET_YEAR]
keep   = alive[~(alive.ret <= TARGET_YEAR)]
print(f'retiring by {TARGET_YEAR}: {retire[MW].sum():,.0f} MW in {len(retire)} units')
print(retire.groupby('Type')[MW].sum().round(0).to_string() if len(retire) else '  none')

# ---- committed additions ----
pr['yr'] = pd.to_numeric(pr['Current Year'], errors='coerce')
add = pr[pr.Status.astype(str).str.strip().isin(COMMITTED) & (pr.yr <= TARGET_YEAR)]
if not INCLUDE_COMMITTED:
    add = add.iloc[0:0]
add_gen = add[add.Type.notna()]
add_bat = pr[(pr.Technology == 'Batteries') &
             pr.Status.astype(str).str.strip().isin(COMMITTED) & (pr.yr <= TARGET_YEAR)]
if not INCLUDE_COMMITTED:
    add_bat = add_bat.iloc[0:0]
print(f'\ncommitted generation online by {TARGET_YEAR}: {add_gen[MW].sum():,.0f} MW '
      f'in {len(add_gen)} units')
print(add_gen.groupby('Type')[MW].agg(['sum','size']).round(0).to_string())
print(f'\ncommitted storage: {add_bat[MW].sum():,.0f} MW in {len(add_bat)} units')

# ---- assemble ----
both = pd.concat([keep[['zone','Type',MW]], add_gen[['zone','Type',MW]]])
fleet = (both.groupby(['zone','Type'])
             .agg(**{'Pmax (MW)': (MW,'sum'), 'n_units': (MW,'size'),
                     'unit_mw_mean': (MW,'mean'), 'unit_mw_max': (MW,'max')})
             .reset_index())
fleet.to_csv(f'{OUT}/fleet_{TARGET_YEAR}.csv', index=False)

base = pd.read_csv(f'{OUT}/fleet_existing.csv')
cmp = (base.groupby('Type')['Pmax (MW)'].sum().rename('2025')
        .to_frame().join(fleet.groupby('Type')['Pmax (MW)'].sum().rename(str(TARGET_YEAR)), how='outer')
        .fillna(0))
cmp['delta'] = cmp[str(TARGET_YEAR)] - cmp['2025']
print(f'\n=== fleet 2025 -> {TARGET_YEAR} (MW) ===')
print(cmp.sort_values(str(TARGET_YEAR), ascending=False).round(0).to_string())
print(f"TOTAL  {cmp['2025'].sum():,.0f} -> {cmp[str(TARGET_YEAR)].sum():,.0f} MW "
      f"({cmp['delta'].sum():+,.0f})")

# ---- storage ----
gs = op[(op.Technology=='Batteries') & op['Status'].astype(str).str.strip().isin(['OP','SB'])]
sto_x = pd.read_excel(f'{E}/3_4_Energy_Storage_Y2025_Early_Release.xlsx', skiprows=2)
ec = [c for c in sto_x.columns if 'Nameplate Energy Capacity' in str(c)][0]
sto_x['MWh'] = pd.to_numeric(sto_x[ec], errors='coerce')
gs = gs.merge(sto_x[['Plant Code','Generator ID','MWh']], on=['Plant Code','Generator ID'], how='left')
dur = (gs.MWh.sum() / gs[MW].sum()) if gs[MW].sum() else 2.0
ab = (add_bat.copy() if INCLUDE_COMMITTED else add_bat.iloc[0:0].copy())
ab['MWh'] = ab[MW] * 2.0
allb = pd.concat([gs[['zone',MW,'MWh']], ab[['zone',MW,'MWh']]])
stor = (allb.groupby('zone').agg(**{'Max Power (MW)': (MW,'sum'),
                                    'Capacity (MWh)': ('MWh','sum'),
                                    'n_units': (MW,'size')}).reset_index())
stor['duration_h'] = stor['Capacity (MWh)'] / stor['Max Power (MW)']
stor.to_csv(f'{OUT}/storage_{TARGET_YEAR}.csv', index=False)
print(f'\n=== storage {TARGET_YEAR} ===');  print(stor.round(1).to_string(index=False))
print(f"TOTAL {stor['Max Power (MW)'].sum():,.0f} MW / {stor['Capacity (MWh)'].sum():,.0f} MWh "
      f"(existing fleet duration {dur:.2f} h)")
