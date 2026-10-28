"""Existing ERCOT fleet by weather zone x technology, from EIA-860 2025 Early Release.

Zone assignment: each plant is mapped to the ERCOT weather zone of its NEAREST
Texas2k Series-25 bus. Texas2k AREA 1-8 are the weather zones, named in the RAW
file's AREA DATA block, and the per-zone bus counts reproduce exactly.
"""
import os, json
import numpy as np, pandas as pd

E   = 'parameter_setting/eia8602025ER'
BUS = 'network/Texas2k/tx2k_s25_bus_coords.csv'
OUT = 'build/out'; os.makedirs(OUT, exist_ok=True)

AREA2ZONE = {1:'FarWest', 2:'North', 3:'West', 4:'South',
             5:'NorthCentral', 6:'SouthCentral', 7:'Coast', 8:'East'}

# EIA-860 Technology -> HOPE Type. 'SolarPV' and 'WindOn' are exact literals
# that GTEP tests for; the rest are free labels.
TECH = {
 'Natural Gas Fired Combined Cycle':'NGCC',
 'Natural Gas Fired Combustion Turbine':'NGCT',
 'Natural Gas Steam Turbine':'NGST',
 'Natural Gas Internal Combustion Engine':'NGCT',
 'Other Natural Gas':'NGCT',
 'Other Gases':'NGCT',
 'Conventional Steam Coal':'Coal',
 'Petroleum Coke':'Coal',
 'Nuclear':'Nuclear',
 'Onshore Wind Turbine':'WindOn',
 'Offshore Wind Turbine':'WindOn',
 'Solar Photovoltaic':'SolarPV',
 'Solar Thermal with Energy Storage':'SolarPV',
 'Solar Thermal without Energy Storage':'SolarPV',
 'Conventional Hydroelectric':'Hydro',
 'Hydroelectric Pumped Storage':'Hydro',
 'Petroleum Liquids':'Oil',
 'Wood/Wood Waste Biomass':'Biomass',
 'Landfill Gas':'Biomass',
 'Municipal Solid Waste':'Biomass',
 'Other Waste Biomass':'Biomass',
 'All Other':'Oil',
}

bus = pd.read_csv(BUS)
bus['zone'] = bus['area'].map(AREA2ZONE)
bus = bus.dropna(subset=['latitude', 'longitude', 'zone'])
BLAT = np.radians(bus.latitude.values); BLON = np.radians(bus.longitude.values)
BZONE = bus.zone.values


def nearest_zone(lat, lon):
    """Great-circle nearest Texas2k bus."""
    la, lo = np.radians(lat), np.radians(lon)
    d = (np.sin((BLAT - la) / 2) ** 2 +
         np.cos(la) * np.cos(BLAT) * np.sin((BLON - lo) / 2) ** 2)
    return BZONE[np.argmin(d)]


plant = pd.read_excel(f'{E}/2___Plant_Y2025_Early_Release.xlsx', skiprows=2)
gen   = pd.read_excel(f'{E}/3_1_Generator_Y2025_Early_Release.xlsx', skiprows=2)
sto   = pd.read_excel(f'{E}/3_4_Energy_Storage_Y2025_Early_Release.xlsx', skiprows=2)

ercot = plant[plant['Balancing Authority Code'].astype(str).str.strip() == 'ERCO'].copy()
ercot['zone'] = [nearest_zone(r.Latitude, r.Longitude) for r in ercot.itertuples()]
print(f'ERCOT plants: {len(ercot)}   zones: {ercot.zone.value_counts().to_dict()}')
p2z = dict(zip(ercot['Plant Code'], ercot['zone']))

g = gen[gen['Plant Code'].isin(p2z)].copy()
g['zone'] = g['Plant Code'].map(p2z)
g['Type'] = g['Technology'].map(TECH)
unmapped = g[g.Type.isna()]['Technology'].value_counts()
if len(unmapped):
    print('\nUNMAPPED technologies (dropped):'); print(unmapped.to_string())
g = g.dropna(subset=['Type'])
g = g[g['Status'].astype(str).str.strip().isin(['OP', 'SB'])]      # operating + standby
g = g[g['Type'] != 'Battery']

MW = 'Nameplate Capacity (MW)'
g[MW] = pd.to_numeric(g[MW], errors='coerce')
g = g.dropna(subset=[MW])
g = g[g[MW] > 0]

fleet = (g.groupby(['zone', 'Type'])
           .agg(**{'Pmax (MW)': (MW, 'sum'), 'n_units': (MW, 'size'),
                   'unit_mw_mean': (MW, 'mean'), 'unit_mw_max': (MW, 'max')})
           .reset_index())
fleet.to_csv(f'{OUT}/fleet_existing.csv', index=False)

print('\n=== existing fleet, MW by zone x type ===')
print(fleet.pivot(index='Type', columns='zone', values='Pmax (MW)')
           .fillna(0).round(0).astype(int).to_string())
print('\n=== unit counts ===')
print(fleet.pivot(index='Type', columns='zone', values='n_units')
           .fillna(0).astype(int).to_string())
print('\n=== totals ===')
t = fleet.groupby('Type').agg(**{'MW': ('Pmax (MW)', 'sum'), 'units': ('n_units', 'sum'),
                                 'mean_unit_MW': ('unit_mw_mean', 'mean')})
print(t.sort_values('MW', ascending=False).round(1).to_string())
print(f"\nTOTAL {t.MW.sum():,.0f} MW across {int(t.units.sum()):,} units")

# ---- storage ----
s = sto[sto['Plant Code'].isin(p2z)].copy()
s['zone'] = s['Plant Code'].map(p2z)
mwcol = [c for c in s.columns if 'Nameplate Energy Capacity' in str(c)]
pcol  = [c for c in s.columns if c.strip() == 'Nameplate Capacity (MW)'] or \
        [c for c in s.columns if 'Nameplate Capacity' in str(c)]
print(f'\nstorage columns used: power={pcol[:1]}  energy={mwcol[:1]}')
s['MWh'] = pd.to_numeric(s[mwcol[0]], errors='coerce') if mwcol else np.nan
gs = gen[gen['Plant Code'].isin(p2z) & (gen['Technology'] == 'Batteries')].copy()
gs['zone'] = gs['Plant Code'].map(p2z)
gs[MW] = pd.to_numeric(gs[MW], errors='coerce')
gs = gs[gs['Status'].astype(str).str.strip().isin(['OP', 'SB'])].dropna(subset=[MW])
gs = gs.merge(s[['Plant Code', 'Generator ID', 'MWh']], on=['Plant Code', 'Generator ID'], how='left')
stor = (gs.groupby('zone')
          .agg(**{'Max Power (MW)': (MW, 'sum'), 'Capacity (MWh)': ('MWh', 'sum'),
                  'n_units': (MW, 'size')}).reset_index())
stor['duration_h'] = stor['Capacity (MWh)'] / stor['Max Power (MW)']
stor.to_csv(f'{OUT}/storage_existing.csv', index=False)
print('\n=== existing storage ===');  print(stor.round(1).to_string(index=False))
print(f"TOTAL {stor['Max Power (MW)'].sum():,.0f} MW / {stor['Capacity (MWh)'].sum():,.0f} MWh")
