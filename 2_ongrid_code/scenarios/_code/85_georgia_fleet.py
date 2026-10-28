"""Georgia Power's existing fleet, from EIA-860, at unit granularity.

The Monte Carlo does not care much about total MW -- it cares about HOW MANY MACHINES
that MW is divided into, because sigma/P = sqrt(q(1-q)/n).  A 4,000 MW block modelled as
one unit and the same block modelled as fifteen give completely different LOLE.  So the
Georgia fleet is taken from EIA-860 2025 Early Release, generator by generator, exactly as
the ERCOT fleet was.

    EIA-860 3_1_Generator, State = GA, Status = OP, Utility Name = "Georgia Power Co"
    -> 132 operating generators, 19,849 MW winter capability

ONE ADJUSTMENT, AND IT MATTERS.  EIA-860 lists a unit at its FULL plant capacity under
the operator's name.  Georgia Power operates Vogtle and Hatch but owns 45.7% and 50.1% of
them, and the same is true of Scherer and Wansley.  Its own accounts report 14,775 MW
OWNED (Facts & Figures 2025), against the 19,849 MW it operates.  The load in the case is
Georgia Power's retail load, which is its ownership share, so the capacity has to be its
ownership share too -- mixing full plant capacity with a partial load would manufacture a
reserve margin that does not exist.

So: UNIT COUNTS and the technology split come from EIA-860, and the MW per technology is
set to the Facts & Figures owned total.  Unit size is owned MW / count.  Both halves are
sourced; the join is the assumption, and it is the conservative one.

    python 85_georgia_fleet.py
    -> out/georgia_fleet.json
"""
import json, re
import numpy as np
import pandas as pd

E = '1_source_data/parameter_setting/eia8602025ER'
OUT = 'build/out'

# Georgia Power Facts & Figures 2025, data as of 31 December 2024, MW OWNED
OWNED_TOTAL = {'gas_oil': 6252.0, 'Coal': 3911.0, 'Nuclear': 3067.0,
               'Hydro': 1112.0, 'SolarPV': 368.0, 'Battery': 65.0}

# EIA-860 technology + prime mover -> the model's unit types
MAP = {
    ('Nuclear', 'ST'): 'Nuclear',
    ('Conventional Steam Coal', 'ST'): 'Coal',
    ('Natural Gas Fired Combined Cycle', 'CT'): 'NGCC',
    ('Natural Gas Fired Combined Cycle', 'CA'): 'NGCC',
    ('Natural Gas Fired Combustion Turbine', 'GT'): 'NGCT',
    ('Natural Gas Internal Combustion Engine', 'IC'): 'NGCT',
    ('Natural Gas Steam Turbine', 'ST'): 'NGST',
    ('Petroleum Liquids', 'GT'): 'Oil',
    ('Petroleum Liquids', 'IC'): 'Oil',
    ('Other Natural Gas', 'FC'): 'NGCT',
    ('Conventional Hydroelectric', 'HY'): 'Hydro',
    ('Hydroelectric Pumped Storage', 'PS'): 'Hydro',
    ('Solar Photovoltaic', 'PV'): 'SolarPV',
    ('Batteries', 'BA'): 'Battery',
}
GASOIL = ('NGCC', 'NGCT', 'NGST', 'Oil')

# forced outage rate and mean time to repair, the same TRE-calibrated GADS values the
# ERCOT cache uses -- held, because there is no Georgia-specific GADS in the public record
FOR_ = {'Coal': 0.0714, 'NGCC': 0.0357, 'NGCT': 0.0476, 'NGST': 0.0595,
        'Nuclear': 0.0476, 'Oil': 0.0595, 'Hydro': 0.60, 'Purchase': 0.0476}
MTTR = {'Coal': 72, 'NGCC': 48, 'NGCT': 36, 'NGST': 60, 'Nuclear': 120,
        'Oil': 36, 'Hydro': 24, 'Purchase': 36}
# Hydro carries a 0.60 "outage" rate in the ERCOT cache: it is an energy limit expressed
# as availability, not a mechanical failure rate.  Held for Georgia for consistency.

g = pd.read_excel(f'{E}/3_1_Generator_Y2025_Early_Release.xlsx', skiprows=2)
g.columns = [re.sub(r'\s+', ' ', str(c)).strip() for c in g.columns]
ga = g[(g.State == 'GA') & (g.Status == 'OP') &
       (g['Utility Name'] == 'Georgia Power Co')].copy()
for c in ('Summer Capacity (MW)', 'Winter Capacity (MW)'):
    ga[c] = pd.to_numeric(ga[c], errors='coerce')
ga['model'] = [MAP.get((t, p)) for t, p in zip(ga.Technology, ga['Prime Mover'])]
missing = ga[ga.model.isna()]
assert len(missing) == 0, f'unmapped EIA-860 rows: {missing.Technology.unique()}'

op = ga.groupby('model').agg(n=('Generator ID', 'size'),
                             winter=('Winter Capacity (MW)', 'sum'),
                             summer=('Summer Capacity (MW)', 'sum'))
print('Georgia Power, EIA-860 2025 ER, operating units (full plant capacity):')
print(op.to_string())
print(f'   {int(op.n.sum())} units, {op.winter.sum():,.0f} MW winter\n')

# scale gas/oil to the owned total, in proportion to winter capability
gasoil_op = op.loc[[t for t in GASOIL if t in op.index], 'winter'].sum()
f_gasoil = OWNED_TOTAL['gas_oil']/gasoil_op
rows = []
for t in op.index:
    n = int(op.loc[t, 'n'])
    if t in GASOIL:
        mw = float(op.loc[t, 'winter'])*f_gasoil
    elif t in OWNED_TOTAL:
        mw = OWNED_TOTAL[t]
    else:
        mw = float(op.loc[t, 'winter'])
    rows.append(dict(Type=t, mw=round(mw, 2), n_units=n))

print(f'ownership scaling: gas and oil {gasoil_op:,.0f} MW operated -> '
      f'{OWNED_TOTAL["gas_oil"]:,.0f} MW owned (x{f_gasoil:.4f});')
print('   nuclear, coal, hydro, solar and battery set directly to the owned totals\n')

fleet = [r for r in rows if r['Type'] not in ('SolarPV', 'Battery')]
solar = next(r for r in rows if r['Type'] == 'SolarPV')
batt = next(r for r in rows if r['Type'] == 'Battery')

# the purchased block: Georgia Power buys from the Southern pool.  Sized so the 2025
# system meets the company's own 26% winter margin against its 16,284 MW peak.
GP_PEAK_2025, TRM = 16284.0, 0.26
firm_owned = sum(r['mw'] for r in fleet)
PURCH = GP_PEAK_2025*(1 + TRM) - (firm_owned + solar['mw'] + batt['mw'])
# treated as 20 independent contracts so a purchase failure is not all-or-nothing;
# that count is assigned, and the LOLE is insensitive to it above about ten
fleet.append(dict(Type='Purchase', mw=round(PURCH, 2), n_units=20))

print(f'{"Type":<10}{"MW owned":>11}{"units":>7}{"MW/unit":>10}{"FOR":>8}{"MTTR":>6}')
for r in fleet:
    r['FOR'] = FOR_[r['Type']]; r['MTTR'] = MTTR[r['Type']]
    print(f'{r["Type"]:<10}{r["mw"]:>11,.1f}{r["n_units"]:>7d}'
          f'{r["mw"]/r["n_units"]:>10,.1f}{r["FOR"]:>8.4f}{r["MTTR"]:>6d}')
tot = sum(r['mw'] for r in fleet)
print(f'{"firm":<10}{tot:>11,.1f}{sum(r["n_units"] for r in fleet):>7d}')
print(f'{"solar":<10}{solar["mw"]:>11,.1f}{solar["n_units"]:>7d}   (VRE, not firm)')
print(f'{"battery":<10}{batt["mw"]:>11,.1f}{batt["n_units"]:>7d}   (storage, 4 h)')
print(f'\ntotal accredited-ish {tot + solar["mw"] + batt["mw"]:,.0f} MW against a '
      f'{GP_PEAK_2025:,.0f} MW peak = {(tot+solar["mw"]+batt["mw"])/GP_PEAK_2025:.3f}x')

json.dump(dict(source='EIA-860 2025 Early Release, 3_1_Generator, State=GA, Status=OP, '
                      'Utility Name="Georgia Power Co"; MW scaled to the owned totals in '
                      'Georgia Power Facts & Figures 2025',
               operated_MW_winter=float(op.winter.sum()),
               operated_units=int(op.n.sum()),
               owned_totals=OWNED_TOTAL, gasoil_scale=float(f_gasoil),
               firm=fleet, solar=solar, battery=batt,
               purchased_MW=float(PURCH), purchased_units=20,
               gp_peak_2025=GP_PEAK_2025, trm_winter=TRM),
          open(f'{OUT}/georgia_fleet.json', 'w'), indent=1)
print(f'\nwritten {OUT}/georgia_fleet.json')
