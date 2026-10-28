"""Sequential Monte Carlo LOLE for (existing fleet + built portfolio), 8 weather years.

Same method as 08_report.py / 10_fixedpoint.py in the results package (two-state
Markov chains per unit from EIA-860 unit counts, NERC-GADS FOR/MTTR calibrated to
TRE, greedy storage dispatch, copper plate, loss-of-load DAYS), generalised to a
planning YEAR and a data-centre multiplier K.
"""
import json, os, sys
import numpy as np, pandas as pd

U = '.'
PKG = '2_ongrid_code/grid_case_assembly'
HERE = os.path.dirname(os.path.abspath(__file__))
ZONES = ['Coast', 'East', 'FarWest', 'North', 'NorthCentral', 'South', 'SouthCentral', 'West']
DCMAP = {'North Central': 'NorthCentral', 'South Central': 'SouthCentral', 'Far West': 'FarWest'}
YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
H = 8760
MTTR = {'NGCC': 48, 'NGCT': 36, 'NGST': 60, 'Coal': 72, 'Nuclear': 120, 'Oil': 36, 'Biomass': 48, 'Hydro': 24}
FOR_ = {'NGCC': 0.0357, 'NGCT': 0.0476, 'NGST': 0.0595, 'Coal': 0.0714, 'Nuclear': 0.0476,
        'Oil': 0.0595, 'Biomass': 0.0892, 'Hydro': 0.60}
WIND_DERATE, SOLAR_DERATE, ETA = 0.7242, 0.7839, 0.922
NEW_STO_H = 4.0
FOR_NEW = {'NGCC': 0.0357, 'NGCT': 0.0476, 'NuC': 0.08}
MTTR_NEW = {'NGCC': 48, 'NGCT': 36, 'NuC': 120}
UNIT_NEW = {'NGCC': 500., 'NGCT': 100., 'NuC': 1200.}     # as in 10_fixedpoint.py


def unit_chain(rng, q, mttr, hours):
    mtbf = mttr*(1 - q)/q
    up = rng.random() > q
    x = np.empty(hours, np.int8); t = 0
    while t < hours:
        d = rng.geometric(1/mtbf) if up else rng.geometric(1/mttr)
        x[t:min(t+d, hours)] = 1 if up else 0
        t += d; up = not up
    return x


def block_trace(rows, rng, hours):
    diff = np.zeros(hours+1); total = 0.0
    for r in rows:
        typ = r['Type']
        if typ in ('WindOn', 'SolarPV'):
            continue
        mw = r['Pmax (MW)']; n = max(1, int(r['n_units'])); c = mw/n
        q = r.get('q', FOR_.get(typ, 0.0)); m = r.get('mttr', MTTR.get(typ, 48))
        total += mw
        if q <= 0:
            continue
        for _ in range(n):
            x = unit_chain(rng, q, m, hours)
            dn = np.diff(np.concatenate(([1], x, [1])))
            for s, e in zip(np.flatnonzero(dn == -1), np.flatnonzero(dn == 1)):
                diff[s] -= c; diff[e] += c
    return total + np.cumsum(diff[:hours])


def dispatch(net, sto_p, sto_e):
    E = sto_e; short = np.zeros(H, bool)
    for h in range(H):
        n = net[h]
        if n > 0:
            d = min(n, sto_p, ETA*E); E -= d/ETA; short[h] = (n - d) > 1e-6
        else:
            E += ETA*min(-n, sto_p, (sto_e - E)/ETA)
    return short


def _raw_loads():
    """Raw ERCOT system load by weather year (MW), from the package's MC cache
    (which stores load x 1.019^5) - identical to reading the Native_Load workbooks."""
    s = pd.read_csv(f'{PKG}/03_results/mc_cache_base_series.csv')
    return {y: s[f'load_{y}'].values[:H]/(1.019**5) for y in YEARS}


def build_case(year, k):
    growth = 1.019 ** (year - 2025)
    fleet = pd.read_csv(f'{HERE}/fleet/fleet_{year}.csv')
    st = pd.read_csv(f'{HERE}/fleet/storage_{year}.csv')
    sto_p, sto_e = st['Max Power (MW)'].sum(), st['Capacity (MWh)'].sum()
    sol = pd.read_csv(f'{PKG}/03_results/_hourly_vre/solar_cf_hourly.csv')
    win = pd.read_csv(f'{PKG}/03_results/_hourly_vre/wind_cf_hourly.csv')
    smw = fleet[fleet.Type == 'SolarPV'].set_index('zone')['Pmax (MW)']
    wmw = fleet[fleet.Type == 'WindOn'].set_index('zone')['Pmax (MW)']
    VRE = {}
    for y in YEARS:
        v = np.zeros(H)
        for z in ZONES:
            if z in smw.index:
                v += smw[z]*sol[(sol.zone == z) & (sol.year == y)].cf.values[:H]*SOLAR_DERATE
            if z in wmw.index:
                v += wmw[z]*win[(win.zone == z) & (win.year == y)].cf.values[:H]*WIND_DERATE
        VRE[y] = v
    dctot = np.zeros(H)
    if k:
        d = pd.read_csv(f'{U}/1_source_data/DC_Load/load_profiles/dc_load_hourly_MW_by_weather_zone.csv', index_col=0, parse_dates=True)
        d = d.rename(columns={c: DCMAP.get(c, c) for c in d.columns})
        d = d[~((d.index.month == 2) & (d.index.day == 29))]
        dctot = d[[c for c in ZONES if c in d.columns]].sum(axis=1).values[:H]*k
    raw = _raw_loads()
    LOAD = {y: raw[y]*growth + dctot for y in YEARS}
    firm0 = fleet[~fleet.Type.isin(['WindOn', 'SolarPV'])]['Pmax (MW)'].sum()
    return dict(fleet=fleet, sto_p=sto_p, sto_e=sto_e, VRE=VRE, LOAD=LOAD, firm0=firm0,
                peak=max(LOAD[y].max() for y in YEARS), year=year, k=k)


def mc(C, built_gen, built_sto_MW, reps, cand_av=None, seed=20260826):
    """LOLE (days/yr) of existing fleet + built portfolio.
    built_gen: {(zone, type): MW}; cand_av: {(zone,type): 8760 per-MW availability} for new VRE."""
    fleet = C['fleet']
    add = []
    for (z, ty), mw in built_gen.items():
        if ty in ('SolarPV', 'WindOn'):
            continue
        add.append(dict(zone=z, Type=ty, **{'Pmax (MW)': mw}, n_units=max(1, int(round(mw/UNIT_NEW[ty]))),
                        q=FOR_NEW[ty], mttr=MTTR_NEW[ty]))
    rows = [r for r in fleet[['Type', 'Pmax (MW)', 'n_units']].to_dict('records')
            if r['Type'] not in ('WindOn', 'SolarPV')] + add
    add_vre = np.zeros(H)
    for (z, ty), mw in built_gen.items():
        if ty in ('SolarPV', 'WindOn'):
            add_vre = add_vre + mw*cand_av[(z, ty)]
    sto_p = C['sto_p'] + built_sto_MW; sto_e = C['sto_e'] + built_sto_MW*NEW_STO_H
    rng = np.random.default_rng(seed); days = 0
    for i in range(reps):
        wy = YEARS[i % len(YEARS)]
        av = block_trace(rows, rng, H)
        sh = dispatch(C['LOAD'][wy] - C['VRE'][wy] - add_vre - av, sto_p, sto_e)
        days += sh.reshape(365, 24).any(1).sum()
    return days/reps, dict(peak=C['peak'], firm=sum(r['Pmax (MW)'] for r in rows), sto_p=sto_p, sto_e=sto_e,
                           new_vre_MW=sum(mw for (z, t), mw in built_gen.items() if t in ('SolarPV', 'WindOn')))


if __name__ == '__main__':
    # regression: the reported 2030 DC portfolio (NGCC 9,593.9 NC + 20,000 SC) should score ~0.09-0.10
    C = build_case(2030, 1.0)
    print('peak', C['peak'])
    L, info = mc(C, {('NorthCentral', 'NGCC'): 9593.9, ('SouthCentral', 'NGCC'): 20000.0}, 0.0, int(sys.argv[1]) if len(sys.argv) > 1 else 96)
    print('LOLE', L, info)
