"""Reliability report: dense LOLE curves, risk-hour diagnostics, convergence, sensitivity.

Speed trick that makes the dense sweep affordable:
  scaling every firm unit's Pmax by k scales the available-MW trace by exactly k,
  because the Markov chains depend only on (q, MTTR), not on capacity.
  So ONE thermal simulation per replication serves every negative add level, and
  positive add levels only need the small added block simulated.
"""
import json, os, sys
import numpy as np, pandas as pd

OUTD = 'build/out'
U    = '.'
ZONES = ['Coast','East','FarWest','North','NorthCentral','South','SouthCentral','West']
ZMAP  = {'COAST':'Coast','EAST':'East','FWEST':'FarWest','NORTH':'North',
         'NCENT':'NorthCentral','SOUTH':'South','SCENT':'SouthCentral','WEST':'West'}
DCMAP = {'North Central':'NorthCentral','South Central':'SouthCentral','Far West':'FarWest'}
YEARS = [2018,2019,2020,2021,2022,2023,2024,2025]
H = 8760
GROWTH = 1.019 ** 5

MTTR = {'NGCC':48,'NGCT':36,'NGST':60,'Coal':72,'Nuclear':120,'Oil':36,'Biomass':48,'Hydro':24}
FOR_ = {'NGCC':0.0357,'NGCT':0.0476,'NGST':0.0595,'Coal':0.0714,'Nuclear':0.0476,
        'Oil':0.0595,'Biomass':0.0892,'Hydro':0.60}
ADD_UNIT, ADD_FOR, ADD_MTTR = 500, 0.0357, 48
WIND_DERATE, SOLAR_DERATE, ETA = 0.7242, 0.7839, 0.922
CCS = {'West':0.2721,'FarWest':0.2721,'Coast':0.3614,'East':0.3614,'North':0.3614,
       'NorthCentral':0.3614,'South':0.3614,'SouthCentral':0.3614}
CCW = {'Coast':0.3083,'South':0.3083,'SouthCentral':0.3083,'North':0.3358,
       'West':0.1568,'FarWest':0.1568,'NorthCentral':0.1568,'East':0.1568}
STO_CC = 0.9329


def load_year(y):
    d = pd.read_excel(f'{U}/parameter_setting/ERCOT_load/Native_Load_{y}.xlsx')
    d = d.rename(columns={c: ZMAP.get(str(c).strip().upper(), str(c).strip()) for c in d.columns})
    t = pd.to_datetime(d[d.columns[0]].astype(str).str.replace(' 24:00',' 00:00'),
                       format='%m/%d/%Y %H:%M', errors='coerce')
    d = d[ZONES].copy(); d['m'] = t.dt.month.values; d['dy'] = t.dt.day.values
    d = d[~((d.m == 2) & (d.dy == 29))].reset_index(drop=True)
    return d[ZONES].sum(axis=1).values[:H]


def unit_chain(rng, q, mttr, hours):
    mtbf = mttr * (1 - q) / q
    up = rng.random() > q
    x = np.empty(hours, np.int8); t = 0
    while t < hours:
        d = rng.geometric(1 / mtbf) if up else rng.geometric(1 / mttr)
        x[t:min(t + d, hours)] = 1 if up else 0
        t += d; up = not up
    return x


def block_trace(rows, rng, hours):
    """Available MW per hour for a list of {Type, Pmax (MW), n_units}."""
    diff = np.zeros(hours + 1); total = 0.0
    for r in rows:
        typ = r['Type']
        if typ in ('WindOn', 'SolarPV'):
            continue
        mw = r['Pmax (MW)']; n = max(1, int(r['n_units'])); c = mw / n
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
    """Greedy storage against a net-load trace. Returns boolean shortfall by hour."""
    E = sto_e; short = np.zeros(H, bool)
    for h in range(H):
        n = net[h]
        if n > 0:
            d = min(n, sto_p, ETA * E); E -= d / ETA; short[h] = (n - d) > 1e-6
        else:
            E += ETA * min(-n, sto_p, (sto_e - E) / ETA)
    return short


def build_case(dc):
    fleet = pd.read_csv(f'{OUTD}/fleet_2030.csv')
    st = pd.read_csv(f'{OUTD}/storage_2030.csv')
    sto_p, sto_e = st['Max Power (MW)'].sum(), st['Capacity (MWh)'].sum()
    sol = pd.read_csv(f'{OUTD}/solar_cf_hourly.csv'); win = pd.read_csv(f'{OUTD}/wind_cf_hourly.csv')
    smw = fleet[fleet.Type == 'SolarPV'].set_index('zone')['Pmax (MW)']
    wmw = fleet[fleet.Type == 'WindOn'].set_index('zone')['Pmax (MW)']
    VRE = {}
    for y in YEARS:
        v = np.zeros(H)
        for z in ZONES:
            if z in smw.index:
                v += smw[z] * sol[(sol.zone == z) & (sol.year == y)].cf.values[:H] * SOLAR_DERATE
            if z in wmw.index:
                v += wmw[z] * win[(win.zone == z) & (win.year == y)].cf.values[:H] * WIND_DERATE
        VRE[y] = v
    dctot = 0.0
    if dc:
        d = pd.read_csv('DC_Load/load_profiles/dc_load_hourly_MW_by_weather_zone.csv',
                        index_col=0, parse_dates=True)
        d = d.rename(columns={c: DCMAP.get(c, c) for c in d.columns})
        d = d[~((d.index.month == 2) & (d.index.day == 29))]
        dctot = d[[c for c in ZONES if c in d.columns]].sum(axis=1).values[:H]
    LOAD = {y: load_year(y) * GROWTH + (dctot if dc else 0.0) for y in YEARS}
    vre_cc = sum(smw.get(z, 0) * CCS[z] + wmw.get(z, 0) * CCW[z] for z in ZONES)
    firm0 = fleet[~fleet.Type.isin(['WindOn', 'SolarPV'])]['Pmax (MW)'].sum()
    return dict(fleet=fleet, sto_p=sto_p, sto_e=sto_e, VRE=VRE, LOAD=LOAD,
                vre_cc=vre_cc, firm0=firm0, peak=max(LOAD[y].max() for y in YEARS))


def prm_of(C, add):
    acc = C['firm0'] + add + C['vre_cc'] + C['sto_p'] * STO_CC
    return acc / C['peak'] - 1


def sweep(C, adds, reps, seed, want_risk_at=None):
    """Dense sweep. Returns LOLE per add, per-replication days, and risk-hour counts."""
    rows = [r for r in C['fleet'][['Type','Pmax (MW)','n_units']].to_dict('records')
            if r['Type'] not in ('WindOn','SolarPV')]
    f0 = C['firm0']
    per_rep = {a: [] for a in adds}
    risk = np.zeros((12, 24))
    rng = np.random.default_rng(seed)
    # calendar index for a 365-day non-leap year starting Jan 1
    cal = pd.date_range('2023-01-01', periods=H, freq='h')
    mo, hr = cal.month.values - 1, cal.hour.values
    for k in range(reps):
        wy = YEARS[k % len(YEARS)]
        base_av = block_trace(rows, rng, H)              # ONE simulation ...
        net0 = C['LOAD'][wy] - C['VRE'][wy]
        for a in adds:
            if a < 0:
                av = base_av * max(0.0, (f0 + a) / f0)   # ... reused by exact scaling
            elif a > 0:
                n = max(1, int(round(a / ADD_UNIT)))
                av = base_av + block_trace(
                    [{'Type':'NGCC','Pmax (MW)':float(a),'n_units':n,
                      'q':ADD_FOR,'mttr':ADD_MTTR}], rng, H)
            else:
                av = base_av
            sh = dispatch(net0 - av, C['sto_p'], C['sto_e'])
            per_rep[a].append(sh.reshape(365, 24).any(1).sum())
            if want_risk_at is not None and a == want_risk_at:
                np.add.at(risk, (mo[sh], hr[sh]), 1)
    out = pd.DataFrame([dict(add_MW=a, PRM=prm_of(C, a),
                             firm_over_peak=(f0 + a) / C['peak'],
                             LOLE_days=float(np.mean(per_rep[a])),
                             se=float(np.std(per_rep[a], ddof=1) / np.sqrt(reps)))
                        for a in adds])
    return out, {a: np.array(v) for a, v in per_rep.items()}, risk / reps


def cross(df, target=0.1):
    """Log-linear interpolation of the LOLE = target crossing, in PRM."""
    d = df.sort_values('PRM').reset_index(drop=True)
    for j in range(1, len(d)):
        y0, y1 = d.LOLE_days[j-1], d.LOLE_days[j]
        if y0 > target >= y1:
            f0, f1 = d.PRM[j-1], d.PRM[j]
            l0, l1 = np.log(max(y0,1e-9)), np.log(max(y1,1e-9))
            return float(f0 + (np.log(target) - l0) * (f1 - f0) / (l1 - l0))
    return None


if __name__ == '__main__':
    REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 96
    ADDS = {'base': [-16000,-12000,-9000,-7000,-6000,-5000,-4000,-3000,-2000,-1000,0,2000],
            'dc':   [14000,17000,20000,22000,24000,25000,26000,27000,28000,30000,32000,35000]}
    RISK = {'base': -2000, 'dc': 28000}
    res = {}
    for case, dc in [('base', False), ('dc', True)]:
        C = build_case(dc)
        df, per_rep, risk = sweep(C, ADDS[case], REPS, 20260826, RISK[case])
        df.to_csv(f'{OUTD}/report_sweep_{case}.csv', index=False)
        pd.DataFrame(risk, index=range(1,13), columns=range(24)).to_csv(f'{OUTD}/report_risk_{case}.csv')
        p = cross(df)
        # convergence: running mean of the replication-level day counts at the risk point
        v = per_rep[RISK[case]]
        conv = pd.DataFrame({'n': np.arange(1, len(v)+1),
                             'mean': np.cumsum(v)/np.arange(1, len(v)+1)})
        conv['se'] = [np.std(v[:i], ddof=1)/np.sqrt(i) if i > 1 else np.nan for i in conv.n]
        conv.to_csv(f'{OUTD}/report_conv_{case}.csv', index=False)
        res[case] = dict(PRM_star=p, peak_MW=float(C['peak']), firm0_MW=float(C['firm0']),
                         vre_cc_MW=float(C['vre_cc']), sto_p_MW=float(C['sto_p']),
                         sto_e_MWh=float(C['sto_e']), reps=REPS,
                         firm_over_peak_at_star=float((C['firm0'] + RISK[case]) / C['peak']))
        print(f'\n=== {case}  peak {C["peak"]:,.0f}  firm0 {C["firm0"]:,.0f}  '
              f'vre_cc {C["vre_cc"]:,.0f}  sto {C["sto_p"]:,.0f} MW')
        print(df.to_string(index=False, float_format=lambda x: f'{x:,.4f}'))
        print(f'  --> PRM* = {p:.2%}' if p else '  --> not bracketed')
    json.dump(res, open(f'{OUTD}/report_summary.json','w'), indent=1)
    print('\n', json.dumps(res, indent=1))
