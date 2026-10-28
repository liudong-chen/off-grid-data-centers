"""LOLE of a given portfolio. Called by run_prm_fixedpoint.jl between HOPE solves.

    python3 11_mc_portfolio.py portfolio.json [out.json]

portfolio.json:
{
  "case": "dc",                     # "base" or "dc" - picks the load record
  "reps": 240,                      # Monte Carlo replications
  "gen":  [{"zone":"SouthCentral","type":"NGCC","mw":20000.0}, ...],
  "storage": [{"zone":"Coast","mw":5000.0,"hours":4.0}, ...]
}

Types must match the candidate file: NGCC, NGCT, NuC, SolarPV, WindOn.

Self-contained by design. Everything about the 2030 existing fleet, the eight
weather years of load, and the existing VRE output is precomputed into
mc_cache_<case>.npz, which sits beside this script - no ERCOT spreadsheets, no
08_report.py, no absolute paths. The only thing read live is the candidate VRE
profile out of the case's gen_availability_timeseries.csv, so that new wind and
solar are scored on exactly the shapes HOPE optimised against.

Requires numpy only - no pandas, no SciPy, no HOPE-side Python.

Paths, in order of precedence:
  OFFGRID_CASE_ROOT   env var, else <script dir>/ModelCases, else
                    <script dir>/TX8z_GTEP_cases
  cache             OFFGRID_MC_CACHE env var, else <script dir>
"""
import csv, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
H = 8760
FOR_NEW  = {'NGCC': 0.0357, 'NGCT': 0.0476, 'NuC': 0.08}
MTTR_NEW = {'NGCC': 48, 'NGCT': 36, 'NuC': 120}
# unit size for new build is read from the candidate file, never hard-coded
ETA = 0.922


def nrows_csv(path):
    with open(path, newline='') as f:
        return sum(1 for _ in csv.reader(f)) - 1


def read_cols(path, names):
    with open(path, newline='') as f:
        r = csv.reader(f); hdr = next(r)
        idx = [hdr.index(n) for n in names]
        cols = [[] for _ in names]
        for row in r:
            if not row:
                continue
            for j, i in enumerate(idx):
                cols[j].append(row[i])
    return cols


def read_availability(path, ncols):
    """The G1..Gn block of gen_availability_timeseries.csv as an (8760, n) array."""
    with open(path, newline='') as f:
        r = csv.reader(f); hdr = next(r)
        idx = [hdr.index(f'G{i+1}') for i in range(ncols)]
        A = np.empty((H, ncols))
        for t, row in enumerate(r):
            if t >= H:
                break
            A[t] = [float(row[i]) for i in idx]
    return A


def case_root(case):
    cands = [os.environ.get('OFFGRID_CASE_ROOT'),
             os.path.join(HERE, 'ModelCases'),
             os.path.join(HERE, 'TX8z_GTEP_cases'),
             os.path.join(HERE, '..', 'Code', 'HOPE', 'ModelCases')]
    for c in cands:
        if not c:
            continue
        p = os.path.join(c, f'TX8z_GTEP_{case}_case', f'Data_TX8z_GTEP_{case}')
        if os.path.isdir(p):
            return p
    raise SystemExit(
        'case data not found. Tried:\n  ' +
        '\n  '.join(str(c) for c in cands if c) +
        '\nSet OFFGRID_CASE_ROOT to the directory holding TX8z_GTEP_<case>_case/.')


def cache_path(case):
    d = os.environ.get('OFFGRID_MC_CACHE', HERE)
    p = os.path.join(d, f'mc_cache_{case}.npz')
    if not os.path.isfile(p):
        raise SystemExit(f'missing {p} — it ships beside this script; '
                         'set OFFGRID_MC_CACHE if you keep it elsewhere.')
    return p


def unit_chain(rng, q, mttr, hours):
    """One unit's 0/1 availability, two-state Markov with geometric holding times."""
    mtbf = mttr*(1 - q)/q
    up = rng.random() > q
    x = np.empty(hours, np.int8); t = 0
    while t < hours:
        d = rng.geometric(1/mtbf) if up else rng.geometric(1/mttr)
        x[t:min(t + d, hours)] = 1 if up else 0
        t += d; up = not up
    return x


def block_trace(rows, rng, hours):
    """Available firm MW per hour, by difference-array accumulation."""
    diff = np.zeros(hours + 1); total = 0.0
    for r in rows:
        mw = r['mw']; n = max(1, int(r['n'])); c = mw/n
        q, m = r['q'], r['mttr']
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
    """Greedy storage against the residual. Returns the hourly shortfall flag."""
    if sto_p <= 0:
        return net > 1e-6
    E = sto_e; short = np.zeros(H, bool)
    for h in range(H):
        n = net[h]
        if n > 0:
            d = min(n, sto_p, ETA*E); E -= d/ETA; short[h] = (n - d) > 1e-6
        else:
            E += ETA*min(-n, sto_p, (sto_e - E)/ETA)
    return short


def lole(spec):
    case = spec.get('case', 'base'); reps = int(spec.get('reps', 240))
    Z = np.load(cache_path(case))
    LOAD, VRE, years = Z['LOAD'], Z['VRE'], Z['years']
    sto_p, sto_e = float(Z['sto_p']), float(Z['sto_e'])

    rows = [dict(mw=float(mw), n=int(n), q=q, mttr=m)
            for mw, n, q, m in zip(Z['fleet_mw'], Z['fleet_n'],
                                   Z['fleet_q'], Z['fleet_mttr'])] \
        if 'fleet_q' in Z else \
        [dict(mw=float(mw), n=int(n),
              q=FOR_FLEET.get(str(t), 0.0), mttr=MTTR_FLEET.get(str(t), 48))
         for t, mw, n in zip(Z['fleet_type'], Z['fleet_mw'], Z['fleet_n'])]

    root = case_root(case)
    NE = nrows_csv(f'{root}/gendata.csv')
    cz, ct, cp, cn = read_cols(f'{root}/gendata_candidate.csv',
                               ['Zone', 'Type', 'Pmax (MW)', 'n_units'])
    add_vre = np.zeros(H); new_firm = 0.0; new_vre = 0.0
    need_avail = any(e['type'] in ('SolarPV', 'WindOn') and float(e['mw']) > 0
                     for e in spec.get('gen', []))
    A = read_availability(f'{root}/gen_availability_timeseries.csv',
                          NE + len(cz)) if need_avail else None

    for e in spec.get('gen', []):
        z, ty, mw = e['zone'], e['type'], float(e['mw'])
        if mw <= 0:
            continue
        if ty in ('SolarPV', 'WindOn'):
            k = [i for i in range(len(cz)) if cz[i] == z and ct[i] == ty]
            if not k:
                raise SystemExit(f'no candidate row for {ty}@{z} in {root}')
            add_vre += mw*A[:, NE + k[0]]
            new_vre += mw
        else:
            if ty not in FOR_NEW:
                raise SystemExit(f'unknown candidate type {ty!r}; expected one of '
                                 f'{sorted(set(FOR_NEW) | {"SolarPV", "WindOn"})}')
            j = [i for i in range(len(cz)) if cz[i] == z and ct[i] == ty]
            if not j:
                raise SystemExit(f'no candidate row for {ty}@{z} in {root}')
            usz = float(cp[j[0]])/max(1, int(float(cn[j[0]])))   # Pmax / n_units
            rows.append(dict(mw=mw, n=max(1, int(round(mw/usz))),
                             q=FOR_NEW[ty], mttr=MTTR_NEW[ty]))
            new_firm += mw
    for s in spec.get('storage', []):
        sto_p += float(s['mw']); sto_e += float(s['mw'])*float(s.get('hours', 4.0))

    rng = np.random.default_rng(20260826); days = 0
    for i in range(reps):
        wy = i % len(years)
        av_t = block_trace(rows, rng, H)
        sh = dispatch(LOAD[wy] - VRE[wy] - add_vre - av_t, sto_p, sto_e)
        days += np.asarray(sh).reshape(365, 24).any(1).sum()

    firm = sum(r['mw'] for r in rows); peak = float(Z['peak'])
    return dict(LOLE_days=days/reps, reps=reps, case=case, peak_MW=peak,
                firm_MW=float(firm), firm_over_peak=float(firm/peak),
                new_firm_MW=float(new_firm), new_vre_MW=float(new_vre),
                storage_MW=float(sto_p), storage_MWh=float(sto_e))


# forced-outage rates for the 2030 existing fleet, calibrated to TRE summer EFORd 4.70%
FOR_FLEET  = {'NGCC': 0.0357, 'NGCT': 0.0476, 'NGST': 0.0595, 'Coal': 0.0714,
              'Nuclear': 0.0476, 'Oil': 0.0595, 'Biomass': 0.0892, 'Hydro': 0.60}
MTTR_FLEET = {'NGCC': 48, 'NGCT': 36, 'NGST': 60, 'Coal': 72, 'Nuclear': 120,
              'Oil': 36, 'Biomass': 48, 'Hydro': 24}

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    spec = json.load(open(sys.argv[1]))
    r = lole(spec)
    if len(sys.argv) > 2:
        json.dump(r, open(sys.argv[2], 'w'), indent=1)
    print(json.dumps(r))
