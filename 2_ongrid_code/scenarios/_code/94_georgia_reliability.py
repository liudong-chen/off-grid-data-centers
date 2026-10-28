"""Georgia's own LOLE-consistent reserve margin, on seven weather years.

THE QUESTION.  Everything else in this study is held to LOLE <= 0.1 days/year: ERCOT's
reserve margins are derived from it, and every off-grid campus in both regions is sized to
it.  The Georgia ON-GRID case alone is held to Georgia Power's 26% winter target, which the
2022 IRP sets on "a combination of economic and reliability metrics" using SERVM -- a
reliability model, but a target chosen on cost, not a margin pinned to a loss-of-load
level.  Those are different objects, and until now Georgia's own LOLE-consistent number
could not be produced because only one weather year existed.

WHAT THIS DOES.  With seven years (2019-2025) now in the cache, it answers two questions
by direct Monte Carlo, using the SAME scorer, the same two-state Markov outage model and
the same 0.1 d/yr criterion as the ERCOT cases:

  1.  Is Georgia Power's inherited fleet already adequate, on its own, with no new build?
  2.  What reserve margin does LOLE = 0.1 d/yr correspond to, for each case?  That is the
      number that is directly comparable with ERCOT's derived margins, and it is what
      belongs beside the 26% in the paper.

The margin is expressed against the SAME peak HOPE applies its constraint to -- each
case's own annual peak from zonedata.csv -- so the answer can be substituted straight into
single_parameter.csv.

Firm capacity is added as NGCT-equivalent units (419 MW, FOR 0.0476, MTTR 36 h), the
cheapest firm resource in the candidate set, so the sweep measures what it takes to reach
the standard rather than what it would cost.

    python 94_georgia_reliability.py [reps]
    -> out/georgia_reliability.json
"""
import importlib.util, json, os, sys
import numpy as np
import pandas as pd

B = 'build'
GA = f'{B}/ga_hope'
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 210      # 30 per weather year
TARGET = 0.1
H = 8760

os.environ['BYONG_MC_CACHE'] = B
spec = importlib.util.spec_from_file_location('mc', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Code', 'HOPE', '11_mc_portfolio.py'))
mc = importlib.util.module_from_spec(spec); spec.loader.exec_module(mc)

# the shared scorer hard-codes the ERCOT directory naming; Georgia's cases are GA1z
mc.case_root = lambda case: [f'{GA}/GA1z_GTEP_{t}_case/Data_GA1z_GTEP_{t}'
                             for t in ('base' if case == 'gabase' else 'dc',)][0]

NEW_UNIT_MW, NEW_Q, NEW_MTTR = 419.0, 0.0476, 36.0


def case_peak(tag):
    d = f'{GA}/GA1z_GTEP_{tag}_case'
    dd = f'{d}/' + [x for x in os.listdir(d) if x.startswith('Data_')][0]
    return float(pd.read_csv(f'{dd}/zonedata.csv')['Demand (MW)'].iloc[0])


def accredited(tag, added_mw):
    """Accredited capacity the way HOPE counts it: CC-weighted, plus what is added."""
    d = f'{GA}/GA1z_GTEP_{tag}_case'
    dd = f'{d}/' + [x for x in os.listdir(d) if x.startswith('Data_')][0]
    g = pd.read_csv(f'{dd}/gendata.csv')
    s = pd.read_csv(f'{dd}/storagedata.csv')
    a = float((g['Pmax (MW)']*g['CC']).sum())
    a += float((s['Max Power (MW)']*s['CC']).sum())
    return a + added_mw          # new firm carries CC = 1.0


def score(cache, added_mw, reps=REPS):
    """LOLE of the inherited fleet plus `added_mw` of firm capacity."""
    Z = np.load(f'{B}/mc_cache_{cache}.npz')
    LOAD, VRE, years = Z['LOAD'], Z['VRE'], Z['years']
    sto_p, sto_e = float(Z['sto_p']), float(Z['sto_e'])
    rows = [dict(mw=float(m), n=int(n), q=float(q), mttr=float(t))
            for m, n, q, t in zip(Z['fleet_mw'], Z['fleet_n'],
                                  Z['fleet_q'], Z['fleet_mttr'])]
    if added_mw > 0:
        rows.append(dict(mw=float(added_mw),
                         n=max(1, int(round(added_mw/NEW_UNIT_MW))),
                         q=NEW_Q, mttr=NEW_MTTR))
    rng = np.random.default_rng(20260826)
    days = 0
    for i in range(reps):
        wy = i % len(years)
        av = mc.block_trace(rows, rng, H)
        sh = mc.dispatch(LOAD[wy] - VRE[wy] - av, sto_p, sto_e)
        days += np.asarray(sh).reshape(365, 24).any(1).sum()
    return days/reps


def solve(cache, tag, lo=0.0, hi=None, iters=11):
    """Bisect on added firm MW until LOLE = TARGET."""
    peak = case_peak(tag)
    if hi is None:
        hi = 0.9*peak
    l0 = score(cache, lo)
    print(f'  {"added MW":>10}{"LOLE (d/yr)":>14}{"margin":>10}')
    print(f'  {lo:>10,.0f}{l0:>14.3f}{100*(accredited(tag, lo)/peak - 1):>9.2f}%'
          f'   <- inherited fleet only')
    if l0 <= TARGET:
        return dict(added_MW=0.0, lole=l0, prm=accredited(tag, 0.0)/peak - 1,
                    peak_MW=peak, binding=False)
    hl = score(cache, hi)
    if hl > TARGET:
        raise SystemExit(f'{tag}: LOLE still {hl:.3f} at +{hi:,.0f} MW; raise the bracket')
    for _ in range(iters):
        mid = 0.5*(lo + hi)
        lm = score(cache, mid)
        print(f'  {mid:>10,.0f}{lm:>14.3f}{100*(accredited(tag, mid)/peak - 1):>9.2f}%')
        if lm > TARGET:
            lo = mid
        else:
            hi = mid
    return dict(added_MW=hi, lole=score(cache, hi), peak_MW=peak,
                prm=accredited(tag, hi)/peak - 1, binding=True)


print(f'Georgia reliability test: {REPS} replications over seven weather years, '
      f'target LOLE {TARGET} d/yr\n')
R = {}
for cache, tag, label in (('gabase', 'base', 'Georgia, no data centres'),
                          ('gadc', 'dc', 'Georgia, with 12.4 GW of data centres')):
    print(f'{label}')
    R[tag] = solve(cache, tag)
    r = R[tag]
    print(f'  -> LOLE {TARGET} d/yr needs {r["added_MW"]:,.0f} MW of added firm capacity, '
          f'a reserve margin of {100*r["prm"]:.2f}%\n')

print('=' * 74)
print(f'{"case":<26}{"derived PRM":>14}{"in the case now":>18}{"Georgia Power":>16}')
NOW = {'base': 0.2600, 'dc': 0.1656}
for tag, label in (('base', 'no data centres'), ('dc', 'with data centres')):
    print(f'{label:<26}{100*R[tag]["prm"]:>13.2f}%{100*NOW[tag]:>17.2f}%'
          f'{"26.00% winter":>16}')
print('=' * 74)

json.dump(dict(reps=REPS, target=TARGET, n_weather_years=7, years='2019-2025',
               new_unit=dict(mw=NEW_UNIT_MW, FOR=NEW_Q, mttr=NEW_MTTR),
               cases={k: {kk: (float(vv) if isinstance(vv, (int, float)) else vv)
                          for kk, vv in v.items()} for k, v in R.items()},
               prm_in_case=NOW,
               note='Derived by the same sequential Monte Carlo, the same two-state '
                    'Markov outage model and the same 0.1 d/yr criterion as the ERCOT '
                    'margins, on seven weather years of Southern Company demand from '
                    'EIA-930 scaled to Georgia Power. The margin is expressed against '
                    "each case's own annual peak, the denominator HOPE applies its "
                    'reserve constraint to.'),
          open(f'{B}/out/georgia_reliability.json', 'w'), indent=1)
print(f'\nwritten {B}/out/georgia_reliability.json')
