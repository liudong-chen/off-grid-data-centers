"""Ramp-up scenario: the 40 GW of data center demand arrives in three stages,
10 GW, 25 GW and 40 GW, at months 0, 12 and 24.  Each stage is a proportional slice of the
17 data centers, so the fleet designs of the diesel-share sweep scale with the slice.

Same monthly accounting as Figures/figures_1_to_3/02_code/plot_figures_1_to_3.py (8% discounted to
day 0, 20 years after the first gas build).  For each stage, diesel can be on site 12 months
after the stage is committed and gas 24 months after; if the grid arrives first, that tranche
is not built.  For each stage the data center takes the better of (i) build gas, then connect
at T_grid and (ii) run diesel on x of the load until the grid, never building gas, as in the
paper's hybrid route.  The waiting route serves each stage from max(stage month, T_grid).

python3 ramp_hybrid.py  ->  ramp_hybrid.json, ramp_hybrid.csv"""
import json, os, csv
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); B = os.path.abspath(os.path.join(H, '..', '..', '..'))
WN = f'{B}/Figures'
HS = json.load(open(f'{WN}/figures_1_to_3/01_raw/hybrid_diesel_summary.json')); P = HS['params']
ONG = json.load(open(f'{WN}/figures_1_to_3/01_raw/ongrid_final.json'))
T = {round(r['share'], 1): r for r in HS['table']}
E, C_ON, MW_PUB, R, T_DSL, T_GAS = ONG['energy_MWh'], P['c_on'], P['MW_pub'], P['R'], P['T_DSL'], P['T_GAS']
RATE = P['RATE']; r_m = R/12.0; e_m = E/12.0
VR_DSL = P['diesel']['vom'] + P['diesel']['hr']*P['diesel_fuel']
N_MONTHS = 240; END = T_GAS + N_MONTHS; NYRS = N_MONTHS/12.0
mo = np.arange(END + 1); disc = (1 + r_m)**(-mo.astype(float))

STAGES = [(0, 10/40), (12, 25/40), (24, 40/40)]          # (month committed, cumulative share of the fleet)
INC = [(t, c - (STAGES[i-1][1] if i else 0.0)) for i, (t, c) in enumerate(STAGES)]


def flows_wait(T_grid):
    earn = np.zeros(END + 1); cost = np.zeros(END + 1)
    for t, g in INC:
        on = mo > max(t, T_grid); earn[on] += g*RATE*MW_PUB; cost[on] += g*C_ON*e_m
    return earn*disc, cost*disc


def stage_routes(x, t, g, T_grid):
    """Two routes for one stage (slice g of the fleet committed at month t). Returns dict of
    route -> (earn, cost, gas_MW, diesel_MW, gas_month)."""
    row = T[x]; f = row['load_fraction_before_gas']
    td, tg = t + T_DSL, t + T_GAS
    out = {}
    # (i) diesel on x of the slice from td, gas at tg, connect at T_grid (only if gas arrives before the grid)
    if tg < T_grid:
        earn = np.zeros(END + 1); cost = np.zeros(END + 1)
        if x > 0:
            A = (mo > td) & (mo <= tg); earn[A] = f*g*RATE*MW_PUB; cost[A] = g*row['fixed_diesel']/12.0 + VR_DSL*f*g*e_m
        Bm = (mo > tg) & (mo <= T_grid); earn[Bm] = g*RATE*MW_PUB; cost[Bm] = g*row['total']/12.0
        C = mo > T_grid; earn[C] = g*RATE*MW_PUB; cost[C] = g*row['c_hybrid_connected_retire']*e_m
        out['gas'] = (earn*disc, cost*disc, g*row['gas_MW'], g*row['diesel_MW'], tg)
    # (ii) diesel on x of the slice until the grid, no gas; diesel capital sunk after connection
    earn = np.zeros(END + 1); cost = np.zeros(END + 1)
    if x > 0 and td < T_grid:
        A = (mo > td) & (mo <= T_grid); earn[A] = f*g*RATE*MW_PUB; cost[A] = g*row['fixed_diesel']/12.0 + VR_DSL*f*g*e_m
        dmw = g*row['diesel_MW']; capex_yr = dmw*P['diesel']['capex']*1000.0*0.1019
    else:
        dmw = 0.0; capex_yr = 0.0
    C = mo > max(t, T_grid); earn[C] = g*RATE*MW_PUB; cost[C] = g*C_ON*e_m + capex_yr/12.0
    out['diesel_only' if dmw > 0 else 'wait'] = (earn*disc, cost*disc, 0.0, dmw, None)
    return out


def ramp(x, T_grid):
    ew, cw = flows_wait(T_grid); v_wait = float((ew - cw).sum())
    earn = np.zeros(END + 1); cost = np.zeros(END + 1); gas = dsl = 0.0; routes = []
    for t, g in INC:
        rs = stage_routes(x, t, g, T_grid)
        build = [k for k in rs if k != 'wait']            # the build-own route: best of the build options
        best = max(build, key=lambda k: float((rs[k][0] - rs[k][1]).sum())) if build else 'wait'
        e_, c_, gm, dm, _ = rs[best]; earn += e_; cost += c_; gas += gm; dsl += dm; routes.append(best)
    v = float((earn - cost).sum())
    return dict(gas_GW=gas/1e3, diesel_GW=dsl/1e3, routes=routes,
                value_vs_wait_B_per_yr=(v - v_wait)/1e9/NYRS,
                extra_earnings_B_per_yr=float((earn - ew).sum())/1e9/NYRS,
                extra_cost_B_per_yr=float((cost - cw).sum())/1e9/NYRS)


def flat_case(x, T_grid):
    global INC
    keep = INC; INC = [(0, 1.0)]
    try:
        return ramp(x, T_grid)
    finally:
        INC = keep


def paper_hybrid(x, T_grid):
    """The paper's hybrid route exactly as in plot_figures_1_to_3.monthly_flows('hyb'): diesel on x of the
    load from month 12 to 24, gas from 24, connected (gas kept, diesel retired) from T_grid."""
    row = T[x]; f = row['load_fraction_before_gas']
    earn = np.zeros(END + 1); cost = np.zeros(END + 1)
    if x > 0:
        A = (mo > T_DSL) & (mo <= T_GAS); earn[A] = f*RATE*MW_PUB; cost[A] = row['fixed_diesel']/12.0 + VR_DSL*f*e_m
    Bm = (mo > T_GAS) & (mo <= T_grid); earn[Bm] = RATE*MW_PUB; cost[Bm] = row['total']/12.0
    C = mo > T_grid; earn[C] = RATE*MW_PUB; cost[C] = row['c_hybrid_connected_retire']*e_m
    ew, cw = flows_wait.__wrapped__(T_grid) if hasattr(flows_wait, '__wrapped__') else _wait_flat(T_grid)
    v = float(((earn - cost)*disc).sum()); v_wait = float((ew - cw).sum())
    return dict(gas_GW=row['gas_MW']/1e3, diesel_GW=row['diesel_MW']/1e3, routes=['gas'],
                value_vs_wait_B_per_yr=(v - v_wait)/1e9/NYRS)


def _wait_flat(T_grid):
    earn = np.zeros(END + 1); cost = np.zeros(END + 1)
    on = mo > T_grid; earn[on] = RATE*MW_PUB; cost[on] = C_ON*e_m
    return earn*disc, cost*disc


def bound(x, fn):
    tf = np.arange(0, 121)
    v = np.array([fn(x, int(t))['value_vs_wait_B_per_yr'] for t in tf])
    neg = np.where(v < 0)[0]
    if len(neg) == 0:
        return None
    after = np.where((v >= 0) & (tf > tf[neg[0]]))[0]
    if len(after) == 0:
        return None
    i = after[0]; return round(float(tf[i-1] + (0 - v[i-1])/(v[i] - v[i-1])), 1)


XS = [0.0, 0.3, 0.6, 1.0]; QS = [36, 50, 72]
out = dict(stages=STAGES, T_DSL=T_DSL, T_GAS=T_GAS, rate=RATE, cases={})
rows = []
for x in XS:
    for q in QS:
        rr = ramp(x, q); ff = flat_case(x, q)
        rec = dict(x=x, queue_months=q, ramp=rr, flat=ff)
        out['cases'][f'x={x:.1f},T={q}'] = rec
        rows.append(dict(x=x, queue=q, ramp_gas_GW=round(rr['gas_GW'], 1), flat_gas_GW=round(ff['gas_GW'], 1),
                         ramp_diesel_GW=round(rr['diesel_GW'], 1), flat_diesel_GW=round(ff['diesel_GW'], 1),
                         ramp_value=round(rr['value_vs_wait_B_per_yr'], 2), flat_value=round(ff['value_vs_wait_B_per_yr'], 2),
                         ramp_routes=' / '.join(rr['routes'])))
    out['cases'][f'x={x:.1f},bound'] = dict(ramp=bound(x, ramp), flat=bound(x, flat_case), paper=bound(x, paper_hybrid))
    print(f'x={x:.1f}  bound: ramp {out["cases"][f"x={x:.1f},bound"]["ramp"]}  flat (same rule) {out["cases"][f"x={x:.1f},bound"]["flat"]}  paper hybrid {out["cases"][f"x={x:.1f},bound"]["paper"]}')
for r in rows:
    print(r)
json.dump(out, open(f'{H}/ramp_hybrid.json', 'w'), indent=1)
with open(f'{H}/ramp_hybrid.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
