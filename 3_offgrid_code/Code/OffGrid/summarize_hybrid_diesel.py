"""Aggregate the diesel-bridge sensitivity to fleet level and put it on the paper's
time axis.

    python3 summarize_hybrid_diesel.py   -> out/hybrid_diesel/hybrid_diesel_summary.{json,csv,md}

Reads out/hybrid_diesel/site_*.json (from run_hybrid_diesel.py) and, when they exist, the HOPE
objectives of TX8z_GTEP_hybrid_dsl<XX>_vre_case (Phase 2).  Without HOPE results the
connected phase is approximated and flagged as such.

TIMELINE (months from the day the campus is ready and waiting for power)
    0            campus ready, orders placed
    T_DSL  = 12  diesel gensets commissioned  -> the campus runs f(x) of its load
    T_GAS  = 24  gas machines commissioned    -> full load, islanded
    T_GAS+M      grid interconnection (M = queue delay beyond the gas build, paper: 26
                 months = 4.2 yr - 2.0 yr; swept 0..48)
    then 240 months of operation counted from T_GAS

    f(x) = D_x / D_100 : the diesel fleet of share x can carry that fraction of the
    load reliably before the gas arrives (D_100 = gensets a diesel-only campus needs)

MONEY (all $ of 2025, r = 8%/yr, monthly compounding; RATE = $400k/MW-month forgone
earnings on the published campus MW, as in HYBRID_CASE_SUMMARY / Fig 5d)
    wait for grid   : earn nothing until T_GAS+M; then on-grid electricity (benchmark)
    off-grid        : earn from T_GAS; islanded premium (c_off - c_on) for 240 months
    hybrid, x       : earn f(x) from T_DSL, all from T_GAS;
                      premium A  (T_DSL..T_GAS)      diesel-only cost on f(x) of load
                      premium B  (T_GAS..T_GAS+M)    islanded design of share x
                      premium C  (after)             connected: HOPE (or placeholder)
    gas-only hybrid : x = 0 with connection (the 25 Sep case)
Everything is reported as NET VALUE AGAINST WAITING FOR THE GRID, so the comparison
is the same one Fig 3c and Fig 5d draw.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))            # Code/OffGrid
OUT = os.environ.get('OFFGRID_OUT', os.path.join(HERE, 'out', 'hybrid_diesel'))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
MC = f'{PROJECT_ROOT}/Code/HOPE/ModelCases'

E = lambda k, d: float(os.environ.get(k, d))
T_DSL, T_GAS, M_PAPER = int(E('T_DSL', 12)), int(E('T_GAS', 24)), int(E('M_QUEUE', 26))
RATE, R, N_MONTHS = E('RATE', 400_000.0), E('R', 0.08), 240
C_ON = E('C_ON', 43.14)                     # $/MWh on-grid societal (dc - base)/E
OBJ_BASE, OBJ_DC = 10.291e9, 19.871e9       # HOPE objectives, 25 Sep (fallback only)
V0 = 2.296e9                                # avoided cost the FULL 62,768 MW gas fleet returned
GAS_FULL = 62_768.0
K_FLEET_DSL_LIFE = 20                       # yr over which the diesel capex is annualised
r_m = R/12.0


def disc(m):
    return (1 + r_m)**(-np.asarray(m, float))


def pv_stream(start, months, monthly):
    """PV of a constant monthly amount paid at the END of months start+1..start+months."""
    t = np.arange(start + 1, start + months + 1)
    return float(monthly*disc(t).sum())


def hope_obj(case):
    for d in (f'{MC}/{case}/output/postprocess_snapshot/metadata.yml', f'{MC}/{case}/output/metadata.yml'):
        if os.path.exists(d):
            for line in open(d):
                if line.strip().startswith('objective_value'):
                    return float(line.split(':')[1])
    return None


def main():
    sites = [json.load(open(p)) for p in sorted(glob.glob(f'{OUT}/site_*.json'))]
    if not sites:
        raise SystemExit('no out/site_*.json yet')
    def isnum(k):
        try:
            float(k); return True
        except ValueError:
            return False
    common = set.intersection(*[set(s['cases']) for s in sites])
    keys = sorted([k for k in common if isnum(k)], key=float)
    extra = sorted(k for k in common if not isnum(k))          # e.g. '0.0_gasopt'
    dsl = sites[0]['diesel']; fuel = sites[0]['diesel_fuel']
    FX_DSL = dsl['fixed_$MWyr']; VR_DSL = dsl['vom'] + dsl['hr']*fuel
    FOM_DSL = dsl['fom']*1000.0
    E_tot = sum(s['energy_MWh'] for s in sites)
    MW_pub = sum(s['mw_published'] for s in sites)
    n_sites = len(sites)
    print(f'{n_sites} campuses, {E_tot/1e6:.1f} TWh/yr, {MW_pub:,.0f} MW published; shares {keys}')

    # ------------------------------------------------ fleet aggregates per share
    rows = []
    for k in keys + extra:
        x = float(k) if isnum(k) else k
        C = [s['cases'][k] for s in sites]
        gas = sum(c['gas_MW'] for c in C); dmw = sum(c['diesel_MW'] for c in C)
        inst = gas + dmw + sum((s['wind_MW'] or 0) + (s['solar_MW'] or 0) for s in sites)
        peak = sum(s['peak_MW'] for s in sites)
        tot = sum(c['total'] for c in C)
        # the load fraction diesel alone can carry, capacity-weighted over campuses
        f = 0.0
        for s in sites:                      # exact year-one load fraction (Monte Carlo, LOLE <= 0.1)
            pa = s['cases'][k].get('phaseA')
            if not isnum(k):
                fx = 0.0
            elif pa is not None:
                fx = pa['load_fraction']
            else:                            # older runs: D_x / D_100 approximation
                d100 = s['cases']['1.0']['diesel_units'] if '1.0' in s['cases'] else None
                fx = min(1.0, s['cases'][k]['diesel_units']/d100) if d100 else 0.0
            f += fx*s['mw_published']
        f /= MW_pub
        served_dsl = sum(c['served_by_type'][5] for c in C)
        rows.append(dict(
            share=x, gas_MW=gas, diesel_MW=dmw, diesel_units=sum(c['diesel_units'] for c in C),
            installed_MW=inst, ratio_to_sum_of_peaks=inst/peak,
            diesel_share_installed=dmw/inst if inst else 0.0,
            LOLE_min=min(c['LOLE'] for c in C), LOLE_max=max(c['LOLE'] for c in C),
            fixed_gas=sum(c['fixed_gas'] for c in C), fixed_diesel=sum(c['fixed_diesel'] for c in C),
            vre_fixed=sum(c['vre_fixed'] for c in C), land=sum(c['land'] for c in C),
            fuel_gas=sum(c['fuel_gas'] for c in C), fuel_diesel=sum(c['fuel_diesel'] for c in C),
            total=tot, lcoe_islanded=tot/E_tot, co2_Mt=sum(c['co2_t'] for c in C)/1e6,
            diesel_TWh=served_dsl/1e6, diesel_Mgal=served_dsl*dsl['hr']/0.1387/1e6,
            load_fraction_before_gas=f,
            diesel_capex_B=dmw*dsl['capex']*1000/1e9, gas_capex_B=None))
    REF = [r for r in rows if not isinstance(r['share'], float)]
    rows = [r for r in rows if isinstance(r['share'], float)]
    T = pd.DataFrame(rows).set_index('share')
    for r in REF:
        print(f"reference {r['share']}: gas {r['gas_MW']:,.0f} MW, {r['ratio_to_sum_of_peaks']:.2f}x peak, "
              f"${r['lcoe_islanded']:.2f}/MWh islanded")
    c_off = T.loc[0.0, 'lcoe_islanded'] if 0.0 in T.index else None

    # ------------------------------------------------ connected phase (HOPE if present)
    obj_base = hope_obj('TX8z_GTEP_base_vre_case') or OBJ_BASE
    obj_dc = hope_obj('TX8z_GTEP_dc_vre_case') or OBJ_DC
    c_on = (obj_dc - obj_base)/E_tot if hope_obj('TX8z_GTEP_dc_vre_case') else C_ON
    c_hyb, c_hyb_src = {}, {}
    for x in T.index:
        K_gas = T.loc[x, 'fixed_gas'] + T.loc[x, 'land'] + T.loc[x, 'vre_fixed']
        K_dsl_backup = T.loc[x, 'fixed_diesel']                           # capex + FOM
        K_dsl_retire = T.loc[x, 'fixed_diesel'] - T.loc[x, 'diesel_MW']*FOM_DSL
        tag = f'dsl{int(round(100*x)):03d}'
        o = hope_obj(f'TX8z_GTEP_hybrid_{tag}_vre_case')          # dsl000 = gas-only x = 0
        if o is not None:
            c_sys = (o - obj_base)/E_tot; src = 'HOPE'
        else:   # placeholder: the full fleet returned V0 of avoided cost; scale by gas MW left
            c_sys = c_on - V0*(T.loc[x, 'gas_MW']/GAS_FULL)/E_tot; src = 'placeholder (no HOPE run yet)'
        c_hyb[x] = dict(backup=c_sys + (K_gas + K_dsl_backup)/E_tot,
                        retire=c_sys + (K_gas + K_dsl_retire)/E_tot, c_sys=c_sys)
        c_hyb_src[x] = src
    T['c_hybrid_connected_backup'] = [c_hyb[x]['backup'] for x in T.index]
    T['c_hybrid_connected_retire'] = [c_hyb[x]['retire'] for x in T.index]
    T['connected_source'] = [c_hyb_src[x] for x in T.index]

    # ------------------------------------------------ lifetime value against waiting
    def value(x, M, mode='retire', connect=True):
        """PV of strategy x at queue delay M, MINUS the PV of waiting for the grid.

        Earnings: the $400k/MW-month rate is forgone ADJUSTED EBITDA, i.e. already net of
        the power bill at benchmark cost.  So electricity enters only as the PREMIUM over
        the on-grid benchmark c_on in every lit month (the convention of 98_hybrid.py and
        Fig 3c/5d), never as the full bill.
        """
        f = T.loc[x, 'load_fraction_before_gas'] if x > 0 else 0.0
        e_m = E_tot/12.0                       # MWh per month at full load
        earn = RATE*MW_pub*(f*pv_stream(T_DSL, T_GAS - T_DSL, 1.0)
                            + pv_stream(T_GAS, N_MONTHS, 1.0))
        earn_wait = RATE*MW_pub*pv_stream(T_GAS + M, N_MONTHS - M, 1.0)   # same horizon end
        prem = 0.0
        if x > 0 and f > 0:                    # A: diesel-only on f of the load; genset fixed cost accrues
            prem += pv_stream(T_DSL, T_GAS - T_DSL,
                              T.loc[x, 'fixed_diesel']/12.0 + (VR_DSL - c_on)*f*e_m)
        prem += pv_stream(T_GAS, M, T.loc[x, 'total']/12.0 - c_on*e_m)             # B: islanded
        if connect:
            prem += pv_stream(T_GAS + M, N_MONTHS - M, (c_hyb[x][mode] - c_on)*e_m)  # C: connected
        else:
            prem += pv_stream(T_GAS + M, N_MONTHS - M, T.loc[x, 'total']/12.0 - c_on*e_m)
        return (earn - earn_wait) - prem

    Ms = list(range(0, 49, 2))
    curves = {}
    if 0.0 in T.index:
        curves['off-grid forever (gas only)'] = [value(0.0, M, connect=False) for M in Ms]
        curves['hybrid, gas only, no diesel'] = [value(0.0, M, 'retire') for M in Ms]
    for x in T.index:
        if x > 0:
            curves[f'hybrid, diesel {int(round(100*x))}% (retire)'] = [value(x, M, 'retire') for M in Ms]
            curves[f'hybrid, diesel {int(round(100*x))}% (backup)'] = [value(x, M, 'backup') for M in Ms]

    def be(series):
        for M, v in zip(Ms, series):
            if v >= 0:
                return M
        return None

    at = {k: v[Ms.index(M_PAPER)] if M_PAPER in Ms else None for k, v in curves.items()}
    T['value_vs_wait_at_paper_queue_B_retire'] = [
        (value(x, M_PAPER, 'retire') if x > 0 else value(0.0, M_PAPER, 'retire'))/1e9 for x in T.index]
    T['value_vs_offgrid_forever_at_paper_queue_B_retire'] = (
        T['value_vs_wait_at_paper_queue_B_retire'] - value(0.0, M_PAPER, connect=False)/1e9) if 0.0 in T.index else np.nan

    # ------------------------------------------------ write
    T.to_csv(f'{OUT}/hybrid_diesel_summary.csv')
    json.dump(dict(params=dict(T_DSL=T_DSL, T_GAS=T_GAS, M_PAPER=M_PAPER, RATE=RATE, R=R, c_on=c_on,
                               diesel=dsl, diesel_fuel=fuel, n_sites=n_sites, E_TWh=E_tot/1e6, MW_pub=MW_pub),
                   table=T.reset_index().to_dict('records'), months=Ms, curves=curves,
                   breakeven_months={k: be(v) for k, v in curves.items()},
                   value_at_paper_queue_B={k: (v/1e9 if v is not None else None) for k, v in at.items()}),
              open(f'{OUT}/hybrid_diesel_summary.json', 'w'), indent=1, default=float)

    with open(f'{OUT}/hybrid_diesel_summary.md', 'w') as fh:
        fh.write(f'# Diesel-bridge sensitivity — {n_sites} campuses, {E_tot/1e6:.1f} TWh/yr\n\n')
        fh.write(f'Diesel: {dsl["S"]} MW gensets, ${dsl["capex"]:.0f}/kW, FOM ${dsl["fom"]:.0f}/kW-yr, '
                 f'VOM ${dsl["vom"]}/MWh, HR {dsl["hr"]}, fuel ${fuel}/MMBtu -> ${VR_DSL:.0f}/MWh; '
                 f'FOR {dsl["q"]}, MTTR {dsl["mttr"]} h, {dsl["poh"]} h/yr service. '
                 f'Timeline: diesel {T_DSL} mo, gas {T_GAS} mo, queue +{M_PAPER} mo; ${RATE:,.0f}/MW-month; r={R}.\n\n')
        fh.write('## Islanded fleet by diesel share\n\n')
        fh.write('| share | gas GW | diesel GW (units) | built/peak | LOLE range | islanded $/MWh | diesel TWh | CO2 Mt | load carried before gas |\n|---|---|---|---|---|---|---|---|---|\n')
        for x, r in T.iterrows():
            fh.write(f'| {x:.1f} | {r.gas_MW/1e3:.1f} | {r.diesel_MW/1e3:.1f} ({r.diesel_units:,.0f}) | {r.ratio_to_sum_of_peaks:.2f} | '
                     f'{r.LOLE_min:.3f}–{r.LOLE_max:.3f} | {r.lcoe_islanded:.2f} | {r.diesel_TWh:.1f} | {r.co2_Mt:.1f} | {r.load_fraction_before_gas:.2f} |\n')
        fh.write('\n## Connected phase ($/MWh, societal) and lifetime value against waiting\n\n')
        fh.write(f'on-grid benchmark c_on = ${c_on:.2f}/MWh\n\n')
        fh.write('| share | connected, diesel retired | connected, diesel kept as backup | source | value vs wait at paper queue ($B, retire) | vs off-grid forever ($B) |\n|---|---|---|---|---|---|\n')
        for x, r in T.iterrows():
            fh.write(f'| {x:.1f} | {r.c_hybrid_connected_retire:.2f} | {r.c_hybrid_connected_backup:.2f} | {r.connected_source} | '
                     f'{r.value_vs_wait_at_paper_queue_B_retire:.1f} | {r.value_vs_offgrid_forever_at_paper_queue_B_retire:.1f} |\n')
        fh.write('\n## Break-even queue delay (months beyond the gas build) by strategy\n\n')
        for k, v in curves.items():
            fh.write(f'- {k}: {be(v)} months; value at +{M_PAPER} mo = ${at[k]/1e9:.1f} B\n')
    print(T[['gas_MW', 'diesel_MW', 'ratio_to_sum_of_peaks', 'lcoe_islanded', 'load_fraction_before_gas',
             'c_hybrid_connected_retire', 'value_vs_wait_at_paper_queue_B_retire']].round(2).to_string())
    print(open(f'{OUT}/hybrid_diesel_summary.md').read().split('## Break-even')[1])


if __name__ == '__main__':
    main()
