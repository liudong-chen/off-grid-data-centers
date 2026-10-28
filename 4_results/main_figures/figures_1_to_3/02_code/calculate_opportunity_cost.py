"""What the waiting time is worth, as an OPPORTUNITY COST rather than as depreciation.

WHY THIS REPLACES 67_wait_sweep.py
----------------------------------
The earlier version charged the wait at the DEPRECIATION of idle processors: the GPUs are
bought, they are written off over four years whether or not they are plugged in, so a
month of waiting destroys a month of book value.  That is a real cost but it is the wrong
one, and it is too small.  The processors and the power arrive TOGETHER: a campus that is
energised two years sooner is not merely a campus whose GPUs stopped ageing on a shelf, it
is a campus that was SELLING COMPUTE for two years that the delayed campus never sells at
all.  The binding constraint is the calendar, not the ledger.  Accelerators obsolete on a
fixed date set by the next generation of silicon, so a month lost at the front of the life
is not recovered at the back - it is gone.  The cost of the wait is therefore the FORGONE
EARNINGS of the whole facility, which is an opportunity cost, and it is what a developer
actually optimises against.

THE RATE
--------
    $400,000 per MW of facility (grid) load per month, forgone earnings while a campus
    is built but not energised.

    THE PRIMARY ANCHOR is CoreWeave's first full reported year, FY2024, because it is the
    only construction that puts a complete year of revenue against a disclosed power
    figure for the same period:

        revenue                             $1.915 B
        adjusted EBITDA                     ~$1.2 B          (62.7% of revenue)
        active power at end of 2024         ~250 MW

        $1.915 B / 250 MW / 12 months  =  $638.3k per MW per month of revenue
        x 62.7% adjusted-EBITDA margin =  $400.0k per MW per month of forgone earnings

    Dividing a FULL YEAR of revenue by the END-of-year power is conservative in the
    denominator: CoreWeave's average active power across 2024 was well below the 250 MW
    it finished with, so the true revenue per operating MW-month in 2024 was higher than
    $638.3k, and $400,000 is a floor on that year's economics rather than a midpoint.

    THREE CROSS-CHECKS from later disclosures, carried through at the same 62.7% margin:

    (1) IREN, FY26 results: "Recent 3-year contracts >$20m revenue per MW
        (IT)".  $20.0M/36 = $555.6k/month of IT load; at PUE 1.13 (this study's campuses
        run 1.10-1.20) that is $491.7k/month of GRID load.
            $491.7k x 0.627  =  $308k/MW/month
    (2) CoreWeave, Q2 2026 earnings call: revenue $2.58B for the quarter,
        "we ended Q2 with 1.5 gigawatts of active power".  $2,580M/3/1,500 = $573.3k.
            $573.3k x 0.627  =  $359k/MW/month
    (3) CoreWeave contracted backlog, same call: $104B against 3.7 GW of contracted
        power, i.e. $28.1M per MW; over a 5-year term $468.7k/MW/month.
            $468.7k x 0.627  =  $294k/MW/month

    BE CLEAR ABOUT THE DIRECTION.  The four constructions span $294k to $400k, and the
    value used is the TOP of that span, not its centre.  A higher rate makes the case for
    going off-grid stronger, so this is the anti-conservative choice for this paper's own
    finding and the sensitivity below is what carries the argument: the two sides of the
    comparison balance at $60,178/MW/month, which is 15% of the rate used and 20% of the
    lowest of the four constructions.  The finding survives any of them.

    FROM REVENUE TO A LOSS.  A delay does not destroy revenue alone; it also avoids the
    cash costs of running, and it DEFERS the accelerator purchase, so the depreciation
    that dominates these companies' income statements is not lost with it.  The quantity
    actually forgone is revenue less cash operating cost, which is what adjusted EBITDA
    approximates.

    THREE HONEST CAVEATS.  The margin is non-GAAP, struck before the depreciation of the
    accelerators, and the argument for using it is precisely that a power delay defers
    that capital spending rather than wasting it -- CoreWeave reported a GAAP operating
    loss in the same periods.  Second, both companies are pure-play AI-cloud operators
    whose revenue per MW is higher than a hyperscaler running mixed workloads, so this
    rate describes the AI campuses this study models and should not be read as an
    all-data-center figure.  Third, these are prices in a market the same filings
    describe as capacity-constrained.

THE DIFFERENCE
--------------
    24 months.  The source frames its delay at two years; this study's own queue
    difference is 4.2 years on-grid against a 2.0-year off-grid build (43_dc_depreciation),
    i.e. 2.2 years, so 24 months is the CONSERVATIVE of the two and is what is reported.

THE COMPARISON
--------------
    extra cost of off-grid electricity  =  (off-grid LCOE - on-grid LCOE) x annual energy,
                                           present-valued as a 20-year annuity at 8%
    value of arriving 24 months earlier =  rate x fleet MW x 24 months,
                                           present-valued monthly at the same 8%

Both sides are present values to the developer at the same discount rate, so the ratio is
a like-for-like magnitude and not a units trick.  There is no break-even left to solve:
the earlier file swept the waiting-time difference because the value of time was unknown
and had to be inferred.  It is no longer unknown, so the sweep and the bound are gone.

    python calculate_opportunity_cost.py
    -> out/opportunity_cost.json
"""
import json

import numpy as np

import os as _os
_PKG = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
OUT = f'{_PKG}/01_raw' if _os.path.isdir(f'{_PKG}/01_raw') else 'build/out'

# ------------------------------------------------------------------------- parameters
RATE_MW_MONTH = 400_000.0     # $/MW of grid load/month, forgone earnings while dark
                              # CoreWeave FY2024: $1.915 B / 250 MW / 12 x 62.7%
RATE_LOW = 294_000.0          # CoreWeave backlog, 5-year contract term, at 62.7% margin
RATE_HIGH = 400_000.0         # CoreWeave FY2024, the construction actually used

# the three public disclosures the rate is triangulated from, kept as data so the
# derivation in the header can be recomputed rather than trusted
DISCLOSED = dict(
    cw_fy2024_revenue=1.915e9, cw_fy2024_adj_ebitda=1.20e9,
    cw_fy2024_active_MW=250.0, cw_fy2024_months=12.0,
    iren_3yr_per_MW_IT=20.0e6, iren_term_months=36.0, pue=1.13,
    cw_q2_revenue=2.58e9, cw_q2_active_MW=1500.0, cw_q2_months=3.0,
    cw_backlog=104.0e9, cw_contracted_MW=3700.0, cw_assumed_term_months=60.0,
    cw_ebitda_margin=1.20e9/1.915e9)
MONTHS = 24.0                 # the delay difference, on-grid against off-grid
FLEET_MW = 40_425.0           # announced design capacity, 17 ERCOT campuses
COINCIDENT_MW = 32_408.0      # the fleet's coincident peak, the conservative denominator

R = 0.08                      # the developer's discount rate, both sides
N_ELEC = 20                   # electricity life, years
ENERGY_MWH = 222.051e6        # data-center energy, MWh/yr

# the two LCOEs this study measured, read from the solved outputs rather than retyped
ON_LCOE = json.load(open(f'{OUT}/ongrid_final.json'))['lcoe']
OFF_LCOE = json.load(open(f'{OUT}/offgrid_summary.json'))['separate']['lcoe']


def af_annual(n, r):
    """Annuity factor, n annual payments in arrears."""
    return (1 - (1 + r)**-n)/r


def af_monthly(n, r):
    """Annuity factor, n monthly payments in arrears at an annual rate r."""
    m = r/12.0
    return (1 - (1 + m)**-n)/m


# ------------------------------------------------------------- the electricity penalty
DL = OFF_LCOE - ON_LCOE
PEN_YR = DL*ENERGY_MWH
PEN_PV = PEN_YR*af_annual(N_ELEC, R)

# ---------------------------------------------------------------- the value of the time
PER_MONTH = RATE_MW_MONTH*FLEET_MW
VAL_UNDISC = PER_MONTH*MONTHS
VAL_PV = PER_MONTH*af_monthly(MONTHS, R)

# net of the power the earlier months would themselves have consumed, so that no part of
# the electricity bill is counted as a benefit
POWER_24MO = OFF_LCOE*ENERGY_MWH*(MONTHS/12.0)
VAL_PV_NET = VAL_PV - OFF_LCOE*ENERGY_MWH/12.0*af_monthly(MONTHS, R)

RATIO = VAL_PV/PEN_PV

# --------------------------------------------------------------- the horizon question
# WHY 20 YEARS, AND WHY NOT 5?
#
# $24.68/MWh is a LEVELISED difference, and the off-grid plant's capital is spread across
# its 20-year life inside it.  Counting only five years of that levelised charge is not a
# shorter view of the same cost - it is five years of a twenty-year mortgage with the
# asset kept, which silently credits the developer with the plant's full residual value.
# So the short horizon UNDERSTATES the cost of self-supply unless a residual is added
# back, and 20 years is the horizon that matches the asset the decision actually creates.
#
# The direction matters more than the argument: a shorter horizon makes self-supply look
# BETTER, not worse, so 20 years is the conservative choice for this paper's own finding.
# At the 4-year GPU book life the ratio is far larger still.
#
# And the choice does not have to be defended at all, because the electricity penalty is
# a bounded annuity: counted FOR EVER at 8% it is only $68.5 B, still far below the value
# of the time.  There is no horizon at which the comparison reverses.
HORIZONS = (4, 5, 10, 15, 20, 25, 30)
PEN_BY_N = {n: PEN_YR*af_annual(n, R) for n in HORIZONS}
PEN_PERP = PEN_YR/R

if __name__ == '__main__':
    W = 52
    print('=' * 78)
    print('THE COST OF WAITING, AS AN OPPORTUNITY COST')
    print('=' * 78)
    print(f'{"rate, forgone earnings":<{W}}${RATE_MW_MONTH:>13,.0f} /MW/month')
    print(f'{"fleet, announced design capacity":<{W}}{FLEET_MW:>14,.0f} MW')
    print(f'{"delay difference, on-grid against off-grid":<{W}}{MONTHS:>14,.0f} months')
    print(f'{"forgone earnings per month":<{W}}${PER_MONTH/1e9:>13,.2f} B')
    print(f'{"over 24 months, undiscounted":<{W}}${VAL_UNDISC/1e9:>13,.1f} B')
    print(f'{f"present value at {R:.0%}, monthly":<{W}}${VAL_PV/1e9:>13,.1f} B')

    print('\n' + '=' * 78)
    print('AGAINST THE ELECTRICITY IT BUYS')
    print('=' * 78)
    print(f'{"off-grid LCOE, one plant per campus":<{W}}${OFF_LCOE:>13,.2f} /MWh')
    print(f'{"on-grid LCOE, the increment to ERCOT":<{W}}${ON_LCOE:>13,.2f} /MWh')
    print(f'{"self-supply penalty":<{W}}${DL:>13,.2f} /MWh')
    print(f'{f"x {ENERGY_MWH/1e6:.1f} TWh/yr":<{W}}${PEN_YR/1e9:>13,.2f} B/yr')
    print(f'{f"present value, {N_ELEC}-year annuity at {R:.0%}":<{W}}'
          f'${PEN_PV/1e9:>13,.1f} B')

    print('\n' + '-' * 78)
    print(f'  value of arriving {MONTHS:.0f} months earlier is {RATIO:>5.2f}x the extra '
          f'cost of the electricity')
    print(f'  the developer pays ${PEN_PV/1e9:,.1f} B to gain ${VAL_PV/1e9:,.1f} B')
    print('-' * 78)

    D = DISCLOSED
    MGN = D['cw_ebitda_margin']
    print(f'\nwhere the rate comes from -- four constructions, two listed operators, '
          f'each at the\nCoreWeave FY2024 adjusted-EBITDA margin of {MGN:.1%}\n')
    tri = [
        ('CoreWeave FY2024, full year over end-of-year active power  <- USED',
         D['cw_fy2024_revenue']/D['cw_fy2024_months']/D['cw_fy2024_active_MW']),
        ('IREN, 3-year contracts >$20M per MW (IT), FY26 results',
         D['iren_3yr_per_MW_IT']/D['iren_term_months']/D['pue']),
        ('CoreWeave, Q2 2026 revenue over end-of-quarter active power',
         D['cw_q2_revenue']/D['cw_q2_months']/D['cw_q2_active_MW']),
        ('CoreWeave, $104B backlog over 3.7 GW, 5-year term',
         D['cw_backlog']/D['cw_contracted_MW']/D['cw_assumed_term_months']),
    ]
    print(f'  {"construction":<62}{"revenue":>11}{"forgone":>11}')
    for lab, v in tri:
        print(f'  {lab:<62}{v:>11,.0f}{v*MGN:>11,.0f}')
    vals = sorted(v*MGN for _l, v in tri)
    print(f'  {"spread of the four":<62}{"":>11}'
          f'{f"{vals[0]:,.0f} - {vals[-1]:,.0f}":>11}')
    print(f'  {"used in this paper":<62}{"":>11}{RATE_MW_MONTH:>11,.0f}'
          f'   -- the TOP of that spread, not its centre')

    print('\nsensitivities -- the conclusion survives all of them\n')
    print(f'  {"variant":<46}{"value $B":>10}{"ratio":>8}')
    for lab, v in (
        (f'headline: {FLEET_MW:,.0f} MW, {MONTHS:.0f} mo, PV at {R:.0%}', VAL_PV),
        ('undiscounted', VAL_UNDISC),
        ('net of the power those months would use', VAL_PV_NET),
        (f'coincident peak {COINCIDENT_MW:,.0f} MW instead of design',
         VAL_PV*COINCIDENT_MW/FLEET_MW),
        (f'rate at the low construction, ${RATE_LOW:,.0f}/MW/month',
         VAL_PV*RATE_LOW/RATE_MW_MONTH),
        (f'rate at the high construction, ${RATE_HIGH:,.0f}/MW/month',
         VAL_PV*RATE_HIGH/RATE_MW_MONTH),
        (f'rate halved to ${RATE_MW_MONTH/2:,.0f}/MW/month', VAL_PV*0.5),
        ('delay difference halved to 12 months',
         PER_MONTH*af_monthly(12, R)),
        ('both halved', 0.5*PER_MONTH*af_monthly(12, R)),
    ):
        print(f'  {lab:<46}{v/1e9:>10,.1f}{v/PEN_PV:>8.2f}x')
    print('\nthe horizon the electricity is counted over -- the finding does not turn '
          'on it\n')
    print(f'  {"years":>7}{"PV of the extra electricity":>30}{"ratio":>9}')
    for n in HORIZONS:
        tag = '   <- the paper' if n == N_ELEC else ('   <- GPU book life' if n == 4
                                                     else '')
        print(f'  {n:>7d}{PEN_BY_N[n]/1e9:>28,.1f} B{VAL_PV/PEN_BY_N[n]:>8.2f}x{tag}')
    print(f'  {"for ever":>7}{PEN_PERP/1e9:>28,.1f} B{VAL_PV/PEN_PERP:>8.2f}x'
          '   <- the annuity is bounded')
    print(f'\n  a shorter horizon makes self-supply look BETTER, so {N_ELEC} years is the '
          f'conservative\n  choice; and even counted for ever the electricity never '
          f'catches the time.')

    brk = PEN_PV/(FLEET_MW*af_monthly(MONTHS, R))
    print(f'\n  the rate at which the two sides balance, at {MONTHS:.0f} months, is '
          f'${brk:,.0f}/MW/month')
    print(f'  -- {100*brk/RATE_MW_MONTH:.0f}% of the rate used, and {100*brk/RATE_LOW:.0f}% of '
          f'the lowest of the four\n     constructions, so the disclosures would have to '
          f'be wrong by more than a factor of\n     {RATE_LOW/brk:.1f} before the finding '
          f'reverses -- and the finding holds at every one of the four.')

    json.dump(dict(rate_USD_MW_month=RATE_MW_MONTH, months=MONTHS, fleet_MW=FLEET_MW,
                   coincident_MW=COINCIDENT_MW, r=R, n_elec=N_ELEC,
                   energy_MWh=ENERGY_MWH, on_lcoe=ON_LCOE, off_lcoe=OFF_LCOE,
                   penalty_per_MWh=DL, penalty_per_yr=PEN_YR, penalty_pv=PEN_PV,
                   value_per_month=PER_MONTH, value_undiscounted=VAL_UNDISC,
                   value_pv=VAL_PV, value_pv_net_of_power=VAL_PV_NET,
                   ratio=RATIO, breakeven_rate_USD_MW_month=brk,
                   penalty_pv_by_horizon={str(n): PEN_BY_N[n] for n in HORIZONS},
                   penalty_pv_perpetuity=PEN_PERP,
                   ratio_perpetuity=VAL_PV/PEN_PERP,
                   horizon_note='20 years matches the life of the plant the decision '
                                'creates, and the LCOE difference already levelises that '
                                "plant's capital over it; a 5-year count would be five "
                                'years of a twenty-year mortgage with the asset kept. A '
                                'shorter horizon makes self-supply look better, so 20 is '
                                'conservative, and the annuity is bounded: counted for '
                                'ever the penalty is only $68.5 B.',
                   rate_low_USD_MW_month=RATE_LOW,
                   rate_high_USD_MW_month=RATE_HIGH,
                   rate_triangulation=DISCLOSED,
                   source='CoreWeave FY2024: revenue $1.915B, adjusted EBITDA ~$1.2B '
                          '(62.7% margin), ~250 MW of active power at year end -> '
                          '$1.915B/250/12 x 62.7% = $400,000 per MW-month of forgone '
                          'earnings. Cross-checked against IREN FY26 results (27 Aug '
                          '2026), "Recent 3-year contracts >$20m revenue per MW (IT)", '
                          'converted from IT to facility load at PUE 1.13; and CoreWeave '
                          'Q2 2026 earnings call, $2.58B quarterly revenue '
                          'over 1.5 GW of active power, and $104B backlog against 3.7 GW '
                          'contracted. All four carried at the FY2024 margin give '
                          '$294k-$400k; the value used is the top of that spread. '
                          'Adjusted EBITDA rather than revenue because a power delay '
                          'defers the accelerator purchase rather than wasting it.',
                   basis='forgone earnings of the whole facility while not energised, '
                         'not depreciation of idle processors; accelerators obsolete on a '
                         'fixed calendar so months lost at the front of life are not '
                         'recovered at the back'),
              open(f'{OUT}/opportunity_cost.json', 'w'), indent=1)
    print(f'\nwritten {OUT}/opportunity_cost.json')
