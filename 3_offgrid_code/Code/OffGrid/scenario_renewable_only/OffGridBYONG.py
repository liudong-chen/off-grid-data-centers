"""OffGridBYONG - the off-grid counterpart to ReliabilityPRM, built for comparison.

The on-grid case is: HOPE picks a least-cost portfolio subject to a reserve margin,
a sequential Monte Carlo scores its true LOLE, and the margin is bisected until the
built portfolio actually meets LOLE = 0.1 days/year.

Off-grid there is no reserve margin to bisect - a MW margin cannot see unit size, and
off-grid that is the whole problem (a 1 GW campus on 3 machines and a 1 GW campus on
30 machines have the same margin and wildly different LOLE). So the criterion IS the
constraint: enumerate integer portfolios of real machines, cost each exactly by merit
order, sort by cost, and simulate up the list until one meets LOLE <= 0.1. The first
that passes is the least-cost design that actually works.

Two decisions, in that order:
  1. WHERE   each campus sits - zone sets gas price, solar capacity factor and land
             value, so the same load has a different least-cost answer in each zone
  2. WHAT    it builds there - integer counts of catalogue machines, plus solar and
             4-hour storage where they pay

CONSISTENCY WITH THE ON-GRID CASE - this is the point of the whole exercise:
  * identical criterion: LOLE <= 0.1 loss-of-load DAYS per year
  * identical outage model: two-state Markov per unit, geometric holding times
  * identical storage treatment: dispatched greedily against the residual, never credited
  * identical weather TREATMENT: cost and design are set on one year (2025), exactly as
    HOPE is, and the reliability check can cycle a stack of weather years exactly as the
    grid Monte Carlo does (scenarios/_code/mc_lole.py, YEARS[i % 8]).  Pass `mc_load` as
    a (Y, 8760) stack to use it.  With a single year the two are identical, which is
    what every run to date has done -- see 24_weather_years.py for why one year is the
    right basis for the DECISION and eight for the PROBABILITY.
  * identical fuel: EIA-923 Page 5 delivered gas by ERCOT zone
  * identical dollar year (2025$) and annualisation convention

Deliberate differences, all forced by the physics of a single site:
  * unit sizes come from a data-centre catalogue, not the utility fleet
  * planned maintenance is explicit - off-grid there is no interconnection to hide behind
  * capacity is lumpy: the decision is an integer count, not a continuous MW
  * no transmission, no network - a single site serves itself

WHAT CHANGED IN THIS VERSION (v2), and why it matters for the results
  The search is the same search; it is just no longer budget-limited. Two exact
  reformulations made it ~300x faster, which is what let the 1.1-1.3 GW sites (S01,
  S02) find a feasible design at all - v1 exhausted its Monte Carlo budget before
  reaching one and reported "no feasible design", which was a statement about the
  budget, not about the site.
    1. Merit-order fuel cost is now evaluated in closed form off the load-duration
       curve instead of by an 8760-hour loop per portfolio. Energy served by the tier
       sitting between cumulative capacity c and c+cap is F(c) - F(c+cap), where
       F(x) = sum_t max(L_t - x, 0). Exact, not approximate - `selftest` checks it.
    2. The Monte Carlo now simulates a BANK of independent machines once per
       replication and evaluates every candidate portfolio against that same bank.
       Unit chains depend only on (FOR, MTTR, POH), never on how many the portfolio
       buys, so the first n chains of the bank are a valid draw for any portfolio
       using n machines of that type. This is also common random numbers across
       designs, which is what makes a cost-ordered walk statistically fair: two
       portfolios one rank apart are compared on the same machine failures.
  One physics correction came with it: v1 subtracted a maintenance outage from
  availability even when the unit was already on forced outage, double-counting the
  loss. Availability is now (not failed) AND (not in maintenance).
"""
import csv, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
H = 8760
TARGET = 0.1                     # loss-of-load days per year - same as on-grid
ETA = 0.922                      # storage one-way efficiency - same as on-grid

# ---------------------------------------------------------------- zone parameters
# Delivered natural gas, EIA-923 Page 5, ERCOT 2025 ($/MMBtu). Zones with no
# reporting plant take the ERCOT-wide average. Same table the on-grid case uses.
GAS = {'FarWest': 3.891, 'North': 3.891, 'West': 3.891, 'East': 3.891,
       'South': 4.361, 'NorthCentral': 3.823, 'SouthCentral': 3.648, 'Coast': 4.079}

# Rural land, Texas Real Estate Research Center, Q3 2025 median $/acre, mapped from
# TRERC regions to ERCOT weather zones.
LAND = {'North':        1844.0,   # Panhandle-South Plains
        'FarWest':       714.0,   # Far West Texas
        'West':         2787.0,   # West Texas
        'East':         9313.0,   # Northeast Texas
        'NorthCentral': 9313.0,   # Northeast Texas - WEAK: DFW-adjacent land trades far above
        'Coast':       11423.0,   # Gulf Coast-Brazos Bottom
        'South':        5970.0,   # South Texas
        'SouthCentral': 7704.0}   # Austin-Waco-Hill Country
ACRES_PER_MW = {'gas': 0.10, 'solar': 5.0, 'storage': 0.05, 'dc': 0.0}   # data center land (0.75 acre/MW) removed : same on and off the grid
LAND_CRF = 0.08                  # land is not depreciated; opportunity cost of capital

# ---------------------------------------------------------------- machine catalogue
# Sizes are what data-centre developers actually buy. The 35 MW aeroderivative is the
# anchor: GE Vernova shipped 29 x LM2500XPRESS at 35 MW to Crusoe for ~1 GW of AI
# data-centre power, and Crusoe is site S01 in this study.
# Costs: EIA capital cost studies (AEO2025 for turbines, AEO2020 for the engine),
# CPI-escalated to 2025$, annualised at CRF 0.1019 (20-year life, 8% real).
CRF = 0.1019
E23, E19 = 1.053, 1.2549
CAT = [
 dict(k='Recip',   name='Reciprocating gas engine',   S=20.0,  capex=1810*E19,
      fom=35.16*E19, vom=5.69*E19, hr=8.295, q=0.030,  mttr=24, poh=300, land='gas',
      src='EIA AEO2020 ICE case; ERCOT has ~34 engines in the 15-20 MW class'),
 dict(k='Aero35',  name='Aeroderivative CT, 35 MW',   S=35.0,  capex=1606*E23,
      fom=9.56*E23,  vom=5.70*E23, hr=9.447, q=0.0476, mttr=36, poh=400, land='gas',
      src='LM2500XPRESS - GE Vernova/Crusoe, 29 units ~1 GW'),
 dict(k='Aero48',  name='Aeroderivative CT, 48 MW',   S=48.0,  capex=1606*E23,
      fom=9.56*E23,  vom=5.70*E23, hr=9.100, q=0.0476, mttr=36, poh=400, land='gas',
      src='LM6000 / PE6000 class'),
 dict(k='FrameCT', name='Frame CT, H-class',          S=419.0, capex=836*E23,
      fom=6.87*E23,  vom=1.24*E23, hr=9.142, q=0.0476, mttr=36, poh=450, land='gas',
      src='EIA AEO2025 H-class CT'),
 dict(k='CC',      name='Combined cycle 1x1, H-class', S=627.0, capex=921*E23,
      fom=15.51*E23, vom=3.33*E23, hr=6.226, q=0.0357, mttr=48, poh=500, land='gas',
      src='EIA AEO2025 H-class CC single-shaft'),
]
for u in CAT:
    u['fixed_$MWyr'] = (u['capex']*CRF + u['fom'])*1000.0

NC = len(CAT)
S_MW  = np.array([u['S'] for u in CAT])
FX_MW = np.array([u['fixed_$MWyr'] for u in CAT])

# Solar and storage, priced to match the on-grid candidate file so the two cases see
# the same technology costs.
SOLAR_FIXED = 117995.0    # $/MW-yr, same as gendata_candidate SolarPV
BATT_FIXED  = 161512.0    # $/MW-yr for a 4-hour battery, same as storagedata_candidate
BATT_HOURS  = 4.0


# ================================================================ costing
def ldc(load):
    """Load-duration machinery for exact merit-order integrals.

    Returns (la, ca, Etot) where la is load sorted ASCENDING and ca is its cumulative
    sum with a leading zero. Then, for any capacity level x,

        F(x) = sum_t max(L_t - x, 0) = (Etot - ca[k]) - (H - k)*x,  k = #{L_t <= x}

    is the energy that must be served ABOVE x. A merit tier occupying the capacity
    band [c, c+cap) therefore serves exactly F(c) - F(c+cap) MWh. This is an identity,
    not an approximation: `selftest` checks it against the hour-by-hour loop.
    """
    la = np.sort(np.asarray(load, float))
    ca = np.concatenate(([0.0], np.cumsum(la)))
    return la, ca, float(la.sum())


def energy_above(x, LD):
    """F(x), vectorised over an array of capacity levels x."""
    la, ca, Etot = LD
    x = np.asarray(x, float)
    k = np.searchsorted(la, x, side='right')
    return np.maximum((Etot - ca[k]) - (len(la) - k)*x, 0.0)


def var_costs(gas):
    """$/MWh by machine at this site's delivered gas price. Sets the merit order."""
    return np.array([u['vom'] + u['hr']*gas for u in CAT])


def merit_fuel_vec(N, VR, LD):
    """Merit-order fuel cost, $/yr, for every row of N at once.

    N is (P, NC) integer counts. Tiers are stacked cheapest-first; anything the
    portfolio cannot serve is unserved, and is NOT priced here - reliability is
    enforced by the LOLE constraint, not by a VOLL term, exactly as on-grid.
    """
    N = np.atleast_2d(N)
    order = np.argsort(VR)
    cum = np.zeros(len(N))
    fuel = np.zeros(len(N))
    Fc = energy_above(cum, LD)
    for i in order:
        cap = N[:, i]*S_MW[i]
        nxt = cum + cap
        Fn = energy_above(nxt, LD)
        fuel += VR[i]*(Fc - Fn)
        cum, Fc = nxt, Fn
    return fuel


def merit_fuel(load_sorted, caps, vr):
    """Reference implementation, hour by hour. Kept for `selftest` only."""
    tot, cum = 0.0, 0.0
    for cap, v in zip(caps, vr):
        served = np.clip(load_sorted - cum, 0.0, cap)
        tot += v*float(served.sum()); cum += cap
    return tot


def annual_cost_vec(N, zone, peak_MW, solar_mw=0.0, batt_mw=0.0):
    """Fixed + land, $/yr, for every row of N. Fuel is added separately."""
    N = np.atleast_2d(N)
    fixed = N @ (S_MW*FX_MW) + solar_mw*SOLAR_FIXED + batt_mw*BATT_FIXED
    acres = ((N @ S_MW)*ACRES_PER_MW['gas'] + solar_mw*ACRES_PER_MW['solar']
             + batt_mw*ACRES_PER_MW['storage'] + peak_MW*ACRES_PER_MW['dc'])
    return fixed + acres*LAND[zone]*LAND_CRF, acres


def annual_cost(load, n, solar_mw, batt_mw, gas, zone):
    """Scalar form: (fixed+land, land, acres) for one portfolio. Used for reporting."""
    n = np.asarray(n, float)
    fixed = float(n @ (S_MW*FX_MW)) + solar_mw*SOLAR_FIXED + batt_mw*BATT_FIXED
    acres = (float(n @ S_MW)*ACRES_PER_MW['gas'] + solar_mw*ACRES_PER_MW['solar']
             + batt_mw*ACRES_PER_MW['storage'] + float(load.max())*ACRES_PER_MW['dc'])
    land = acres*LAND[zone]*LAND_CRF
    return fixed, land, acres


# ================================================================ Monte Carlo
_POH = [u['poh'] for u in CAT]
_POH_PREFIX = np.concatenate(([0], np.cumsum(_POH)[:-1]))
_POH_CYCLE = float(sum(_POH))


def maint_start(i, j):
    """Hour at which unit j of machine type i begins its planned overhaul.

    An off-grid campus is one owner, one site, one O&M contract - nobody overhauls two
    turbines in the same week if they can help it. So the schedule is the one a
    competent operator would run: windows are PACKED END TO END around the year, in a
    fixed global order that interleaves the machine types (global index g = i + NC*j).
    Two consequences, both wanted:

      * a portfolio whose total overhaul hours fit inside the year gets ZERO concurrent
        planned outages, because a subset of a non-overlapping schedule cannot overlap;
      * a portfolio needing more overhaul hours than the year contains gets exactly the
        concurrency it must - the windows wrap and stack ceil(T/8760) deep.

    And because the start depends only on (i, j), never on how many units the portfolio
    holds, the schedule is consistent between a design simulated alone and the same
    design simulated inside a batch. That is what `selftest` verifies.
    """
    return (j*_POH_CYCLE + _POH_PREFIX[i]) % H


def _bank(key, q, mttr, poh, nmax, typ=0):
    """(nmax, H) uint8 availability for nmax independent, identical machines.

    Two-state Markov with geometric holding times, generated in bulk: draw enough
    up/down durations per unit, cumulate them into failure intervals, and lay the
    intervals down with a difference array. Then AND with the planned-outage mask.

    NESTING is the property that makes the bank shareable: the first m rows must be
    the same draw whatever nmax is, or two portfolios in the same batch would be
    compared on different machine failures purely because a third portfolio in the
    batch wanted more units. Each stream (up times, down times, initial state) is
    therefore drawn from its OWN generator, keyed on (seed, rep, type), and row-major
    filling does the rest. `selftest` checks batch == single as a result.
    """
    if nmax <= 0:
        return np.zeros((0, H), np.uint8)
    B = np.ones((nmax, H), np.uint8)
    if q > 0:
        mtbf = mttr*(1 - q)/q
        cyc = max(4, int(3*H/(mtbf + mttr)) + 8)
        lu, ld = np.log1p(-1/mtbf), np.log1p(-1/mttr)
        U = np.maximum(np.ceil(np.log(
            np.random.default_rng(key + (0,)).random((nmax, cyc)))/lu), 1.0)
        D = np.maximum(np.ceil(np.log(
            np.random.default_rng(key + (1,)).random((nmax, cyc)))/ld), 1.0)
        start_down = np.random.default_rng(key + (2,)).random(nmax) <= q
        U[start_down, 0] = 0.0                       # begin the year already failed
        seq = np.empty((nmax, 2*cyc))
        seq[:, 0::2], seq[:, 1::2] = U, D
        t = np.cumsum(seq, axis=1)                   # boundaries; downs are odd slots
        st = np.clip(t[:, 0::2], 0, H).astype(np.int32)      # down starts
        en = np.clip(t[:, 1::2], 0, H).astype(np.int32)      # down ends
        diff = np.zeros((nmax, H + 1), np.int32)
        rows = np.repeat(np.arange(nmax), cyc)
        np.add.at(diff, (rows, st.ravel()), -1)
        np.add.at(diff, (rows, en.ravel()), 1)
        B = (1 + np.cumsum(diff[:, :H], axis=1) > 0).astype(np.uint8)
    if poh and poh > 0:
        idx = np.arange(int(poh))
        for j in range(nmax):
            B[j, (idx + int(maint_start(typ, j))) % H] = 0
    return B


def _cumbank(seed, rep, nmax_by_type):
    """For each machine type, the cumulative count of available units among the first
    n of the bank, as an (nmax+1, H) int16 array. Row n is exactly what a portfolio
    holding n of that type sees."""
    out = []
    for i, u in enumerate(CAT):
        nm = int(nmax_by_type[i])
        B = _bank((int(seed), int(rep), i), u['q'], u['mttr'], u['poh'], nm, typ=i)
        c = np.zeros((nm + 1, H), np.int16)
        if nm:
            c[1:] = np.cumsum(B, axis=0, dtype=np.int16)
        out.append(c)
    return out


def _days_from_avail(netload, A, batt_mw):
    """Loss-of-load DAYS per portfolio. A is (P, H) available MW."""
    if batt_mw <= 0:
        short = netload[None, :] > A + 1e-6
        return short.reshape(len(A), 365, 24).any(2).sum(1)
    P = len(A)
    be = batt_mw*BATT_HOURS
    E = np.full(P, be); short = np.zeros((P, H), bool)
    for h in range(H):
        n = netload[h] - A[:, h]
        pos = n > 0
        d = np.where(pos, np.minimum(np.minimum(n, batt_mw), ETA*E), 0.0)
        E = np.where(pos, E - d/ETA,
                     E + ETA*np.minimum(np.minimum(-n, batt_mw), (be - E)/ETA))
        short[:, h] = pos & ((n - d) > 1e-6)
    return short.reshape(P, 365, 24).any(2).sum(1)


def lole_batch(load, N, reps, solar_mw=0.0, solar_cf=None, batt_mw=0.0,
               seed=20260826, chunk=384):
    """LOLE in days/year for every portfolio in N, sharing one machine bank per
    replication. Common random numbers across designs by construction.

    `load` is either one 8,760-hour year or a (Y, 8760) stack of weather years; with a
    stack, replication r runs against year r % Y, which is the cycling the grid Monte
    Carlo uses (mc_lole.py: wy = YEARS[i % len(YEARS)]).  One year reduces to the old
    behaviour exactly, so existing calls are unchanged.  `solar_cf` may be given the
    same way, and is matched year for year with the load when both are stacks.
    """
    N = np.atleast_2d(np.asarray(N, np.int64))
    P = len(N)
    LD = np.asarray(load, float)
    if LD.ndim == 1:
        LD = LD[None, :]
    Y = len(LD)
    NET = LD.copy()
    if solar_mw > 0 and solar_cf is not None:
        SC = np.asarray(solar_cf, float)
        if SC.ndim == 1:
            SC = SC[None, :]
        if len(SC) != Y and len(SC) == 1:
            SC = np.repeat(SC, Y, axis=0)
        NET -= solar_mw*SC
    nmax = N.max(0)
    days = np.zeros(P, np.int64)
    for r in range(int(reps)):
        netload = NET[r % Y]
        cum = _cumbank(seed, r, nmax)
        for a in range(0, P, chunk):
            b = min(a + chunk, P)
            A = np.zeros((b - a, H), np.float32)
            for i in range(NC):
                if nmax[i]:
                    A += S_MW[i]*cum[i][N[a:b, i]].astype(np.float32)
            days[a:b] += _days_from_avail(netload, A, batt_mw)
        del cum
    return days/float(reps)


def lole_storage(load, n, reps, batt_mw, hours=BATT_HOURS, solar_mw=0.0,
                 solar_cf=None, seed=20260826, chunk=64):
    """LOLE of ONE portfolio with a battery, batching over REPLICATIONS.

    Storage dispatch is sequential in time - the state of charge at hour h depends on
    hour h-1 - so the 8760-step loop cannot be removed. What can be removed is running
    it once per replication: availability for `chunk` replications is stacked into a
    (chunk, 8760) array and the loop walks all of them together, which turns 8760*reps
    array operations into 8760*reps/chunk. Same model, same answer, ~60x faster.
    """
    n = np.asarray(n, np.int64)
    netload = np.asarray(load, float).copy()
    if solar_mw > 0 and solar_cf is not None:
        netload -= solar_mw*np.asarray(solar_cf, float)
    be = batt_mw*hours
    days = 0
    for a in range(0, int(reps), chunk):
        k = min(chunk, int(reps) - a)
        A = np.zeros((k, H), np.float32)
        for r in range(k):
            cum = _cumbank(seed, a + r, n)
            for i in range(NC):
                if n[i]:
                    A[r] += S_MW[i]*cum[i][n[i]].astype(np.float32)
        E = np.full(k, be); short = np.zeros((k, H), bool)
        for h in range(H):
            net = netload[h] - A[:, h]
            pos = net > 0
            dis = np.where(pos, np.minimum(np.minimum(net, batt_mw), ETA*E), 0.0)
            E = np.where(pos, E - dis/ETA,
                         E + ETA*np.minimum(np.minimum(-net, batt_mw), (be - E)/ETA))
            short[:, h] = pos & ((net - dis) > 1e-6)
        days += int(short.reshape(k, 365, 24).any(2).sum())
    return days/float(reps)


def lole(load, fleet, solar_mw, solar_cf, batt_mw, reps, seed=20260826):
    """Single-portfolio form, kept so the v1 call signature still works.
    `fleet` is a list of (unit_MW, count, FOR, MTTR, POH)."""
    n = np.zeros(NC, np.int64)
    for mw, cnt, q, mttr, poh in fleet:
        j = [i for i in range(NC) if abs(CAT[i]['S'] - mw) < 1e-9]
        if not j:
            raise SystemExit(f'unit size {mw} MW is not in the catalogue')
        n[j[0]] += int(cnt)
    return float(lole_batch(load, n[None, :], reps, solar_mw, solar_cf,
                            batt_mw, seed=seed)[0])


# ================================================================ the search
#
# WHY THE SEARCH IS STRUCTURED AND NOT A COST-ORDERED WALK
#
# The obvious algorithm - cost every integer portfolio, sort, simulate down the list
# until one meets LOLE <= 0.1 - does not terminate in useful time here, and the reason
# is worth stating because it is the substantive finding of this section.
#
# Big machines are cheaper on BOTH axes. At $3.89/MMBtu an H-class frame CT costs
# $96.9k/MW-yr fixed against $182.4k for an aeroderivative, and an H-class combined
# cycle burns $27.7/MWh of gas against $41.4. So every cheap design is a handful of
# very large machines - and at a 1.1 GW campus a single 627 MW combined cycle IS 55%
# of peak, so losing one is a blackout. Four of them, 2.19x peak, still score LOLE
# 0.61. The cost-ordered list therefore opens with tens of thousands of designs that
# are all cheap and all uninsurable, and the designs that work sit far down it.
#
# Two structural facts make the problem easy instead:
#
#   1. LOLE DOES NOT DEPEND ON THE ZONE. Zone sets gas price, land value and solar
#      capacity factor - all cost terms. The machines and their failure statistics are
#      the same everywhere. So the feasible set is computed ONCE per campus and priced
#      eight times, not searched eight times.
#
#   2. THE FEASIBLE SET IS UPWARD-CLOSED. Adding a machine never raises LOLE, so
#      feasibility has a staircase boundary, and cost is increasing in every count once
#      the portfolio covers peak (extra small machines land at the top of the merit
#      order, where they add fixed cost and displace no fuel). The optimum is therefore
#      on the MINIMAL frontier of the feasible set, which is found by bisection instead
#      of by scanning.
#
# So: fix the counts of the two big machines, bisect on how much granular capacity has
# to sit on top of them, repeat for every big-machine combination, and keep the minimal
# feasible points. That pool is the frontier; each zone then just picks its cheapest
# member.


def enumerate_site(load, gas, lo_mult=1.0, hi_mult=2.4, proxy_max=2.0,
                   step_frac=0.03, max_rows=400_000):
    """Integer gas portfolios in the capacity band, with an analytic LOLE proxy.

    The proxy is a normal approximation to available capacity against each day's peak,
    summed over days - it is only used to ORDER and prune, never to decide. Everything
    that survives is costed exactly and simulated exactly.
    """
    from scipy.special import ndtr
    pk = float(load.max())
    A = np.array([(1 - u['q'])*(1 - u['poh']/H) for u in CAT])
    dpk = np.asarray(load, float).reshape(365, 24).max(1)
    for frac in (step_frac, step_frac*2, step_frac*4):
        st = [max(1, int(round(frac*pk/s))) for s in S_MW]
        g = [np.arange(0, int(np.ceil(hi_mult*pk/S_MW[i])) + 1, st[i]) for i in range(3)]
        if len(g[0])*len(g[1])*len(g[2]) <= max_rows:
            break
    G0, G1, G2 = [x.ravel() for x in np.meshgrid(*g, indexing='ij')]
    base = G0*S_MW[0] + G1*S_MW[1] + G2*S_MW[2]
    n4max = int(np.ceil(hi_mult*pk/S_MW[4])); n3max = int(np.ceil(hi_mult*pk/S_MW[3]))
    keep = []
    for n4 in range(n4max + 1):
        for n3 in range(n3max + 1):
            fx = n4*S_MW[4] + n3*S_MW[3]
            if fx > hi_mult*pk:
                break
            ins = base + fx
            m = (ins >= lo_mult*pk) & (ins <= hi_mult*pk)
            if not m.any():
                continue
            N = np.zeros((int(m.sum()), NC), np.int32)
            N[:, 0], N[:, 1], N[:, 2] = G0[m], G1[m], G2[m]
            N[:, 3], N[:, 4] = n3, n4
            keep.append(N)
    if not keep:
        return np.zeros((0, NC), np.int32), np.zeros(0)
    N = np.vstack(keep)
    mu = N @ (S_MW*A); sd = np.sqrt(np.maximum(N @ (S_MW**2 * A*(1 - A)), 1e-9))
    ph = np.empty(len(N))
    for a in range(0, len(N), 4000):
        b = min(a + 4000, len(N))
        ph[a:b] = ndtr((dpk[None, :] - mu[a:b, None])/sd[a:b, None]).sum(1)
    k = ph <= proxy_max
    return N[k], ph[k]


# granular top-up directions, as MW shares over (Recip, Aero35, Aero48)
DIRS = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                 [.5, .5, 0], [.5, 0, .5], [0, .5, .5], [1/3, 1/3, 1/3]], float)


def _gran(d, K):
    """Integer counts of the three granular machines realising K MW along direction d."""
    n = np.zeros(3, np.int64)
    for i in range(3):
        if d[i] > 0:
            n[i] = int(round(d[i]*K/S_MW[i]))
    return n


def _blocks(pk, hi_mult, max_blocks=160):
    """(n_FrameCT, n_CC) combinations whose capacity alone fits under the cap.

    At a large campus the two big-machine counts run into the dozens, so the lattice is
    thinned in BOTH dimensions rather than by slicing a flattened list - thinning one
    axis only would silently fix the other, and the frame/combined-cycle split is
    exactly the trade-off being searched. Zero and the maximum are always kept.
    """
    n3max = int(hi_mult*pk/S_MW[3]); n4max = int(hi_mult*pk/S_MW[4])
    sa = sb = 1
    while ((n3max//sa) + 1)*((n4max//sb) + 1) > max_blocks:
        if (n3max//sa) >= (n4max//sb):
            sa += 1
        else:
            sb += 1
    A = sorted({*range(0, n3max + 1, sa), n3max})
    B = sorted({*range(0, n4max + 1, sb), n4max})
    return [(a, b) for b in B for a in A if a*S_MW[3] + b*S_MW[4] <= hi_mult*pk]


def feasible_frontier(load, hi_mult=3.5, screen=40, iters=9, refine=3,
                      seed=17, verbose=False, keep_frac=0.20, keep_min=40,
                      mc_load=None):
    """Minimal portfolios meeting LOLE <= TARGET. Zone-independent by construction.

    For every (FrameCT, CC) combination, bisect on the granular capacity that has to
    sit on top along each of seven mix directions; keep the smallest that passes. Then
    walk the pool downhill - try deleting one machine at a time - until nothing can be
    removed without breaking the target. What comes back is a set of minimal feasible
    designs; costing is somebody else's job.
    """
    load = np.asarray(load, float)
    # The DECISION is taken on the cost year, as HOPE's is; the PROBABILITY may be
    # estimated over a stack of weather years, as the grid Monte Carlo's is.  The
    # capacity band is set by the tightest year the reliability check will see.
    ML = load if mc_load is None else np.atleast_2d(np.asarray(mc_load, float))
    pk = float(np.asarray(ML, float).max())
    tasks = []                                   # (n3, n4, dir index, lo, hi)
    for (n3, n4) in _blocks(pk, hi_mult):
        room = hi_mult*pk - (n3*S_MW[3] + n4*S_MW[4])
        if room < 0:
            continue
        for di in range(len(DIRS)):
            tasks.append([n3, n4, di, 0.0, room])

    def build(t, K):
        n = np.zeros(NC, np.int64)
        n[:3] = _gran(DIRS[t[2]], K)
        n[3], n[4] = t[0], t[1]
        return n

    # drop any (block, direction) that cannot pass even at full granular capacity
    N = np.array([build(t, t[4]) for t in tasks])
    L = lole_batch(ML, N, screen, seed=seed)
    alive = [t for t, l in zip(tasks, L) if l <= TARGET]
    verbose and print(f'   {len(alive)}/{len(tasks)} (block,direction) pairs can pass',
                      flush=True)
    if not alive:
        return np.zeros((0, NC), np.int64)

    for _ in range(iters):                       # synchronous bisection
        mids = [0.5*(t[3] + t[4]) for t in alive]
        N = np.array([build(t, m) for t, m in zip(alive, mids)])
        L = lole_batch(ML, N, screen, seed=seed)
        for t, m, l in zip(alive, mids, L):
            if l <= TARGET:
                t[4] = m
            else:
                t[3] = m

    pool = {tuple(int(x) for x in build(t, t[4])) for t in alive}

    # Prune before refining. Refinement is the expensive step, and a design that is
    # already far off the pace cannot be rescued by deleting one machine - the most a
    # single deletion can save is one machine's fixed cost. So designs are costed at
    # ERCOT-average gas and land and anything more than `keep_frac` above the best is
    # dropped. The margin is wide relative to the spread of zonal gas prices (3.65 to
    # 4.36 $/MMBtu, worth about 4% of total cost), so this cannot change which design
    # wins in any zone.
    if len(pool) > keep_min:
        P = np.array(sorted(pool), np.int64)
        gbar = float(np.mean(list(GAS.values())))
        cbar = merit_fuel_vec(P, var_costs(gbar), ldc(load)) + P @ (S_MW*FX_MW)
        cut = cbar.min()*(1 + keep_frac)
        keep = np.flatnonzero(cbar <= cut)
        if len(keep) < keep_min:
            keep = np.argsort(cbar)[:keep_min]
        pool = {tuple(int(x) for x in P[j]) for j in keep}
        verbose and print(f'   pruned to {len(pool)} designs within '
                          f'{keep_frac:.0%} of best before refining', flush=True)

    # The downhill pass decides whether a design is MINIMAL, so it runs at more
    # replications than the bisection: at 40 replications the standard error on a LOLE
    # of 0.1 is about 0.05, and a noisy accept here would ship an under-built design.
    for _ in range(refine):                      # downhill: delete one machine at a time
        cur = sorted(pool)
        cand, src = [], []
        for n in cur:
            for i in range(NC):
                if n[i] > 0:
                    m = list(n); m[i] -= 1
                    cand.append(m); src.append(n)
        if not cand:
            break
        cand = np.array(cand, np.int64)
        L = lole_batch(ML, cand, max(screen, 150), seed=seed + 101)
        added = False
        for c, s, l in zip(cand, src, L):
            if l <= TARGET:
                pool.add(tuple(int(x) for x in c)); pool.discard(s); added = True
        if not added:
            break
    out = np.array(sorted(pool), np.int64)
    verbose and print(f'   frontier: {len(out)} minimal feasible designs', flush=True)
    return out


def design_site(load, zone, solar_cf=None, frontier=None, confirm=400,
                solar_mw=0.0, batt_mw=0.0, verbose=False, seed_confirm=23,
                hi_mult=3.5, screen=40, mc_load=None, **kw):
    """Least-cost portfolio at this site in this zone that meets LOLE <= 0.1.

    Pass `frontier` (from `feasible_frontier`) to price a campus in a second zone
    without re-running the reliability search - the feasible set does not depend on
    the zone. The chosen design is re-simulated at `confirm` replications on an
    independent random stream before it is returned, so a design that only cleared the
    screen by luck is caught and the next-cheapest is taken instead.
    """
    load = np.asarray(load, float)
    # cost and merit order on the cost year; the confirmation simulated over whatever
    # weather years the caller hands in (one, by default -- see lole_batch)
    ML = load if mc_load is None else np.atleast_2d(np.asarray(mc_load, float))
    pk = float(load.max()); energy = float(load.sum())
    gas = GAS[zone]
    if frontier is None:
        frontier = feasible_frontier(load, hi_mult=hi_mult, screen=screen,
                                     verbose=verbose, mc_load=mc_load)
    if not len(frontier):
        return None
    VR = var_costs(gas)
    LD = ldc(np.maximum(load - solar_mw*solar_cf, 0.0) if solar_mw else load)
    cost = merit_fuel_vec(frontier, VR, LD) + \
        annual_cost_vec(frontier, zone, pk, solar_mw, batt_mw)[0]
    o = np.argsort(cost)

    # the cheapest portfolio that merely covers peak, ignoring reliability entirely -
    # the benchmark the reliability premium is measured against
    Nb, _ = enumerate_site(load, gas, lo_mult=1.0, hi_mult=1.35, proxy_max=np.inf)
    base = float((merit_fuel_vec(Nb, VR, LD)
                  + annual_cost_vec(Nb, zone, pk)[0]).min()) if len(Nb) else np.nan

    for j in o:
        n = frontier[j]
        Lc = float(lole_batch(ML, n[None, :], confirm, solar_mw, solar_cf,
                              batt_mw, seed=seed_confirm)[0])
        if Lc <= TARGET:
            inst = float(n @ S_MW)
            fixed, land, acres = annual_cost(load, n, solar_mw, batt_mw, gas, zone)
            c = float(cost[j])
            return dict(zone=zone, n=[int(x) for x in n], installed_MW=inst,
                        ratio=inst/pk, solar_MW=solar_mw, batt_MW=batt_mw,
                        LOLE=Lc, cost=c, n_frontier=int(len(frontier)),
                        fixed_cost=fixed, land_cost=land, acres=acres, gas=gas,
                        fuel_cost=c - fixed - land, lcoe=c/energy, energy_MWh=energy,
                        peak_MW=pk, unit_count=int(n.sum()),
                        cheapest_ignoring_reliability=base,
                        premium=c/base - 1 if base == base else None,
                        mix={CAT[i]['k']: int(n[i]) for i in range(NC) if n[i]})
    return None


# ================================================================ siting + driver
def zonal_solar_cf(case_dir, zone):
    """The SAME candidate solar profile HOPE optimised against, from the case files."""
    import pandas as pd
    g = pd.read_csv(f'{case_dir}/gendata.csv')
    c = pd.read_csv(f'{case_dir}/gendata_candidate.csv')
    k = c.index[(c.Zone == zone) & (c.Type == 'SolarPV')]
    if not len(k):
        return None
    av = pd.read_csv(f'{case_dir}/gen_availability_timeseries.csv',
                     usecols=[f'G{len(g)+int(k[0])+1}'])
    return av.values[:H, 0]


def solar_value(load, zone, solar_cf, n, gas=None):
    """Marginal value of solar at a FIXED firm portfolio.

    For a flat data-centre load solar has no capacity value - it cannot cover the
    night, so the firm fleet is unchanged - which means the only question is whether
    its fixed cost beats the fuel it displaces at the top of the merit order. Both
    sides are computed exactly, so this is a decision, not an assumption.
    """
    gas = GAS[zone] if gas is None else gas
    VR = var_costs(gas); n = np.asarray(n)
    base = merit_fuel_vec(n[None, :], VR, ldc(load))[0]
    grid = np.linspace(0, float(load.min())/max(solar_cf.max(), 1e-9), 9)[1:]
    out = []
    for s in grid:
        net = np.maximum(load - s*solar_cf, 0.0)
        saved = base - merit_fuel_vec(n[None, :], VR, ldc(net))[0]
        extra = s*SOLAR_FIXED + s*ACRES_PER_MW['solar']*LAND[zone]*LAND_CRF
        out.append(dict(solar_MW=float(s), fuel_saved=float(saved),
                        solar_cost=float(extra), net=float(saved - extra)))
    return out


def run(sites, case_dir, zones=None, screen=40, confirm=400, verbose=True,
        home_only=False, hi_mult=3.5):
    """For every campus, design it in its home zone and in every alternative zone.

    The reliability search runs ONCE per campus - the feasible set is a property of the
    machines and the load shape, not of where the campus sits - and the resulting
    frontier is then priced in each zone. Siting is a pure cost comparison over a
    common feasible set, which is also what makes the zone rankings meaningful.
    """
    zones = zones or list(GAS)
    CF = {z: zonal_solar_cf(case_dir, z) for z in zones}
    out = []
    for s in sites:
        load = np.asarray(s['load'], float)
        zl = [s['zone']] if home_only else zones
        fr = feasible_frontier(load, hi_mult=hi_mult, screen=screen, verbose=verbose)
        rows = []
        for z in zl:
            r = design_site(load, z, CF[z], frontier=fr, confirm=confirm)
            if r is None:
                continue
            r['site'] = s['id']; r['home_zone'] = s['zone']
            rows.append(r)
        if not rows:
            verbose and print(f"{s['id']:<5} no feasible design found", flush=True)
            continue
        home = next((r for r in rows if r['zone'] == s['zone']), None)
        best = min(rows, key=lambda r: r['cost'])
        for r in rows:
            r['is_home'] = (r['zone'] == s['zone'])
            r['vs_home_pct'] = (r['cost']/home['cost'] - 1)*100 if home else None
        out.append(dict(site=s['id'], home=home, best=best, all_zones=rows))
        if verbose:
            mix = ' '.join(f'{k}x{v}' for k, v in best['mix'].items())
            print(f"{s['id']:<5} peak {load.max():>6,.0f} MW  home {s['zone']:<13}"
                  f"${home['lcoe']:>6.2f}/MWh   best {best['zone']:<13}"
                  f"${best['lcoe']:>6.2f}/MWh ({best['vs_home_pct']:+5.1f}%)  {mix}",
                  flush=True)
    return out


# ================================================================ self-test
def selftest(seed=1):
    """Checks the two reformulations against their reference implementations."""
    rng = np.random.default_rng(seed)
    load = 800 + 300*rng.random(H)
    ok = True

    # 1. vectorised merit-order fuel == hour-by-hour loop
    VR = var_costs(GAS['West']); LD = ldc(load); order = np.argsort(VR)
    ls = np.sort(load)[::-1]
    N = rng.integers(0, [30, 20, 15, 3, 3], size=(25, NC))
    fv = merit_fuel_vec(N, VR, LD)
    fr = np.array([merit_fuel(ls, [n[i]*S_MW[i] for i in order], VR[order]) for n in N])
    e1 = float(np.max(np.abs(fv - fr)/np.maximum(fr, 1)))
    print(f'  merit-order fuel  vectorised vs loop   max rel err {e1:.3e}'
          f'   {"OK" if e1 < 1e-12 else "FAIL"}')
    ok &= e1 < 1e-12

    # 2. the bank reproduces the right availability statistics
    for i in (1, 4):
        u = CAT[i]
        B = _bank((7, 0, i), u['q'], u['mttr'], u['poh'], 400, typ=i)
        emp = B.mean(); th = (1 - u['q'])*(1 - u['poh']/H)
        e2 = abs(emp - th)
        print(f"  {u['k']:<8} availability  bank {emp:.4f}  theory {th:.4f}"
              f"   {'OK' if e2 < 0.004 else 'FAIL'}")
        ok &= e2 < 0.004

    # 3. batched LOLE vs the same design simulated alone
    n = np.array([0, 0, 0, 0, 3])
    Lb = lole_batch(load, np.vstack([n, n + [0, 5, 0, 0, 0]]), 120, seed=5)
    La = lole_batch(load, n[None, :], 120, seed=5)[0]
    e3 = abs(Lb[0] - La)
    print(f'  batched vs single LOLE   {Lb[0]:.4f} vs {La:.4f}'
          f'   {"OK" if e3 < 1e-9 else "FAIL"}')
    ok &= e3 < 1e-9
    print('  ' + ('all checks pass' if ok else 'SOMETHING FAILED'))
    return ok


if __name__ == '__main__':
    selftest()
