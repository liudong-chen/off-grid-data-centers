"""offgrid_mip -- the off-grid campus design problem as a mixed-integer program.

THE PROBLEM
-----------
One data-center campus, islanded. Choose

    n_j  in Z+      j = 1..5      number of machines of each catalogue type
                                   (recip 20 MW, aero 35, aero 48, frame CT 419, CC 627)
    s    >= 0        MW of solar
    w    >= 0        MW of wind
    b    >= 0        MW of 4-hour battery      (s, w, b continuous, or on a step grid)

to minimise the annual cost

    sum_j n_j S_j phi_j  +  s phi_S + w phi_W + b phi_B  +  land  +  fuel(n, s, w, b)

subject to the reliability standard the on-grid case is planned to,

    LOLE(n, s, w, b)  <=  0.1 loss-of-load days per year,

scored by the same sequential Monte Carlo (two-state Markov outages, packed planned
maintenance, greedy storage dispatch) that `offgrid_model.py` uses. Everything in the
catalogue, the cost tables and the Monte Carlo is imported from that file unchanged, so
a design found here is scored on exactly the same machines and the same random streams
as a design found by the frontier search.

WHY A MIP NEEDS A HEURISTIC FOR THE LOLE CONSTRAINT
---------------------------------------------------
Every term in the objective is linear once the hourly dispatch is made explicit: fuel is
an LP over 8,760 hours with each machine type's output capped by the units bought, solar
and wind by their profiles times the MW built, and the battery by a state-of-charge chain.
The one thing that is not linear -- not even convex -- is LOLE. It is the expectation of
the indicator "some hour of this day was short" over a Markov process with 24-48 h repair
times, and its value depends on how the capacity is divided into machines, not only on how
much there is (a 1 GW campus on three machines and on thirty have the same MW margin and
wildly different LOLE). There is no exact linear reformulation.

So the LOLE constraint is enforced by a linear surrogate inside the MIP and corrected by
the Monte Carlo outside it:

  1. INSIDE the MIP, reliability is written as an N-k rule: the firm capacity left after
     losing ANY k of the machines the design owns must still cover the peak,
         sum_j n_j S_j a_j - (the k largest owned units) + credited VRE/battery >= P,
     with a_j = (1 - FOR_j)(1 - POH_j/8760) the expected availability of a machine,
     one linear constraint per loss pattern (l_1..l_5, sum l = k) switched by nested
     binaries yc[j,c] = [n_j >= c], and the VRE + battery capacity credit measured on the
     load's 50 tightest hours and capped at 15% of peak. This surrogate SEES unit size:
     six 627 MW machines and thirty 35 MW machines with the same MW margin face very
     different N-2 requirements. k starts at 2 (one unit in overhaul, one failed).
  2. OUTSIDE the MIP: solve, simulate the returned design with the Monte Carlo, and if it
     misses the target add one EXACT no-good cut -- LOLE is monotone in every unit, so
     every machine portfolio dominated by a rejected one is rejected too, and the next
     solve must own strictly more of at least one type -- and, if LOLE was more than
     three times the target, raise k by one. Two to five solves close the loop; the first
     design the Monte Carlo passes (with a 10% screening buffer) is the cheapest design
     satisfying every cut, i.e. a sequential search in cost order over MIP optima.
  3. A LOCAL POLISH then tries to remove one machine at a time (or one step of s, w, b)
     and keeps any change that is cheaper and still passes at 150 replications; the final
     design is confirmed at 400 replications on an independent random stream, exactly
     as `design_site` does, adding the cheapest machine of an owned type until it passes.

This is the off-grid twin of ReliabilityPRM.jl: a cost-minimising optimiser under a
linear reliability constraint, tightened by the Monte Carlo until the built portfolio
actually meets LOLE = 0.1 -- except that here the constraint is per campus and counts
machines, which is exactly what the on-grid margin never needed to do.

WHAT WAS TRIED AND DROPPED (so nobody tries it again)
  * bisection on a MW reserve margin: cheap to run, but the MIP satisfies any MW floor
    with the cheapest big machines, so at a 268 MW campus it walked to three frame CTs and
    a combined cycle (7x peak) while twelve aeroderivatives would have done;
  * lifting a firm-MW floor by the Monte Carlo's shortfall quantile: same failure mode --
    the shortfall of a lumpy design is the whole load, so the floor forbids the small-unit
    answer; a MW floor is not a valid cut for a granularity constraint;
  * no-good cuts that let solar, wind or the battery escape by one 10 MW step: the MIP
    shuffles VRE by 10 MW a time and never buys a machine.

USAGE
-----
    import offgrid_mip as om
    r = om.design(load, zone='FarWest', solar_cf=cf_s, wind_cf=cf_w, verbose=True)
    # r['n'], r['solar_MW'], r['wind_MW'], r['batt_MW'], r['LOLE'], r['cost'], r['lcoe'] ...

    python3 offgrid_mip.py            # self-test on a synthetic load
    python3 offgrid_mip.py S01        # one real campus (paths at the foot of the file)

Requires scipy >= 1.9 (scipy.optimize.milp, the HiGHS MILP solver) and numpy.
"""
import os, sys, time, json
import importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('og', os.path.join(HERE, 'offgrid_model.py'))
og = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(og)

H, NC, TARGET = og.H, og.NC, og.TARGET
S_MW, FX_MW, CAT = og.S_MW, og.FX_MW, og.CAT

# ------------------------------------------------------------- extra parameters
# Costs priced to the on-grid candidate file so both cases see the same technology.
WIND_FIXED = 157009.0            # $/MW-yr, gendata_candidate WindOn (TX8z vre case)
SOLAR_FIXED = og.SOLAR_FIXED     # 117,995 $/MW-yr
BATT_FIXED = og.BATT_FIXED       # 161,512 $/MW-yr, 4-hour
BATT_HOURS = og.BATT_HOURS
ETA = og.ETA
ACRES = dict(og.ACRES_PER_MW, wind=1.0)    # direct-impact area only, as 68_offgrid.py
AVAIL = np.array([(1 - u['q'])*(1 - u['poh']/H) for u in CAT])   # expected availability

# The published v4 results (4_results/main_figures/figures_1_to_3/02_code/68_offgrid.py) anchor the
# catalogue capex so that the frame CT and the CC carry the on-grid candidate file's fixed
# cost ($103,382 and $118,650 /MW-yr); every 'CT-class' machine (recip, aeros, frame CT)
# takes the CT factor. Off-grid and on-grid then see the same technology prices.
ONGRID_FIXED = {'CC': 118650.0, 'CT': 103382.0}
_CLASS = {'Recip': 'CT', 'Aero35': 'CT', 'Aero48': 'CT', 'FrameCT': 'CT', 'CC': 'CC'}
ANCHORED = False


def anchor_capex(ongrid=ONGRID_FIXED):
    """Scale catalogue capex to the on-grid candidate costs, in place (idempotent)."""
    global ANCHORED
    if ANCHORED:
        return
    by = {u['k']: u for u in CAT}
    f = {k: ((ongrid[k]/1000.0 - u['fom'])/og.CRF)/u['capex']
         for k, u in (('CC', by['CC']), ('CT', by['FrameCT']))}
    for u in CAT:
        u['capex'] *= f[_CLASS[u['k']]]
        u['fixed_$MWyr'] = (u['capex']*og.CRF + u['fom'])*1000.0
    FX_MW[:] = [u['fixed_$MWyr'] for u in CAT]          # in place: og.FX_MW is the same array
    assert abs(by['CC']['fixed_$MWyr'] - ongrid['CC']) < 1.0
    assert abs(by['FrameCT']['fixed_$MWyr'] - ongrid['FrameCT' if 'FrameCT' in ongrid else 'CT']) < 1.0
    ANCHORED = True
    return f


# ======================================================================= inputs
class Site:
    """Everything the optimiser needs about one campus in one zone."""

    def __init__(self, load, zone, solar_cf=None, wind_cf=None, gas=None, land=None,
                 mc_load=None):
        self.load = np.asarray(load, float)[:H]
        self.zone = zone
        self.gas = og.GAS[zone] if gas is None else float(gas)
        self.land = og.LAND[zone] if land is None else float(land)
        self.cf_s = None if solar_cf is None else np.asarray(solar_cf, float)[:H]
        self.cf_w = None if wind_cf is None else np.asarray(wind_cf, float)[:H]
        self.mc_load = self.load if mc_load is None else np.atleast_2d(np.asarray(mc_load, float))
        self.peak = float(self.load.max())
        self.energy = float(self.load.sum())
        self.VR = og.var_costs(self.gas)                 # $/MWh by machine, sets merit order

    def net(self, s, w):
        """Load net of VRE, one 8,760-hour year (may be negative: the battery charges)."""
        x = self.load.copy()
        if s > 0 and self.cf_s is not None:
            x -= s*self.cf_s
        if w > 0 and self.cf_w is not None:
            x -= w*self.cf_w
        return x

    def capacity_credits(self, s=0.0, w=0.0, top=50, n=None):
        """Capacity credit of solar, wind and a 4-hour battery on THIS load's tightest hours.

        Credit = mean capacity factor over the `top` highest net-load hours -- the same
        top-hours rule the on-grid case applies to its VRE candidates. The battery credit
        is the share of the tightest hours it can bridge at full power: a 4-hour battery
        against a data-center load whose daily peak lasts longer than four hours earns
        less than 1.0 MW of credit per MW.
        """
        net = self.net(s, w)
        idx = np.argsort(net)[-top:]
        cs = float(self.cf_s[idx].mean()) if self.cf_s is not None else 0.0
        cw = float(self.cf_w[idx].mean()) if self.cf_w is not None else 0.0
        # On an island the shortages a battery must bridge are machine outages, and they
        # last as long as the repair (24-48 h), not as long as a load peak. A 4-hour
        # battery therefore covers BATT_HOURS of an outage of mean length MTTR: its credit
        # is BATT_HOURS / MTTR, capacity-weighted over the design's machines (or the
        # catalogue when no design is given). About 0.1 -- a battery buys granularity,
        # not firm capacity, which is what the Monte Carlo has always said.
        if n is None or not np.any(np.asarray(n) > 0):
            mttr = float(np.mean([u['mttr'] for u in CAT]))
        else:
            nn = np.asarray(n, float)
            mttr = float((nn*S_MW*np.array([u['mttr'] for u in CAT])).sum()/(nn*S_MW).sum())
        cb = float(min(1.0, BATT_HOURS/max(mttr, BATT_HOURS)))
        return cs, cw, cb


# ==================================================================== the MIP
def build_and_solve(site, margin, cc, step=1, vre_step=None, allow=(True, True, True),
                    nmax=None, time_limit=120.0, derate_dispatch=False, mip_gap=1e-3,
                    cuts=(), nogood=(), credit_cap=0.15, k_out=2, vre_cut_frac=0.0,
                    sens=(), cost_cap=None, center=None, radius=4):
    """Solve the design MIP at reserve margin `margin`. Returns a dict or None.

    Decision variables (in this order):
        n[0..4]        integer machine counts
        s, w, b        MW of solar, wind, battery (continuous, or integer multiples of
                       `vre_step` MW when given)
        yc[j, c]       binaries: yc = 1 iff n_j >= c, c = 1..k  (for the N-k rule)
        g[j, t]        dispatch of type j in hour t          (T*5)
        ch[t], dis[t]  battery charge / discharge             (2T)
        soc[t]         state of charge at the end of hour t   (T)

    `step` sub-samples the year (every step-th hour, costs scaled back up) when a full
    8,760-hour LP is more than the machine wants; step=1 is exact.
    `cuts` is a list of firm-MW floors: each adds  sum_j S_j a_j n_j >= value.
    `sens` is a list of Monte-Carlo sensitivity cuts (dicts from `sensitivity_cut`):
    each adds  sum_j g_j (n_j - n*_j) + g_s (s - s*) + g_w (w - w*) + g_b (b - b*) >= rhs,
    a first-order cut on log LOLE around a rejected design (Bloom 1983 / PSR-style
    Benders reliability cut, gradients by common-random-number finite differences).
    `nogood` is a list of rejected designs (n*, s*, w*, b*); for each, the MIP must own
    strictly more of at least one machine type:  OR_j ( n_j >= n*_j + 1 ), linearised with
    one binary per type. Because LOLE is monotone in every unit, every machine portfolio
    dominated by a rejected one is rejected too, so on the machine counts this cut is
    exact. `credit_cap` caps the VRE + battery capacity credit at that share of peak.
    """
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import coo_matrix

    L = site.load[::step]; T = len(L)
    cf_s = site.cf_s[::step] if (site.cf_s is not None and allow[0]) else np.zeros(T)
    cf_w = site.cf_w[::step] if (site.cf_w is not None and allow[1]) else np.zeros(T)
    scale = H/T                                   # hours represented by each sample
    cs, cw, cb = cc
    P = site.peak
    if nmax is None:                              # generous box: 3.5x peak of any one type
        nmax = np.ceil(3.5*P/S_MW).astype(int)
    cap_dispatch = S_MW*(AVAIL if derate_dispatch else 1.0)

    # ---- variable layout
    iN = np.arange(NC); iS, iW, iB = NC, NC + 1, NC + 2
    K = int(k_out)
    iYC = NC + 3 + np.arange(NC*K).reshape(NC, K)   # yc[j, c-1] = 1 iff n_j >= c
    off = NC + 3 + NC*K
    iG = off + np.arange(NC*T).reshape(NC, T); off += NC*T
    iCH = off + np.arange(T); off += T
    iDIS = off + np.arange(T); off += T
    iSOC = off + np.arange(T); off += T
    NZ = NC + (3 if vre_cut_frac > 0 else 0)
    iZ = (off + np.arange(len(nogood)*NZ).reshape(len(nogood), NZ)) if len(nogood) else None
    off += len(nogood)*NZ
    iV = off; off += 1                            # credited VRE + battery MW, capped
    n_sens_terms = sum(int((np.asarray(q['gn']) > 0).sum()) + sum(1 for key, ok in (('gs', allow[0]), ('gw', allow[1]), ('gb', allow[2])) if q[key] > 0 and ok) for q in sens)
    iSA = off + np.arange(2*n_sens_terms); off += 2*n_sens_terms   # (d, u) pairs for one-sided cuts
    sens_aux = list(iSA)
    NV = off

    # ---- objective
    c = np.zeros(NV)
    c[iN] = S_MW*FX_MW + S_MW*ACRES['gas']*site.land*og.LAND_CRF
    c[iS] = SOLAR_FIXED + ACRES['solar']*site.land*og.LAND_CRF
    c[iW] = WIND_FIXED + ACRES['wind']*site.land*og.LAND_CRF
    c[iB] = BATT_FIXED + ACRES['storage']*site.land*og.LAND_CRF
    for j in range(NC):
        c[iG[j]] = site.VR[j]*scale
    const = P*ACRES['dc']*site.land*og.LAND_CRF   # the campus's own land, fixed

    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0

    def add(coefs, l, h):
        nonlocal r
        for col, v in coefs:
            rows.append(r); cols.append(col); vals.append(v)
        lo.append(l); hi.append(h); r += 1

    # (1) hourly balance: sum_j g + s cf_s + w cf_w + dis - ch >= L   (surplus curtailed)
    for t in range(T):
        coefs = [(iG[j, t], 1.0) for j in range(NC)]
        if cf_s[t] > 0: coefs.append((iS, cf_s[t]))
        if cf_w[t] > 0: coefs.append((iW, cf_w[t]))
        coefs += [(iDIS[t], 1.0), (iCH[t], -1.0)]
        add(coefs, L[t], np.inf)
    # (2) dispatch capped by units bought: g_jt - cap_j n_j <= 0
    for j in range(NC):
        for t in range(T):
            add([(iG[j, t], 1.0), (iN[j], -cap_dispatch[j])], -np.inf, 0.0)
    # (3) battery: power limits and the state-of-charge chain (cyclic year)
    for t in range(T):
        add([(iCH[t], 1.0), (iB, -1.0)], -np.inf, 0.0)
        add([(iDIS[t], 1.0), (iB, -1.0)], -np.inf, 0.0)
        add([(iSOC[t], 1.0), (iB, -BATT_HOURS)], -np.inf, 0.0)
        prev = iSOC[t - 1] if t > 0 else iSOC[T - 1]
        # soc_t - soc_{t-1} - eta*scale*ch_t + scale/eta*dis_t = 0
        add([(iSOC[t], 1.0), (prev, -1.0), (iCH[t], -ETA*scale), (iDIS[t], scale/ETA)], 0.0, 0.0)
    # (4) capacity credit of solar, wind and battery, capped at credit_cap * P: a flat
    #     data-center load cannot be carried through the night on VRE, so the credit is a
    #     small correction to a firm machine fleet, never a substitute for it
    add([(iV, 1.0), (iS, -cs), (iW, -cw), (iB, -cb)], -np.inf, 0.0)
    add([(iV, 1.0)], -np.inf, credit_cap*P)
    #     firm-capacity margin
    coefs = [(iN[j], S_MW[j]*AVAIL[j]) for j in range(NC)] + [(iV, 1.0)]
    add(coefs, (1 + margin)*P, np.inf)
    # (5) the N-k rule: the firm capacity left after losing ANY k of the machines the
    #     design owns must still cover the peak. With the types in the catalogue this is
    #     one constraint per loss pattern l (l_j units of type j lost, sum l = k), relaxed
    #     by big-M whenever the design owns fewer than l_j units of type j:
    #        firm - sum_j l_j S_j a_j  >=  P - M * sum_j (1 - yc[j, l_j])
    #     yc[j, c] = 1 iff n_j >= c  (nested binaries). k is raised by the outer loop
    #     whenever the Monte Carlo rejects a design, so the surrogate tightens exactly
    #     where the machine-level physics says it must: few big machines need a low k,
    #     many small ones a higher one.
    Mbig = 6.0*P + K*S_MW.max()
    for j in range(NC):
        for lvl in range(1, K + 1):
            add([(iN[j], 1.0), (iYC[j, lvl - 1], -float(lvl))], 0.0, np.inf)            # yc=1 -> n>=lvl
            add([(iN[j], 1.0), (iYC[j, lvl - 1], -float(nmax[j]))], -np.inf, lvl - 1.0)  # n>=lvl -> yc=1
    firm_coefs = [(iN[k], S_MW[k]*AVAIL[k]) for k in range(NC)] + [(iV, 1.0)]
    add([(iN[j], 1.0) for j in range(NC)], K + 1.0, np.inf)   # own more than k machines
    from itertools import product
    for pat in product(range(K + 1), repeat=NC):
        if sum(pat) != K:
            continue
        coefs = list(firm_coefs); rhs = P + sum(pat[j]*S_MW[j]*AVAIL[j] for j in range(NC))
        for j in range(NC):
            if pat[j] > 0:
                coefs.append((iYC[j, pat[j] - 1], -Mbig)); rhs -= Mbig
        add(coefs, rhs, np.inf)
    # (6) firm-MW floors, if the caller supplies any
    for v in cuts:
        add([(iN[k], S_MW[k]*AVAIL[k]) for k in range(NC)], float(v), np.inf)
    # (6b) Monte-Carlo sensitivity cuts on log LOLE around rejected designs, ONE-SIDED:
    #        sum_k g_k d_k >= rhs,   0 <= d_k <= max(x_k - x*_k, 0)
    #     i.e. the ADDITIONS alone must buy the log-reduction; removals are not charged.
    #     A two-sided hyperplane charges "3 frame CTs removed" at the CT slope and so cuts
    #     off every design of a different composition (23 aeros + CC was excluded by a cut
    #     measured at 3 CT + CC). One-sided, the cut is weaker but it does not fence the
    #     search into the machine family it happened to be measured on.
    #     Linearised with d_k <= x_k - x*_k + x*_k (1 - u_k),  d_k <= X_k u_k,  u_k binary.
    for q in sens:
        terms = [(iN[j], float(q['gn'][j]), float(q['n0'][j]), float(nmax[j])) for j in range(NC) if q['gn'][j] > 0]
        for ix, key, g0, ok in ((iS, 'gs', 's0', allow[0]), (iW, 'gw', 'w0', allow[1]), (iB, 'gb', 'b0', allow[2])):
            if q[key] > 0 and ok:
                terms.append((ix, float(q[key]), float(q[g0]), 6.0*P))
        if not terms:
            continue
        coefs = []
        for ix, g, x0, X in terms:
            iD, iU = sens_aux.pop(0), sens_aux.pop(0)
            add([(iD, 1.0), (ix, -1.0), (iU, x0)], -np.inf, 0.0)      # d <= x - x0 + x0 (1-u)
            add([(iD, 1.0), (iU, -X)], -np.inf, 0.0)                  # d <= X u
            coefs.append((iD, g))
        add(coefs, float(q['rhs']), np.inf)
    # (7) exact no-good cuts on Monte-Carlo-rejected designs (LOLE is monotone in every
    #     variable, so everything dominated by a rejected design is rejected with it):
    #     the design must own strictly more of at least one thing,
    #        x_k >= (x*_k + step_k) z_k  for every k,   sum_k z_k >= 1
    #     The cut is written on the machine counts only: reliability is bought with
    #     machines, while solar, wind and the battery enter on their energy economics and
    #     the capped credit above. (A cut that also let a 10 MW step of VRE escape it
    #     just made the MIP shuffle VRE by 10 MW a time.)
    #     With vre_cut_frac > 0 the cut may also be escaped by adding at least that share
    #     of peak in solar, wind or battery -- the setting that lets the model decide
    #     whether renewables plus storage can stand in for machines.
    for q, (nq, sq, wq, bq) in enumerate(nogood):
        for k in range(NC):
            add([(iN[k], 1.0), (iZ[q, k], -(float(nq[k]) + 1.0))], 0.0, np.inf)
        if vre_cut_frac > 0:
            for k, (ix, xq, ok) in enumerate(((iS, sq, allow[0]), (iW, wq, allow[1]), (iB, bq, allow[2]))):
                if ok:
                    add([(ix, 1.0), (iZ[q, NC + k], -(float(xq) + vre_cut_frac*P))], 0.0, np.inf)
                else:
                    add([(iZ[q, NC + k], 1.0)], 0.0, 0.0)
        add([(iZ[q, k], 1.0) for k in range(NZ)], 1.0, np.inf)

    # (8) improvement stage: the objective must beat the incumbent (c.x <= cost_cap)
    if cost_cap is not None:
        add([(i, float(c[i])) for i in range(NV) if c[i] != 0.0], -np.inf, float(cost_cap))
    # (9) local branching (Fischetti & Lodi): stay within `radius` machine changes of the
    #     incumbent, sum_j |n_j - n_c_j| <= radius, so the improvement search walks the
    #     neighbourhood of a known-good design instead of the whole lattice
    if center is not None:
        iD_ = list(range(NV, NV + NC)); NV2 = NV + NC
        for j in range(NC):
            add([(iD_[j], 1.0), (iN[j], -1.0)], -float(center[j]), np.inf)   # d >= n - nc
            add([(iD_[j], 1.0), (iN[j], 1.0)], float(center[j]), np.inf)     # d >= nc - n
        add([(iD_[j], 1.0) for j in range(NC)], -np.inf, float(radius))
        c = np.concatenate([c, np.zeros(NC)]); NV = NV2
    A = coo_matrix((vals, (rows, cols)), shape=(r, NV)).tocsr()
    cons = LinearConstraint(A, np.array(lo), np.array(hi))

    lb = np.zeros(NV); ub = np.full(NV, np.inf)
    ub[iN] = nmax
    if not allow[0]: ub[iS] = 0.0
    if not allow[1]: ub[iW] = 0.0
    if not allow[2]: ub[iB] = 0.0
    integ = np.zeros(NV); integ[iN] = 1
    ub[iYC.ravel()] = 1.0; integ[iYC.ravel()] = 1
    if iZ is not None:
        ub[iZ.ravel()] = 1.0; integ[iZ.ravel()] = 1
    for i in range(1, len(iSA), 2):               # the u's are binary; the d's continuous >= 0
        ub[iSA[i]] = 1.0; integ[iSA[i]] = 1
    if vre_step:                                  # s, w, b on a step grid: scale the columns
        for i in (iS, iW, iB):
            A_col = A[:, i].toarray().ravel()
            # substitute x = vre_step * k, k integer: multiply the column and the cost
            A = A.tolil(); A[:, i] = A_col*vre_step; A = A.tocsr()
            c[i] *= vre_step; integ[i] = 1
            ub[i] = np.floor(ub[i]/vre_step) if np.isfinite(ub[i]) else np.inf
        cons = LinearConstraint(A, np.array(lo), np.array(hi))

    res = milp(c, constraints=cons, integrality=integ, bounds=Bounds(lb, ub),
               options=dict(time_limit=time_limit, mip_rel_gap=mip_gap, disp=False))
    if res.x is None:
        return None
    x = res.x
    n = np.rint(x[iN]).astype(int)
    k = vre_step if vre_step else 1.0
    s, w, b = (0.0 if float(x[i])*k < 0.5 else float(x[i])*k for i in (iS, iW, iB))  # snap solver dust
    fuel = float(sum(site.VR[j]*x[iG[j]].sum() for j in range(NC)))*scale
    return dict(n=n, s=s, w=w, b=b, obj=float(res.fun) + const, fuel=fuel,
                status=res.message, margin=margin, T=T)


def zonal_wind_cf(case_dir, zone):
    """The candidate onshore-wind profile HOPE optimised against, from the case files."""
    import pandas as pd
    g = pd.read_csv(f'{case_dir}/gendata.csv')
    c = pd.read_csv(f'{case_dir}/gendata_candidate.csv')
    k = c.index[(c.Zone == zone) & (c.Type == 'WindOn')]
    if not len(k):
        return None
    av = pd.read_csv(f'{case_dir}/gen_availability_timeseries.csv',
                     usecols=[f'G{len(g)+int(k[0])+1}'])
    return av.values[:H, 0]


# ================================================================ reliability
def lole_of(site, n, s, w, b, reps, seed=20260826):
    """LOLE (days/yr) of one design, on the site's Monte Carlo weather years."""
    ML = site.mc_load
    if ML.ndim == 1:
        ML = ML[None, :]
    NET = ML.copy()
    if s > 0 and site.cf_s is not None:
        NET -= s*site.cf_s[None, :]
    if w > 0 and site.cf_w is not None:
        NET -= w*site.cf_w[None, :]
    n = np.asarray(n, np.int64)
    if b > 0:
        # battery dispatch is sequential; lole_storage batches over replications
        return float(og.lole_storage(NET[0] if len(NET) == 1 else NET.mean(0), n, reps, b,
                                     seed=seed))
    return float(og.lole_batch(NET if len(NET) > 1 else NET[0], n[None, :], reps, seed=seed)[0])


def cost_of(site, n, s, w, b, fuel=None):
    """Annual cost breakdown, $/yr, by exact merit order (no dispatch model needed
    when there is no battery; with a battery the MIP's own dispatch cost is used)."""
    n = np.asarray(n, float)
    fixed = float(n @ (S_MW*FX_MW)) + s*SOLAR_FIXED + w*WIND_FIXED + b*BATT_FIXED
    acres = (float(n @ S_MW)*ACRES['gas'] + s*ACRES['solar'] + w*ACRES['wind']
             + b*ACRES['storage'] + site.peak*ACRES['dc'])
    land = acres*site.land*og.LAND_CRF
    if fuel is None:
        net = np.maximum(site.net(s, w), 0.0)
        fuel = float(og.merit_fuel_vec(n.astype(np.int64)[None, :], site.VR, og.ldc(net))[0])
    return dict(fixed=fixed, land=land, acres=acres, fuel=fuel, total=fixed + land + fuel)


# ============================================================ sensitivity cut
def sensitivity_cut(site, n, s, w, b, L0, reps, seed, allow=(True, True, True),
                    vre_delta=None, mw_delta=None, floor=None, target=None, max_step=2.5):
    """First-order cut on log LOLE around a rejected design (n, s, w, b) with LOLE L0.

    Re-scores the design under the SAME random stream with one more unit of each type
    and with +vre_delta MW of solar, wind and battery (common random numbers, so the
    differences are far less noisy than the levels). Gradients are on log LOLE, which is
    close to linear in firm capacity; negative estimates (noise) are clipped to zero.
    LOLE levels are floored at `floor` (default half an event over the replications) so
    a zero-shortfall sample still yields a finite, conservative slope.
    The cut  sum_j g_j (n_j - n_j*) + g_s (s - s*) + g_w (w - w*) + g_b (b - b*) >= log(L0/TARGET)
    asks the next design to buy at least the log-reduction the target needs, priced at
    the simulator's own marginal values -- one MC-derived Benders cut per rejected design.
    """
    P = site.peak
    if vre_delta is None:
        vre_delta = 0.10*P
    if mw_delta is None:
        # far above the target the shortfall is deep (a solar night, a lost big unit) and
        # one small machine barely moves LOLE, so a 5%-of-peak secant under-states what a
        # fleet of small machines would do and the cut wrongly forces big units (Georgia
        # 660 MW campus: aero slope measured 0.01/unit at LOLE 38, then 4 CT + CC was the
        # only route left). Use a quarter of peak there, 5% near the target.
        mw_delta = P*(0.25 if L0 > 10*TARGET else 0.05)
    if floor is None:
        floor = 0.5/max(reps, 1)
    lg = lambda v: np.log(max(float(v), floor))
    l0 = lg(L0)
    gn = np.zeros(NC); evals = {}
    for j in range(NC):
        # secant, not tangent: add enough units of type j to move firm capacity by at
        # least `mw_delta` (one big machine, several small ones), so the log difference is
        # well above Monte Carlo noise; slope is then per unit
        dj = max(1, int(np.ceil(mw_delta/(S_MW[j]*AVAIL[j]))))
        n2 = np.asarray(n).copy(); n2[j] += dj
        Lj = lole_of(site, n2, s, w, b, reps, seed=seed)
        gn[j] = max(0.0, l0 - lg(Lj))/dj; evals[f"{CAT[j]['k']}+{dj}"] = Lj
    gs = gw = gb = 0.0
    if allow[0] and site.cf_s is not None:
        Ls = lole_of(site, n, s + vre_delta, w, b, reps, seed=seed); gs = max(0.0, l0 - lg(Ls))/vre_delta
        evals['solar+%.0f' % vre_delta] = Ls
    if allow[1] and site.cf_w is not None:
        Lw = lole_of(site, n, s, w + vre_delta, b, reps, seed=seed); gw = max(0.0, l0 - lg(Lw))/vre_delta
        evals['wind+%.0f' % vre_delta] = Lw
    if allow[2]:
        Lb = lole_of(site, n, s, w, b + vre_delta, reps, seed=seed); gb = max(0.0, l0 - lg(Lb))/vre_delta
        evals['batt+%.0f' % vre_delta] = Lb
    # trust region: log LOLE steepens as the margin grows, so a slope measured far above
    # the target under-states what the last units buy and a one-shot extrapolation
    # over-builds (the pooled 32 GW plant jumped from LOLE 166 to 0.000). Ask for at most
    # `max_step` log units per cut; the next iteration measures a fresh slope.
    # the cut is one-sided and its slopes are noisy secants, so ask for a little more than
    # the bare log-gap (x1.3): a step that lands short costs a whole extra solve+simulate
    rhs = min(1.3*(l0 - np.log(TARGET if target is None else target)), max_step)
    return dict(n0=np.asarray(n, float), s0=s, w0=w, b0=b, gn=gn, gs=gs, gw=gw, gb=gb, rhs=rhs,
                lole=L0, evals=evals)



def polish_confirm(site, n, s, w, b, allow=(True, True, True), vre_step=None, reps=400, confirm=2000,
                   seed=20260826, seed_confirm=23, verbose=False):
    """Stages 3 and 4 of `design`, callable on any design: single-unit removals and swaps and
    solar/wind/battery steps (+-5% of peak, drop, halve), kept when cheaper and passing at
    `reps`; then independent confirmation at `confirm` replications, adding the cheapest
    owned-type machine until it passes. Returns (n, s, w, b, LOLE_confirm, cost dict)."""
    n = np.asarray(n, np.int64).copy(); s, w, b = float(s), float(w), float(b)

    def confirmed(n_, s_, w_, b_):
        return lole_of(site, n_, s_, w_, b_, confirm, seed=seed_confirm)

    # the incoming design must itself pass the confirmation before it is polished
    for _ in range(12):
        Lc = confirmed(n, s, w, b)
        if Lc <= TARGET:
            break
        j = int(np.argmin(S_MW*FX_MW*np.where(n > 0, 1.0, 1.5)))   # prefer a type already owned
        n[j] += 1
        verbose and print(f'   confirm failed ({Lc:.3f}); added one {CAT[j]["k"]}', flush=True)
    C = cost_of(site, n, s, w, b)
    vs = vre_step or max(10.0, 0.05*site.peak)
    improved = True
    while improved:
        improved = False
        cands = []
        for j in range(NC):
            if n[j] > 0:
                m2 = n.copy(); m2[j] -= 1; cands.append((m2, s, w, b))          # remove one
                for j2 in range(NC):
                    if j2 != j:
                        m3 = m2.copy(); m3[j2] += 1; cands.append((m3, s, w, b))  # swap one
        for ds in (-vs, vs):
            if allow[0] and site.cf_s is not None and s + ds >= 0: cands.append((n.copy(), s + ds, w, b))
            if allow[1] and site.cf_w is not None and w + ds >= 0: cands.append((n.copy(), s, w + ds, b))
            if allow[2] and b + ds >= 0: cands.append((n.copy(), s, w, b + ds))
        for f_ in (0.0, 0.5):                       # and: drop or halve each of solar, wind, battery
            if s > 0: cands.append((n.copy(), f_*s, w, b))
            if w > 0: cands.append((n.copy(), s, f_*w, b))
            if b > 0: cands.append((n.copy(), s, w, f_*b))
        scored = []
        for (n2, s2, w2, b2) in cands:
            C2 = cost_of(site, n2, s2, w2, b2)
            if C2['total'] < C['total']*(1 - 1e-4):
                scored.append((C2['total'], n2, s2, w2, b2, C2))
        for _tot, n2, s2, w2, b2, C2 in sorted(scored, key=lambda z: z[0]):
            L2 = lole_of(site, n2, s2, w2, b2, reps, seed=seed + 101)
            if L2 <= TARGET:
                L3 = confirmed(n2, s2, w2, b2)          # a move is kept only if it also passes the confirmation
                if L3 <= TARGET:
                    n, s, w, b, C, Lc = n2, s2, w2, b2, C2, L3; improved = True
                    verbose and print(f'   polish: n={[int(v) for v in n]} s={s:.0f} w={w:.0f} b={b:.0f} '
                                      f'LOLE {L3:.3f} ${C["total"]/1e6:,.1f}M', flush=True)
                    break
    return n, s, w, b, Lc, cost_of(site, n, s, w, b)


# ==================================================================== the loop
def design(load, zone, solar_cf=None, wind_cf=None, allow=(True, True, True),
           vre_step=None, step=6, screen=240, confirm=2000, m_lo=0.0, k0=2, k_max=6,
           iters=12, improve_iters=15, improve_reps=400, improve_radius=6, screen_buffer=0.1, k_step_lole=3.0, credit_cap=0.15, vre_cut_frac=0.0,
           sens_cuts=False, polish=True, verbose=False, mc_load=None, time_limit=120.0,
           seed=20260826, seed_confirm=23, **site_kw):
    """Least-cost off-grid design meeting LOLE <= 0.1 d/yr. See the module docstring.

    allow      = (solar, wind, battery) -- set an entry False to exclude that technology
    vre_step   = MW grid for s, w, b (None: continuous)
    step       = hour sub-sampling of the dispatch LP (1 = every hour)
    screen     = Monte Carlo replications used inside the bisection
    confirm    = replications for the final independent confirmation
    sens_cuts  = add a Monte-Carlo sensitivity (Benders-style) cut on log LOLE after each
                 rejected design, alongside the exact no-good cut
    improve_iters = solves in the improvement stage (0 disables it): with the heuristic
                 surrogates dropped and only the exact no-good cuts kept, ask the MIP for a
                 design cheaper than the incumbent, simulate it at `improve_reps`, accept or
                 cross it off, until nothing cheaper is left or the budget is spent
    """
    t0 = time.time()
    site = Site(load, zone, solar_cf, wind_cf, mc_load=mc_load, **site_kw)
    cc = site.capacity_credits()
    log = []
    verbose and print(f'  {zone}: peak {site.peak:,.0f} MW, gas ${site.gas:.2f}, '
                      f'credits solar {cc[0]:.2f} wind {cc[1]:.2f} battery {cc[2]:.2f}', flush=True)

    def evaluate(sol, reps, sd):
        L = lole_of(site, sol['n'], sol['s'], sol['w'], sol['b'], reps, seed=sd)
        C = cost_of(site, sol['n'], sol['s'], sol['w'], sol['b'],
                    fuel=sol['fuel'] if sol['b'] > 0 else None)
        return L, C

    # ---- 1. lazy LOLE cuts: solve, simulate, cut, repeat
    # The MIP returns the cheapest design that satisfies the linear surrogates and every
    # cut so far. If the Monte Carlo rejects it, every design with no more firm machine
    # capacity is rejected with it (LOLE is monotone in every unit), so one linear cut
    # removes that whole set and the next solve must add real machines. The step is one
    # smallest machine when the miss is small and larger when LOLE is far above target.
    # Each rejected design tightens the N-k rule by one unit and adds one exact no-good
    # cut (see build_and_solve (5) and (7)). The N-k rule carries the granularity, so
    # the loop normally closes in two to four solves.
    cuts, nogood, sens, best = [], [], [], None
    m, k = m_lo, k0
    for it in range(iters):
        # after three rejections the VRE escape from the no-good cuts is closed: near the
        # target the loop otherwise shuffles 10%-of-peak parcels of wind/solar between
        # near-identical designs (LOLE 0.12-0.29) for dozens of solves before it buys the
        # machine it needs; the sensitivity cut's marginal values still let VRE compete
        vcf = vre_cut_frac if len(nogood) < 3 else 0.0
        sol = build_and_solve(site, m, cc, step=step, vre_step=vre_step, allow=allow,
                              time_limit=time_limit, cuts=cuts, nogood=nogood, k_out=k,
                              credit_cap=credit_cap, vre_cut_frac=vcf, sens=sens)
        if sol is None:
            verbose and print(f'   iteration {it}: MIP infeasible', flush=True)
            return None
        L, C = evaluate(sol, screen, seed)
        firm = float(sol['n'] @ (S_MW*AVAIL))
        ok = L <= TARGET*(1 - screen_buffer)          # a screen at 120 reps is noisy: keep a margin
        log.append(dict(it=it, k=k, n=[int(v) for v in sol['n']], s=sol['s'], w=sol['w'],
                        b=sol['b'], firm_MW=firm, lole=L, cost=C['total'], feasible=ok,
                        status=sol['status']))
        verbose and print(f'   it {it} (N-{k}): n={[int(v) for v in sol["n"]]} s={sol["s"]:.0f} '
                          f'w={sol["w"]:.0f} b={sol["b"]:.0f}  firm {firm:,.0f} MW  '
                          f'LOLE {L:.3f}  ${C["total"]/1e6:,.1f}M  {"pass" if ok else "fail"}'
                          f'{"" if sol["status"].startswith("Optim") else "  [" + sol["status"][:20] + "]"}',
                          flush=True)
        if ok:
            best = (sol, C, L)
            break
        nogood.append((sol['n'].copy(), sol['s'], sol['w'], sol['b']))
        # Was this a MACHINE design that failed, or a design leaning on solar / wind / the
        # battery at credits the Monte Carlo has just shown to be too generous? Only the
        # first says anything about unit granularity (raise k) or about the marginal value
        # of machines near this design (sensitivity cut). A solar-night shortfall makes LOLE
        # a step function of small-unit count, and a cut measured there wrongly forces big
        # units (Georgia 660 MW campus: aero slope 0.05/unit at LOLE 38 -> 4 CT + CC at
        # 3.5x peak, 4% dearer than 23 aeros + CC). For a VRE-reliant failure the answer is
        # the credit re-measurement below plus the no-good cut.
        vre_reliant = (cc[0]*sol['s'] + cc[1]*sol['w'] + cc[2]*sol['b']) > 0.15*site.peak
        if sens_cuts and not vre_reliant:
            q = sensitivity_cut(site, sol['n'], sol['s'], sol['w'], sol['b'], L, 2*screen, seed, allow=allow,
                                target=TARGET*(1 - screen_buffer))
            sens.append(q)
            log[-1]['sens'] = dict(gn=[float(v) for v in q['gn']], gs=q['gs'], gw=q['gw'], gb=q['gb'],
                                   rhs=q['rhs'], evals={k_: float(v) for k_, v in q['evals'].items()})
            verbose and print('      sens cut: dlogLOLE per unit ' +
                              ' '.join(f"{CAT[j]['k']}={q['gn'][j]:.2f}" for j in range(NC)) +
                              f"  per 100 MW solar={100*q['gs']:.2f} wind={100*q['gw']:.2f} "
                              f"batt={100*q['gb']:.2f}  need {q['rhs']:.2f}", flush=True)
        if L > k_step_lole*TARGET and k < k_max and not vre_reliant:
            k += 1                # far off target: tighten the N-k rule; near it, the cut suffices
        # ELCC-style update of the capacity credits: re-measure them on the tightest hours
        # of the load NET of the rejected design's own solar and wind. A campus that leans
        # on solar moves its tightest hours into the night, where solar is worth nothing;
        # the credit falls and the next solve sees it. Credits only ever tighten.
        if sol['s'] > 0 or sol['w'] > 0 or sol['b'] > 0:
            cn = site.capacity_credits(sol['s'], sol['w'], n=sol['n'])
            cc = tuple(min(a_, b_) for a_, b_ in zip(cc, cn))
            verbose and print(f'      credits re-measured on the net load: solar {cc[0]:.2f} '
                              f'wind {cc[1]:.2f} battery {cc[2]:.2f}', flush=True)
    if best is None:
        # the loop did not close in `iters` solves (near the target the surrogates cannot
        # tell 0.13 from 0.09 and the MIP shuffles small units and VRE parcels): take the
        # rejected design with the lowest LOLE and add the cheapest owned-type machine until
        # the screen passes, then polish and confirm as usual. The incumbent check
        # (incumbent_check.py) guards the cost of what comes out.
        q = min(log, key=lambda e: e['lole']); n = np.array(q['n']); s_, w_, b_ = q['s'], q['w'], q['b']
        for _ in range(12):
            L = lole_of(site, n, s_, w_, b_, screen, seed=seed)
            if L <= TARGET*(1 - screen_buffer):
                break
            j = int(np.argmin(S_MW*FX_MW*np.where(n > 0, 1.0, 1.5))); n[j] += 1
        C = cost_of(site, n, s_, w_, b_)
        sol = dict(n=n, s=s_, w=w_, b=b_, fuel=C['fuel']); best = (sol, C, L)
        log.append(dict(it=len(log), k=k, n=[int(v) for v in n], s=s_, w=w_, b=b_, firm_MW=float(n @ (S_MW*AVAIL)),
                        lole=L, cost=C['total'], feasible=bool(L <= TARGET*(1 - screen_buffer)), status='fallback: cheapest-unit additions'))
        verbose and print(f'   loop did not close in {iters} solves; fallback n={[int(v) for v in n]} LOLE {L:.3f}', flush=True)
    sol, C, L = best

    # ---- 1b. improvement stage: exact local search on the machine lattice
    # Stage 1 stops at the first design that passes the screen, i.e. the cheapest design
    # under the SURROGATES (N-k, sensitivity cuts), which need not be the cheapest feasible
    # design. Here the surrogates are dropped altogether and the Monte Carlo itself is the
    # oracle: enumerate every machine portfolio within `improve_radius` unit changes of the
    # incumbent (5 types, radius 4: ~1,000 portfolios), cost each exactly (merit order,
    # incumbent's solar/wind/battery held), and test the ones cheaper than the incumbent in
    # ascending cost order at `improve_reps` replications, in vectorised batches. The first
    # that passes is the new incumbent; the search re-centres on it and repeats. It stops
    # when no cheaper portfolio in the neighbourhood passes: a local optimum of the exact
    # problem, reached by the same test every design must satisfy. A portfolio dominated by
    # a rejected one (no more of any machine) is skipped without testing (LOLE is monotone).
    sol, C, L = best
    C_inc = cost_of(site, sol['n'], sol['s'], sol['w'], sol['b'])
    best = (sol, C_inc, L); n_impr = 0; n_tested = 0
    from itertools import product
    # stage-1 rejections were screened at `screen` replications against a 10% buffer, so a
    # design at true LOLE 0.05 can sit in `nogood` by bad luck (Georgia GA22: 9 aeros
    # screened 0.100, confirmed 0.048). Only clear failures (> 2x target) carry over as
    # proven-infeasible; everything else is re-tested here at improve_reps if it is cheaper.
    rejected = [np.asarray(e['n']) for e in log if e.get('lole', 0.0) > 2*TARGET and 'n' in e]
    Rr = int(improve_radius)
    while improve_iters and n_tested < improve_iters*20:
        n0 = np.asarray(sol['n']); s_, w_, b_ = sol['s'], sol['w'], sol['b']
        cands = []
        for dv in product(range(-Rr, Rr + 1), repeat=NC):
            if 0 < sum(abs(x) for x in dv) <= Rr:
                n2 = n0 + np.array(dv)
                if (n2 >= 0).all() and n2.sum() > 0:
                    cands.append(n2)
        if not cands:
            break
        NN = np.array(cands, np.int64)
        VR = site.VR; net = np.maximum(site.net(s_, w_), 0.0)
        fuel = og.merit_fuel_vec(NN, VR, og.ldc(net))
        fixed = NN @ (S_MW*FX_MW) + s_*SOLAR_FIXED + w_*WIND_FIXED + b_*BATT_FIXED
        acres = (NN @ S_MW)*ACRES['gas'] + s_*ACRES['solar'] + w_*ACRES['wind'] + b_*ACRES['storage'] + site.peak*ACRES['dc']
        tot = fixed + acres*site.land*og.LAND_CRF + fuel
        order = np.argsort(tot)
        order = [i for i in order if tot[i] < C_inc['total']*(1 - 1e-4)
                 and not any((NN[i] <= rj).all() for rj in rejected)]
        found = False
        for k0_ in range(0, len(order), 25):
            idx = order[k0_:k0_ + 25]
            if b_ > 0:
                Ls = np.array([lole_of(site, NN[i], s_, w_, b_, improve_reps, seed=seed + 7) for i in idx])
            else:
                NET = np.atleast_2d(site.mc_load).copy()
                if s_ > 0 and site.cf_s is not None: NET -= s_*site.cf_s[None, :]
                if w_ > 0 and site.cf_w is not None: NET -= w_*site.cf_w[None, :]
                Ls = og.lole_batch(NET[0] if len(NET) == 1 else NET, NN[idx], improve_reps, seed=seed + 7)
            n_tested += len(idx)
            for i, Li in zip(idx, Ls):
                if Li <= TARGET:
                    n_new = NN[i]; C_new = cost_of(site, n_new, s_, w_, b_)
                    log.append(dict(it=len(log), stage='improve', n=[int(v) for v in n_new], s=s_, w=w_, b=b_,
                                    lole=float(Li), cost=C_new['total'], feasible=True, tested=n_tested))
                    verbose and print(f'   improve: n={[int(v) for v in n_new]}  LOLE {Li:.3f}  ${C_new["total"]/1e6:,.1f}M  '
                                      f'(was ${C_inc["total"]/1e6:,.1f}M; {n_tested} portfolios tested)', flush=True)
                    sol = dict(sol, n=n_new); C_inc = C_new; L = float(Li); best = (sol, C_inc, L); n_impr += 1
                    found = True
                    break
                rejected.append(NN[i])
            if found or n_tested >= improve_iters*20:
                break
        if not found:
            log.append(dict(it=len(log), stage='improve', status='local optimum', cost=C_inc['total'], tested=n_tested))
            verbose and print(f'   improve: no cheaper portfolio within {Rr} unit changes passes '
                              f'({n_tested} tested) -> local optimum ${C_inc["total"]/1e6:,.1f}M', flush=True)
            break
    sol, C, L = best

    # ---- 2 + 3. polish and independent confirmation (polish_confirm)
    n, s, w, b, Lc, C = polish_confirm(site, sol['n'], sol['s'], sol['w'], sol['b'], allow=allow, vre_step=vre_step,
                                       reps=improve_reps, confirm=confirm, seed=seed, seed_confirm=seed_confirm,
                                       verbose=verbose) if polish else (sol['n'].copy(), sol['s'], sol['w'], sol['b'], None, None)
    if not polish:
        n = np.asarray(n, np.int64)
        for _ in range(12):
            Lc = lole_of(site, n, s, w, b, confirm, seed=seed_confirm)
            if Lc <= TARGET:
                break
            j = int(np.argmin(S_MW*FX_MW*np.where(n > 0, 1.0, 1.5))); n[j] += 1
    C = cost_of(site, n, s, w, b)
    inst = float(n @ S_MW)
    # split as the published files do: fixed_cost / land_cost are the machines (and the
    # campus's own land); vre_cost carries solar, wind and battery with their land
    vre_fixed = s*SOLAR_FIXED + w*WIND_FIXED + b*BATT_FIXED
    vre_land = (s*ACRES['solar'] + w*ACRES['wind'] + b*ACRES['storage'])*site.land*og.LAND_CRF
    return dict(zone=zone, n=[int(v) for v in n], installed_MW=inst, ratio=inst/site.peak,
                solar_MW=s, wind_MW=w, batt_MW=b, LOLE=Lc, cost=C['total'],
                fixed_cost=C['fixed'] - vre_fixed, land_cost=C['land'] - vre_land,
                vre_cost=vre_fixed + vre_land, fuel_cost=C['fuel'],
                acres=C['acres'], gas=site.gas, lcoe=C['total']/site.energy,
                energy_MWh=site.energy, peak_MW=site.peak, unit_count=int(n.sum()),
                mix={CAT[i]['k']: int(n[i]) for i in range(NC) if n[i]},
                margin=m, n_minus_k=k, n_cuts=len(nogood), n_sens_cuts=len(sens), credit_cap=credit_cap,
                vre_cut_frac=vre_cut_frac, capacity_credits=dict(solar=cc[0], wind=cc[1], battery=cc[2]),
                mip_log=log, seconds=time.time() - t0,
                n_improvements=n_impr, confirm_reps=confirm,
                method='mip+lazy-lole-cuts' + ('+mc-sensitivity' if sens_cuts else '') + ('+improve' if improve_iters else ''))


# ================================================================ self-test
def selftest():
    """Synthetic 1 GW campus: MIP design vs the frontier search on the same criterion."""
    rng = np.random.default_rng(1)
    load = 800 + 200*rng.random(H)
    hours = np.arange(H) % 24
    cf_s = np.clip(np.sin((hours - 6)/12*np.pi), 0, None)*0.8*(0.7 + 0.3*rng.random(H))
    cf_w = np.clip(0.35 + 0.3*rng.standard_normal(H), 0, 1)
    print('offgrid_mip self-test, synthetic 1 GW campus, zone FarWest')
    t = time.time()
    r = design(load, 'FarWest', cf_s, cf_w, verbose=True)
    print(f'  MIP  design: {r["mix"]}  solar {r["solar_MW"]:.0f} wind {r["wind_MW"]:.0f} '
          f'batt {r["batt_MW"]:.0f} MW  LOLE {r["LOLE"]:.3f}  ${r["lcoe"]:.2f}/MWh  '
          f'{r["installed_MW"]/r["peak_MW"]:.2f}x  [{time.time()-t:.0f}s]')
    t = time.time()
    fr = og.feasible_frontier(load)
    ref = og.design_site(load, 'FarWest', frontier=fr)
    print(f'  frontier search (gas only): {ref["mix"]}  LOLE {ref["LOLE"]:.3f}  '
          f'${ref["lcoe"]:.2f}/MWh  {ref["ratio"]:.2f}x  [{time.time()-t:.0f}s]')
    r2 = design(load, 'FarWest', allow=(False, False, False), verbose=False)
    print(f'  MIP, gas only:              {r2["mix"]}  LOLE {r2["LOLE"]:.3f}  '
          f'${r2["lcoe"]:.2f}/MWh  {r2["ratio"]:.2f}x')
    return r, ref, r2


if __name__ == '__main__':
    if len(sys.argv) == 1:
        selftest()
    else:
        # one real campus: paths follow run_offgrid.py's conventions
        import pandas as pd
        CD = os.environ.get('OFFGRID_CASE', '2_ongrid_code/HOPE/ModelCases/TX8z_GTEP_dc_vre_case/Data_TX8z_GTEP_dc_vre')
        R = os.environ.get('OFFGRID_LOAD', '1_source_data/DC_Load/load_profiles')
        d = pd.read_csv(f'{R}/dc_load_hourly_MW.csv', index_col=0, parse_dates=True)
        d = d[~((d.index.month == 2) & (d.index.day == 29))]
        meta = pd.read_csv(f'{R}/site_parameters_and_results.csv')
        ZM = {'Far West': 'FarWest', 'North Central': 'NorthCentral', 'South Central': 'SouthCentral'}
        for sid in sys.argv[1].split(','):
            row = meta[meta.site_id == sid].iloc[0]
            zone = ZM.get(row.ercot_weather_zone, row.ercot_weather_zone)
            load = d[sid].values[:H]
            cf_s = og.zonal_solar_cf(CD, zone)
            cf_w = zonal_wind_cf(CD, zone)
            r = design(load, zone, cf_s, cf_w, verbose=True)
            print(json.dumps({k: v for k, v in r.items() if k != 'mip_log'}, indent=1, default=float))
