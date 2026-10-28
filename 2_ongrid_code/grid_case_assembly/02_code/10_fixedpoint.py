"""PRM fixed point: expansion -> Monte Carlo -> adjust PRM -> repeat.

The expansion step is a HiGHS LP written against the SAME case CSVs HOPE reads
(gendata, gendata_candidate, storagedata*, linedata, load and availability
timeseries, single_parameter), reproducing HOPE's GTEP structure:

  min  sum_n capex_n x_n + sum_s capex_s x_s + sum_t w_t ( sum voc p + VOLL u )
  s.t. p_e,t <= Pmax_e AF_e a_e,t                 existing dispatch
       p_n,t <= x_n    AF_n a_n,t                 candidate dispatch, RHS is a variable
       storage SOC balance, cyclic within each representative day
       zonal balance with line flows and unserved energy
       sum_g CC_g cap_g + sum_s CC_s cap_s >= (1+PRM) * annual peak      <- the RA constraint
       thermal headroom >= spin_requirement * load

Operations are solved on weighted representative days; the RA constraint uses the
true 8760 annual peak. The Monte Carlo that judges the result always runs full
8760 x 8 weather years, so representative days affect only which portfolio is
chosen, never the reliability verdict.

Usage:  python 10_fixedpoint.py [case] [reps] [ndays]
"""
import json, os, sys, time
import numpy as np, pandas as pd
from scipy import sparse
from scipy.optimize import linprog
import importlib.util

spec = importlib.util.spec_from_file_location('r', os.path.join(os.path.dirname(os.path.abspath(__file__)), '08_report.py'))
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)

CASE   = sys.argv[1] if len(sys.argv) > 1 else 'base'
REPS   = int(sys.argv[2]) if len(sys.argv) > 2 else 240
NDAYS  = int(sys.argv[3]) if len(sys.argv) > 3 else 24
ROOT   = f'Code/HOPE/ModelCases/TX8z_GTEP_{CASE}_case/Data_TX8z_GTEP_{CASE}'
OUTD   = M.OUTD
ZONES  = M.ZONES
H      = 8760
NEW_STO_H = 4.0
FOR_NEW = {'NGCC':0.0357, 'NGCT':0.0476, 'NuC':0.08, 'SolarPV':0.0, 'WindOn':0.0}
MTTR_NEW= {'NGCC':48, 'NGCT':36, 'NuC':120}
UNIT_NEW= {'NGCC':500., 'NGCT':100., 'NuC':1200.}       # Pmax/n_units from the candidate file


# ----------------------------------------------------------------- case data
def load_case():
    g  = pd.read_csv(f'{ROOT}/gendata.csv')
    c  = pd.read_csv(f'{ROOT}/gendata_candidate.csv')
    se = pd.read_csv(f'{ROOT}/storagedata.csv')
    sn = pd.read_csv(f'{ROOT}/storagedata_candidate.csv')
    ln = pd.read_csv(f'{ROOT}/linedata.csv')
    av = pd.read_csv(f'{ROOT}/gen_availability_timeseries.csv')
    ld = pd.read_csv(f'{ROOT}/load_timeseries_regional.csv')
    sp = pd.read_csv(f'{ROOT}/single_parameter.csv').iloc[0]
    ceil = float(os.environ.get('CEILING_MW', 0))
    if ceil:                       # sensitivity: relax the per-zone candidate build ceiling
        c['Pmax (MW)'] = ceil
        sn['Max Power (MW)'] = ceil
    A  = av[[f'G{i+1}' for i in range(len(g)+len(c))]].values          # 8760 x 97
    # Sensitivity: apply the SAME vintage/curtailment derate to new VRE that the
    # existing fleet carries. Without this, candidate wind runs at CF 0.466 against an
    # operating fleet at 0.337 - new build looks both cheaper and more reliable.
    if os.environ.get('VRE_DERATE_NEW'):
        NE = len(g)
        for k in range(len(c)):
            t = c.Type.values[k]
            if t == 'WindOn':   A[:, NE+k] *= 0.7242
            elif t == 'SolarPV': A[:, NE+k] *= 0.7839
    zd = pd.read_csv(f'{ROOT}/zonedata.csv').set_index('Zone_id')['Demand (MW)']
    # HOPE stores load as a per-unit profile scaled by the zonal peak in zonedata
    L  = ld[ZONES].values * zd[ZONES].values[None, :]                 # 8760 x 8, MW
    return dict(g=g, c=c, se=se, sn=sn, ln=ln, A=A, L=L, sp=sp, zd=zd,
                peak_override=None)


def rep_days(D, nd):
    """Pick nd representative days by k-means on the daily net-load shape.
    The single highest-net-load day is always kept as its own singleton."""
    net = (D['L'].sum(1) - (D['A'][:, len(D['g']):] * 0).sum(1)).reshape(365, 24)
    vre = np.zeros(H)
    for j, r in D['g'].iterrows():
        if r.Type in ('SolarPV','WindOn'):
            vre += r['Pmax (MW)'] * D['A'][:, j]
    net = (D['L'].sum(1) - vre).reshape(365, 24)
    pk  = int(net.max(1).argmax())
    X   = net.copy(); mu, sd = X.mean(), X.std()
    Xn  = (X - mu)/sd
    rng = np.random.default_rng(7)
    idx = np.array([i for i in range(365) if i != pk])
    k   = nd - 1
    cen = Xn[rng.choice(idx, k, replace=False)]
    for _ in range(60):
        d   = ((Xn[idx][:,None,:] - cen[None,:,:])**2).sum(2)
        lab = d.argmin(1)
        for j in range(k):
            if (lab==j).any(): cen[j] = Xn[idx][lab==j].mean(0)
    days, wts = [], []
    for j in range(k):
        mem = idx[lab==j]
        if not len(mem): continue
        med = mem[((Xn[mem]-cen[j])**2).sum(1).argmin()]
        days.append(int(med)); wts.append(float(len(mem)))
    days.append(pk); wts.append(1.0)
    return np.array(days), np.array(wts)


# ----------------------------------------------------------------- expansion LP
def expand(D, prm, days, wts, verbose=False):
    g, c, se, sn, ln = D['g'], D['c'], D['se'], D['sn'], D['ln']
    NE, NC, NSE, NSN, NL, NZ = len(g), len(c), len(se), len(sn), len(ln), len(ZONES)
    hrs = np.concatenate([np.arange(d*24, d*24+24) for d in days])
    T   = len(hrs); ND = len(days)
    w   = np.repeat(wts, 24)
    A   = D['A'][hrs]; L = D['L'][hrs]
    VOLL = float(D['sp']['VOLL']); SPIN = float(D['sp']['spin_requirement'])
    peak = float(D.get('peak_override') or D['L'].sum(1).max())
    zi   = {z:i for i,z in enumerate(ZONES)}

    # ---- variable blocks
    n = 0
    def blk(sz):
        nonlocal n
        s = n; n += sz; return s
    o_xg  = blk(NC)                # candidate gen MW
    o_xs  = blk(NSN)               # candidate storage MW
    o_pe  = blk(NE*T); o_pn = blk(NC*T)
    o_dce = blk(NSE*T); o_che = blk(NSE*T); o_soce = blk(NSE*T)
    o_dcn = blk(NSN*T); o_chn = blk(NSN*T); o_socn = blk(NSN*T)
    o_f   = blk(NL*T);  o_u   = blk(NZ*T)
    NV = n

    lo = np.zeros(NV); hi = np.full(NV, np.inf)
    hi[o_xg:o_xg+NC]  = c['Pmax (MW)'].values
    hi[o_xs:o_xs+NSN] = sn['Max Power (MW)'].values
    for l in range(NL):
        lo[o_f+l*T:o_f+(l+1)*T] = -ln['Reverse Capacity (MW)'].values[l]
        hi[o_f+l*T:o_f+(l+1)*T] =  ln['Forward Capacity (MW)'].values[l]

    obj = np.zeros(NV)
    obj[o_xg:o_xg+NC]  = c['Cost ($/MW/yr)'].values.astype(float)
    obj[o_xs:o_xs+NSN] = sn['Cost ($/MW/yr)'].values.astype(float)
    for e in range(NE): obj[o_pe+e*T:o_pe+(e+1)*T] = g['Cost ($/MWh)'].values[e]*w
    for k in range(NC): obj[o_pn+k*T:o_pn+(k+1)*T] = c['Cost ($/MWh)'].values[k]*w
    for z in range(NZ): obj[o_u+z*T:o_u+(z+1)*T]   = VOLL*w

    R, Cc, V, ub = [], [], [], []      # inequality rows
    ER, EC, EV, be = [], [], [], []    # equality rows
    r = 0
    def ineq(cols, vals, rhs):
        nonlocal r
        R.extend([r]*len(cols)); Cc.extend(cols); V.extend(vals); ub.append(rhs); r += 1
    er = 0
    def eq(cols, vals, rhs):
        nonlocal er
        ER.extend([er]*len(cols)); EC.extend(cols); EV.extend(vals); be.append(rhs); er += 1

    # existing dispatch bound (constant RHS) -> fold into hi[]
    for e in range(NE):
        cap = g['Pmax (MW)'].values[e]*g['AF'].values[e]
        hi[o_pe+e*T:o_pe+(e+1)*T] = cap*A[:, e]
    # candidate dispatch: p - af*a*x <= 0
    for k in range(NC):
        af = c['AF'].values[k]
        for t in range(T):
            ineq([o_pn+k*T+t, o_xg+k], [1.0, -af*A[t, NE+k]], 0.0)
    # existing storage limits
    for s in range(NSE):
        P, E = se['Max Power (MW)'].values[s], se['Capacity (MWh)'].values[s]
        hi[o_dce+s*T:o_dce+(s+1)*T] = P
        hi[o_che+s*T:o_che+(s+1)*T] = P
        hi[o_soce+s*T:o_soce+(s+1)*T] = E
    # candidate storage limits reference the investment variable
    for s in range(NSN):
        for t in range(T):
            ineq([o_dcn+s*T+t, o_xs+s], [1.0, -1.0], 0.0)
            ineq([o_chn+s*T+t, o_xs+s], [1.0, -1.0], 0.0)
            ineq([o_socn+s*T+t, o_xs+s], [1.0, -NEW_STO_H], 0.0)
    # SOC balance, cyclic within each representative day
    def soc_rows(o_soc, o_dc, o_ch, eta_c, eta_d, ns):
        for s in range(ns):
            for d in range(ND):
                for h in range(24):
                    t  = d*24+h; tp = d*24+((h-1) % 24)
                    eq([o_soc+s*T+t, o_soc+s*T+tp, o_ch+s*T+t, o_dc+s*T+t],
                       [1.0, -1.0, -eta_c[s], 1.0/eta_d[s]], 0.0)
    soc_rows(o_soce, o_dce, o_che, se['Charging efficiency'].values,
             se['Discharging efficiency'].values, NSE)
    soc_rows(o_socn, o_dcn, o_chn, sn['Charging efficiency'].values,
             sn['Discharging efficiency'].values, NSN)
    # zonal balance
    gz = [zi[z] for z in g.Zone]; cz = [zi[z] for z in c.Zone]
    sez = [zi[z] for z in se.Zone]; snz = [zi[z] for z in sn.Zone]
    lf = [zi[z] for z in ln.From_zone]; lt = [zi[z] for z in ln.To_zone]
    for z in range(NZ):
        for t in range(T):
            cols, vals = [], []
            for e in range(NE):
                if gz[e]==z: cols.append(o_pe+e*T+t); vals.append(1.0)
            for k in range(NC):
                if cz[k]==z: cols.append(o_pn+k*T+t); vals.append(1.0)
            for s in range(NSE):
                if sez[s]==z:
                    cols += [o_dce+s*T+t, o_che+s*T+t]; vals += [1.0, -1.0]
            for s in range(NSN):
                if snz[s]==z:
                    cols += [o_dcn+s*T+t, o_chn+s*T+t]; vals += [1.0, -1.0]
            for l in range(NL):
                if lt[l]==z: cols.append(o_f+l*T+t); vals.append(1.0)
                if lf[l]==z: cols.append(o_f+l*T+t); vals.append(-1.0)
            cols.append(o_u+z*T+t); vals.append(1.0)
            eq(cols, vals, L[t, z])
    # RA constraint  -> -(sum CC cap) <= -(1+prm) peak
    fixed = float((g['Pmax (MW)']*g['CC']).sum() + (se['Max Power (MW)']*se['CC']).sum())
    cols = [o_xg+k for k in range(NC)] + [o_xs+s for s in range(NSN)]
    vals = [-float(c['CC'].values[k]) for k in range(NC)] + \
           [-float(sn['CC'].values[s]) for s in range(NSN)]
    ineq(cols, vals, -((1+prm)*peak - fixed))
    # spinning reserve: thermal headroom >= SPIN * load
    th_e = [e for e in range(NE) if g['Flag_thermal'].values[e]==1]
    th_c = [k for k in range(NC) if c['Flag_thermal'].values[k]==1]
    for t in range(T):
        cols, vals = [], []
        rhs = -SPIN*L[t].sum()
        for e in th_e:
            cols.append(o_pe+e*T+t); vals.append(1.0)
            rhs += g['Pmax (MW)'].values[e]*g['AF'].values[e]*A[t, e]
        for k in th_c:
            cols.append(o_pn+k*T+t); vals.append(1.0)
            cols.append(o_xg+k);     vals.append(-c['AF'].values[k]*A[t, NE+k])
        ineq(cols, vals, rhs)

    Aub = sparse.csr_matrix((V, (R, Cc)), shape=(r, NV))
    Aeq = sparse.csr_matrix((EV, (ER, EC)), shape=(er, NV))
    t0 = time.time()
    res = linprog(obj, A_ub=Aub, b_ub=np.array(ub), A_eq=Aeq, b_eq=np.array(be),
                  bounds=np.c_[lo, hi], method='highs')
    if not res.success:
        raise RuntimeError(res.message)
    x = res.x
    built_g = {(c.Zone[k], c.Type[k]): float(x[o_xg+k]) for k in range(NC) if x[o_xg+k] > 1.0}
    built_s = {sn.Zone[s]: float(x[o_xs+s]) for s in range(NSN) if x[o_xs+s] > 1.0}
    unserved = float(sum(x[o_u+z*T+t]*w[t] for z in range(NZ) for t in range(T)))
    if verbose:
        print(f'   LP {NV:,} vars / {r+er:,} rows solved in {time.time()-t0:.1f}s  '
              f'obj ${res.fun/1e9:.3f}B  unserved {unserved:,.0f} MWh')
    return dict(gen=built_g, sto=built_s, obj=float(res.fun), unserved=unserved,
                capex=float(obj[:o_pe] @ x[:o_pe]))


# ----------------------------------------------------------------- MC on a built portfolio
def mc(D, built, reps, C):
    """LOLE of (2030 existing fleet + built portfolio), full 8760 x 8 weather years."""
    fleet = C['fleet'].copy()
    add = []
    for (z, ty), mw in built['gen'].items():
        if ty in ('SolarPV', 'WindOn'):
            continue
        add.append(dict(zone=z, Type=ty, **{'Pmax (MW)': mw},
                        n_units=max(1, int(round(mw/UNIT_NEW[ty]))),
                        q=FOR_NEW[ty], mttr=MTTR_NEW[ty]))
    rows = [r for r in fleet[['Type','Pmax (MW)','n_units']].to_dict('records')
            if r['Type'] not in ('WindOn','SolarPV')] + add
    # new VRE uses the case-file (undervated) profile, matching what the LP saw
    g, c = D['g'], D['c']
    NE = len(g)
    newvre = {}
    for (z, ty), mw in built['gen'].items():
        if ty in ('SolarPV','WindOn'):
            k = c.index[(c.Zone == z) & (c.Type == ty)][0]
            newvre[(z, ty)] = mw * D['A'][:, NE+k]
    add_vre = sum(newvre.values()) if newvre else np.zeros(H)
    sto_p = C['sto_p'] + sum(built['sto'].values())
    sto_e = C['sto_e'] + sum(built['sto'].values())*NEW_STO_H
    rng = np.random.default_rng(20260826); days = 0
    for i in range(reps):
        wy = M.YEARS[i % len(M.YEARS)]
        av = M.block_trace(rows, rng, H)
        sh = M.dispatch(C['LOAD'][wy] - C['VRE'][wy] - add_vre - av, sto_p, sto_e)
        days += sh.reshape(365,24).any(1).sum()
    return days/reps, dict(peak=C['peak'], firm=sum(r['Pmax (MW)'] for r in rows),
                           sto_p=sto_p, sto_e=sto_e, new_vre_MW=sum(
                               mw for (z,t),mw in built['gen'].items()
                               if t in ('SolarPV','WindOn')))


# ----------------------------------------------------------------- the loop
if __name__ == '__main__':
    D = load_case()
    C = M.build_case(CASE == 'dc')          # 8-weather-year load/VRE for the Monte Carlo
    D['peak_override'] = C['peak']          # size the RA constraint on the same peak
    days, wts = rep_days(D, NDAYS)
    print(f'annual peak: case file {D["L"].sum(1).max():,.0f} MW (2025 weather) -> '
          f'{C["peak"]:,.0f} MW (worst of 8 weather years, used for RA)')
    print(f'case={CASE}  {len(days)} representative days (weights {wts.sum():.0f} d), '
          f'{REPS} MC replications')
    prm, hist = float(D['sp']['planning_reserve_margin']), []
    lo_p, hi_p = None, None
    for it in range(1, 9):
        b = expand(D, prm, days, wts, verbose=True)
        lole, info = mc(D, b, REPS, C)
        newg = sum(b['gen'].values()); news = sum(b['sto'].values())
        mix = {}
        for (z, t), mw in b['gen'].items(): mix[t] = mix.get(t, 0)+mw
        mix['Storage4h'] = news
        print(f'  it{it}  PRM {prm:7.3%}   built {newg+news:>8,.0f} MW '
              f'{ {k:round(v/1000,1) for k,v in sorted(mix.items())} }   LOLE {lole:7.3f}')
        hist.append(dict(it=it, PRM=prm, LOLE=lole, built=mix,
                         capex_Byr=b['capex']/1e9, obj_Byr=b['obj']/1e9,
                         firm_MW=info['firm'], peak_MW=info['peak'],
                         sto_MW=info['sto_p'], new_vre_MW=info['new_vre_MW']))
        if abs(lole-0.1) < 0.02 and it > 1:
            print('  converged'); break
        if lole > 0.1: lo_p = prm
        else:          hi_p = prm
        if lo_p is not None and hi_p is not None:
            prm = 0.5*(lo_p+hi_p)
        elif lole > 0.1:
            prm += max(0.01, 0.02*np.log10(max(lole/0.1, 1.1)))
        else:
            prm -= 0.01
    json.dump(hist, open(f'{OUTD}/fixedpoint_{CASE}.json','w'), indent=1)
    print('\n', json.dumps(hist[-1], indent=1))
