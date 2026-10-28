"""GAS-ONLY variant: no solar/wind/battery; the wind lever (G3) is dropped.
Georgia -> ERCOT, one parameter at a time, with the MIP (frontier-search version:
Figures/fig5_package/02_code/82_georgia_decomp.py). G0 is read from the Georgia run.

    python3 run_offgrid_mip_georgia_decomp.py [G1|G2|G3|G4]    # one step (parallel), or all
    -> out/georgia_decomp_<step>.json, and out/georgia_decomp.json when all four exist
"""
import os, sys, json, time
import numpy as np, pandas as pd
import offgrid_mip as om

CD = os.environ.get('BYONG_CASE'); R = os.environ.get('BYONG_LOAD')
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f'{HERE}/out'
GA_GAS, GA_LAND, TX_GAS, TX_LAND = 5.688, 4670.0, 3.953, 2560.0
GA_SOLAR_CF, FLEET_MW, GA_N, CAMELLIA = 0.265*0.88, 12400.0, 31, 3200.0
om.anchor_capex()
KW = dict(credit_cap=1.0, vre_cut_frac=0.10, sens_cuts=True)


def prof(zone, typ):
    g = pd.read_csv(f'{CD}/gendata.csv'); c = pd.read_csv(f'{CD}/gendata_candidate.csv')
    k = c.index[(c.Zone == zone) & (c.Type == typ)][0]
    return pd.read_csv(f'{CD}/gen_availability_timeseries.csv', usecols=[f'G{len(g)+int(k)+1}']).values[:om.H, 0]


SCF_GA = prof('Coast', 'SolarPV'); SCF_GA = SCF_GA*(GA_SOLAR_CF/SCF_GA.mean())
WCF_TX = prof('North', 'WindOn')
d = pd.read_csv(f'{R}/dc_load_hourly_MW.csv', index_col=0, parse_dates=True)
d = d[~((d.index.month == 2) & (d.index.day == 29))].drop(columns=['TOTAL_MW'])
pk = d.max(); shape = (pk/pk.sum()).sort_values(ascending=False); src = list(shape.index)


def fleet(n_campus):
    if n_campus == len(src):
        return [(s, float(FLEET_MW*shape[s])) for s in src]
    rest = FLEET_MW - CAMELLIA; q = np.linspace(0, 1, n_campus - 1)
    w = np.interp(q, np.linspace(0, 1, len(src)), [shape[x] for x in src]); w = w/w.sum()
    F = 100.0/rest
    for _ in range(50):
        lo = w < F
        if not lo.any():
            break
        w = np.where(lo, F, w*(1 - lo.sum()*F)/w[~lo].sum())
    ar = [src[min(int(round(x*(len(src)-1))), len(src)-1)] for x in q]
    return [(src[0], CAMELLIA)] + [(ar[i], float(rest*w[i])) for i in range(n_campus - 1)]


def run(tag, n_campus, gas, land, wind):
    t0 = time.time(); tot = dict(cost=0.0, e=0.0, mw=0.0, pk=0.0, wind=0.0, solar=0.0, batt=0.0); sites = []
    for site, mw in fleet(n_campus):
        v = d[site].values[:om.H]; load = v*(mw/v.max())
        r = om.design(load, 'Georgia', SCF_GA, WCF_TX if wind else None, gas=gas, land=land,
                      allow=(False, False, False), **KW)
        if r is None:
            print(f'   {tag}: no design for {mw:,.0f} MW', flush=True); continue
        tot['cost'] += r['cost']; tot['e'] += r['energy_MWh']; tot['mw'] += r['installed_MW']; tot['pk'] += r['peak_MW']
        tot['wind'] += r['wind_MW']; tot['solar'] += r['solar_MW']; tot['batt'] += r['batt_MW']
        sites.append(dict(site=site, peak_MW=r['peak_MW'], mix=r['mix'], installed_MW=r['installed_MW'],
                          wind_MW=r['wind_MW'], solar_MW=r['solar_MW'], batt_MW=r['batt_MW'], LOLE=r['LOLE'],
                          cost=r['cost'], lcoe=r['lcoe']))
        print(f"   {tag[:3]} {site:<6}{mw:>7,.0f} MW  {r['mix']}  w{r['wind_MW']:.0f} s{r['solar_MW']:.0f} b{r['batt_MW']:.0f}  "
              f"LOLE {r['LOLE']:.3f}  ${r['lcoe']:.2f}", flush=True)
    lc = tot['cost']/tot['e']
    print(f"{tag:<34} {n_campus:3d} campuses  gas ${gas:.3f}  land ${land:,.0f}  wind {'yes' if wind else 'no '}  "
          f"->  {tot['mw']:8,.0f} MW ({tot['mw']/tot['pk']:.2f}x)  ${lc:7.2f}/MWh   [{time.time()-t0:.0f}s]", flush=True)
    return dict(tag=tag, n_campus=n_campus, gas=gas, land=land, wind=wind, installed_MW=tot['mw'],
                peak_MW=tot['pk'], cost=tot['cost'], energy_MWh=tot['e'], lcoe=lc, ratio=tot['mw']/tot['pk'],
                wind_MW=tot['wind'], solar_MW=tot['solar'], batt_MW=tot['batt'], sites=sites, method='mip')


CASES = {'G1': ('G1  + ERCOT campus sizes', len(src), GA_GAS, GA_LAND, False),
         'G2': ('G2  + ERCOT cheapest gas', GA_N, TX_GAS, GA_LAND, False),
         'G4': ('G4  + ERCOT land', GA_N, GA_GAS, TX_LAND, False)}
steps = sys.argv[1:] or list(CASES)
for st in steps:
    if os.path.exists(f'{OUT}/georgia_decomp_gasonly_{st}.json'):
        continue
    json.dump(run(*CASES[st]), open(f'{OUT}/georgia_decomp_gasonly_{st}.json', 'w'), indent=1, default=float)

if all(os.path.exists(f'{OUT}/georgia_decomp_gasonly_{st}.json') for st in CASES) and os.path.exists(f'{OUT}/offgrid_mip_georgia_gasonly_sites.json'):
    S = json.load(open(f'{OUT}/offgrid_mip_georgia_gasonly_sites.json'))
    c = sum(r['cost'] for r in S); e = sum(r['energy_MWh'] for r in S); mw = sum(r['installed_MW'] for r in S); p = sum(r['peak_MW'] for r in S)
    G0 = dict(tag='G0  Georgia as it is', n_campus=len(S), gas=GA_GAS, land=GA_LAND, wind=False, installed_MW=mw,
              peak_MW=p, cost=c, energy_MWh=e, lcoe=c/e, ratio=mw/p, wind_MW=0.0,
              solar_MW=sum(r['solar_MW'] for r in S), batt_MW=sum(r['batt_MW'] for r in S), method='mip')
    R_ = [G0] + [json.load(open(f'{OUT}/georgia_decomp_gasonly_{st}.json')) for st in CASES]
    for r in R_:
        r.pop('sites', None)
    E = json.load(open(f'{OUT}/offgrid_mip_gasonly.json')); ref = sum(r['cost'] for r in E)/sum(r['energy_MWh'] for r in E)
    json.dump(dict(cases=R_, ercot_reference_lcoe=ref,
                   note='MIP designs (lazy LOLE cuts, exact neighbourhood improvement, 2,000-rep confirmation); each step changes ONE parameter from the Georgia case towards ERCOT, holding '
                        'the rest Georgian; steps are measured against G0 and are not constrained to sum to the total'),
              open(f'{OUT}/georgia_decomp_gasonly.json', 'w'), indent=1, default=float)
    for r in R_:
        print(f'{r["tag"]:<34} {r["lcoe"]:8.2f} {r["lcoe"]-R_[0]["lcoe"]:+8.2f}')
    print(f'written {OUT}/georgia_decomp_gasonly.json')
