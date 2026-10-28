"""polish_confirm pass over the free-siting zone designs (out/S1_free_siting_sites*.json) and the
Georgia decomposition step files (out/georgia_decomp_G*.json), so every design in the package
carries the same final stages.   python3 repolish_s1_decomp.py s1|decomp"""
import os, sys, json, glob, shutil, time
import numpy as np, pandas as pd
import offgrid_mip as om
WHAT = sys.argv[1]; CD = os.environ['OFFGRID_CASE']; R = os.environ['OFFGRID_LOAD']; HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f'{HERE}/out'
om.anchor_capex(); K = ['Recip', 'Aero35', 'Aero48', 'FrameCT', 'CC']
d = pd.read_csv(f'{R}/dc_load_hourly_MW.csv', index_col=0, parse_dates=True); d = d[~((d.index.month == 2) & (d.index.day == 29))].drop(columns=['TOTAL_MW'])


def zonal_cf(zone, typ):
    g = pd.read_csv(f'{CD}/gendata.csv'); c = pd.read_csv(f'{CD}/gendata_candidate.csv'); k = c.index[(c.Zone == zone) & (c.Type == typ)]
    return None if not len(k) else pd.read_csv(f'{CD}/gen_availability_timeseries.csv', usecols=[f'G{len(g)+int(k[0])+1}']).values[:om.H, 0]


def redo(site, mix, s, w, b, allow):
    n0 = np.array([mix.get(k, 0) for k in K])
    n, s, w, b, Lc, C = om.polish_confirm(site, n0, s, w, b, allow=allow)
    return dict(mix={K[i]: int(n[i]) for i in range(5) if n[i]}, installed_MW=float(n @ om.S_MW), solar_MW=s, wind_MW=w, batt_MW=b,
                LOLE=Lc, cost=C['total'], lcoe=C['total']/site.energy, confirm_reps=2000, repolished=True)


if WHAT == 's1':
    GAS = json.load(open(f'{HERE}/zonal_gas_2030.json')); CF_S = {z: zonal_cf(z, 'SolarPV') for z in GAS}; CF_W = {z: zonal_cf(z, 'WindOn') for z in GAS}
    for f in glob.glob(f'{OUT}/S1_free_siting_sites*.json'):
        rows = json.load(open(f)); shutil.copy(f, f.replace('.json', '_prepolish.json'))
        for r in rows:
            load = d[r['site']].values[:om.H]
            for z, zr in r['zones'].items():
                site = om.Site(load, z, CF_S[z], CF_W[z], gas=GAS[z]); t = time.time()
                new = redo(site, zr['mix'], zr['solar_MW'], zr['wind_MW'], zr.get('batt_MW', 0.0), (True, CF_W[z] is not None, True))
                ch = ' CHANGED' if new['mix'] != zr['mix'] or abs(new['cost'] - zr['cost']) > 1 else ''
                print(f"{r['site']} {z:<13} ${zr['lcoe']:.2f} -> ${new['lcoe']:.2f}{ch} [{time.time()-t:.0f}s]", flush=True)
                zr.update(new)
        json.dump(rows, open(f, 'w'), indent=1, default=float); print('written', f, flush=True)
else:
    GA_GAS, GA_LAND, TX_GAS, TX_LAND = 5.688, 4670.0, 3.953, 2560.0
    scf = zonal_cf('Coast', 'SolarPV'); scf = scf*(0.265*0.88/scf.mean()); wcf = zonal_cf('North', 'WindOn')
    for st in ('G1', 'G2', 'G3', 'G4'):
        f = f'{OUT}/georgia_decomp_{st}.json'
        if not os.path.exists(f):
            continue
        D = json.load(open(f)); shutil.copy(f, f.replace('.json', '_prepolish.json'))
        gas, land, wind = D['gas'], D['land'], D['wind']; tot = dict(cost=0.0, mw=0.0, w=0.0, s=0.0, b=0.0)
        for srec in D['sites']:
            v = d[srec['site']].values[:om.H]; load = v*(srec['peak_MW']/v.max())
            site = om.Site(load, 'Georgia', scf, wcf if wind else None, gas=gas, land=land); t = time.time()
            new = redo(site, srec['mix'], srec['solar_MW'], srec['wind_MW'], srec.get('batt_MW', 0.0), (True, bool(wind), True))
            ch = ' CHANGED' if new['mix'] != srec['mix'] or abs(new['cost'] - srec['cost']) > 1 else ''
            print(f"{st} {srec['site']:<6} ${srec['lcoe']:.2f} -> ${new['lcoe']:.2f}{ch} [{time.time()-t:.0f}s]", flush=True)
            srec.update(new)
            tot['cost'] += new['cost']; tot['mw'] += new['installed_MW']; tot['w'] += new['wind_MW']; tot['s'] += new['solar_MW']; tot['b'] += new['batt_MW']
        # G2-G4 keep the 31-campus fleet: the G0 design of the same campus (a confirmed MIP design
        # on the same load and machines) is a feasible design for the step as well, so the step's
        # cost is min(re-optimised design, G0 design re-costed at the step's gas/land). The land step
        # is ~0.1% of cost, below the search's own noise (~1%), and without this a cheaper land price
        # came out dearer (+0.85 $/MWh). Both candidates are designs of the same method.
        if st != 'G1':
            G0S = json.load(open(f'{OUT}/offgrid_mip_georgia_sites.json')); n_g0 = 0
            tot = dict(cost=0.0, mw=0.0, w=0.0, s=0.0, b=0.0)
            for i, srec in enumerate(D['sites']):
                g0 = G0S[i]; v = d[srec['site']].values[:om.H]; load = v*(g0['peak_MW']/v.max())
                site = om.Site(load, 'Georgia', scf, wcf if wind else None, gas=gas, land=land)
                n0 = np.array([g0['mix'].get(k, 0) for k in K]); C0 = om.cost_of(site, n0, g0['solar_MW'], 0.0, g0['batt_MW'])
                if C0['total'] < srec['cost']:
                    n_g0 += 1
                    srec.update(mix=g0['mix'], installed_MW=g0['installed_MW'], wind_MW=0.0, solar_MW=g0['solar_MW'], batt_MW=g0['batt_MW'],
                                LOLE=g0['LOLE'], cost=C0['total'], lcoe=C0['total']/site.energy, from_G0_design=True)
                tot['cost'] += srec['cost']; tot['mw'] += srec['installed_MW']; tot['w'] += srec['wind_MW']; tot['s'] += srec['solar_MW']; tot['b'] += srec['batt_MW']
            print(f'{st}: {n_g0} campuses keep their G0 design (cheaper at this step)', flush=True)
        D.update(cost=tot['cost'], installed_MW=tot['mw'], wind_MW=tot['w'], solar_MW=tot['s'], batt_MW=tot['b'],
                 lcoe=tot['cost']/D['energy_MWh'], ratio=tot['mw']/D['peak_MW'])
        json.dump(D, open(f, 'w'), indent=1, default=float); print('written', f, flush=True)

if WHAT == 'decomp' and all(os.path.exists(f'{OUT}/georgia_decomp_{st}.json') for st in ('G1', 'G2', 'G3', 'G4')):
    S = json.load(open(f'{OUT}/offgrid_mip_georgia_sites.json'))
    c = sum(r['cost'] for r in S); e = sum(r['energy_MWh'] for r in S); mw = sum(r['installed_MW'] for r in S); p = sum(r['peak_MW'] for r in S)
    G0 = dict(tag='G0  Georgia as it is', n_campus=len(S), gas=GA_GAS, land=GA_LAND, wind=False, installed_MW=mw, peak_MW=p, cost=c,
              energy_MWh=e, lcoe=c/e, ratio=mw/p, wind_MW=0.0, solar_MW=sum(r['solar_MW'] for r in S), batt_MW=sum(r['batt_MW'] for r in S), method='mip')
    R_ = [G0]
    for st in ('G1', 'G2', 'G3', 'G4'):
        D = json.load(open(f'{OUT}/georgia_decomp_{st}.json')); D.pop('sites', None); R_.append(D)
    E = json.load(open(f'{OUT}/offgrid_mip_final.json')); ref = sum(r['cost'] for r in E)/sum(r['energy_MWh'] for r in E)
    json.dump(dict(cases=R_, ercot_reference_lcoe=ref, note='MIP designs (lazy LOLE cuts, exact neighbourhood improvement, 2,000-rep '
              'confirmation); each step changes ONE parameter from the Georgia case towards ERCOT, holding the rest Georgian; steps are '
              'measured against G0 and are not constrained to sum to the total'), open(f'{OUT}/georgia_decomp.json', 'w'), indent=1, default=float)
    for r in R_:
        print(f'{r["tag"]:<34} {r["lcoe"]:8.2f} {r["lcoe"]-R_[0]["lcoe"]:+8.2f}')
