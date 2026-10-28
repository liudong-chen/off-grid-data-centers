"""Build the Phase-2 (grid-connected) HOPE cases for the diesel-bridge sensitivity.

One HOPE case per diesel share x, INCLUDING x = 0 (dsl000: the gas-only off-grid fleet,
the main off-grid reference  -- no campus wind/solar/battery).  Same construction as build_hybrid_case.py (the
gas fleet the campus built during the bridge enters as EXISTING capacity, so HOPE pays
its fuel but never its capital), with two differences:

  * the gas fleet is the STRIPPED fleet of share x (what is left after x of the gas MW
    was replaced by diesel gensets during the bridge);
  * the diesel gensets are added as existing rows too, at their true variable cost
    (~$197/MWh), so the model keeps them as BACKUP: they will only run in hours the
    rest of the system cannot cover, which with VOLL = $35,000/MWh is essentially never.
    Retiring them instead changes nothing inside HOPE (they never dispatch and the PRM
    is slack in every hybrid case), so retire-vs-backup is settled OUTSIDE the model
    by the diesel FOM (summarize_hybrid_diesel.py).  Set DSL_MODE=retire to leave them
    out of the case anyway.

    python build_hybrid_diesel_cases.py [0.1,0.2,...]        (default: every share found)
    -> TX8z_GTEP_hybrid_dsl<XX>_vre_case/   + btm_manifest.json in each

Then run HOPE on each case exactly as for TX8z_GTEP_hybrid_vre_case, and afterwards
    python ../../OffGrid/summarize_hybrid_diesel.py
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BYONG = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
SRC = os.path.join(HERE, 'TX8z_GTEP_dc_vre_case')
SRC_DATA = 'Data_TX8z_GTEP_dc_vre'
DESIGNS = os.environ.get('HYB_DESIGNS', f'{BYONG}/Code/OffGrid/out/hybrid_diesel/all_sites.json')
PREFIX = os.environ.get('HYB_PREFIX', 'hybrid')   # hyb2p -> connection-aware fleets (out/hybrid_2phase_nocredit)
GASF = f'{BYONG}/Code/OffGrid/zonal_gas_2030.json'
DSL_MODE = os.environ.get('DSL_MODE', 'backup')          # backup | retire

E23, E19 = 1.053, 1.2549
CAT = {   # index order matches OffGridBYONG.CAT; diesel is the sixth
    0: dict(k='Recip',   S=20.0,  hr=8.295, vom=5.69*E19, q=0.030,  type='NGCT'),
    1: dict(k='Aero35',  S=35.0,  hr=9.447, vom=5.70*E23, q=0.0476, type='NGCT'),
    2: dict(k='Aero48',  S=48.0,  hr=9.100, vom=5.70*E23, q=0.0476, type='NGCT'),
    3: dict(k='FrameCT', S=419.0, hr=9.142, vom=1.24*E23, q=0.0476, type='NGCT'),
    4: dict(k='CC',      S=627.0, hr=6.226, vom=3.33*E23, q=0.0357, type='NGCC'),
}
CO2_GAS, CO2_DSL = 0.05306, 0.07413      # t/MMBtu
BATT_HOURS = 4.0


def zonal_cf(case_data, zone, tech):
    """The zonal candidate profile HOPE (and the MIP designs) used for this technology."""
    g = pd.read_csv(f'{case_data}/gendata.csv')
    c = pd.read_csv(f'{case_data}/gendata_candidate.csv')
    k = c.index[(c.Zone == zone) & (c.Type == tech)]
    if not len(k):
        return None
    av = pd.read_csv(f'{case_data}/gen_availability_timeseries.csv', usecols=[f'G{len(g)+int(k[0])+1}'])
    return av.values[:8760, 0]


def zone_fleet(sites, key):
    z = {}
    for r in sites:
        c = r['cases'][key]
        d = z.setdefault(r['zone'], dict(n=np.zeros(6), wind=0.0, solar=0.0, batt=0.0, peak=0.0, sites=[]))
        d['n'] += np.array(c['n'], float)
        if r.get('vre_kept', True):               # main case (27 Sep): gas + diesel only
            d['wind'] += r['wind_MW'] or 0.0
            d['solar'] += r['solar_MW'] or 0.0
            d['batt'] += r['batt_MW'] or 0.0
        d['peak'] += r['peak_MW']
        d['sites'].append(r['site'])
    return z


def build(sites, key, gas, dsl):
    x = float(key)
    tag = f'dsl{int(round(100*x)):03d}' + ('' if DSL_MODE == 'backup' else '_retire')
    DST = os.path.join(HERE, f'TX8z_GTEP_{PREFIX}_{tag}_vre_case')
    DST_DATA = f'Data_TX8z_GTEP_{PREFIX}_{tag}_vre'
    fleet = zone_fleet(sites, key)

    os.makedirs(DST, exist_ok=True)
    D = f'{DST}/{DST_DATA}'
    shutil.copytree(f'{SRC}/{SRC_DATA}', D, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(f'{SRC}/Settings', f'{DST}/Settings', dirs_exist_ok=True)
    stale = f'{DST}/_from_dc_case_DELETE_ME'
    for d in os.listdir(DST):
        if d.startswith('output') or d == SRC_DATA:
            os.makedirs(stale, exist_ok=True)
            if not os.path.exists(f'{stale}/{d}'):
                os.rename(f'{DST}/{d}', f'{stale}/{d}')
    for f in ('gendata.csv', 'gen_availability_timeseries.csv', 'storagedata.csv'):
        shutil.copyfile(f'{SRC}/{SRC_DATA}/{f}', f'{D}/{f}')
    shutil.copyfile(f'{SRC}/Settings/HOPE_model_settings.yml', f'{DST}/Settings/HOPE_model_settings.yml')

    gen = pd.read_csv(f'{D}/gendata.csv')
    cand = pd.read_csv(f'{D}/gendata_candidate.csv')
    n_gen0, n_cand = len(gen), len(cand)
    tmpl = {t: gen[gen.Type == t].iloc[0].copy() for t in ('NGCC', 'NGCT', 'WindOn', 'SolarPV')}

    rows, manifest, profiles = [], [], {}
    for zone in sorted(fleet):
        f, g = fleet[zone], gas[zone]
        for i in range(5):
            n = int(round(f['n'][i]))
            if n <= 0:
                continue
            c = CAT[i]
            r = tmpl[c['type']].copy()
            r['Pmax (MW)'] = c['S']*n; r['Pmin (MW)'] = 0.0; r['Zone'] = zone; r['Flag_RET'] = 0
            r['Cost ($/MWh)'] = round(c['vom'] + c['hr']*g, 4)
            r['EF'] = round(c['hr']*CO2_GAS, 6)
            r['CC'] = 1.0; r['AF'] = 1.0; r['FOR'] = c['q']; r['n_units'] = n
            rows.append(r)
            manifest.append(dict(row=n_gen0 + len(rows) - 1, zone=zone, machine=c['k'], units=n,
                                 MW=c['S']*n, type=c['type'], var_cost_USD_MWh=float(r['Cost ($/MWh)'])))
        nd = int(round(f['n'][5]))
        if nd > 0 and DSL_MODE == 'backup':
            r = tmpl['NGCT'].copy()                 # standard Type label; identified via the manifest
            r['Pmax (MW)'] = dsl['S']*nd; r['Pmin (MW)'] = 0.0; r['Zone'] = zone; r['Flag_RET'] = 0
            r['Cost ($/MWh)'] = round(dsl['vom'] + dsl['hr']*dsl['fuel'], 4)
            r['EF'] = round(dsl['hr']*CO2_DSL, 6)
            r['CC'] = 1.0; r['AF'] = 1.0; r['FOR'] = dsl['q']; r['n_units'] = nd
            rows.append(r)
            manifest.append(dict(row=n_gen0 + len(rows) - 1, zone=zone, machine='Diesel', units=nd,
                                 MW=dsl['S']*nd, type='NGCT', var_cost_USD_MWh=float(r['Cost ($/MWh)'])))
        for kind, tp in (('wind', 'WindOn'), ('solar', 'SolarPV')):
            mw = f[kind]
            if mw <= 0.05:
                continue
            r = tmpl[tp].copy()
            r['Pmax (MW)'] = round(mw, 3); r['Pmin (MW)'] = 0.0; r['Zone'] = zone; r['Flag_RET'] = 0
            r['Cost ($/MWh)'] = 0.0; r['EF'] = 0.0; r['AF'] = 1.0; r['FOR'] = 0.0; r['n_units'] = 1
            rows.append(r)
            idx = n_gen0 + len(rows) - 1
            pr = zonal_cf(f'{SRC}/{SRC_DATA}', zone, tp)
            if pr is None:
                raise SystemExit(f'no candidate {tp} profile for {zone}')
            profiles[idx] = pr
            manifest.append(dict(row=idx, zone=zone, machine=kind, units=1, MW=float(r['Pmax (MW)']),
                                 type=tp, var_cost_USD_MWh=0.0))

    gen2 = pd.concat([gen, pd.DataFrame(rows)], ignore_index=True)
    gen2.to_csv(f'{D}/gendata.csv', index=False)
    n_new = len(rows)

    af = pd.read_csv(f'{D}/gen_availability_timeseries.csv')
    cols = {c: af[c] for c in af.columns if not c.startswith('G')}
    for i in range(n_gen0):
        cols[f'G{i+1}'] = af[f'G{i+1}']
    for j in range(n_new):
        pr = profiles.get(n_gen0 + j)
        cols[f'G{n_gen0+j+1}'] = pd.Series(1.0, index=af.index) if pr is None else pd.Series(pr)
    for j in range(n_cand):
        cols[f'G{n_gen0+n_new+j+1}'] = af[f'G{n_gen0+j+1}']
    out = pd.DataFrame(cols)
    out.to_csv(f'{D}/gen_availability_timeseries.csv', index=False)
    src_af = pd.read_csv(f'{SRC}/{SRC_DATA}/gen_availability_timeseries.csv')
    assert all(np.allclose(src_af[f'G{i+1}'], out[f'G{i+1}']) for i in range(n_gen0))
    assert all(np.allclose(src_af[f'G{n_gen0+j+1}'], out[f'G{n_gen0+n_new+j+1}']) for j in range(n_cand))

    st = pd.read_csv(f'{D}/storagedata.csv')
    brows = []
    for zone in sorted(fleet):
        mw = fleet[zone]['batt']
        if mw <= 0.05:
            continue
        b = st[st.Zone == zone].iloc[0].copy() if (st.Zone == zone).any() else st.iloc[0].copy()
        b['Zone'] = zone; b['Max Power (MW)'] = round(mw, 3); b['Capacity (MWh)'] = round(mw*BATT_HOURS, 3); b['n_units'] = 1
        brows.append(b)
    if brows:
        pd.concat([st, pd.DataFrame(brows)], ignore_index=True).to_csv(f'{D}/storagedata.csv', index=False)

    sp = f'{DST}/Settings/HOPE_model_settings.yml'
    txt = open(sp).read().replace(f'DataCase: {SRC_DATA}/', f'DataCase: {DST_DATA}/')
    open(sp, 'w').write(txt)

    tot = sum(m['MW'] for m in manifest)
    json.dump(dict(case=os.path.basename(DST), data_dir=DST_DATA, diesel_share=x, diesel_mode=DSL_MODE,
                   n_existing_rows_before=n_gen0, n_btm_rows=n_new, n_candidate_rows=n_cand,
                   total_btm_MW=tot, units=manifest), open(f'{DST}/btm_manifest.json', 'w'), indent=1)
    gmw = sum(m['MW'] for m in manifest if m['type'] in ('NGCC', 'NGCT') and m['machine'] != 'Diesel')
    dmw = sum(m['MW'] for m in manifest if m['machine'] == 'Diesel')
    print(f'built {os.path.basename(DST)}: gas {gmw:,.0f} MW, diesel {dmw:,.0f} MW, {n_new} rows; '
          f'candidates now G{n_gen0+n_new+1}..G{n_gen0+n_new+n_cand}')


def main():
    sites = json.load(open(DESIGNS))
    gas = json.load(open(GASF))
    dsl = dict(sites[0]['diesel']); dsl['fuel'] = sites[0]['diesel_fuel']
    def isnum(k):
        try:
            float(k); return True
        except ValueError:
            return False
    keys = sorted((k for k in sites[0]['cases'] if isnum(k)), key=float)   # includes 0.0 -> dsl000
    if len(sys.argv) > 1:
        keys = [f'{float(s):.1f}' for s in sys.argv[1].split(',')]
    for k in keys:
        build(sites, k, gas, dsl)


if __name__ == '__main__':
    main()
