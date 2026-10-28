"""Phase-2 (grid-connected) HOPE cases for the diesel-first hybrid on the paper's OTHER cases.

    python3 build_hybrid_diesel_cases_other.py 2035_k0p5      # 20 GW fleet  -> TX8z_GTEP_hyb20_dsl<XXX>_case
    python3 build_hybrid_diesel_cases_other.py 2035_k3        # 121 GW fleet -> TX8z_GTEP_hyb120_dsl<XXX>_case
    python3 build_hybrid_diesel_cases_other.py georgia        # Georgia      -> GA1z_GTEP_hyb_dsl<XXX>_case
    (optional second argument: shares, e.g. 0.3,0.6,1.0; default every share in all_sites.json incl. 0)

Same construction as build_hybrid_diesel_cases.py (the campus gas fleet enters the on-grid
case of that scenario as EXISTING capacity at its zonal fuel cost; the gensets as existing
rows at ~$197/MWh, i.e. backup that never dispatches), with the source case and the fuel
prices of the scenario:
    2035_k0p5 : TX8z_GTEP_dc20_case  / Data_TX8z_GTEP_dc20,  2035 zonal gas (as the MIP designs)
    2035_k3   : TX8z_GTEP_dc120_case / Data_TX8z_GTEP_dc120, 2035 zonal gas
    georgia   : GA1z_GTEP_dc_case    / Data_GA1z_GTEP_dc,    gas $5.688 (zone 'Georgia')
Designs: 3_offgrid_code/out/hybrid_diesel_<tag>/all_sites.json (run_hybrid_diesel_first_cases.py).
Then run HOPE on each case (run_hybrid_diesel_cases_other.jl) and
    python3 ../../../3_offgrid_code/summarize_hybrid_diesel_other.py <tag>
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

import build_hybrid_diesel_cases as B

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
TAG = sys.argv[1]
SPEC = {
    '2035_k0p5': dict(src='TX8z_GTEP_dc20_case', data='Data_TX8z_GTEP_dc20', dst='TX8z_GTEP_hyb20_dsl{tag}_case',
                      dst_data='Data_TX8z_GTEP_hyb20_dsl{tag}',
                      gas={'Coast': 5.0, 'East': 4.866, 'FarWest': 3.997, 'NorthCentral': 5.187, 'South': 5.348,
                           'SouthCentral': 5.044, 'West': 4.501, 'North': 4.168}),
    '2035_k3': dict(src='TX8z_GTEP_dc120_case', data='Data_TX8z_GTEP_dc120', dst='TX8z_GTEP_hyb120_dsl{tag}_case',
                    dst_data='Data_TX8z_GTEP_hyb120_dsl{tag}',
                    gas={'Coast': 5.0, 'East': 4.866, 'FarWest': 3.997, 'NorthCentral': 5.187, 'South': 5.348,
                         'SouthCentral': 5.044, 'West': 4.501, 'North': 4.168}),
    'georgia': dict(src='GA1z_GTEP_dc_case', data='Data_GA1z_GTEP_dc', dst='GA1z_GTEP_hyb_dsl{tag}_case',
                    dst_data='Data_GA1z_GTEP_hyb_dsl{tag}', gas={'Georgia': 5.688}),
}[TAG]
DESIGNS = f'{PROJECT_ROOT}/3_offgrid_code/out/hybrid_diesel_{TAG}/all_sites.json'


def build(sites, key, gas, dsl):
    x = float(key)
    tag = f'{int(round(100*x)):03d}'
    SRC, SRC_DATA = os.path.join(HERE, SPEC['src']), SPEC['data']
    DST = os.path.join(HERE, SPEC['dst'].format(tag=tag)); DST_DATA = SPEC['dst_data'].format(tag=tag)
    fleet = B.zone_fleet(sites, key)

    os.makedirs(DST, exist_ok=True)
    D = f'{DST}/{DST_DATA}'
    shutil.copytree(f'{SRC}/{SRC_DATA}', D, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*BEFORE*'))
    shutil.copytree(f'{SRC}/Settings', f'{DST}/Settings', dirs_exist_ok=True)
    for f in ('gendata.csv', 'gen_availability_timeseries.csv', 'storagedata.csv'):
        shutil.copyfile(f'{SRC}/{SRC_DATA}/{f}', f'{D}/{f}')

    gen = pd.read_csv(f'{D}/gendata.csv')
    cand = pd.read_csv(f'{D}/gendata_candidate.csv')
    n_gen0, n_cand = len(gen), len(cand)
    tmpl = {t: gen[gen.Type == t].iloc[0].copy() for t in ('NGCC', 'NGCT') if (gen.Type == t).any()}
    if 'NGCT' not in tmpl:
        tmpl['NGCT'] = tmpl['NGCC'].copy(); tmpl['NGCT']['Type'] = 'NGCT'
    if 'NGCC' not in tmpl:
        tmpl['NGCC'] = tmpl['NGCT'].copy(); tmpl['NGCC']['Type'] = 'NGCC'

    rows, manifest = [], []
    for zone in sorted(fleet):
        f, g = fleet[zone], gas[zone]
        for i in range(5):
            n = int(round(f['n'][i]))
            if n <= 0:
                continue
            c = B.CAT[i]
            r = tmpl[c['type']].copy()
            r['Pmax (MW)'] = c['S']*n; r['Pmin (MW)'] = 0.0; r['Zone'] = zone; r['Flag_RET'] = 0
            r['Cost ($/MWh)'] = round(c['vom'] + c['hr']*g, 4)
            r['EF'] = round(c['hr']*B.CO2_GAS, 6)
            r['CC'] = 1.0; r['AF'] = 1.0; r['FOR'] = c['q']; r['n_units'] = n
            rows.append(r)
            manifest.append(dict(row=n_gen0 + len(rows) - 1, zone=zone, machine=c['k'], units=n,
                                 MW=c['S']*n, type=c['type'], var_cost_USD_MWh=float(r['Cost ($/MWh)'])))
        nd = int(round(f['n'][5]))
        if nd > 0 and B.DSL_MODE == 'backup':
            r = tmpl['NGCT'].copy()
            r['Pmax (MW)'] = dsl['S']*nd; r['Pmin (MW)'] = 0.0; r['Zone'] = zone; r['Flag_RET'] = 0
            r['Cost ($/MWh)'] = round(dsl['vom'] + dsl['hr']*dsl['fuel'], 4)
            r['EF'] = round(dsl['hr']*B.CO2_DSL, 6)
            r['CC'] = 1.0; r['AF'] = 1.0; r['FOR'] = dsl['q']; r['n_units'] = nd
            rows.append(r)
            manifest.append(dict(row=n_gen0 + len(rows) - 1, zone=zone, machine='Diesel', units=nd,
                                 MW=dsl['S']*nd, type='NGCT', var_cost_USD_MWh=float(r['Cost ($/MWh)'])))

    gen2 = pd.concat([gen, pd.DataFrame(rows)], ignore_index=True)
    gen2.to_csv(f'{D}/gendata.csv', index=False)
    n_new = len(rows)

    af = pd.read_csv(f'{D}/gen_availability_timeseries.csv')
    cols = {c: af[c] for c in af.columns if not c.startswith('G')}
    for i in range(n_gen0):
        cols[f'G{i+1}'] = af[f'G{i+1}']
    for j in range(n_new):
        cols[f'G{n_gen0+j+1}'] = pd.Series(1.0, index=af.index)
    for j in range(n_cand):
        cols[f'G{n_gen0+n_new+j+1}'] = af[f'G{n_gen0+j+1}']
    out = pd.DataFrame(cols)
    out.to_csv(f'{D}/gen_availability_timeseries.csv', index=False)
    src_af = pd.read_csv(f'{SRC}/{SRC_DATA}/gen_availability_timeseries.csv')
    assert all(np.allclose(src_af[f'G{i+1}'], out[f'G{i+1}']) for i in range(n_gen0))
    assert all(np.allclose(src_af[f'G{n_gen0+j+1}'], out[f'G{n_gen0+n_new+j+1}']) for j in range(n_cand))

    sp = f'{DST}/Settings/HOPE_model_settings.yml'
    txt = open(sp).read().replace(f'DataCase: {SRC_DATA}/', f'DataCase: {DST_DATA}/')
    assert f'DataCase: {DST_DATA}/' in txt
    open(sp, 'w').write(txt)

    tot = sum(m['MW'] for m in manifest)
    json.dump(dict(case=os.path.basename(DST), scenario=TAG, source_case=SPEC['src'], data_dir=DST_DATA, diesel_share=x,
                   diesel_mode=B.DSL_MODE, n_existing_rows_before=n_gen0, n_btm_rows=n_new, n_candidate_rows=n_cand,
                   total_btm_MW=tot, units=manifest), open(f'{DST}/btm_manifest.json', 'w'), indent=1)
    gmw = sum(m['MW'] for m in manifest if m['machine'] != 'Diesel')
    dmw = sum(m['MW'] for m in manifest if m['machine'] == 'Diesel')
    print(f'built {os.path.basename(DST)}: gas {gmw:,.0f} MW, diesel {dmw:,.0f} MW, {n_new} rows; '
          f'candidates now G{n_gen0+n_new+1}..G{n_gen0+n_new+n_cand}')


if __name__ == '__main__':
    sites = json.load(open(DESIGNS))
    dsl = dict(sites[0]['diesel']); dsl['fuel'] = sites[0]['diesel_fuel']

    def isnum(k):
        try:
            float(k); return True
        except ValueError:
            return False
    keys = sorted((k for k in sites[0]['cases'] if isnum(k)), key=float)
    if len(sys.argv) > 2:
        keys = [f'{float(s):.1f}' for s in sys.argv[2].split(',')]
    for k in keys:
        build(sites, k, SPEC['gas'], dsl)
