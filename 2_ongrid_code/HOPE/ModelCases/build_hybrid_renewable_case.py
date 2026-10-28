"""Build the grid-connected HOPE case for the pure-renewable hybrid.

The 17 data centers are islanded on solar, wind and batteries only (the pure-renewable off-grid
design, 3_offgrid_code/scenario_renewable_only/renewable_only_sites.json) until the grid arrives,
then connect.  As in build_hybrid_diesel_cases.py, the plant the data centers built enters HOPE as
EXISTING capacity (no capital in the objective), at its zone, with the zonal candidate profiles the
design was sized on; the batteries keep their own power and energy (free duration).  HOPE then
decides what else the system builds.  The sunk capital is added outside the model by
3_offgrid_code/summarize_hybrid_renewable.py.

    python build_hybrid_renewable_case.py   ->  TX8z_GTEP_hybrid_ren4h_vre_case/  (+ btm_manifest.json)
Run HOPE on it exactly as for TX8z_GTEP_hybrid_dsl060_vre_case, then
    python ../../../3_offgrid_code/summarize_hybrid_renewable.py
"""
import json, os, shutil
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
SRC, SRC_DATA = os.path.join(HERE, 'TX8z_GTEP_dc_vre_case'), 'Data_TX8z_GTEP_dc_vre'
# : the 4-hour-battery design (run_renewable_only.py with OFFGRID_BATT_HOURS=4); E = 4P exactly.
DESIGN = f'{PROJECT_ROOT}/3_offgrid_code/scenario_renewable_only/renewable_only_4h_sites.json'
DST, DST_DATA = os.path.join(HERE, 'TX8z_GTEP_hybrid_ren4h_vre_case'), 'Data_TX8z_GTEP_hybrid_ren4h_vre'


def zonal_cf(case_data, zone, tech):
    g = pd.read_csv(f'{case_data}/gendata.csv'); c = pd.read_csv(f'{case_data}/gendata_candidate.csv')
    k = c.index[(c.Zone == zone) & (c.Type == tech)]
    av = pd.read_csv(f'{case_data}/gen_availability_timeseries.csv', usecols=[f'G{len(g)+int(k[0])+1}'])
    return av.values[:8760, 0]


def main():
    sites = json.load(open(DESIGN))
    fleet = {}
    for s in sites:
        d = fleet.setdefault(s['zone'], dict(wind=0.0, solar=0.0, batt_MW=0.0, batt_MWh=0.0, sites=[]))
        d['wind'] += s['wind_MW']; d['solar'] += s['solar_MW']
        d['batt_MW'] += s['batt_MW']; d['batt_MWh'] += s['batt_MWh']; d['sites'].append(s['site'])
    if os.path.exists(DST):
        raise SystemExit(f'{DST} exists; move it away first')
    os.makedirs(DST)
    D = f'{DST}/{DST_DATA}'
    shutil.copytree(f'{SRC}/{SRC_DATA}', D, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(f'{SRC}/Settings', f'{DST}/Settings')

    gen = pd.read_csv(f'{D}/gendata.csv'); cand = pd.read_csv(f'{D}/gendata_candidate.csv')
    n_gen0, n_cand = len(gen), len(cand)
    tmpl = {t: gen[gen.Type == t].iloc[0].copy() for t in ('WindOn', 'SolarPV')}
    rows, manifest, profiles = [], [], {}
    for zone in sorted(fleet):
        for kind, tp in (('wind', 'WindOn'), ('solar', 'SolarPV')):
            mw = fleet[zone][kind]
            if mw <= 0.05:
                continue
            r = tmpl[tp].copy()
            r['Pmax (MW)'] = round(mw, 3); r['Pmin (MW)'] = 0.0; r['Zone'] = zone; r['Flag_RET'] = 0
            r['Cost ($/MWh)'] = 0.0; r['EF'] = 0.0; r['AF'] = 1.0; r['FOR'] = 0.0; r['n_units'] = 1
            r['CC'] = float(cand[(cand.Zone == zone) & (cand.Type == tp)].CC.iloc[0])   # the zone's own ELCC
            rows.append(r)
            idx = n_gen0 + len(rows) - 1
            profiles[idx] = zonal_cf(f'{SRC}/{SRC_DATA}', zone, tp)
            manifest.append(dict(row=idx, zone=zone, machine=kind, MW=float(r['Pmax (MW)']), type=tp))
    pd.concat([gen, pd.DataFrame(rows)], ignore_index=True).to_csv(f'{D}/gendata.csv', index=False)
    n_new = len(rows)

    af = pd.read_csv(f'{D}/gen_availability_timeseries.csv')
    cols = {c: af[c] for c in af.columns if not c.startswith('G')}
    for i in range(n_gen0):
        cols[f'G{i+1}'] = af[f'G{i+1}']
    for j in range(n_new):
        cols[f'G{n_gen0+j+1}'] = pd.Series(profiles[n_gen0 + j][:len(af)], index=af.index)
    for j in range(n_cand):
        cols[f'G{n_gen0+n_new+j+1}'] = af[f'G{n_gen0+j+1}']
    out = pd.DataFrame(cols); out.to_csv(f'{D}/gen_availability_timeseries.csv', index=False)
    assert all(np.allclose(af[f'G{n_gen0+j+1}'], out[f'G{n_gen0+n_new+j+1}']) for j in range(n_cand))

    st = pd.read_csv(f'{D}/storagedata.csv'); brows = []
    for zone in sorted(fleet):
        f = fleet[zone]
        if f['batt_MW'] <= 0.05:
            continue
        b = st[st.Zone == zone].iloc[0].copy() if (st.Zone == zone).any() else st.iloc[0].copy()
        b['Zone'] = zone; b['Max Power (MW)'] = round(f['batt_MW'], 3); b['Capacity (MWh)'] = round(4.0*f['batt_MW'], 3)          # 4-hour batteries
        assert abs(f['batt_MWh'] - 4.0*f['batt_MW']) < 1e-3*max(f['batt_MW'], 1), 'design is not 4-hour'
        b['n_units'] = 1
        brows.append(b)
        manifest.append(dict(zone=zone, machine='battery', MW=f['batt_MW'], MWh=f['batt_MWh'], type='Battery'))
    pd.concat([st, pd.DataFrame(brows)], ignore_index=True).to_csv(f'{D}/storagedata.csv', index=False)

    sp1 = pd.read_csv(f'{D}/single_parameter.csv')          # same PRM as the diesel hybrid cases (run_hybrid_diesel_cases.jl)
    sp1.loc[0, 'planning_reserve_margin'] = 0.13043960378211986; sp1.to_csv(f'{D}/single_parameter.csv', index=False)
    sp = f'{DST}/Settings/HOPE_model_settings.yml'
    txt = open(sp).read()                                    # read first: open(..., 'w') truncates
    assert f'DataCase: {SRC_DATA}/' in txt
    open(sp, 'w').write(txt.replace(f'DataCase: {SRC_DATA}/', f'DataCase: {DST_DATA}/'))
    json.dump(dict(case=os.path.basename(DST), data_dir=DST_DATA, design=os.path.relpath(DESIGN, PROJECT_ROOT),
                   n_existing_rows_before=n_gen0, n_btm_rows=n_new, n_candidate_rows=n_cand,
                   wind_MW=sum(f['wind'] for f in fleet.values()), solar_MW=sum(f['solar'] for f in fleet.values()),
                   batt_MW=sum(f['batt_MW'] for f in fleet.values()), batt_MWh=sum(f['batt_MWh'] for f in fleet.values()),
                   units=manifest), open(f'{DST}/btm_manifest.json', 'w'), indent=1)
    print(f'built {os.path.basename(DST)}: {n_new} renewable rows, {len(brows)} battery rows; candidates now '
          f'G{n_gen0+n_new+1}..G{n_gen0+n_new+n_cand}')


if __name__ == '__main__':
    main()
