"""Build the ERCOT 2030 HYBRID (bridge) case from the on-grid DC case.

THE SCENARIO.  The campus builds its own plant and runs islanded while it waits in the
interconnection queue.  When the queue clears, campus AND plant energise together: the
campus can buy from the grid, and the plant can sell into it.  This script builds the
POST-CONNECTION world -- Phase 2 -- as a HOPE case.

WHAT IS AND IS NOT A DECISION HERE.  The plant is already built and paid for, so it
enters as EXISTING capacity: the model pays its fuel and O&M but never its capital,
which the data centre sank during the bridge.  Everything else stays a decision --
the system is still free to build generation, storage and transmission on top.  Three
objectives then bracket the whole question:

    obj_base    no data centres                     (TX8z_GTEP_base_vre_case)
    obj_dc      DC load, grid builds everything     (TX8z_GTEP_dc_vre_case)
    obj_hybrid  DC load + the sunk off-grid fleet   (this case)

    value the fleet returns to the system = obj_dc - obj_hybrid
    societal cost of the hybrid           = (obj_hybrid - obj_base) + fleet capital

THE FLEET.  Exactly the published off-grid design -- the same 17 campus plants that
solve to LOLE <= 0.1 islanded -- aggregated to one row per zone and machine class.
Nothing is re-optimised: the hybrid differs from the off-grid case only in what happens
after the queue clears, which is what makes the comparison clean.

TWO FILE-FORMAT TRAPS, both handled below:

  1  gen_availability_timeseries.csv columns are POSITIONAL.  GTEP.jl builds
     Ordered_gen_nm = ["G$g" for g = 1:(Num_gen+Num_Cgen)], so G1..G_Num_gen are the
     gendata rows in order and the rest are the candidates.  Appending rows to
     gendata.csv therefore RENUMBERS every candidate column.  Left alone, the candidate
     wind and solar profiles would silently be read as the new units' profiles.  This
     script rewrites the whole file in the correct order.

  2  power.csv is written one row per generator row, in the same order, with no
     grouping.  So the new units keep standard Type labels ('NGCC'/'NGCT'/'WindOn'/
     'SolarPV') -- no risk of a new string meeting code that switches on Type -- and
     btm_manifest.json records their row indices so the post-processing can pick them
     out exactly.

    python build_hybrid_case.py
    -> TX8z_GTEP_hybrid_vre_case/  and its btm_manifest.json
"""
import json
import os
import shutil

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
SRC = os.path.join(HERE, 'TX8z_GTEP_dc_vre_case')
DST = os.path.join(HERE, 'TX8z_GTEP_hybrid_vre_case')
SRC_DATA, DST_DATA = 'Data_TX8z_GTEP_dc_vre', 'Data_TX8z_GTEP_hybrid_vre'

DESIGN = f'{PROJECT_ROOT}/4_results/main_figures/figures_1_to_3/01_raw/offgrid_sites.json'
GASF = f'{PROJECT_ROOT}/3_offgrid_code/zonal_gas_2030.json'
# Renewable designs require per-site exports: site_<name>/{wind,solar}_cf.csv.
CFDIR = os.environ.get('OFFGRID_CF_DIR')

# The machine catalogue, copied from offgrid_model.py.  vom is already escalated there;
# variable cost is vom + hr*gas, which is what the model's 'Cost ($/MWh)' column means.
E23, E19 = 1.053, 1.2549
CAT = {
    'Recip':   dict(S=20.0,  hr=8.295, vom=5.69*E19, q=0.030,  type='NGCT'),
    'Aero35':  dict(S=35.0,  hr=9.447, vom=5.70*E23, q=0.0476, type='NGCT'),
    'Aero48':  dict(S=48.0,  hr=9.100, vom=5.70*E23, q=0.0476, type='NGCT'),
    'FrameCT': dict(S=419.0, hr=9.142, vom=1.24*E23, q=0.0476, type='NGCT'),
    'CC':      dict(S=627.0, hr=6.226, vom=3.33*E23, q=0.0357, type='NGCC'),
}
CO2_T_PER_MMBTU = 0.05306        # reproduces the incumbent NGCC EF of 0.3304 t/MWh
BATT_HOURS = 4.0


def zone_fleet(sites):
    """The published design, aggregated to zone x machine class, plus VRE and storage."""
    z = {}
    for r in sites:
        d = z.setdefault(r['zone'], dict(mix={}, wind=0.0, solar=0.0, batt=0.0,
                                         sites=[], peak=0.0))
        for k, n in r['mix'].items():
            d['mix'][k] = d['mix'].get(k, 0) + n
        d['wind'] += r.get('wind_MW', 0.0) or 0.0
        d['solar'] += r.get('solar_MW', 0.0) or 0.0
        d['batt'] += r.get('batt_MW', 0.0) or 0.0
        d['peak'] += r['peak_MW']
        d['sites'].append(r['site'])
    return z


def cf_profile(sites, kind):
    """Capacity-weighted hourly CF for one zone's wind or solar, from the case exports.

    Each campus carries its own resource, so a zone with wind at two campuses gets the
    average weighted by how much each of them built -- not a simple mean, which would
    let a campus that built 3 MW pull the profile as hard as one that built 300.
    """
    num, den = None, 0.0
    for site, mw in sites:
        if mw <= 0:
            continue
        if not CFDIR:
            raise SystemExit('Set OFFGRID_CF_DIR to the per-site renewable profile export directory.')
        p = f'{CFDIR}/site_{site}/{kind}_cf.csv'
        if not os.path.exists(p):
            raise SystemExit(f'missing {p}')
        d = pd.read_csv(p)                       # the exports carry a 'cf' header
        if len(d) != 8760:
            d = pd.read_csv(p, header=None)
        v = d.iloc[:, 0].astype(float).values
        if len(v) != 8760:
            raise SystemExit(f'{p}: expected 8760 hours, got {len(v)}')
        num = v*mw if num is None else num + v*mw
        den += mw
    return None if num is None else num/den


def main():
    sites = json.load(open(DESIGN))
    gas = json.load(open(GASF))
    fleet = zone_fleet(sites)

    # Copy the INPUTS only, and copy the data folder straight to its new name rather
    # than copying and renaming -- this folder cannot delete files, so a half-finished
    # run must not leave a directory behind that a re-run then trips over.
    os.makedirs(DST, exist_ok=True)
    D = f'{DST}/{DST_DATA}'
    shutil.copytree(f'{SRC}/{SRC_DATA}', D, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(f'{SRC}/Settings', f'{DST}/Settings', dirs_exist_ok=True)
    # anything a previous run left in the way is moved aside, not deleted
    stale = f'{DST}/_from_dc_case_DELETE_ME'
    for d in os.listdir(DST):
        if d.startswith('output') or d == SRC_DATA:
            os.makedirs(stale, exist_ok=True)
            tgt = f'{stale}/{d}'
            if not os.path.exists(tgt):
                os.rename(f'{DST}/{d}', tgt)
    # a re-run starts from the ORIGINAL inputs, never from its own last output
    for f in ('gendata.csv', 'gen_availability_timeseries.csv', 'storagedata.csv'):
        shutil.copyfile(f'{SRC}/{SRC_DATA}/{f}', f'{D}/{f}')
    shutil.copyfile(f'{SRC}/Settings/HOPE_model_settings.yml',
                    f'{DST}/Settings/HOPE_model_settings.yml')

    gen = pd.read_csv(f'{D}/gendata.csv')
    cand = pd.read_csv(f'{D}/gendata_candidate.csv')
    n_gen0, n_cand = len(gen), len(cand)

    # one template row per technology, so every flag and column we do not set on purpose
    # keeps the value the incumbents use
    tmpl = {t: gen[gen.Type == t].iloc[0].copy() for t in ('NGCC', 'NGCT', 'WindOn', 'SolarPV')}

    rows, manifest, profiles = [], [], {}
    for zone in sorted(fleet):
        f = fleet[zone]
        g = gas[zone]
        for k in sorted(f['mix'], key=lambda k: -CAT[k]['S']*f['mix'][k]):
            n, c = f['mix'][k], CAT[k]
            r = tmpl[c['type']].copy()
            r['Pmax (MW)'] = c['S']*n
            r['Pmin (MW)'] = 0.0
            r['Zone'] = zone
            r['Flag_RET'] = 0
            r['Cost ($/MWh)'] = round(c['vom'] + c['hr']*g, 4)
            r['EF'] = round(c['hr']*CO2_T_PER_MMBTU, 6)
            r['CC'] = 1.0
            r['AF'] = 1.0
            r['FOR'] = c['q']
            r['n_units'] = n
            rows.append(r)
            manifest.append(dict(row=n_gen0 + len(rows) - 1, zone=zone, machine=k,
                                 units=n, MW=c['S']*n, type=c['type'],
                                 var_cost_USD_MWh=float(r['Cost ($/MWh)'])))
        for kind, key, tp in (('wind', 'wind_MW', 'WindOn'), ('solar', 'solar_MW', 'SolarPV')):
            mw = f[kind]
            if mw <= 0.05:
                continue
            pairs = [(r['site'], r.get(key, 0.0) or 0.0) for r in sites if r['zone'] == zone]
            r = tmpl[tp].copy()
            r['Pmax (MW)'] = round(mw, 3)
            r['Pmin (MW)'] = 0.0
            r['Zone'] = zone
            r['Flag_RET'] = 0
            r['Cost ($/MWh)'] = 0.0
            r['EF'] = 0.0
            r['AF'] = 1.0
            r['FOR'] = 0.0
            r['n_units'] = 1
            rows.append(r)
            idx = n_gen0 + len(rows) - 1
            profiles[idx] = cf_profile(pairs, kind)
            manifest.append(dict(row=idx, zone=zone, machine=kind, units=1,
                                 MW=float(r['Pmax (MW)']), type=tp, var_cost_USD_MWh=0.0))

    gen2 = pd.concat([gen, pd.DataFrame(rows)], ignore_index=True)
    gen2.to_csv(f'{D}/gendata.csv', index=False)
    n_new = len(rows)

    # ---- the positional availability file, rebuilt in the order GTEP.jl expects
    af = pd.read_csv(f'{D}/gen_availability_timeseries.csv')
    tcols = [c for c in af.columns if not c.startswith('G')]
    cols = {c: af[c] for c in tcols}
    for i in range(n_gen0):                       # existing units keep their profiles
        cols[f'G{i+1}'] = af[f'G{i+1}']
    for j in range(n_new):                        # the fleet slots in behind them
        idx = n_gen0 + j
        pr = profiles.get(idx)
        cols[f'G{idx+1}'] = pd.Series(1.0, index=af.index) if pr is None else pd.Series(pr)
    for j in range(n_cand):                       # candidates shift up by n_new
        cols[f'G{n_gen0 + n_new + j + 1}'] = af[f'G{n_gen0 + j + 1}']
    out = pd.DataFrame(cols)
    out.to_csv(f'{D}/gen_availability_timeseries.csv', index=False)

    # Check the remap rather than trusting it: a silently mis-shifted candidate column
    # would hand a solar profile to a gas turbine and the case would still solve.
    import numpy as np
    src_af = pd.read_csv(f'{SRC}/{SRC_DATA}/gen_availability_timeseries.csv')
    assert all(np.allclose(src_af[f'G{i+1}'], out[f'G{i+1}']) for i in range(n_gen0)), \
        'existing availability columns moved'
    assert all(np.allclose(src_af[f'G{n_gen0+j+1}'], out[f'G{n_gen0+n_new+j+1}'])
               for j in range(n_cand)), 'candidate availability columns mis-shifted'

    # ---- the campuses' own batteries
    st = pd.read_csv(f'{D}/storagedata.csv')
    brows = []
    for zone in sorted(fleet):
        mw = fleet[zone]['batt']
        if mw <= 0.05:
            continue
        b = st[st.Zone == zone].iloc[0].copy() if (st.Zone == zone).any() else st.iloc[0].copy()
        b['Zone'] = zone
        b['Max Power (MW)'] = round(mw, 3)
        b['Capacity (MWh)'] = round(mw*BATT_HOURS, 3)
        b['n_units'] = 1
        brows.append(b)
    if brows:
        pd.concat([st, pd.DataFrame(brows)], ignore_index=True).to_csv(
            f'{D}/storagedata.csv', index=False)

    # ---- settings: point at the new data folder, and turn shadow prices ON so the
    #      price-based cross-check has a planning-reserve dual to work with
    sp = f'{DST}/Settings/HOPE_model_settings.yml'
    txt = open(sp).read().replace(f'DataCase: {SRC_DATA}/', f'DataCase: {DST_DATA}/')
    txt = txt.replace('write_shadow_prices: 0', 'write_shadow_prices: 1')
    open(sp, 'w').write(txt)

    tot = sum(m['MW'] for m in manifest)
    json.dump(dict(case=os.path.basename(DST), data_dir=DST_DATA,
                   n_existing_rows_before=n_gen0, n_btm_rows=n_new,
                   n_candidate_rows=n_cand, total_btm_MW=tot,
                   note=('rows are 0-based indices into gendata.csv AND into power.csv, '
                         'which HOPE writes one row per generator in the same order'),
                   units=manifest),
              open(f'{DST}/btm_manifest.json', 'w'), indent=1)

    print(f'built {DST}')
    print(f'  gendata rows {n_gen0} -> {len(gen2)}   candidates {n_cand} (now G'
          f'{n_gen0+n_new+1}..G{n_gen0+n_new+n_cand})')
    print(f'  fleet added {tot:,.0f} MW in {n_new} rows across {len(fleet)} zones')
    for zone in sorted(fleet):
        f = fleet[zone]
        mw = sum(CAT[k]['S']*n for k, n in f['mix'].items())
        print(f'    {zone:13s} gas {mw:8,.0f} MW  wind {f["wind"]:6.1f}  '
              f'solar {f["solar"]:5.1f}  batt {f["batt"]:5.1f}   peak {f["peak"]:8,.0f}')


if __name__ == '__main__':
    main()
