"""On-grid numbers from the solved HOPE cases -> the figure packages' 01_raw folders.

    python3 prepare_grid_results.py            (ModelCases found relative to this file)
    python3 prepare_grid_results.py <ModelCases>

Writes
  figures_1_to_3/01_raw/ongrid_final.json   ERCOT 2030: new build, cost, LCOE (69_ongrid_overbuild.py logic)
  figure_5/01_raw/ongrid_final.json     the same file
  figure_5/01_raw/grid_newbuild.json    new-build capacity factor, ERCOT and Georgia (97_grid_newbuild.py)
  figure_5/01_raw/hope/{gabase,gadc}/   metadata.yml, capacity.csv, system_cost.csv copied from the cases
  figure_4/01_raw/ongrid2035.json       2035 pairs, 20 and 121 GW (77_ongrid_2035.py logic)

Conventions kept from the scripts this replaces:
  - the TOTAL attributable cost is the difference of the two solver objectives (metadata.yml);
    system_cost.csv's Inv_cost column is not used for ERCOT (it under-reports a large build);
  - investment is rebuilt from primitives, new MW x annualised fixed cost of the candidate;
  - transmission attributable to the data centers is the dc-minus-base INCREMENT per corridor;
  - the grid's gas is reported in units of the candidate machine, Pmax / n_units of the
    candidate file (627 MW combined cycle ; it was 351 MW before).
"""
import json
import os
import re
import shutil
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = os.path.dirname(os.path.dirname(HERE))                         # Figures
MC = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(f'{NEW}/../Code/HOPE/ModelCases')
RAW123, RAW4, RAW5 = f'{NEW}/figures_1_to_3/01_raw', f'{NEW}/figure_4/01_raw', f'{NEW}/figure_5/01_raw'

DC_ENERGY_MWh = 222_051_000.0                       # ERCOT 2030 data-center energy, from the load model
NGCC_FIXED = 118_650.0                              # $/MW-yr, 2030 candidate (ERCOT 2026 LTSA)
FIX35 = {'NGCC': 113755.0, 'NGCT': 96576.0, 'WindOn': 149412.0, 'SolarPV': 108393.0, 'NuC': 706973.0}
DC_ENERGY_2035 = {'dc20': 111_025_500.0, 'dc120': 666_153_000.0}   # k x 222.051 TWh


def datadir(case):
    for ln in open(f'{MC}/{case}/Settings/HOPE_model_settings.yml'):
        m = re.match(r'\s*DataCase\s*:\s*([^#\s]+)', ln)
        if m:
            return f'{MC}/{case}/' + m.group(1).rstrip('/')
    raise SystemExit(f'no DataCase line for {case}')


def objective(case):
    for q in (f'{MC}/{case}/output/postprocess_snapshot/metadata.yml', f'{MC}/{case}/output/metadata.yml'):
        if os.path.exists(q):
            t = open(q).read()
            st = re.search(r'termination_status:\s*"?([A-Z_]+)', t)
            assert st and st.group(1) == 'OPTIMAL', f'{case}: {st.group(1) if st else "no status"}'
            return float(re.search(r'objective_value:\s*([0-9.eE+-]+)', t).group(1))
    raise SystemExit(f'no metadata for {case}')


def new_build(case):
    c = pd.read_csv(f'{MC}/{case}/output/capacity.csv')
    n = c[(c.New_Build == 1) & (c['Capacity_FIN (MW)'] > 1.0)]
    return {(z, t): float(v) for z, t, v in zip(n['Zone'], n['Technology'], n['Capacity_FIN (MW)'])}


def new_lines(case):
    l = pd.read_csv(f'{MC}/{case}/output/line.csv')
    l = l[l.New_Build == 1]
    fwd = l.columns[3]                       # 'Forward Capacity (MW)'
    return {(a, b): float(v) for a, b, v in zip(l['From_zone'], l['To_zone'], l[fwd]) if float(v) > 1.0}


def new_gen_mwh(case):
    p = pd.read_csv(f'{MC}/{case}/output/power.csv', usecols=['Technology', 'New_Build', 'AnnSum'])
    return float(p[p.New_Build == 1]['AnnSum'].sum())


CATALOGUE = (627.0, 419.0, 1111.0)          # EIA AEO2026 TRE reference plants: CC single-shaft, CT frame; nuclear block


def unit_mw(case, tech):
    """Machine size of the grid's new build: Pmax / n_units of the candidate file, snapped to the
    catalogue size it was set from (100,000 / 159 = 628.9 -> 627)."""
    c = pd.read_csv(f'{datadir(case)}/gendata_candidate.csv')
    r = c[c.Type == tech].iloc[0]
    u = float(r['Pmax (MW)'])/max(1, int(r['n_units']))
    near = min(CATALOGUE, key=lambda k: abs(k - u))
    return near if abs(near - u)/near < 0.01 else u


def prm(case):
    return float(pd.read_csv(f'{datadir(case)}/single_parameter.csv').iloc[0]['planning_reserve_margin'])


# ------------------------------------------------------------------ ERCOT 2030
B, D = 'TX8z_GTEP_base_vre_case', 'TX8z_GTEP_dc_vre_case'
obj = {k: objective(k) for k in (B, D)}
nb, nl = {k: new_build(k) for k in (B, D)}, {k: new_lines(k) for k in (B, D)}
by_zone, by_tech = {}, {}
for (z, t), mw in nb[D].items():
    by_zone[z] = by_zone.get(z, 0.0) + mw
    by_tech[t] = by_tech.get(t, 0.0) + mw
new_MW = sum(by_zone.values())
base_MW = sum(nb[B].values())
inv = new_MW*NGCC_FIXED
cost = obj[D] - obj[B]
inc = {k: nl[D].get(k, 0.0) - nl[B].get(k, 0.0) for k in set(nl[D]) | set(nl[B])}
lines = [[a, b, mw] for (a, b), mw in sorted(inc.items()) if mw > 1.0]
gen_mwh = new_gen_mwh(D) - new_gen_mwh(B)
U = unit_mw(D, 'NGCC')
ONG = dict(new_MW=new_MW, by_zone=by_zone, by_tech=by_tech, inv=inv, opr_increment=cost - inv, cost=cost,
           energy_MWh=DC_ENERGY_MWh, lcoe=cost/DC_ENERGY_MWh, n_units=new_MW/U, unit_MW=U,
           lines=lines, base_new_MW=base_MW,
           lines_dc={f'{a}->{b}': v for (a, b), v in nl[D].items()},
           lines_base={f'{a}->{b}': v for (a, b), v in nl[B].items()},
           obj_dc=obj[D], obj_base=obj[B], new_gen_MWh=gen_mwh, new_gen_cf=gen_mwh/(new_MW*8760.0),
           displaced_older_MWh=gen_mwh - DC_ENERGY_MWh, prm_dc=prm(D), prm_base=prm(B),
           cases=dict(base=B, dc=D))
for raw in (RAW123, RAW5):
    json.dump(ONG, open(f'{raw}/ongrid_final.json', 'w'), indent=1)
print(f'ERCOT 2030  new {new_MW:,.0f} MW ({new_MW/U:.1f} x {U:.0f} MW)  cost ${cost/1e9:.4f} B/yr  '
      f'${cost/DC_ENERGY_MWh:.2f}/MWh  cf {ONG["new_gen_cf"]:.3f}  PRM {prm(D):.4f}  lines {lines or "none"}')

# ------------------------------------------------------- new-build CF, ERCOT + Georgia
PAIRS = {'ercot': (B, D), 'georgia': ('GA1z_GTEP_base_case', 'GA1z_GTEP_dc_case')}
GNB = {}
for name, (b, d) in PAIRS.items():
    mw = sum(new_build(d).values()) - sum(new_build(b).values())
    mwh = new_gen_mwh(d) - new_gen_mwh(b)
    GNB[name] = dict(new_MW=mw, new_gen_MWh=mwh, new_gen_cf=mwh/(mw*8760.0), cases=dict(base=b, dc=d),
                     prm=dict(base=prm(b), dc=prm(d)))
    print(f'{name:8s} new {mw:10,.1f} MW  cf {mwh/(mw*8760.0):.4f}')
json.dump(GNB, open(f'{RAW5}/grid_newbuild.json', 'w'), indent=1)

# ------------------------------------------------------------ Georgia HOPE copies
for tag, case in (('gabase', 'GA1z_GTEP_base_case'), ('gadc', 'GA1z_GTEP_dc_case')):
    dst = f'{RAW5}/hope/{tag}'; os.makedirs(dst, exist_ok=True)
    for f in ('capacity.csv', 'system_cost.csv'):
        shutil.copy(f'{MC}/{case}/output/{f}', f'{dst}/{f}')
    src = f'{MC}/{case}/output/postprocess_snapshot/metadata.yml'
    shutil.copy(src if os.path.exists(src) else f'{MC}/{case}/output/metadata.yml', f'{dst}/metadata.yml')
print(f'Georgia  obj base {objective("GA1z_GTEP_base_case")/1e9:.4f}  dc {objective("GA1z_GTEP_dc_case")/1e9:.4f} B/yr')

# --------------------------------------------------------------------- 2035
C35 = {'base': 'TX8z_GTEP_base2035_case', 'dc20': 'TX8z_GTEP_dc20_case', 'dc120': 'TX8z_GTEP_dc120_case'}
LC = pd.read_csv(f'{datadir(C35["dc120"])}/linedata_candidate.csv')
LINE_COST = {(a, b): c*1e6/f for a, b, f, c in zip(LC['From_zone'], LC['To_zone'], LC['Forward Capacity (MW)'], LC['Cost (M$)'])}
o35 = {k: objective(c) for k, c in C35.items()}
g35 = {k: new_build(c) for k, c in C35.items()}
l35 = {k: new_lines(c) for k, c in C35.items()}
R = dict(base=dict(obj=o35['base'], new_MW=sum(g35['base'].values()),
                   by_zone_tech={f'{t}/{z}': v for (z, t), v in g35['base'].items()},
                   lines={f'{a}->{b}': v for (a, b), v in l35['base'].items()}, prm=prm(C35['base'])))
for k in ('dc20', 'dc120'):
    dg = {key: g35[k].get(key, 0.0) - g35['base'].get(key, 0.0) for key in set(g35[k]) | set(g35['base'])}
    dg = {key: v for key, v in dg.items() if abs(v) > 1.0}
    dl = {key: l35[k].get(key, 0.0) - l35['base'].get(key, 0.0) for key in set(l35[k]) | set(l35['base'])}
    dl = {key: v for key, v in dl.items() if abs(v) > 1.0}
    cost = o35[k] - o35['base']
    inv_gen = sum(FIX35[t]*v for (z, t), v in dg.items())
    inv_line = sum(LINE_COST[key]*v for key, v in dl.items() if v > 0)
    E = DC_ENERGY_2035[k]
    by_tech, by_zone = {}, {}
    for (z, t), v in dg.items():
        by_tech[t] = by_tech.get(t, 0.0) + v; by_zone[z] = by_zone.get(z, 0.0) + v
    R[k] = dict(new_MW=sum(dg.values()), by_tech=by_tech, by_zone=by_zone,
                by_zone_tech={f'{t}/{z}': v for (z, t), v in dg.items()},
                lines={f'{a}->{b}': v for (a, b), v in dl.items()}, new_line_MW=sum(v for v in dl.values() if v > 0),
                inv=inv_gen, inv_line=inv_line, opr_increment=cost - inv_gen - inv_line, cost=cost,
                energy_MWh=E, lcoe=cost/E, obj_dc=o35[k], obj_base=o35['base'], base_new_MW=sum(g35['base'].values()),
                unit_MW=unit_mw(C35[k], 'NGCC'), prm=prm(C35[k]))
    print(f'2035 {k:6s} new {R[k]["new_MW"]:10,.0f} MW  {({t: round(v) for t, v in by_tech.items()})}  '
          f'lines {R[k]["new_line_MW"]:,.0f} MW  ${cost/1e9:.3f} B/yr  ${cost/E:.2f}/MWh  PRM {R[k]["prm"]:.4f}')
json.dump(R, open(f'{RAW4}/ongrid2035.json', 'w'), indent=1)
print('written ongrid_final.json (x2), grid_newbuild.json, hope/, ongrid2035.json')
