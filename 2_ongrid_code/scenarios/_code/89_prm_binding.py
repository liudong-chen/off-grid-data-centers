"""Does the planning reserve margin actually bind in each solved case?

WHY THE QUESTION MATTERS MORE THAN THE PRM ITSELF.  The margin is a fixed point: HOPE
builds at PRM_k, the Monte Carlo scores what it built, and the PRM is bisected until the
portfolio delivers 0.1 days/year.  Every case is a different system, so in principle every
case needs its own fixed point.  But the PRM enters the optimisation through exactly one
inequality,

    sum CC x Pmax (existing) + sum CC x built (candidates) + storage  >=  (1 + PRM) x peak

and if that inequality is SLACK at the solution, then the build was decided by economics
and not by adequacy.  Moving the PRM within the slack changes nothing at all: not the
portfolio, not the cost, not the $/MWh the paper reports.  Re-deriving the margin is then
a reporting exercise -- the number goes in the reliability table -- and not a reason to
redo the economics.

If it BINDS, the opposite: the margin is setting the build, the cost is a function of it,
and a stale PRM makes the cost wrong.

So this computes the slack, in MW and as a share of the requirement, for every solved
case, using HOPE's own RA expression from src/GTEP.jl.  It answers the practical question
-- which cases have to be re-run, and which only have to be re-scored -- without solving
anything.

    python 89_prm_binding.py
    -> out/prm_binding.json
"""
import json, os
import numpy as np
import pandas as pd

U = '2_ongrid_code/HOPE/ModelCases'
B = 'build'
OUT = f'{B}/out'
TX = ['Coast', 'East', 'FarWest', 'North', 'NorthCentral', 'South', 'SouthCentral', 'West']

# case tag -> (directory holding Data_* and output/, zone list)
CASES = {
    'base_vre  (2030 base)': (f'{U}/TX8z_GTEP_base_vre_case', TX),
    # the 2030 DC case data lives here locally (it is the corrected case the figures
    # are drawn from); its HOPE output comes back from the device under a different root
    'dc_vre    (2030 dc)':   (f'{B}/TX8z_GTEP_cases_vre/TX8z_GTEP_dc_case', TX,
                              f'{U}/TX8z_GTEP_dc_vre_case'),
    'base2035':              (f'{U}/TX8z_GTEP_base2035_case', TX),
    'dc20      (2035, 20 GW)':  (f'{U}/TX8z_GTEP_dc20_case', TX),
    'dc120     (2035, 121 GW)': (f'{U}/TX8z_GTEP_dc120_case', TX),
    'ga_base   (Georgia base)': (f'{B}/ga_hope/GA1z_GTEP_base_case', ['Georgia'],
                                 f'{U}/GA1z_GTEP_base_case'),
    'ga_dc     (Georgia dc)':   (f'{B}/ga_hope/GA1z_GTEP_dc_case', ['Georgia'],
                                 f'{U}/GA1z_GTEP_dc_case'),
}


def datadir(case):
    d = [x for x in os.listdir(case) if x.startswith('Data_') and
         os.path.isdir(f'{case}/{x}') and not x.endswith('_BEFORE_2035_UPDATE')]
    return f'{case}/{d[0]}'


rows = []
for tag, spec in CASES.items():
    case, zones = spec[0], spec[1]
    outdir = spec[2] if len(spec) > 2 else case
    if not os.path.isdir(case) or not os.path.exists(f'{outdir}/output/capacity.csv'):
        print(f'{tag:26s} not solved / not staged, skipped')
        continue
    dd = datadir(case)
    g = pd.read_csv(f'{dd}/gendata.csv')
    c = pd.read_csv(f'{dd}/gendata_candidate.csv')
    se = pd.read_csv(f'{dd}/storagedata.csv')
    sn = pd.read_csv(f'{dd}/storagedata_candidate.csv')
    sp = pd.read_csv(f'{dd}/single_parameter.csv').iloc[0]
    zd = pd.read_csv(f'{dd}/zonedata.csv').set_index('Zone_id')['Demand (MW)']
    ld = pd.read_csv(f'{dd}/load_timeseries_regional.csv')
    prm = float(sp['planning_reserve_margin'])

    # HOPE's PK_system: the maximum over hours of the summed zonal load
    L = ld[zones].values*zd[zones].values[None, :]
    peak = float(L.sum(1).max())

    cap = pd.read_csv(f'{outdir}/output/capacity.csv')
    built = {f'{t}/{z}': float(v) for t, z, v, k in
             zip(cap.Technology, cap.Zone, cap['Capacity_FIN (MW)'], cap.EC_Category)
             if k == 'Candidate'}
    esf = f'{outdir}/output/es_capacity.csv'
    sto_new = 0.0
    if os.path.exists(esf):
        es = pd.read_csv(esf)
        col = 'Capacity (MW)'
        sto_new = float(sum(v*float(sn['CC'].iloc[0]) for v, k in
                            zip(es[col], es.EC_Category) if k == 'Candidate'))

    acc_exist = float((g['Pmax (MW)']*g['CC']).sum()
                      + (se['Max Power (MW)']*se['CC']).sum())
    ccmap = {f'{t}/{z}': cc for t, z, cc in zip(c.Type, c.Zone, c['CC'])}
    acc_new = sum(mw*ccmap.get(k, 0.0) for k, mw in built.items())
    acc = acc_exist + acc_new + sto_new
    req = (1 + prm)*peak
    slack = acc - req

    # the PRM the built portfolio would just satisfy: how far the margin could rise
    # before this build stops being feasible
    prm_max = acc/peak - 1.0
    rows.append(dict(case=tag, prm_in_file=prm, peak_MW=peak, requirement_MW=req,
                     accredited_MW=acc, accredited_existing_MW=acc_exist,
                     accredited_new_MW=acc_new + sto_new, slack_MW=slack,
                     slack_pct_of_requirement=100*slack/req,
                     prm_headroom=prm_max, binds=bool(slack < 0.005*req)))
    print(f'{tag:26s} PRM {prm:7.4f}  peak {peak:9,.0f}  need {req:9,.0f}  '
          f'have {acc:9,.0f}  slack {slack:+9,.0f} MW ({100*slack/req:+6.2f}%)  '
          f'{"BINDS" if slack < 0.005*req else "slack"}')

print('\nreading of the table')
print('  BINDS  the margin set the build.  A stale PRM makes the COST wrong, and the case')
print('         has to be re-solved at the re-derived margin.')
print('  slack  the build was decided by economics, and the margin could rise to the')
print('         "PRM headroom" column before it would change anything.  Re-deriving the')
print('         margin is then a reporting step: the portfolio, and every $/MWh in the')
print('         paper, is unaffected.  The LOLE of what was built still has to be scored.')
print()
for r in rows:
    print(f'  {r["case"]:26s} PRM headroom {r["prm_headroom"]:6.3f} '
          f'against {r["prm_in_file"]:.4f} in the file'
          f'{"   <-- re-solve" if r["binds"] else ""}')

json.dump(rows, open(f'{OUT}/prm_binding.json', 'w'), indent=1, default=float)
print(f'\nwritten {OUT}/prm_binding.json')
