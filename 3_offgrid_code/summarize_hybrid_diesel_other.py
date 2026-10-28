"""Fleet-level summary of the diesel-first hybrid for the OTHER scenarios (20 GW, 121 GW, Georgia).

    python3 summarize_hybrid_diesel_other.py 2035_k0p5|2035_k3|georgia
    -> out/hybrid_diesel_<tag>/hybrid_diesel_summary.{json,csv}

Per share x: fleet (gas / genset MW, ratio to summed peak), islanded cost components and
$/MWh, year-one load fraction, and -- when the HOPE cases exist and are solved -- the
connected cost with gensets retired:
    c_connected = (obj_hybrid - obj_base)/E + (gas capital + land + genset capex)/E
using the scenario's own base and dc cases (same formula as summarize_hybrid_diesel.py).
Env: OFFGRID_MC (ModelCases dir; default ../2_ongrid_code/HOPE/ModelCases relative to this file).
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..'))
MC = os.environ.get('OFFGRID_MC', f'{PROJECT_ROOT}/2_ongrid_code/HOPE/ModelCases')
TAG = sys.argv[1]
OUT = f'{HERE}/out/hybrid_diesel_{TAG}'
HOPE = {'2035_k0p5': dict(base='TX8z_GTEP_base2035_case', dc='TX8z_GTEP_dc20_case', hyb='TX8z_GTEP_hyb20_dsl{tag}_case'),
        '2035_k3': dict(base='TX8z_GTEP_base2035_case', dc='TX8z_GTEP_dc120_case', hyb='TX8z_GTEP_hyb120_dsl{tag}_case'),
        'georgia': dict(base='GA1z_GTEP_base_case', dc='GA1z_GTEP_dc_case', hyb='GA1z_GTEP_hyb_dsl{tag}_case')}[TAG]


def hope_obj(case):
    for p in (f'{MC}/{case}/output/postprocess_snapshot/metadata.yml', f'{MC}/{case}/output/metadata.yml'):
        if os.path.exists(p):
            for line in open(p):
                if line.strip().startswith('objective_value'):
                    return float(line.split(':')[1])
    return None


sites = [json.load(open(p)) for p in sorted(glob.glob(f'{OUT}/site_*.json'))]
if not sites and os.path.exists(f'{OUT}/all_sites.json'):
    sites = json.load(open(f'{OUT}/all_sites.json'))
if not sites:
    raise SystemExit(f'no {OUT}/site_*.json or all_sites.json yet')


def isnum(k):
    try:
        float(k); return True
    except ValueError:
        return False


keys = sorted(set.intersection(*[set(s['cases']) for s in sites]), key=float)
dsl = sites[0]['diesel']; VR_DSL = dsl['vom'] + dsl['hr']*sites[0]['diesel_fuel']; FOM_DSL = dsl['fom']*1000.0
E_tot = sum(s['energy_MWh'] for s in sites); MW_pub = sum(s['mw_published'] for s in sites)
peak = sum(s['peak_MW'] for s in sites)
print(f'{TAG}: {len(sites)} campuses, {E_tot/1e6:.1f} TWh/yr, {MW_pub:,.0f} MW, {peak:,.0f} MW summed peak; shares {keys}')

rows = []
for k in keys:
    x = float(k); C = [s['cases'][k] for s in sites]
    gas = sum(c['gas_MW'] for c in C); dmw = sum(c['diesel_MW'] for c in C); inst = gas + dmw
    tot = sum(c['total'] for c in C)
    n = np.sum([c['n'] for c in C], axis=0)
    rows.append(dict(share=x, gas_MW=gas, diesel_MW=dmw, diesel_units=int(n[5]), installed_MW=inst,
                     ratio_to_sum_of_peaks=inst/peak, n_Recip=int(n[0]), n_Aero35=int(n[1]), n_Aero48=int(n[2]),
                     n_FrameCT=int(n[3]), n_CC=int(n[4]),
                     LOLE_min=min(c['LOLE'] for c in C), LOLE_max=max(c['LOLE'] for c in C),
                     fixed_gas=sum(c['fixed_gas'] for c in C), fixed_diesel=sum(c['fixed_diesel'] for c in C),
                     land=sum(c['land'] for c in C), fuel_gas=sum(c['fuel_gas'] for c in C),
                     fuel_diesel=sum(c['fuel_diesel'] for c in C), total=tot, lcoe_islanded=tot/E_tot,
                     load_fraction_before_gas=x,
                     year1_fixed_diesel=sum(c['phaseA']['fixed_diesel'] for c in C),
                     year1_fuel_diesel=sum(c['phaseA']['fuel_diesel'] for c in C),
                     year1_energy_MWh=sum(c['phaseA']['energy_MWh'] for c in C)))
T = pd.DataFrame(rows).set_index('share')

obj_base, obj_dc = hope_obj(HOPE['base']), hope_obj(HOPE['dc'])
c_on = (obj_dc - obj_base)/E_tot if (obj_base and obj_dc) else None
conn, src = [], []
for x in T.index:
    o = hope_obj(HOPE['hyb'].format(tag=f'{int(round(100*x)):03d}'))
    if o is None or obj_base is None:
        conn.append(np.nan); src.append('no HOPE run'); continue
    K = T.loc[x, 'fixed_gas'] + T.loc[x, 'land'] + T.loc[x, 'fixed_diesel'] - T.loc[x, 'diesel_MW']*FOM_DSL
    conn.append((o - obj_base)/E_tot + K/E_tot); src.append('HOPE')
T['c_connected_retire'] = conn; T['connected_source'] = src
T.to_csv(f'{OUT}/hybrid_diesel_summary.csv')
json.dump(dict(case=TAG, n_campuses=len(sites), energy_MWh=E_tot, mw_published=MW_pub, peak_MW=peak,
               c_on=c_on, obj_base=obj_base, obj_dc=obj_dc, diesel=dsl, diesel_fuel=sites[0]['diesel_fuel'],
               VR_DSL=VR_DSL, FOM_DSL=FOM_DSL, hope_cases=HOPE,
               table={f'{x:.1f}': {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()}
                      for x, r in T.to_dict('index').items()}),
          open(f'{OUT}/hybrid_diesel_summary.json', 'w'), indent=1, default=float)
pd.set_option('display.width', 200)
print(T[['gas_MW', 'diesel_MW', 'installed_MW', 'ratio_to_sum_of_peaks', 'lcoe_islanded', 'c_connected_retire', 'connected_source']].round(2))
print(f'on-grid (dc - base)/E: {c_on if c_on is None else round(c_on, 2)} $/MWh')
