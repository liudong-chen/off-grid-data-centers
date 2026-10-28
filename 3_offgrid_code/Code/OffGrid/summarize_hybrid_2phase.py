"""Fleet summary: off-grid-sized hybrid vs connection-aware (two-phase) hybrid, ERCOT.

    python3 summarize_hybrid_2phase.py [out/hybrid_2phase] [tag]
    -> <dir>/hybrid_2phase_summary.{csv,json}

Connected cost of the new fleet is ESTIMATED from the HOPE result of the off-grid-sized
fleet at the same x, shifted by the fitted grid value of the gas that is no longer there:
    G_new = G_old + sum_k credit_k (MW_k,old - MW_k,new)        (credit: CC 71.9k, peakers 3.9k $/MW-yr)
    c_conn_new = (K_new + G_new)/E,   K = gas capital + land + diesel capex (no FOM)
When TX8z_GTEP_<HYB_PREFIX>_dsl<XXX>_vre_case is solved (HYB_PREFIX default hyb2p), the HOPE
objective replaces the estimate:  c_conn = (obj - obj_base)/E + K_new/E.
Levelised gas-phase cost = PV(cost, months 24..264)/PV(energy, same months).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D2 = sys.argv[1] if len(sys.argv) > 1 else f'{HERE}/out/hybrid_2phase'
REF = json.load(open(os.environ.get('HYB_REF', f'{HERE}/out/hybrid_diesel/hybrid_diesel_summary.json')))
S_MW = np.array([20., 35., 48., 419., 627., 3.])
CRED = np.array([float(os.environ.get('EVAL_PK', 3861.))]*4 + [float(os.environ.get('EVAL_CC', 71947.)), 0.])
FOM = 35e3
T_GAS, T_CON, END = 24, 50, 264
r_m = 1.08**(1/12) - 1; mo = np.arange(END + 1); disc = (1 + r_m)**(-mo.astype(float))
W_ISL = disc[(mo > T_GAS) & (mo <= T_CON)].sum()/12; W_POST = disc[mo > T_CON].sum()/12

MC = os.environ.get('BYONG_MC', os.path.abspath(f'{HERE}/../HOPE/ModelCases'))
PREFIX = os.environ.get('HYB_PREFIX', 'hyb2p')


def hope_obj(case):
    for q in (f'{MC}/{case}/output/postprocess_snapshot/metadata.yml', f'{MC}/{case}/output/metadata.yml'):
        if os.path.exists(q):
            for line in open(q):
                if line.strip().startswith('objective_value'):
                    return float(line.split(':')[1])
    return None


OBJ_BASE = hope_obj('TX8z_GTEP_base_vre_case') or 10.291e9
P = REF['params']; E = P['E_TWh']*1e6
ref = pd.DataFrame(REF['table']).set_index('share')
new = json.load(open(f'{D2}/all_sites.json'))
SRC = {a['site']: a for a in json.load(open(f'{HERE}/out/hybrid_diesel/all_sites.json'))}
keys = sorted(set.intersection(*[set(s['cases']) for s in new]), key=float)
rows = []
for k in keys:
    x = float(k)
    C = [s['cases'][k] for s in new]
    # off-grid-sized fleet per site (fallback if the 2-phase fleet is worse on its own metric -- screening noise)
    n_new = np.zeros(6); n_old = np.zeros(6); tot = dict(fixed_gas=0, land=0, fixed_diesel=0, fuel_gas=0, fuel_diesel=0, total=0)
    for s_, c in zip(new, C):
        if c['pv_2phase_offgrid_fleet'] < c['pv_2phase']:
            c = SRC[s_['site']]['cases']['0.0_gasopt' if x == 0 else k] | dict(offgrid_n=c['offgrid_n'], n=c['offgrid_n'])
            c['pv_2phase'] = c['pv_2phase_offgrid_fleet'] = 0
        use_old = c['pv_2phase_offgrid_fleet'] < c['pv_2phase']
        n_old += np.array(c['offgrid_n'])
        n_new += np.array(c['offgrid_n'] if use_old else c['n'])
        for f in tot:
            tot[f] += c[f]
    r = ref.loc[x]
    K_old = r.fixed_gas + r.land + r.fixed_diesel - r.diesel_MW*FOM
    G_old = r.c_hybrid_connected_retire*E - K_old
    K_new = tot['fixed_gas'] + tot['land'] + tot['fixed_diesel'] - n_new[5]*3*FOM
    G_new = G_old + float(((n_old - n_new)*S_MW*CRED).sum())
    c_new = (K_new + G_new)/E; src = 'estimate (grid credit)'
    o = hope_obj(f'TX8z_GTEP_{PREFIX}_dsl{int(round(100*x)):03d}_vre_case')
    if o is not None:
        c_new = (o - OBJ_BASE + K_new)/E; src = 'HOPE'
    lev = lambda isl, conn: (W_ISL*isl + W_POST*conn*E)/((W_ISL + W_POST)*E)
    rows.append(dict(share=x,
                     gas_MW_offgrid=r.gas_MW, gas_MW_2p=float(n_new[:5] @ S_MW[:5]), diesel_MW=r.diesel_MW,
                     CC_GW_offgrid=n_old[4]*627/1e3, CC_GW_2p=n_new[4]*627/1e3,
                     peaker_GW_offgrid=float(n_old[:4] @ S_MW[:4])/1e3, peaker_GW_2p=float(n_new[:4] @ S_MW[:4])/1e3,
                     before_conn_offgrid=r.lcoe_islanded, before_conn_2p=tot['total']/E,
                     diesel_TWh_2p=None,
                     after_conn_offgrid=r.c_hybrid_connected_retire, after_conn_2p=c_new, after_conn_source=src,
                     gasphase_offgrid_never=r.lcoe_islanded,
                     gasphase_hyb_offgrid_fleet=lev(r.total, r.c_hybrid_connected_retire),
                     gasphase_hyb_2p=lev(tot['total'], c_new)))
T = pd.DataFrame(rows).set_index('share').drop(columns='diesel_TWh_2p')
T['gas_cut_MW'] = T.gas_MW_offgrid - T.gas_MW_2p
T['saving_$MWh'] = T.gasphase_hyb_offgrid_fleet - T.gasphase_hyb_2p
T.to_csv(f'{D2}/hybrid_2phase_summary.csv')
json.dump(dict(W_ISL=W_ISL, W_POST=W_POST, credit=CRED.tolist(), E_MWh=E, table=T.reset_index().to_dict('records')),
          open(f'{D2}/hybrid_2phase_summary.json', 'w'), indent=1, default=float)
pd.set_option('display.width', 250)
print(T.round(2).to_string())
