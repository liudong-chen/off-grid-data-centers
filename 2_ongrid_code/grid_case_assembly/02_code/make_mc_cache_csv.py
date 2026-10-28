"""Same cache, but as plain CSV so Julia reads it with CSV.jl - already a HOPE dep."""
import os
import numpy as np, pandas as pd, importlib.util
spec = importlib.util.spec_from_file_location('r', os.path.join(os.path.dirname(os.path.abspath(__file__)), '08_report.py'))
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
FOR_ = {'NGCC':0.0357,'NGCT':0.0476,'NGST':0.0595,'Coal':0.0714,'Nuclear':0.0476,
        'Oil':0.0595,'Biomass':0.0892,'Hydro':0.60}
MTTR = {'NGCC':48,'NGCT':36,'NGST':60,'Coal':72,'Nuclear':120,'Oil':36,
        'Biomass':48,'Hydro':24}
for case, dc in [('base', False), ('dc', True)]:
    C = M.build_case(dc); f = C['fleet']
    firm = f[~f.Type.isin(['WindOn','SolarPV'])].reset_index(drop=True)
    pd.DataFrame({'Type': firm.Type, 'mw': firm['Pmax (MW)'], 'n_units': firm.n_units,
                  'FOR': [FOR_.get(t,0.0) for t in firm.Type],
                  'MTTR': [MTTR.get(t,48) for t in firm.Type]}
                 ).to_csv(f'mc_cache_{case}_fleet.csv', index=False)
    d = {}
    for j, y in enumerate(M.YEARS):
        d[f'load_{y}'] = C['LOAD'][y]; d[f'vre_{y}'] = C['VRE'][y]
    pd.DataFrame(d).to_csv(f'mc_cache_{case}_series.csv', index=False,
                           float_format='%.4f')
    pd.DataFrame([{'sto_p_MW': C['sto_p'], 'sto_e_MWh': C['sto_e'],
                   'peak_MW': C['peak'], 'n_years': len(M.YEARS),
                   'eta': 0.922}]).to_csv(f'mc_cache_{case}_meta.csv', index=False)
    print(f'{case}: fleet {len(firm)} rows, series {len(d)} cols, peak {C["peak"]:,.0f} MW')
