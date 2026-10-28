"""Precompute everything 11_mc_portfolio.py needs, so it carries no dependency on the
ERCOT Excel load files or on any container path. Run once, here."""
import os
import numpy as np, importlib.util
spec = importlib.util.spec_from_file_location('r', os.path.join(os.path.dirname(os.path.abspath(__file__)), '08_report.py'))
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
for case, dc in [('base', False), ('dc', True)]:
    C = M.build_case(dc)
    f = C['fleet']
    firm = f[~f.Type.isin(['WindOn','SolarPV'])]
    np.savez_compressed(
        f'build/mc_cache_{case}.npz',
        years=np.array(M.YEARS),
        LOAD=np.stack([C['LOAD'][y] for y in M.YEARS]),
        VRE=np.stack([C['VRE'][y] for y in M.YEARS]),
        fleet_type=firm.Type.values.astype('U12'),
        fleet_mw=firm['Pmax (MW)'].values.astype(float),
        fleet_n=firm.n_units.values.astype(int),
        sto_p=np.array(C['sto_p']), sto_e=np.array(C['sto_e']),
        peak=np.array(C['peak']))
    print(f'mc_cache_{case}.npz: {len(firm)} firm rows, peak {C["peak"]:,.0f} MW, '
          f'storage {C["sto_p"]:,.0f} MW / {C["sto_e"]:,.0f} MWh')
