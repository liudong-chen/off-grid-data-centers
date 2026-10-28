"""Scenario: workload mix.

Rebuilds the 17 ERCOT data center load profiles with one workload for every data center,
keeping everything else of DC_Load/build_dc_load.py unchanged (same traces, seed, weather,
announced capacity, design convention and the cooling level M_0 of each site's archetype,
which describes the building rather than the workload). The static share s follows the
workload. Mixes:
    training     w = (1, 0, 0)   every data center runs training only
    inference    w = (0, 1, 0)   inference only
    colocation   w = (0, 0, 1)   colocation-hall shape only
The base case is the archetype mix of the paper (DC_Load/load_profiles).
    python3 build_profiles.py  ->  profiles/<mix>/ (same files as DC_Load/load_profiles)
"""
import importlib.util, os, sys
H = os.path.dirname(os.path.abspath(__file__))
B = os.path.abspath(os.path.join(H, '..', '..', '..'))
MIXES = {'training': (1.0, 0.0, 0.0), 'inference': (0.0, 1.0, 0.0), 'colocation': (0.0, 0.0, 1.0)}
for name, w in MIXES.items():
    spec = importlib.util.spec_from_file_location('bdl', f'{B}/DC_Load/build_dc_load.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.OUT = f'{H}/profiles/{name}'; os.makedirs(m.OUT, exist_ok=True)
    m.ARCHETYPE = {k: w for k in m.ARCHETYPE}           # M0 stays by archetype
    sys.argv = ['build_dc_load.py']
    print(f'==== {name} {w}', flush=True)
    m.main()
