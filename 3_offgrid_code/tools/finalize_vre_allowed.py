"""Final numbers of the sensitivity case "renewables allowed next to gas".

Inputs (same MIP as the main case, run_offgrid_mip.py with solar, wind and 4-hour
batteries allowed, credit cap removed):
  out/sensitivity_with_vre/offgrid_mip_final.json   ERCOT 2030 (data center land already removed)
  out/offgrid_mip_2035_k0p5_sites.json              ERCOT 2035, 20 GW
  out/offgrid_mip_2035_k3_sites.json                ERCOT 2035, 121 GW
The two 2035 files were solved on 18 Sep and missed the 8 Oct removal of the data
center's own land (tools/remove_dc_land.py); this script removes it the same way
(backup in out/_before_dcland_20261008/) and compares each case with its gas-only run.
Writes out/sensitivity_with_vre/vre_allowed_summary.json.
Run from 3_offgrid_code:  python3 tools/finalize_vre_allowed.py
"""
import json, os, shutil
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BK = f'{HERE}/out/_before_dcland_20261008'
DC = 0.75

def load(rel): return json.load(open(f'{HERE}/{rel}'))

def remove_dc_land(rel):
    d = load(rel); changed = 0
    for r in d:
        vre = 5.0*(r.get('solar_MW') or 0) + 1.0*(r.get('wind_MW') or 0) + 0.05*(r.get('batt_MW') or 0)
        base = r['acres'] - vre; gas_only = 0.10*r['installed_MW']
        dca = DC*r['peak_MW']
        if abs(base - gas_only) < 0.01*gas_only + 1:      # already without data center land
            continue
        assert abs(base - (gas_only + dca)) < 0.01*(gas_only + dca) + 1, (rel, r['site'])
        dl = r['land_cost']*dca/base
        r['land_cost'] -= dl; r['cost'] -= dl; r['acres'] -= dca; r['lcoe'] = r['cost']/r['energy_MWh']
        changed += 1
    if changed:
        dst = f'{BK}/{rel}'; os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst): shutil.copy2(f'{HERE}/{rel}', dst)
        json.dump(d, open(f'{HERE}/{rel}', 'w'), indent=1, default=float)
    return d, changed

def agg(d):
    E = sum(r['energy_MWh'] for r in d); C = sum(r['cost'] for r in d)
    s = sum(r.get('solar_MW') or 0 for r in d); w = sum(r.get('wind_MW') or 0 for r in d)
    b = sum(r.get('batt_MW') or 0 for r in d); G = sum(r['installed_MW'] for r in d)
    return dict(gas_MW=G, solar_MW=s, wind_MW=w, batt_MW=b, total_MW=G + s + w + b,
                vre_share=(s + w + b)/(G + s + w + b), cost_Byr=C/1e9, lcoe=C/E,
                utilization=E/8760/(G + s + w + b))

CASES = [('ERCOT 2030', 'out/sensitivity_with_vre/offgrid_mip_final.json', 'out/offgrid_mip_gasonly.json'),
         ('ERCOT 2035, 20 GW', 'out/offgrid_mip_2035_k0p5_sites.json', 'out/offgrid_mip_2035_k0p5_gasonly_sites.json'),
         ('ERCOT 2035, 121 GW', 'out/offgrid_mip_2035_k3_sites.json', 'out/offgrid_mip_2035_k3_gasonly_sites.json')]
out = {}
for lab, vre_rel, gas_rel in CASES:
    dv, n = remove_dc_land(vre_rel); dg = load(gas_rel)
    a, g = agg(dv), agg(dg)
    a['gas_only_lcoe'] = g['lcoe']; a['gas_only_total_MW'] = g['total_MW']
    a['saving_per_MWh'] = g['lcoe'] - a['lcoe']; a['dc_land_removed_sites'] = n
    out[lab] = a
    print(f"{lab:20s} total {a['total_MW']:9,.0f} MW (gas {a['gas_MW']:,.0f}, wind {a['wind_MW']:,.0f}, solar {a['solar_MW']:,.0f}, "
          f"batt {a['batt_MW']:,.0f}; VRE {a['vre_share']:.1%})  ${a['lcoe']:.2f} vs gas-only ${g['lcoe']:.2f} "
          f"(saving {a['saving_per_MWh']:.2f})  util {a['utilization']:.1%}  fixed {n}")
json.dump(out, open(f'{HERE}/out/sensitivity_with_vre/vre_allowed_summary.json', 'w'), indent=1)
