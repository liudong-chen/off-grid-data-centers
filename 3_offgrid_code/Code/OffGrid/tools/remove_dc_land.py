"""Remove the data center's own land from every off-grid / hybrid cost.

The data center needs its land on or off the grid, so it is not part of the supply
comparison. It entered the off-grid objective as a constant per site (0.75 acre/MW of
peak x zonal price x 8%), so no design changes; only the reported costs do.
Backs up every touched file to out/_before_dcland_20261008/<relpath> first.
Run from Code/OffGrid:   python3 tools/remove_dc_land.py
"""
import json, os, shutil, glob

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BK = f'{HERE}/out/_before_dcland_20261008'
LAND = {'North': 1844.0, 'FarWest': 714.0, 'West': 2787.0, 'East': 9313.0,
        'NorthCentral': 9313.0, 'Coast': 11423.0, 'South': 5970.0, 'SouthCentral': 7704.0,
        'Georgia': 4670.0}
DC, CRF = 0.75, 0.08
log = []

def load(rel):
    return json.load(open(f'{HERE}/{rel}'))

def save(rel, d):
    src = f'{HERE}/{rel}'; dst = f'{BK}/{rel}'
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)
    json.dump(d, open(src, 'w'), indent=1, default=float)

def fix_mip(r, tag):
    """MIP record: acres = 0.1 gas + 5 s + 1 w + 0.05 b + 0.75 peak; land_cost excludes VRE land."""
    dca = DC*r['peak_MW']
    vre = 5.0*(r.get('solar_MW') or 0) + 1.0*(r.get('wind_MW') or 0) + 0.05*(r.get('batt_MW') or 0)
    base = r['acres'] - vre
    chk = 0.10*r['installed_MW'] + dca
    assert abs(base - chk) < 0.01*chk + 1, (tag, base, chk)
    dl = r['land_cost']*dca/base
    r['land_cost'] -= dl; r['cost'] -= dl; r['acres'] -= dca
    r['lcoe'] = r['cost']/r['energy_MWh']
    return dl

def fix_hyb_case(c, peak, energy, zone, tag):
    dca = DC*peak
    base = 0.10*(c['gas_MW'] + c['diesel_MW']) + dca
    price = c['land']/base/CRF
    assert abs(price - LAND[zone]) < 0.01*LAND[zone], (tag, price, zone)
    dl = dca*LAND[zone]*CRF
    c['land'] -= dl; c['total'] -= dl; c['lcoe'] = c['total']/energy
    return dl

def mip_list(rel):
    d = load(rel); tot = 0
    for r in d: tot += fix_mip(r, rel)
    save(rel, d); log.append((rel, tot))

def mip_pool(rel):
    d = load(rel); tot = 0
    for k, r in d.items(): tot += fix_mip(r, f'{rel}:{k}')
    save(rel, d); log.append((rel, tot))

# 1. MIP site and pool files
for rel in ['out/offgrid_mip_gasonly.json', 'out/offgrid_mip_2035_k0p5_gasonly_sites.json',
            'out/offgrid_mip_2035_k3_gasonly_sites.json', 'out/offgrid_mip_georgia_gasonly_sites.json',
            'scenario_two_unit_sizes/two_unit_sizes_sites.json', 'out/sensitivity_with_vre/offgrid_mip_final.json']:
    mip_list(rel)
for rel in ['out/offgrid_mip_pool_gasonly.json', 'out/offgrid_mip_2035_k0p5_gasonly_pool.json',
            'out/offgrid_mip_2035_k3_gasonly_pool.json', 'out/offgrid_mip_georgia_gasonly_pool.json',
            'scenario_two_unit_sizes/two_unit_sizes_pool.json', 'out/sensitivity_with_vre/offgrid_mip_pool_final.json']:
    mip_pool(rel)

# 2. hybrid sweeps (site_*.json and all_sites.json in each folder)
for folder in ['out/hybrid_diesel', 'out/hybrid_diesel_2035_k0p5', 'out/hybrid_diesel_2035_k3',
               'out/hybrid_diesel_georgia', 'out/hybrid_2phase']:
    rels = [os.path.relpath(p, HERE) for p in sorted(glob.glob(f'{HERE}/{folder}/site_*.json'))]
    rels += [f'{folder}/all_sites.json'] if os.path.exists(f'{HERE}/{folder}/all_sites.json') else []
    for rel in rels:
        d = load(rel); S = d if isinstance(d, list) else [d]; tot = 0
        for s in S:
            z = s['zone']
            for k, c in s['cases'].items():
                tot += fix_hyb_case(c, s['peak_MW'], s['energy_MWh'], z, f'{rel}:{s["site"]}:{k}')
        save(rel, d); log.append((rel, tot))

# 3. pure-renewable scenario
rel = 'scenario_renewable_only/renewable_only_sites.json'
d = load(rel); tot = 0
for r in d:
    dca = DC*r['peak_MW']; dl = r['land_cost']*dca/r['acres']
    r['land_cost'] -= dl; r['total_cost'] -= dl; r['acres'] -= dca; r['lcoe'] = r['total_cost']/r['energy_MWh']; tot += dl
save(rel, d); log.append((rel, tot))
rel2 = 'scenario_renewable_only/renewable_only_summary.json'
s = load(rel2); E = sum(r['energy_MWh'] for r in d)
s['cost_Byr'] = sum(r['total_cost'] for r in d)/1e9; s['lcoe'] = s['cost_Byr']*1e9/E; s['acres'] = sum(r['acres'] for r in d)
save(rel2, s)

# 4. free siting
rel = 'out/S1_free_siting_gasonly_sites.json'
d = load(rel); tot = 0
for r in d:
    dh = DC*r['peak_MW']*LAND[r['home_zone']]*CRF; db = DC*r['peak_MW']*LAND[r['best_zone']]*CRF
    r['home_cost'] -= dh; r['best_cost'] -= db
    r['home_lcoe'] = r['home_cost']/r['energy_MWh']; r['best_lcoe'] = r['best_cost']/r['energy_MWh']; tot += dh
save(rel, d); log.append((rel, tot))
rel2 = 'out/S1_free_siting_gasonly.json'
s = load(rel2); E = s['energy_MWh']
s['offgrid_home_cost'] = sum(r['home_cost'] for r in d); s['offgrid_free_cost'] = sum(r['best_cost'] for r in d)
s['offgrid_home_lcoe'] = s['offgrid_home_cost']/E; s['offgrid_free_lcoe'] = s['offgrid_free_cost']/E
s['saving_per_MWh'] = s['offgrid_home_lcoe'] - s['offgrid_free_lcoe']; s['saving_Byr'] = (s['offgrid_home_cost'] - s['offgrid_free_cost'])/1e9
save(rel2, s)

# 5. Georgia decomposition
def fix_decomp_case(c):
    tot = 0
    for r in c['sites']:
        dl = DC*r['peak_MW']*c['land']*CRF; r['cost'] -= dl; r['lcoe'] = r['cost']/(r['cost']/r['lcoe'] + 0) if False else None; tot += dl
    return tot
for tag in ['G1', 'G2', 'G4']:
    rel = f'out/georgia_decomp_gasonly_{tag}.json'
    c = load(rel); tot = 0
    for r in c['sites']:
        e = r['cost']/r['lcoe']; dl = DC*r['peak_MW']*c['land']*CRF
        r['cost'] -= dl; r['lcoe'] = r['cost']/e; tot += dl
    c['cost'] -= tot; c['lcoe'] = c['cost']/c['energy_MWh']
    save(rel, c); log.append((rel, tot))
rel = 'out/georgia_decomp_gasonly.json'
dd = load(rel)
ga = load('out/offgrid_mip_georgia_gasonly_sites.json')
for c in dd['cases']:
    t = c['tag'].split()[0]
    if t == 'G0':
        c['cost'] = sum(r['cost'] for r in ga); c['lcoe'] = c['cost']/c['energy_MWh']
    else:
        g = load(f'out/georgia_decomp_gasonly_{t}.json'); c['cost'] = g['cost']; c['lcoe'] = g['lcoe']
er = load('out/offgrid_mip_gasonly.json')
dd['ercot_reference_lcoe'] = sum(r['cost'] for r in er)/sum(r['energy_MWh'] for r in er)
dd['note'] = dd.get('note', '') + ' | : data center land removed from all costs.'
save(rel, dd)

for rel, t in log:
    print(f'{rel:<62} -{t/1e6:8.3f} M$/yr')
print('ERCOT 2030 fleet lcoe', dd['ercot_reference_lcoe'])
