"""Write the GAS-ONLY off-grid designs in the file shapes the figure code reads.

    python3 prepare_offgrid_results.py <Code/OffGrid/out dir>

Reads  offgrid_mip_gasonly.json, offgrid_mip_pool_gasonly.json (ERCOT 2030, allow=(F,F,F)),
       offgrid_mip_2035_k{0p5,3}_gasonly_{sites,pool}.json, offgrid_mip_georgia_gasonly_{sites,pool}.json,
       georgia_decomp_gasonly.json, hybrid_diesel/all_sites.json, hybrid_diesel/hybrid_diesel_summary.json
Writes into the three packages' 01_raw: offgrid_{sites,pool,summary}.json (SAME names as the
published package so 69b/87 run unchanged -- but gas-only), offgrid2035_*, offgrid_georgia_*,
georgia_decomp.json, hybrid_*.json.  Then run decompose_overbuild.py and calculate_opportunity_cost.py
on figures_1_to_3/01_raw.
"""
import json
import os
import shutil
import sys

SRC = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
NEW = os.path.dirname(os.path.dirname(HERE))
RAW23, RAW4, RAW5 = f'{NEW}/figures_1_to_3/01_raw', f'{NEW}/figure_4/01_raw', f'{NEW}/figure_5/01_raw'


def rec(r):
    keep = ['site', 'home_zone', 'zone', 'peak_MW', 'energy_MWh', 'n', 'mix', 'unit_count', 'installed_MW',
            'ratio', 'LOLE', 'solar_MW', 'wind_MW', 'batt_MW', 'gas', 'fixed_cost', 'land_cost', 'vre_cost',
            'fuel_cost', 'cost', 'lcoe', 'acres', 'n_minus_k', 'n_cuts', 'capacity_credits', 'seconds']
    o = {k: r[k] for k in keep if k in r}
    o['method'] = 'mip, gas-only (allow solar/wind/battery = False)'
    return o


def summary(sites, pool, extra):
    e = sum(r['energy_MWh'] for r in sites); c = sum(r['cost'] for r in sites)
    s = dict(peak_MW=sum(r['peak_MW'] for r in sites), installed_MW=sum(r['installed_MW'] for r in sites),
             cost=c, energy=e, lcoe=c/e, wind_MW=0.0, solar_MW=0.0, batt_MW=0.0)
    p = {z: dict(installed_MW=r['installed_MW'], cost=r['cost'], lcoe=r['lcoe'], wind_MW=0.0, solar_MW=0.0,
                 batt_MW=0.0, ratio=r['ratio'], mix=r['mix'], peak_MW=r['peak_MW'], LOLE=r['LOLE'])
         for z, r in pool.items()}
    return dict(extra, separate=s, pooled=p, method='mip gas-only (offgrid_mip.py, allow=(False,False,False), 2,000-rep confirmation)')


def emit(src_sites, src_pool, dest, name, extra):
    if not (os.path.exists(src_sites) and os.path.exists(src_pool)):
        print(f'   {name}: inputs missing ({src_sites}, {src_pool}), skipped'); return
    S = [rec(r) for r in json.load(open(src_sites))]
    P = {z: rec(r) for z, r in json.load(open(src_pool)).items()}
    json.dump(S, open(f'{dest}/{name}_sites.json', 'w'), indent=1, default=float)
    json.dump(P, open(f'{dest}/{name}_pool.json', 'w'), indent=1, default=float)
    sm = summary(S, P, extra)
    json.dump(sm, open(f'{dest}/{name}_summary.json', 'w'), indent=1, default=float)
    best = min(P.values(), key=lambda r: r['cost'])
    print(f"   {name}: {len(S)} campuses {sm['separate']['installed_MW']:,.0f} MW ${sm['separate']['lcoe']:.2f}/MWh; "
          f"pooled {best['zone']} {best['installed_MW']:,.0f} MW ${best['lcoe']:.2f}/MWh")


for dest in (RAW23, RAW5):
    emit(f'{SRC}/offgrid_mip_gasonly.json', f'{SRC}/offgrid_mip_pool_gasonly.json', dest, 'offgrid', {})
emit(f'{SRC}/offgrid_mip_georgia_gasonly_sites.json', f'{SRC}/offgrid_mip_georgia_gasonly_pool.json', RAW5,
     'offgrid_georgia', dict(region='GeorgiaPower', year=2030, gas=5.688, land=4670.0, solar_cf=None,
                             wind_available=False, n_campuses=31, fleet_MW=12400.0))
for k, tag in ((0.5, 'k0p5'), (3.0, 'k3')):
    emit(f'{SRC}/offgrid_mip_2035_{tag}_gasonly_sites.json', f'{SRC}/offgrid_mip_2035_{tag}_gasonly_pool.json',
         RAW4, f'offgrid2035_{tag}', dict(k=k, year=2035))
if os.path.exists(f'{SRC}/georgia_decomp_gasonly.json'):
    shutil.copyfile(f'{SRC}/georgia_decomp_gasonly.json', f'{RAW5}/georgia_decomp.json'); print('   georgia_decomp.json (gas-only)')
for f in ('all_sites.json', 'hybrid_diesel_summary.json', 'hybrid_diesel_sites.csv'):
    if os.path.exists(f'{SRC}/hybrid_diesel/{f}'):
        shutil.copyfile(f'{SRC}/hybrid_diesel/{f}', f'{RAW23}/' + ('hybrid_all_sites.json' if f == 'all_sites.json' else f)); print(f'   {f}')
# hybrid (diesel-first) summaries for the other scenarios -> fig4 / fig5 raw
for tag, dest, name in (('2035_k0p5', RAW4, 'hybrid_2035_k0p5_summary.json'), ('2035_k3', RAW4, 'hybrid_2035_k3_summary.json'),
                        ('georgia', RAW5, 'hybrid_georgia_summary.json')):
    src = f'{SRC}/hybrid_diesel_{tag}/hybrid_diesel_summary.json'
    if os.path.exists(src):
        shutil.copyfile(src, f'{dest}/{name}'); print(f'   {name}')
if os.path.exists(f'{SRC}/hybrid_diesel/hybrid_diesel_summary.json'):
    shutil.copyfile(f'{SRC}/hybrid_diesel/hybrid_diesel_summary.json', f'{RAW5}/hybrid_ercot_summary.json'); print('   hybrid_ercot_summary.json (fig5)')
for f in ('S1_free_siting_gasonly.json', 'S1_free_siting_gasonly_sites.json'):
    if os.path.exists(f'{SRC}/{f}'):
        shutil.copyfile(f'{SRC}/{f}', f'{RAW23}/{f}'); print(f'   {f}')
print('done; now: python3 decompose_overbuild.py ../01_raw && python3 calculate_opportunity_cost.py')
