"""GAS-ONLY variant: solar/wind/battery disallowed everywhere, tag suffix _gasonly.
The MIP (offgrid_mip.design) on the paper's other cases, with exactly the inputs the
published frontier-search runs used:

    python3 run_offgrid_mip_cases.py georgia      # fig5: 31 Georgia Power campuses, 12,400 MW
    python3 run_offgrid_mip_cases.py 2035 0.5     # fig4: 2035 inputs, 20 GW fleet
    python3 run_offgrid_mip_cases.py 2035 3.0     # fig4: 2035 inputs, 121 GW fleet
    -> out/offgrid_mip_<case>_sites.json, out/offgrid_mip_<case>_pool.json

Georgia follows Figures/fig5_package/02_code/80_offgrid_georgia.py (fleet construction, gas
$5.688, land $4,670, ERCOT-Coast solar shape at 23.3%, no wind, 2030 capex anchor).
2035 follows Figures/fig4_package/02_code/76_offgrid_2035.py (loads x k, capex re-anchored to
the 2035 NGCC/NGCT $113,755/$96,576, 2035 zonal gas, wind $149,412, solar $108,393, VRE
profiles +3%). The 4-hour battery is not in the paper's 2035 set; it is scaled by the same
2035/2030 ratio as solar (x0.919) so it is not disadvantaged.
Env: BYONG_CASE, BYONG_LOAD as run_offgrid_mip.py; BYONG_FREE/BYONG_SENS default ON here.
"""
import os, sys, json, time
import numpy as np, pandas as pd
import offgrid_mip as om

CASE = sys.argv[1]
CD = os.environ.get('BYONG_CASE', 'Code/HOPE/ModelCases/TX8z_GTEP_dc_vre_case/Data_TX8z_GTEP_dc_vre')
R = os.environ.get('BYONG_LOAD', 'DC_Load/load_profiles')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'); os.makedirs(OUT, exist_ok=True)
FREE = os.environ.get('BYONG_FREE', '1') == '1'
SENS = os.environ.get('BYONG_SENS', '1') == '1'
KW = dict(credit_cap=1.0 if FREE else 0.15, vre_cut_frac=0.10 if FREE else 0.0, sens_cuts=SENS)
ZM = {'Far West': 'FarWest', 'North Central': 'NorthCentral', 'South Central': 'SouthCentral'}

d = pd.read_csv(f'{R}/dc_load_hourly_MW.csv', index_col=0, parse_dates=True)
d = d[~((d.index.month == 2) & (d.index.day == 29))].drop(columns=['TOTAL_MW'])
meta = pd.read_csv(f'{R}/site_parameters_and_results.csv')
meta['zone'] = meta.ercot_weather_zone.map(lambda z: ZM.get(z, z))


def zonal_cf(zone, typ, uplift=1.0):
    g = pd.read_csv(f'{CD}/gendata.csv'); c = pd.read_csv(f'{CD}/gendata_candidate.csv')
    k = c.index[(c.Zone == zone) & (c.Type == typ)]
    if not len(k):
        return None
    av = pd.read_csv(f'{CD}/gen_availability_timeseries.csv', usecols=[f'G{len(g)+int(k[0])+1}'])
    v = av.values[:om.H, 0]*uplift
    assert v.max() <= 1.0
    return v


def report(tag, r, t0):
    mix = ' '.join(f'{k}x{v}' for k, v in r['mix'].items())
    print(f"{tag:<16}{r['zone']:<13}{r['peak_MW']:>9,.0f}  {mix:<36}{r['solar_MW']:>7,.0f}"
          f"{r['wind_MW']:>7,.0f}{r['batt_MW']:>6,.0f}{r['installed_MW']:>9,.0f}{r['ratio']:>6.2f}"
          f"{r['LOLE']:>7.3f}{r['lcoe']:>8.2f}{r['n_cuts']:>5}{time.time()-t0:>6.0f}s", flush=True)


def run(loads, zones, cfs, cfw, pool_zones, tag, extra):
    """loads: dict name -> 8760 array; zones: name -> zone; cfs/cfw: zone -> profile."""
    print(f"{'campus':<16}{'zone':<13}{'peak':>9}  {'portfolio':<36}{'solar':>7}{'wind':>7}{'batt':>6}"
          f"{'MW':>9}{'xpk':>6}{'LOLE':>7}{'$/MWh':>8}{'cuts':>5}{'s':>7}", flush=True)
    rows = []
    for name, load in loads.items():
        z = zones[name]; t0 = time.time()
        r = om.design(load, z, cfs.get(z), cfw.get(z), allow=(False, False, False),
                      **KW, **extra(z))
        if r is None:
            print(f'{name}: NO DESIGN', flush=True); continue
        r.update(site=name, home_zone=z); rows.append(r); report(name, r, t0)
        json.dump(rows, open(f'{OUT}/offgrid_mip_{tag}_gasonly_sites.json', 'w'), indent=1, default=float)
    pk = sum(r['peak_MW'] for r in rows); mw = sum(r['installed_MW'] for r in rows)
    c = sum(r['cost'] for r in rows); e = sum(r['energy_MWh'] for r in rows)
    print(f'\nSEPARATE {len(rows)} campuses: {pk:,.0f} MW peak  {mw:,.0f} MW built ({mw/pk:.2f}x)  '
          f'${c/1e9:.3f} B/yr  ${c/e:.2f}/MWh', flush=True)
    tot = np.sum(list(loads.values()), axis=0); pool = {}
    for z in pool_zones:
        t0 = time.time()
        r = om.design(tot, z, cfs.get(z), cfw.get(z), allow=(False, False, False),
                      time_limit=300.0, **KW, **extra(z))
        if r is None:
            print(f'POOL {z}: NO DESIGN', flush=True); continue
        pool[z] = r; report('POOLED', r, t0)
        json.dump(pool, open(f'{OUT}/offgrid_mip_{tag}_gasonly_pool.json', 'w'), indent=1, default=float)


if CASE == 'georgia':
    GA_GAS, GA_LAND, GA_SOLAR_CF = 5.688, 4670.0, 0.265*0.88
    GA_FLEET_MW, GA_N, CAMELLIA_MW = 12400.0, 31, 3200.0
    f = om.anchor_capex(); print(f'capex anchored (2030): CT x{f["CT"]:.5f} CC x{f["CC"]:.5f}')
    scf = zonal_cf('Coast', 'SolarPV'); scf = scf*(GA_SOLAR_CF/scf.mean())
    # the fleet, exactly as 80_offgrid_georgia.py builds it
    pk = d.max(); shape = (pk/pk.sum()).sort_values(ascending=False)
    rest_MW = GA_FLEET_MW - CAMELLIA_MW; src = list(shape.index)
    q_src = np.linspace(0, 1, len(src)); q_new = np.linspace(0, 1, GA_N - 1)
    w = np.interp(q_new, q_src, [shape[s] for s in src]); w = w/w.sum()
    FLOOR = 100.0/rest_MW
    for _ in range(50):
        lo = w < FLOOR
        if not lo.any():
            break
        w = np.where(lo, FLOOR, w*(1 - lo.sum()*FLOOR)/w[~lo].sum())
    arche = [src[min(int(round(q*(len(src)-1))), len(src)-1)] for q in q_new]
    sizes = [('GA01_Camellia', src[0], CAMELLIA_MW)] + \
            [(f'GA{i+2:02d}', arche[i], float(rest_MW*w[i])) for i in range(GA_N - 1)]
    loads = {n: d[s].values[:om.H]*(mw/d[s].values[:om.H].max()) for n, s, mw in sizes}
    print(f'Georgia: {len(loads)} campuses, {sum(v.max() for v in loads.values()):,.0f} MW peak, '
          f'gas ${GA_GAS}, land ${GA_LAND:,.0f}, solar CF {scf.mean():.3f}, no wind', flush=True)
    run(loads, {n: 'Georgia' for n in loads}, {'Georgia': scf}, {}, ['Georgia'], 'georgia',
        lambda z: dict(gas=GA_GAS, land=GA_LAND))

elif CASE == '2035':
    K = float(sys.argv[2]); tag = f'2035_k{K:g}'.replace('.', 'p')
    f = om.anchor_capex({'CC': 113755.0, 'CT': 96576.0})
    print(f'capex anchored (2035 NGCC/NGCT): CT x{f["CT"]:.5f} CC x{f["CC"]:.5f}')
    ratio = 108393.0/117995.0
    om.WIND_FIXED, om.SOLAR_FIXED, om.BATT_FIXED = 149412.0, 108393.0, om.BATT_FIXED*ratio
    G35 = {'Coast': 5.0, 'East': 4.866, 'FarWest': 3.997, 'NorthCentral': 5.187, 'South': 5.348,
           'SouthCentral': 5.044, 'West': 4.501, 'North': 4.168}   # fig4 offgrid2035_*_sites.json
    zones = dict(zip(meta.site_id, meta.zone))
    loads = {s: d[s].values[:om.H]*K for s in meta.site_id}
    cfs = {z: zonal_cf(z, 'SolarPV', 1.03) for z in G35}; cfw = {z: zonal_cf(z, 'WindOn', 1.03) for z in G35}
    print(f'2035 x{K:g}: wind ${om.WIND_FIXED:,.0f} solar ${om.SOLAR_FIXED:,.0f} batt ${om.BATT_FIXED:,.0f}/MW-yr; '
          f'fleet {sum(v.max() for v in loads.values()):,.0f} MW sum of peaks', flush=True)
    run(loads, zones, cfs, cfw, ['FarWest', 'SouthCentral', 'NorthCentral'], tag, lambda z: dict(gas=G35[z]))
