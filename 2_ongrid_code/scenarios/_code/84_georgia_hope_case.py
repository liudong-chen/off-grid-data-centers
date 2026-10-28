"""Package the Georgia Power case as HOPE ModelCases, ready to run.

Writes, in the same layout as the ERCOT cases:

    GA1z_GTEP_base_case/   Data_GA1z_GTEP_base/    Settings/  case_info.json
    GA1z_GTEP_dc_case/     Data_GA1z_GTEP_dc/      Settings/  case_info.json

    HOPE.run_hope("ModelCases/GA1z_GTEP_base_case/")
    HOPE.run_hope("ModelCases/GA1z_GTEP_dc_case/")

ONE ZONE, so `transmission_expansion` is 0 and the two line files are headers only.  That
is the case design, not a simplification of convenience: Georgia Power is a single
integrated system, no public transfer limits exist for it, and the point of the comparison
is that Georgia has no cheap-gas corner to move generation to.  ERCOT's whole on-grid
advantage was putting 29.6 GW on $3.95 Permian gas and shipping the power; that option
does not exist here.

EVERYTHING IN THE DATA IS IN GEORGIA_PARAMETERS.md WITH A SOURCE.  The two things that
are derived rather than found, and are the ones to argue about:

  * the purchased-capacity block, sized so the 2025 system meets Georgia Power's own 26%
    winter target reserve margin against its 16,284 MW peak.  A utility that buys from the
    Southern pool has to be represented somehow and this is the transparent way.
  * the solar capacity credit.  Georgia Power credits standalone solar at 5% because it
    plans to an extreme winter morning.  A normal weather year does not contain that
    event: in the 2025 SOCO data the highest hour is 22 January 09:00 and the second is
    28 July 17:00, 0.2% below, and the data-centre load tips the binding hour into June.
    The shipped cases carry the credit DERIVED from the solar profile at each case's own
    top-20 load hours; `--cc 0.05` writes the Georgia Power convention instead, and the
    difference is worth $3.67/MWh.  Run both.

THE PLANNING RESERVE MARGIN IN THE SHIPPED FILES IS GEORGIA POWER'S 26% WINTER TARGET,
not a derived fixed point.  It is a different object from ERCOT's, which this study
derives from a 0.1 day/yr LOLE criterion, and the difference is itself a finding: running
Georgia to its own standard roughly doubles the reserve the data-centre load must carry.
Re-derive it with the fixed-point code if you want the LOLE-consistent number; both belong
in the paper.

    python 84_georgia_hope_case.py [--cc 0.05]
    -> ga_hope/GA1z_GTEP_{base,dc}_case/
"""
import json, os, shutil, sys
import numpy as np
import pandas as pd

B = 'build'
OUT = f'{B}/ga_hope'
SRC_SETTINGS = '2_ongrid_code/HOPE/ModelCases/TX8z_GTEP_dc_vre_case/Settings'
SOCO = '1_source_data/parameter_setting/SOCO_hourly_2025.csv'
CD = f'{B}/TX8z_GTEP_cases_vre/TX8z_GTEP_dc_case/Data_TX8z_GTEP_dc'
R = '1_source_data/DC_Load/load_profiles'
H, Z = 8760, 'Georgia'
CC_OVERRIDE = None
if '--cc' in sys.argv:
    CC_OVERRIDE = float(sys.argv[sys.argv.index('--cc') + 1])

GP_PEAK_2025, GROWTH, DC_SHARE = 16284.0, 8205.0, 0.80
TRM_WINTER, TRM_SUMMER = 0.26, 0.1625   # Georgia Power long-term seasonal targets,
                                        # 2022 IRP, reaffirmed 2023 IRP Update
GAS, LAND = 5.688, 4670.0
# GAS: measured, not assumed.  EIA withholds the STATE electric-power series for Georgia
# (N3045GA3A) to protect Southern Company, but the PLANT-LEVEL receipts in EIA-923
# Schedule 2 Page 5 are not withheld, and they are the same source the ERCOT zonal prices
# come from.  Quantity-weighted over the 15 Georgia plants reporting in 2025:
# $4.369/MMBtu (344.7 million MMBtu; Georgia Power's own four plants weight to $4.382,
# so the state figure is not driven by non-utility plants).  ERCOT measures $3.137 the
# same way and shifts it additively by +$1.319 to reach its 2030 level of $4.456; the
# SAME shift applied to Georgia gives 4.369 + 1.319 = $5.688/MMBtu.  The previous value,
# $5.35, was Henry Hub $5.00 plus an assumed $0.35 Southeast basis adder.
SOLAR_CF = 0.265*0.88
CC_BATT, FLEET_MW, CAMELLIA = 0.95, 12400.0, 3200.0
# The existing fleet comes from 85_georgia_fleet.py: EIA-860 2025 ER unit counts for
# Georgia Power Co, with MW scaled to the owned totals in Facts & Figures 2025.  It is
# read rather than retyped so the HOPE case and the Monte Carlo cache cannot drift apart.
FL = json.load(open(f'{B}/out/georgia_fleet.json'))
OWNED = [(r['Type'], r['mw'], r['n_units']) for r in FL['firm']
         if r['Type'] != 'Purchase'] + [('SolarPV', FL['solar']['mw'],
                                         FL['solar']['n_units'])]
PURCH = FL['purchased_MW']; PURCH_N = FL['purchased_units']
BATT_MW = FL['battery']['mw']

COST = {'NGCC': 118650.0, 'NGCT': 103382.0, 'SolarPV': 117995.0}
BATT = 161512.0
HR = {'NGCC': 6.226, 'NGCT': 9.142}
VOM = {'NGCC': 3.57, 'NGCT': 4.17}
HR['NGST'] = 10.5
VOM['NGST'] = 4.17
FUEL = {'NGCC': HR['NGCC']*GAS + VOM['NGCC'], 'NGCT': HR['NGCT']*GAS + VOM['NGCT'],
        'NGST': HR['NGST']*GAS + VOM['NGST'],
        'Coal': 10.0*2.20 + 4.5, 'Nuclear': 8.0, 'Hydro': 0.0, 'SolarPV': 0.0,
        'Oil': 10.5*18.0 + 5.0,                      # distillate peakers, EIA-923 level
        'Purchase': HR['NGCT']*GAS + VOM['NGCT']}
# FOR and MTTR are the same TRE-calibrated GADS values the ERCOT case uses, and they are
# the values the Monte Carlo cache carries -- 85_georgia_fleet.py is the single source.
FOR_ = {r['Type']: r['FOR'] for r in FL['firm']}
FOR_['SolarPV'] = 0.0
EF = {'NGCC': 0.3304, 'NGCT': 0.3304, 'NGST': 0.3304, 'Coal': 0.9500, 'Nuclear': 0.0,
      'Hydro': 0.0, 'Oil': 0.7000, 'SolarPV': 0.0, 'Purchase': 0.3304}

# ---------------------------------------------------------------------------- load
s = pd.read_csv(SOCO)
soco = pd.to_numeric(s['MW'], errors='coerce').values[:H]
gp_base = soco*((GP_PEAK_2025 + (1 - DC_SHARE)*GROWTH)/soco.max())

d = pd.read_csv(f'{R}/dc_load_hourly_MW.csv', index_col=0, parse_dates=True)
d = d[~((d.index.month == 2) & (d.index.day == 29))].drop(columns=['TOTAL_MW'])
pk = d.max(); shape = (pk/pk.sum()).sort_values(ascending=False); src = list(shape.index)
rest = FLEET_MW - CAMELLIA
q = np.linspace(0, 1, 30)
w = np.interp(q, np.linspace(0, 1, len(src)), [shape[x] for x in src]); w = w/w.sum()
F = 100.0/rest
for _ in range(50):
    lo = w < F
    if not lo.any():
        break
    w = np.where(lo, F, w*(1 - lo.sum()*F)/w[~lo].sum())
ar = [src[min(int(round(x*(len(src)-1))), len(src)-1)] for x in q]
sizes = [(src[0], CAMELLIA)] + [(ar[i], float(rest*w[i])) for i in range(30)]
dc = np.zeros(H)
for site, mw in sizes:
    v = d[site].values[:H]
    dc += v*(mw/v.max())
gp_dc = gp_base + dc

# --------------------------------------------------------------------- solar profile
g0 = pd.read_csv(f'{CD}/gendata.csv'); c0 = pd.read_csv(f'{CD}/gendata_candidate.csv')
kk = c0.index[(c0.Zone == 'Coast') & (c0.Type == 'SolarPV')][0]
scf = pd.read_csv(f'{CD}/gen_availability_timeseries.csv',
                  usecols=[f'G{len(g0)+int(kk)+1}']).values[:H, 0]
scf = scf*(SOLAR_CF/scf.mean())

# a real calendar for the timeseries files, 2025 non-leap.
#
# HOPE cross-checks EVERY timeseries file against the load file ROW BY ROW on
# 'Time Period' and 'Hours'.  So the four time columns are built ONCE, here, and both
# writers below paste the same object in front of their data.  Writing them twice is how
# the first version of this script shipped a case that failed with
#   "Timeseries time mapping mismatch between reference and gen_availability_timeseries"
# because the availability file carried no time columns at all.
#
# AND THE COLUMNS DO NOT MEAN WHAT THEIR NAMES SUGGEST.  In HOPE's full-chronology mode
# there is exactly ONE representative period covering the whole year, so
#
#     Time Period  is the period INDEX and is 1 on every row.  Numbering it 1..8760
#                  declares 8760 representative periods and trips
#                  "use representative-day mode for multiple Time Period values".
#     Hours        is the CUMULATIVE hour within that period, 1..8760 -- not the hour
#                  of the day.  1..24 repeating is the representative-day convention.
#     Month, Day   are the real calendar, and are what make the output readable.
#
# Verified against Data_TX8z_GTEP_dc/load_timeseries_regional.csv, which the solver
# accepts: Time Period unique = {1}, Hours = 1..8760.
CAL = pd.date_range('2025-01-01', periods=H, freq='h')
TIME = pd.DataFrame({'Time Period': np.ones(H, dtype=int), 'Month': CAL.month,
                     'Day': CAL.day, 'Hours': np.arange(1, H + 1)})


# The binding season, as a boolean mask over the 8,760 hours.  Georgia Power plans to a
# 26% WINTER target and a 16.25% summer target, and the winter requirement binds in both
# cases -- 22,586 against 20,793 MW in the base case, 32,249 against 32,165 with the data
# centres.  See georgia_case.tex.
WINTER = np.asarray((CAL.month <= 2) | (CAL.month == 12))
SUMMER = np.asarray((CAL.month >= 6) & (CAL.month <= 8))


def cc_solar(load, n=20):
    """Solar capacity credit: mean output over the tightest hours OF THE BINDING SEASON.

    The earlier version averaged over the top-20 hours overall.  That is the right rule
    in ERCOT, where the peak and the binding requirement are the same summer afternoons.
    In Georgia they are not: the 26% winter target is high enough that WINTER sets the
    capacity requirement even after 12.4 GW of flat data-centre load pushes the raw peak
    into June.  Crediting solar for showing up in hours that do not set the requirement
    inflated the credit to 0.63 in the base case and 0.74 with the data centres; over the
    tightest winter hours the same rule gives 0.135 and 0.128.

    Those two are within 0.7 pp of each other, so ONE value is used for both cases -- the
    base-case figure, the reference system against which the increment is measured.  That
    keeps the reported increment dc - base on a single accreditation basis, which the two
    different credits did not.
    """
    if CC_OVERRIDE is not None:
        return CC_OVERRIDE
    idx = np.flatnonzero(WINTER)
    return float(scf[idx[np.argsort(-load[idx])[:n]]].mean())


def write(tag, load, k):
    case = f'{OUT}/GA1z_GTEP_{tag}_case'
    dat = f'{case}/Data_GA1z_GTEP_{tag}'
    os.makedirs(dat, exist_ok=True)
    shutil.copytree(SRC_SETTINGS, f'{case}/Settings', dirs_exist_ok=True)
    y = open(f'{case}/Settings/HOPE_model_settings.yml').read()
    y = y.replace('Data_TX8z_GTEP_dc_vre/', f'Data_GA1z_GTEP_{tag}/')
    y = y.replace('transmission_expansion: 1', 'transmission_expansion: 0')
    # The ERCOT Settings this is copied from use a barrier interior point without
    # crossover.  On the Georgia base case that returned a point 32.7% above the optimum
    # found by an independent LP on identical inputs, and because the reported number is
    # the difference dc - base, two solves that stop short by different amounts do not
    # give an increment of anything.  Crossover is on here by construction, not by hand.
    gs = f'{case}/Settings/gurobi_settings.yml'
    if os.path.exists(gs):
        t = open(gs).read()
        t = t.replace('Crossover: 0', 'Crossover: 1')
        t = t.replace('NumericFocus: 0', 'NumericFocus: 1')
        open(gs, 'w').write(t)
    # the inherited comments describe ERCOT; correct them so a shipped case cannot
    # mislead a later reader about what its own settings mean
    y = y.replace("planning_reserve_mode: 1         # system RA at ERCOT's 13.75% Board target",
                  "planning_reserve_mode: 1         # system RA at Georgia Power's 26% WINTER target")
    y = y.replace('transmission_expansion: 0     # 17 candidate 345 kV corridors in linedata_candidate.csv',
                  'transmission_expansion: 0     # one zone: Georgia Power is a single integrated system')
    y = y.replace('clean_energy_policy: 0           # Texas has no binding RPS',
                  'clean_energy_policy: 0           # Georgia has no binding RPS')
    open(f'{case}/Settings/HOPE_model_settings.yml', 'w').write(y)
    assert f'Data_GA1z_GTEP_{tag}/' in y and 'transmission_expansion: 0' in y

    peak = float(load.max())
    CCS = CC_COMMON
    pd.DataFrame([{'Zone_id': Z, 'Demand (MW)': peak, 'State': 'GA'}]
                 ).to_csv(f'{dat}/zonedata.csv', index=False)
    pd.concat([TIME, pd.DataFrame({Z: load/peak, 'NI': 0.0})], axis=1
              ).to_csv(f'{dat}/load_timeseries_regional.csv', index=False)

    ge = pd.DataFrame([{'Pmax (MW)': mw, 'Pmin (MW)': 0.0, 'Zone': Z, 'Type': t,
                        'Flag_RET': 0, 'Flag_RPS': 0,
                        'Flag_thermal': int(t in ('NGCC', 'NGCT', 'NGST', 'Coal',
                                                  'Oil', 'Purchase')),
                        'Flag_VRE': int(t == 'SolarPV'), 'Flag_mustrun': 0,
                        'Cost ($/MWh)': FUEL[t], 'EF': EF[t],
                        'CC': CCS if t == 'SolarPV' else 1.0, 'AF': 1.0,
                        'FOR': FOR_[t], 'n_units': n}
                       for t, mw, n in OWNED + [('Purchase', PURCH, PURCH_N)]])
    ge.to_csv(f'{dat}/gendata.csv', index=False)

    cd = pd.DataFrame([{'Pmax (MW)': 100000.0, 'Pmin (MW)': 0.0, 'Zone': Z, 'Type': t,
                        'Cost ($/MW/yr)': COST[t], 'Cost ($/MWh)': FUEL[t],
                        'Flag_RPS': 0, 'Flag_thermal': int(t != 'SolarPV'),
                        'Flag_VRE': int(t == 'SolarPV'), 'Flag_mustrun': 0, 'EF': EF[t],
                        'CC': CCS if t == 'SolarPV' else 1.0, 'AF': 1.0, 'FOR': FOR_[t],
                        'n_units': 285} for t in ('NGCC', 'NGCT', 'SolarPV')])
    cd.to_csv(f'{dat}/gendata_candidate.csv', index=False)

    A = np.ones((H, len(ge) + len(cd)))
    A[:, [t for t in range(len(ge))][list(ge.Type).index('SolarPV')]] = scf
    A[:, len(ge) + list(cd.Type).index('SolarPV')] = scf
    pd.concat([TIME, pd.DataFrame(A, columns=[f'G{i+1}' for i in range(A.shape[1])])],
              axis=1).to_csv(f'{dat}/gen_availability_timeseries.csv', index=False)

    pd.DataFrame([{'Zone': Z, 'Type': 'Battery', 'Capacity (MWh)': BATT_MW*4.0,
                   'Max Power (MW)': BATT_MW, 'Charging efficiency': 0.922,
                   'Discharging efficiency': 0.922, 'Cost ($/MWh)': 0.0, 'EF': 0.0,
                   'CC': CC_BATT, 'Charging Rate': 1.0, 'Discharging Rate': 1.0}]
                 ).to_csv(f'{dat}/storagedata.csv', index=False)
    pd.DataFrame([{'Zone': Z, 'Type': 'Battery', 'Capacity (MWh)': 200000.0,
                   'Max Power (MW)': 50000.0, 'Charging efficiency': 0.922,
                   'Discharging efficiency': 0.922, 'Cost ($/MW/yr)': BATT,
                   'Cost ($/MWh)': 0.0, 'EF': 0.0, 'CC': CC_BATT,
                   'Charging Rate': 1.0, 'Discharging Rate': 1.0}]
                 ).to_csv(f'{dat}/storagedata_candidate.csv', index=False)

    lc = ['From_zone', 'To_zone', 'Forward Capacity (MW)', 'Reverse Capacity (MW)',
          'Cost (M$)', 'X', 'Loss (%)']
    pd.DataFrame(columns=lc).to_csv(f'{dat}/linedata.csv', index=False)
    pd.DataFrame(columns=lc).to_csv(f'{dat}/linedata_candidate.csv', index=False)
    for f in ('carbonpolicies.csv', 'rpspolicies.csv'):
        src_f = '2_ongrid_code/HOPE/ModelCases/TX8z_GTEP_dc_vre_case/Data_TX8z_GTEP_dc_vre/' + f
        if os.path.exists(src_f):
            t = pd.read_csv(src_f)
            t.to_csv(f'{dat}/{f}', index=False)

    # HOPE carries ONE reserve-margin constraint against the annual peak, and Georgia
    # Power plans against TWO seasonal targets at once.  The correct single number is
    # therefore the binding seasonal requirement expressed against the annual peak.
    req = max((1 + TRM_WINTER)*float(load[WINTER].max()),
              (1 + TRM_SUMMER)*float(load[SUMMER].max()))
    prm = req/peak - 1.0
    pd.DataFrame([{'VOLL': 35000.0, 'planning_reserve_margin': prm, 'BigM': 1e6,
                   'PT_RPS': 1e9, 'PT_emis': 1e9, 'spin_requirement': 0.03,
                   'delta_spin': 1/6, 'Inv_bugt_gen': 3.37e11,
                   'Inv_bugt_line': 3.3e10, 'Inv_bugt_storage': 5.5e10}]
                 ).to_csv(f'{dat}/single_parameter.csv', index=False)

    json.dump(dict(region='GeorgiaPower', year=2030, zones=1, k=k,
                   fleet_MW=FLEET_MW if k else 0.0, n_campuses=31 if k else 0,
                   peak_MW=peak, energy_TWh=float(load.sum()/1e6),
                   dc_coincident_MW=float(dc.max()) if k else 0.0,
                   dc_energy_TWh=float(dc.sum()/1e6) if k else 0.0,
                   gas=GAS, land=LAND, solar_cf=float(scf.mean()),
                   cc_solar=CCS, cc_solar_source=('Georgia Power 2023 IRP Update, '
                                                  'standalone winter'
                                                  if CC_OVERRIDE is not None else
                                                  'derived: mean solar output over the '
                                                  'top-20 tightest hours of the BINDING '
                                                  'season (winter), taken from the base '
                                                  'case and applied to both'),
                   prm=prm, prm_source=f'the binding one of Georgia Power\'s two seasonal '
                                       f'targets ({100*TRM_WINTER:.2f}% winter, '
                                       f'{100*TRM_SUMMER:.2f}% summer, 2022 IRP) '
                                       f'expressed against this case\'s annual peak. '
                                       f'NOT a derived LOLE fixed point.',
                   owned_MW=sum(m for _, m, _n in OWNED) + BATT_MW, purchased_MW=PURCH,
                   fleet_source=FL['source'], fleet_units=int(sum(n for _, _m, n in OWNED)),
                   wind_available=False,
                   note='built by 84_georgia_hope_case.py; see GEORGIA_PARAMETERS.md'),
              open(f'{case}/case_info.json', 'w'), indent=1)
    print(f'{tag:5s} peak {peak:9,.0f} MW  {load.sum()/1e6:7.1f} TWh  '
          f'solar CC {CCS:.4f}  -> {case}')
    return peak, CCS


os.makedirs(OUT, exist_ok=True)
print(f'fleet from EIA-860: {sum(n for _, _m, n in OWNED)} units, '
      f'{sum(m for _, m, _n in OWNED):,.0f} MW owned + {BATT_MW:,.0f} MW battery')
print(f'purchased-capacity block {PURCH:,.0f} MW '
      f'(26% margin on {GP_PEAK_2025:,.0f} MW less what is owned), '
      f'{PURCH_N} contracts')
# One accreditation basis for both cases: derived from the BASE system's tightest winter
# hours.  The data-centre case's own derivation gives 0.128 against the base's 0.135, a
# difference of 0.7 pp; using one value keeps the reported increment dc - base clean.
CC_COMMON = cc_solar(gp_base)
print(f'solar capacity credit, common to both cases: {CC_COMMON:.4f}'
      f'   (data-centre case alone would give {cc_solar(gp_dc):.4f})')

write('base', gp_base, 0.0)
write('dc', gp_dc, 1.0)

# ------------------------------------------------------------------ HOPE's own pre-check
# HOPE aligns every timeseries file against the load file row by row and refuses the case
# if they disagree.  Run the identical comparison here, so a broken case is caught by the
# builder in a second rather than by the solver after a model build.
TCOLS = ['Time Period', 'Month', 'Day', 'Hours']
REF = pd.read_csv(f'{CD}/load_timeseries_regional.csv', usecols=TCOLS)[TCOLS]
for tag in ('base', 'dc'):
    dat = f'{OUT}/GA1z_GTEP_{tag}_case/Data_GA1z_GTEP_{tag}'
    for f in ('load_timeseries_regional.csv', 'gen_availability_timeseries.csv'):
        A = pd.read_csv(f'{dat}/{f}')
        assert len(A) == H, f'{tag}/{f}: {len(A)} rows, expected {H}'
        for c in TCOLS:
            assert c in A.columns, f'{tag}/{f}: no {c!r} column'
            # against the ERCOT case the solver already accepts, not merely against
            # this case's own other file -- two files can agree and both be wrong
            assert (A[c].values == REF[c].values).all(), \
                f'{tag}/{f}: {c} does not match the working ERCOT time mapping'
    print(f'{tag:5s} time mapping matches the ERCOT reference across {H:,} rows '
          f'(Time Period = 1, Hours = 1..{H:,})')
print(f'\nsolar capacity credit convention: '
      f'{"Georgia Power 5%" if CC_OVERRIDE is not None else "derived over the top-20 tightest WINTER hours, one value for both cases"}')
print('run:  HOPE.run_hope("ModelCases/GA1z_GTEP_base_case/")')
print('      HOPE.run_hope("ModelCases/GA1z_GTEP_dc_case/")')
