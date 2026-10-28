"""Two data-only changes to the DC case, each with a citable basis. No model change.

WHY THESE TWO. The case builds no transmission because no zone ever has surplus
zero-marginal-cost energy trapped behind a constraint: renewable curtailment is exactly
zero in all 8,760 hours, every zonal price is set by gas, and the whole system sits in a
$27-47/MWh band whose median inter-zonal spread is $0.00. Congestion rent cannot exceed
the fuel-cost spread, so it stays near zero however tight the network is made - which the
N-1 run confirmed: an 18% tighter grid moved rent 6x, changed no decision, and cost the
system $0.5 M/yr. The binding constraint is therefore not the network. It is that no VRE
is ever built, so no surplus is ever created. New wind at its best zone scores a
benefit/cost of 0.80; it needs 1.00.

  A. GAS PRICE. The case uses $3.891/MMBtu, the 2025 realised delivered price from
     EIA-923 fuel receipts. That is a historical spot price standing in for a 2030
     planning year. EIA's Annual Energy Outlook 2026 projects Henry Hub
     at $5-6/MMBtu in real 2025 dollars through the early 2030s. The case is already
     in 2025 dollars, so the dollar years match. $5.00 is taken - the bottom of that
     range, deliberately conservative. Zonal differentials are preserved by scaling,
     so the relative merit order between zones is untouched.

  B. NEW-BUILD VRE DERATE. One scalar derate (wind 0.7242, solar 0.7839) is currently
     applied to BOTH the operating fleet and new build. It is calibrated so the 2025
     modelled fleet reproduces observed ERCOT output, which is right for units that
     exist and wrong for units that do not, because the observed gap contains two
     components that do not belong to new build:

       curtailment  ~5%   ERCOT curtailed >8 TWh of wind and solar in 2024 out of
                          ~150 TWh generated. HOPE decides curtailment endogenously,
                          so leaving it in the input capacity factor double-counts it.
       vintage      ~8-10% The operating fleet is full of pre-2015 turbines with lower
                          hub heights and smaller rotors than the modern power curve
                          the model applies. New build IS the modern machine.
       wake + availability + electrical ~13-15%  These are physical and DO apply.

     So new build carries 1 - 0.15 = 0.85 (wind) and 0.90 (solar); the operating fleet
     keeps its calibrated 0.7242 / 0.7839 unchanged. Capacity credits are left alone,
     which is conservative: a higher capacity factor would raise ELCC too.

VARIANTS WRITTEN
  gasA    gas price only                      - the single best-supported change
  gasA_derateB  gas price + new-build derate  - both together

Each is a full copy of the case data directory, ready to point HOPE at.
"""
import os, shutil
import numpy as np, pandas as pd

SRC = ('2_ongrid_code/HOPE/ModelCases/TX8z_GTEP_dc_vre_case/Data_TX8z_GTEP_dc_vre')
OUT = 'build/out/case_variants'

GAS_OLD = {'FarWest': 3.891, 'North': 3.891, 'West': 3.891, 'South': 4.361,
           'NorthCentral': 3.823, 'SouthCentral': 3.648, 'Coast': 4.079,
           'East': 3.891}
GAS_NEW_BASE = 5.00           # AEO2026, low end of $5-6/MMBtu real 2025$, early 2030s
SCALE = GAS_NEW_BASE/3.891
GAS_NEW = {z: round(v*SCALE, 3) for z, v in GAS_OLD.items()}

HR = {'NGCC': 6.226, 'NGCT': 9.142, 'NGST': 10.5}
VOM = {'NGCC': 3.57, 'NGCT': 4.17, 'NGST': 4.17}

DERATE_OLD = {'WindOn': 0.7242, 'SolarPV': 0.7839}
DERATE_NEW = {'WindOn': 0.85, 'SolarPV': 0.90}


def gas_costs(df):
    """Recompute Cost ($/MWh) = VOM + heat rate x new zonal gas, gas units only."""
    d = df.copy(); n = 0
    for i, r in d.iterrows():
        if r['Type'] in HR:
            old = float(r['Cost ($/MWh)'])
            new = round(VOM[r['Type']] + HR[r['Type']]*GAS_NEW[r['Zone']], 2)
            d.at[i, 'Cost ($/MWh)'] = new
            n += 1
    return d, n


def derate(av, g, gc):
    """Scale the CANDIDATE wind and solar availability columns only."""
    d = av.copy()
    cols = [c for c in av.columns if c.startswith('G')]
    names = ([('exist', r.Type) for r in g.itertuples()] +
             [('cand', r.Type) for r in gc.itertuples()])
    touched, clipped = 0, 0
    for i, c in enumerate(cols):
        if i >= len(names):
            break
        kind, typ = names[i]
        if kind == 'cand' and typ in DERATE_NEW:
            f = DERATE_NEW[typ]/DERATE_OLD[typ]
            v = d[c].values.astype(float)*f
            clipped += int((v > 1.0).sum())
            d[c] = np.minimum(v, 1.0)
            touched += 1
    return d, touched, clipped


def build(name, do_gas, do_derate):
    dst = f'{OUT}/Data_{name}'
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(SRC, dst)
    log = []
    g = pd.read_csv(f'{SRC}/gendata.csv'); gc = pd.read_csv(f'{SRC}/gendata_candidate.csv')
    if do_gas:
        g2, n1 = gas_costs(g); gc2, n2 = gas_costs(gc)
        g2.to_csv(f'{dst}/gendata.csv', index=False)
        gc2.to_csv(f'{dst}/gendata_candidate.csv', index=False)
        log.append(f'gas: {n1} existing + {n2} candidate rows repriced')
    if do_derate:
        av = pd.read_csv(f'{SRC}/gen_availability_timeseries.csv')
        av2, t, cl = derate(av, g, gc)
        av2.to_csv(f'{dst}/gen_availability_timeseries.csv', index=False)
        log.append(f'derate: {t} candidate VRE columns scaled, {cl} hours clipped at 1.0')
    return dst, log


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    print('gas price, $/MMBtu:')
    for z in sorted(GAS_OLD):
        print(f'  {z:<14}{GAS_OLD[z]:>7.3f} -> {GAS_NEW[z]:>6.3f}')
    print(f'  ERCOT-wide      {3.891:>7.3f} -> {GAS_NEW_BASE:>6.3f}   '
          f'(x{SCALE:.4f}; AEO2026 $5-6/MMBtu real 2025$, early 2030s)')

    g = pd.read_csv(f'{SRC}/gendata_candidate.csv')
    print('\nresulting candidate variable cost, $/MWh:')
    print(f"  {'zone':<14}{'NGCC old':>10}{'NGCC new':>10}{'NGCT old':>10}{'NGCT new':>10}")
    for z in sorted(GAS_OLD):
        o1 = float(g[(g.Type == 'NGCC') & (g.Zone == z)]['Cost ($/MWh)'].iloc[0])
        o2 = float(g[(g.Type == 'NGCT') & (g.Zone == z)]['Cost ($/MWh)'].iloc[0])
        print(f'  {z:<14}{o1:>10.2f}{VOM["NGCC"]+HR["NGCC"]*GAS_NEW[z]:>10.2f}'
              f'{o2:>10.2f}{VOM["NGCT"]+HR["NGCT"]*GAS_NEW[z]:>10.2f}')
    sp_old = HR['NGCC']*(max(GAS_OLD.values())-min(GAS_OLD.values()))
    sp_new = HR['NGCC']*(max(GAS_NEW.values())-min(GAS_NEW.values()))
    print(f'\n  max inter-zonal NGCC cost spread  ${sp_old:.2f} -> ${sp_new:.2f}/MWh')

    for name, dg, dd in (('gasA', True, False), ('gasA_derateB', True, True)):
        dst, log = build(name, dg, dd)
        print(f'\n{name}: {dst}')
        for l in log:
            print(f'   {l}')
