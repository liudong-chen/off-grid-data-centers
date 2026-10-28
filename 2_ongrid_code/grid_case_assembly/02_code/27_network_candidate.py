"""Rebuild the transmission candidate set with a non-binding ceiling, and report headroom.

TWO CHANGES.

1. THE CEILING IS NOW NON-BINDING. It was 3,000 MW on every corridor, which is an
   odd quantity on a set of corridors ranging from 1,371 to 24,135 MW: on five of
   seventeen it exceeded the whole existing corridor, on five others it was a fifth
   of it. Because `inv_dcs_bin: 0` makes the line investment variable continuous,
   3,000 MW was never a lump the model had to swallow whole - it was a cap - but it
   was still an arbitrary cap. It is raised to 30,000 MW, above the largest existing
   corridor in the system, so that what the model builds is decided by economics and
   not by the bound. Cost per MW-year is unchanged, so nothing about the economics
   moves; only the constraint disappears.

2. THE TABLE NOW CARRIES HEADROOM. Existing capability, the highest flow the solved
   case ever puts on the corridor, and the difference. This is the quantity that
   actually explains the build decision: a corridor with 14 GW of unused headroom is
   not a candidate for expansion no matter how it is priced.

Cost is quoted per MW-year rather than as a block price, because that is the figure
that does not change when the ceiling does.
"""
import numpy as np, pandas as pd

CASE = 'Code/HOPE/ModelCases/TX8z_GTEP_dc_vre_case/Data_TX8z_GTEP_dc_vre'
OUT = 'build/out'
UNIT, CRF, OLD_BLOCK, NEW_BLOCK = 1100.0, 0.0839, 3000.0, 30000.0

# Highest |flow| on each corridor over the 8,760 hours of the solved DC case, and the
# number of hours it spends within 1% of its bound. From output/power_flow.csv.
FLOW = {
    ('North', 'NorthCentral'): (4440, 96), ('East', 'North'): (2768, 84),
    ('FarWest', 'North'): (1521, 77), ('North', 'West'): (1371, 77),
    ('NorthCentral', 'SouthCentral'): (8549, 0), ('Coast', 'SouthCentral'): (14528, 0),
    ('Coast', 'East'): (4538, 0), ('FarWest', 'West'): (8758, 0),
    ('East', 'NorthCentral'): (11556, 0), ('SouthCentral', 'West'): (1742, 0),
    ('East', 'SouthCentral'): (1780, 0), ('FarWest', 'SouthCentral'): (1203, 0),
    ('South', 'SouthCentral'): (3161, 0), ('Coast', 'South'): (6060, 0),
    ('NorthCentral', 'West'): (9593, 0), ('FarWest', 'NorthCentral'): (3065, 0),
    ('South', 'West'): (503, 0)}

ln = pd.read_csv(f'{CASE}/linedata.csv')
lc = pd.read_csv(f'{CASE}/linedata_candidate.csv')
lc['miles'] = lc['Cost (M$)']*1e6/(UNIT*OLD_BLOCK*CRF)
lc['per_MWyr'] = UNIT*lc.miles*CRF                       # invariant to the ceiling

new = lc[['From_zone', 'To_zone']].copy()
new['Forward Capacity (MW)'] = NEW_BLOCK
new['Reverse Capacity (MW)'] = NEW_BLOCK
new['Cost (M$)'] = (lc.per_MWyr*NEW_BLOCK/1e6).round(3)
new['X'] = (lc.X*OLD_BLOCK/NEW_BLOCK).round(6)           # reactance scales inversely
new['Loss (%)'] = 0.0
new.to_csv(f'{OUT}/linedata_candidate_open.csv', index=False)

T = ln.merge(lc[['From_zone', 'To_zone', 'miles', 'per_MWyr']],
             on=['From_zone', 'To_zone'])
T['flow'] = [FLOW[(a, b)][0] for a, b in zip(T.From_zone, T.To_zone)]
T['tight'] = [FLOW[(a, b)][1] for a, b in zip(T.From_zone, T.To_zone)]
T['headroom'] = T['Forward Capacity (MW)'] - T.flow
T['util'] = T.flow/T['Forward Capacity (MW)']*100
T = T.sort_values('util', ascending=False)

print(f"{'corridor':<27}{'existing':>9}{'peak flow':>10}{'headroom':>10}{'util':>7}"
      f"{'hrs at cap':>11}{'length':>8}{'$/MW-yr':>9}")
for r in T.itertuples():
    print(f'{r.From_zone+"-"+r.To_zone:<27}{getattr(r,"_3"):>9,.0f}{r.flow:>10,.0f}'
          f'{r.headroom:>10,.0f}{r.util:>6.0f}%{r.tight:>11,}{r.miles:>7.0f}mi'
          f'{r.per_MWyr:>9,.0f}')
print(f'\n{"TOTAL":<27}{T["Forward Capacity (MW)"].sum():>9,.0f}{T.flow.sum():>10,.0f}'
      f'{T.headroom.sum():>10,.0f}{T.flow.sum()/T["Forward Capacity (MW)"].sum()*100:>6.0f}%')
print(f'\ncorridors that ever reach their bound: {(T.tight>0).sum()} of {len(T)}')
print(f'ceiling raised {OLD_BLOCK:,.0f} -> {NEW_BLOCK:,.0f} MW per corridor; '
      f'cost per MW-year unchanged at ${T.per_MWyr.min():,.0f}-{T.per_MWyr.max():,.0f}')

# ---------------------------------------------------------------- LaTeX table
ZL = {'FarWest': 'Far West', 'NorthCentral': 'North Central',
      'SouthCentral': 'South Central'}
zl = lambda z: ZL.get(z, z)
b = ['\\begin{tabular}{@{}llrrrrrr@{}}', '\\toprule',
     'From & To & Existing & Peak flow & Headroom & Util. & At cap & Expansion \\\\',
     ' & & (MW) & (MW) & (MW) & (\\%) & (h/yr) & (\\$/MW-yr) \\\\', '\\midrule']
for r in T.sort_values(['From_zone', 'To_zone']).itertuples():
    b.append(f'{zl(r.From_zone)} & {zl(r.To_zone)} & {getattr(r,"_3"):,.0f} & '
             f'{r.flow:,.0f} & {r.headroom:,.0f} & {r.util:.0f} & '
             f'{r.tight if r.tight else "--"} & {r.per_MWyr:,.0f} \\\\')
b += ['\\midrule',
      f'\\textbf{{17 corridors}} & & \\textbf{{{T["Forward Capacity (MW)"].sum():,.0f}}} & '
      f'\\textbf{{{T.flow.sum():,.0f}}} & \\textbf{{{T.headroom.sum():,.0f}}} & '
      f'\\textbf{{{T.flow.sum()/T["Forward Capacity (MW)"].sum()*100:.0f}}} & & \\\\',
      '\\bottomrule', '\\end{tabular}']
open('build/tex_tables/t_network.tex', 'w').write('\n'.join(b) + '\n')
print(f'\nwritten {OUT}/linedata_candidate_open.csv and tex_tables/t_network.tex')
