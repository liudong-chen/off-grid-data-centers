"""Figures 1, 2 and 3 (Figures).  python3 plot_figures_1_to_3.py  ->  04_figure/fig{1,2,3}_*.{pdf,png}, 03_panel_data/fig{1,2,3}_panel_*.csv  Three approaches: off-grid (diesel share x,
never connects), hybrid (x, connects at the ERCOT queue, diesel retired), on-grid.  Focal cases:
off-grid gas only (x = 0), off-grid x = 0.6, on-grid; hybrid x = 0.6 as the extra strategy.
The x = 0 ... 1.0 sweep appears only as a band, whisker or a compact line panel.

FIGURE 1  construction
  a  where the capacity lands            b  the over-build decomposed
  c  capacity built for every x by machine class, with on-grid
FIGURE 2  cost
  a  cost of electricity against x
  b  annual cost from first power to grid connection, split four ways, all x; utilization below
FIGURE 3  early access
  a  present value per year of the horizon: earnings, electricity cost, net, every approach
  b  earnings before the grid arrives against the electricity cost above on-grid, by queue length; the bound
Everything is read from solved outputs (01_raw); nothing is retyped.
"""
import json
import re
LETTERS = False   # panel letters off (added in layout)
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Wedge
from shapely.geometry import shape

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O, OUT, DATA = f'{_PKG}/01_raw', f'{_PKG}/04_figure', f'{_PKG}/03_panel_data'
os.makedirs(OUT, exist_ok=True); os.makedirs(DATA, exist_ok=True)
MM = 1/25.4
W2 = 183*MM

GRID, HYB, SELF = '#2a78d6', '#e8663a', '#96331a'
cmap = matplotlib.colormaps['copper_r']
DSL = '#3d4f5c'                       # gensets: slate, so they read as "not gas"
INK, INK2, INK3 = '#161616', '#63625f', '#9a9995'
LAND, EDGE = '#f5f4f1', '#dcdad5'
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 6.6,
    'axes.linewidth': 0.5, 'axes.edgecolor': INK2, 'axes.labelsize': 7.2, 'axes.labelcolor': INK,
    'xtick.labelsize': 6.4, 'ytick.labelsize': 6.4, 'xtick.color': INK2, 'ytick.color': INK2,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.0, 'ytick.major.size': 2.0,
    'text.color': INK, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'legend.frameon': False})


def clean(ax, keep=('left', 'bottom')):
    for sp in ('top', 'right', 'left', 'bottom'):
        ax.spines[sp].set_visible(sp in keep)
    for sp in keep:
        ax.spines[sp].set_position(('outward', 3))


def letters(fig, axes, chars, dx=0.036, dy=0.012):
    fig.canvas.draw()
    for ax_, ch in zip(axes, chars):
        b = ax_.get_position()
        fig.text(max(b.x0 - dx, 0.004), min(b.y1 + dy, 0.985), ch, fontsize=8, fontweight='bold',
                 va='bottom', ha='left', color=INK)


# ============================================================ the solved numbers
S = json.load(open(f'{O}/offgrid_summary.json'))            # GAS-ONLY, 17 plants
SITES = json.load(open(f'{O}/offgrid_sites.json'))
ONG = json.load(open(f'{O}/ongrid_final.json'))
OB = json.load(open(f'{O}/overbuild.json'))
OC = json.load(open(f'{O}/opportunity_cost.json'))
HYB_SITES = json.load(open(f'{O}/hybrid_all_sites.json'))
HS = json.load(open(f'{O}/hybrid_diesel_summary.json'))
T = pd.DataFrame(HS['table']).set_index('share').sort_index()
XS = [x for x in T.index]
P = HS['params']
E = ONG['energy_MWh']
sep = S['separate']
SZ = {'Recip': 20.0, 'Aero35': 35.0, 'Aero48': 48.0, 'FrameCT': 419.0, 'CC': 627.0}
CLASS_ORDER = ['CC', 'FrameCT', 'Aero48', 'Aero35', 'Recip']

print('FIGURE 2/3 inputs')
print(f'   off-grid (gas only)  {sep["installed_MW"]:9,.0f} MW  ${sep["cost"]/1e9:6.3f} B  ${sep["lcoe"]:6.2f}/MWh')
print(f'   on-grid              {ONG["new_MW"]:9,.0f} MW  ${ONG["cost"]/1e9:6.3f} B  ${ONG["lcoe"]:6.2f}/MWh')
assert abs(T.loc[0.0, 'lcoe_islanded'] - sep['lcoe']) < 0.05, 'hybrid x=0 must be the gas-only off-grid design'

# hybrid fleets by class, GW, for every x
ROWS = []
for x in XS:
    k = f'{x:.1f}'
    n = np.zeros(6)
    for s in HYB_SITES:
        n += np.array(s['cases'][k]['n'], float)
    ROWS.append(dict(share=x, CC=n[4]*627/1e3, FrameCT=n[3]*419/1e3, Aero=(n[1]*35 + n[2]*48)/1e3,
                     Recip=n[0]*20/1e3, Diesel=n[5]*3/1e3, units_CC=int(n[4]), units_FrameCT=int(n[3]),
                     units_Aero35=int(n[1]), units_Aero48=int(n[2]), units_Recip=int(n[0]), units_Diesel=int(n[5]),
                     installed_GW=T.loc[x, 'installed_MW']/1e3, built_per_peak=T.loc[x, 'ratio_to_sum_of_peaks'],
                     year1_load_fraction=T.loc[x, 'load_fraction_before_gas']))
FL = pd.DataFrame(ROWS).set_index('share')
FL.to_csv(f'{DATA}/fig1_panel_c_capacity_by_x.csv')

P = HS['params']
T_DSL, T_GAS, M_PAPER = P['T_DSL'], P['T_GAS'], P['M_PAPER']
T_CON = T_GAS + M_PAPER
HORIZON = 72

# ==================================================================== FIGURE 1  construction
Z = {k: shape(v) for k, v in json.load(open(f'{O}/tx_zones.json')).items()}
ZORDER = ['North', 'NorthCentral', 'East', 'FarWest', 'West', 'SouthCentral', 'Coast', 'South']
X0, X1, Y0, Y1 = -106.8, -93.3, 25.7, 36.7


def rings(g):
    for p in (g.geoms if g.geom_type == 'MultiPolygon' else [g]):
        x, y = p.exterior.xy
        yield np.asarray(x), np.asarray(y)



figh = 136*MM
FS1 = 6.4                                          # one text size for the whole of Fig 1
f1 = plt.figure(figsize=(W2, figh))
g1 = f1.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 0.85],
                     wspace=0.30, hspace=0.46, left=0.075, right=0.985, top=0.965, bottom=0.105)
TOTC = '#7b4fa0'                                   # total built (panel d): purple, distinct from diesel slate
PK_SUM = FL.loc[0.0, 'installed_GW']/FL.loc[0.0, 'built_per_peak']
SZK = 26.0                                         # marker area per (capacity / peak)^2
msz = lambda r_: SZK*r_**2

# ---- a  where the capacity lands; marker size = capacity built / peak load served
ax = f1.add_subplot(g1[0, 0]); ax_map = ax
for z in ZORDER:
    for x, y in rings(Z[z]):
        ax.fill(x, y, color=LAND, lw=0.4, ec=EDGE, zorder=2)
for x, y in rings(Z['outside']):
    ax.fill(x, y, color='#eae9e5', lw=0.4, ec=EDGE, zorder=2)
ZC = {z: (Z[z].centroid.x, Z[z].centroid.y) for z in ZORDER}
M = pd.read_csv(f'{O}/site_parameters_and_results.csv').set_index('site_id')
for r in SITES:
    ax.scatter([M.loc[r['site'], 'lon']], [M.loc[r['site'], 'lat']], s=msz(r['installed_MW']/r['peak_MW']),
               color=SELF, alpha=0.72, lw=0.45, edgecolor='white', zorder=9)
for k, mw in ONG['lines_dc'].items():
    a_, b_ = k.split('->')
    ax.annotate('', xy=ZC[b_], xytext=ZC[a_], zorder=8,
                arrowprops=dict(arrowstyle='-', color=GRID, lw=1.6, alpha=0.90, linestyle=(0, (4, 2))))
    xm, ym = 0.5*(ZC[a_][0]+ZC[b_][0]), 0.5*(ZC[a_][1]+ZC[b_][1])
    ax.text(xm + 0.16, ym + 1.15, f'{mw/1000:.1f} GW', fontsize=FS1, color=GRID, ha='left', va='bottom',
            fontweight='bold', zorder=12)
R_ON = ONG['new_MW']/1e3/PK_SUM
for z, mw in ONG['by_zone'].items():
    ax.scatter([ZC[z][0]], [ZC[z][1]], s=msz(R_ON), color=GRID, alpha=0.88, lw=0.8, edgecolor='white', zorder=11)
    ax.text(ZC[z][0], ZC[z][1] - 0.45, ' '.join(re.findall('[A-Z][a-z]+', z)), fontsize=FS1, color=GRID, ha='center', va='top', zorder=12)
ax.set_xlim(X0 - 2.6, X1 + 0.3); ax.set_ylim(Y0 - 0.2, Y1)
ax.set_aspect(1/np.cos(np.radians(31.0))); ax.axis('off'); ax.set_anchor('C')
_c = g1[0, 0].get_position(f1)                      # let the map use the whole cell plus the margins
ax.set_position([0.0, _c.y0 - 0.05, _c.x1 + 0.02, _c.height + 0.075])
h1 = [Line2D([], [], marker='o', ms=5, color=SELF, alpha=0.72, lw=0, mec='white', label='Off-grid gas only (x = 0)'),
      Line2D([], [], marker='o', ms=5, color=GRID, alpha=0.88, lw=0, mec='white', label='On-grid'),
      Line2D([], [], color=GRID, lw=1.6, alpha=0.90, linestyle=(0, (4, 2)), label='Transmission built\nfor demand growth')]
l1 = ax.legend(handles=h1, loc='lower left', bbox_to_anchor=(0.0, 0.0), fontsize=FS1, handlelength=1.6, handletextpad=0.4,
               labelspacing=0.45, borderpad=0.0, borderaxespad=0.0, ncol=1, frameon=False)
ax.add_artist(l1)
h2 = [ax.scatter([], [], s=msz(v), color=INK3, alpha=0.6, lw=0.45, edgecolor='white', label=f'{v:.1f}×') for v in (1.0, 1.5, 2.0, 2.5)]
ax.legend(handles=h2, loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=FS1, ncol=1, handletextpad=0.6, columnspacing=0.8, labelspacing=1.45,
          borderpad=0.0, borderaxespad=0.0, frameon=False, title='Circle size:\ncapacity / peak', title_fontsize=FS1, alignment='left')

# ---- b  the over-build decomposed
ax = f1.add_subplot(g1[0, 1]); ax_b1 = ax
SMALL = OB['outage_pooling'] + OB['lumpiness']
steps = [('Off-grid\ngas only', OB['A'], SELF), ('Overbuild\nrequired\nfor\nreliability', -SMALL, '#c85a34'),
         ('Overbuild\nfor non-\ncoincident\npeak', -OB['demand'], '#f4a077'), ('Existing\nERCOT\ngeneration', -OB['leaving_grid'], '#7fb2e8'),
         ('On-grid', OB['C'], GRID)]
run = 0
for i, (lab, v, c) in enumerate(steps):
    if i in (0, len(steps)-1):
        ax.bar(i, v/1000, 0.66, color=c, lw=0, zorder=3)
        ax.text(i, v/1000 + 1.5, f'{v/1000:.1f}', ha='center', fontsize=FS1, fontweight='bold', color=c); run = v
    else:
        ax.bar(i, v/1000, 0.66, bottom=run/1000, color=c, lw=0, zorder=3)
        ax.plot([i - 0.33, i + 0.33], [run/1000]*2, color='white', lw=0.6, zorder=5)
        ax.text(i, run/1000 + 1.2, f'−{abs(v)/1000:.1f}', ha='center', va='bottom', fontsize=FS1, fontweight='bold', color=c)
        run += v
        ax.plot([i - 0.33, i + 0.67], [run/1000]*2, color=INK3, lw=0.4, ls=(0, (2, 2)), zorder=2)
ax.set_xticks(range(len(steps)))
ax.set_xlim(-0.6, len(steps)-0.4); ax.set_ylim(0, 80)
ax.set_xticklabels([s_[0] for s_ in steps], linespacing=1.05, fontsize=FS1)
ax.set_ylabel('Capacity built (GW)', fontsize=FS1); ax.tick_params(labelsize=FS1); clean(ax)


# ---- c  capacity built for every diesel share x, by machine class, with on-grid on the left.
#         Off-grid keeps the diesel; the hybrid is the same plant and retires the diesel at connection.
ax = f1.add_subplot(g1[1, :]); ax_c1 = ax
XF = 0.6
DSLC = '#3d4f5c'
CLS = (('CC', 1.0, 'Gas, 627 MW units'), ('FrameCT', 0.55, 'Gas, 419 MW units'), ('Aero', 0.25, 'Gas, 20–48 MW units'))
wb1 = 0.66
POSC = {x: 1 + i for i, x in enumerate(XS)}
ax.bar(0, ONG['new_MW']/1e3, wb1, color=GRID, lw=0, zorder=3)
ax.text(0, ONG['new_MW']/1e3 + 1.2, f'{ONG["new_MW"]/1e3:.1f}', ha='center', va='bottom', fontsize=FS1, color=GRID, fontweight='bold')
for x in XS:
    xp = POSC[x]; bot = 0.0
    for key, al, _ in CLS:
        v = FL.loc[x, key] + (FL.loc[x, 'Recip'] if key == 'Aero' else 0)
        ax.bar(xp, v, wb1, bottom=bot, color=SELF, alpha=al, lw=0, zorder=3)
        bot += v
    gas_top = bot
    d = FL.loc[x, 'Diesel']
    if d > 0:
        ax.bar(xp, d, wb1, bottom=bot, color=DSLC, lw=0, zorder=3)
        bot += d
    lab_ = f'{bot:.1f}' + (f'\n{FL.loc[x, "built_per_peak"]:.2f}× peak' if x in (0.0, XF) else '')
    ax.text(xp, bot + 1.2, lab_, ha='center', va='bottom', fontsize=FS1, color=INK, fontweight='bold', linespacing=1.05)
PK_SUM = FL.loc[0.0, 'installed_GW']/FL.loc[0.0, 'built_per_peak']
ax.axhline(PK_SUM, color=INK2, lw=0.6, ls=(0, (1, 2)), zorder=4)
ax.text(-0.55, PK_SUM + 0.8, f'Data center\npeak {PK_SUM:.0f} GW', ha='left', va='bottom', fontsize=FS1, color=INK2, linespacing=1.0)
ax.set_xticks([0] + [POSC[x] for x in XS]); ax.set_xticklabels(['On-grid', '0\n(gas only)'] + [f'{x:.1f}' for x in XS[1:]], fontsize=FS1)
ax.set_xlim(-0.6, len(XS) + 0.5); ax.set_ylim(0, 100)
ax.set_xlabel('Initial demand share served by diesel', fontsize=FS1)
ax.set_ylabel('Capacity built (GW)', fontsize=FS1); ax.tick_params(labelsize=FS1)
ax.legend(handles=[Patch(facecolor=SELF, alpha=al, lw=0, label=l) for _, al, l in CLS]
          + [Patch(facecolor=DSLC, lw=0, label='Diesel, 3 MW units (retired by the hybrid at connection)'),
             Patch(facecolor=GRID, lw=0, label=f'Grid gas, {ONG["new_MW"]/ONG["n_units"]:.0f} MW units')],
          loc='upper left', bbox_to_anchor=(0.0, 1.02), ncol=3, fontsize=FS1, handlelength=1.1, handleheight=0.85, handletextpad=0.4,
          columnspacing=1.2, labelspacing=0.35, borderpad=0.0)
clean(ax)

if LETTERS: letters(f1, [ax_map, ax_b1, ax_c1], 'abc')
f1.savefig(f'{OUT}/fig1_construction.pdf'); f1.savefig(f'{OUT}/fig1_construction.png', dpi=400)

# ==================================================================== FIGURE 2  cost
C_ON = P['c_on']
RATE, R_ = P['RATE'], P['R']
MW_PUB, N_MONTHS = P['MW_pub'], 240
r_m = R_/12.0
VR_DSL = P['diesel']['vom'] + P['diesel']['hr']*P['diesel_fuel']
END = T_GAS + N_MONTHS                     # one calendar horizon for every approach: 20 years after the gas build
e_m = E/12.0


def monthly_flows(kind, x=None, T_grid=None):
    """(earnings, electricity cost) per month, months 0..END, discounted to day 0, $.
    on:  nothing until the grid (T_grid, default ERCOT queue), then on-grid electricity.
    off: diesel on x of the load from T_DSL (x > 0), gas at T_GAS, NEVER connects: the
         before-connection cost of fleet x for the rest of the horizon.
    hyb: as off until the grid arrives, then connected (HOPE, diesel retired)."""
    T_grid = T_CON if T_grid is None else T_grid
    earn = np.zeros(END + 1); cost = np.zeros(END + 1)
    mo = np.arange(END + 1)
    if kind == 'on':
        on = mo > T_grid
        earn[on] = RATE*MW_PUB; cost[on] = C_ON*e_m
    else:
        f = T.loc[x, 'load_fraction_before_gas']
        if x > 0:
            A = (mo > T_DSL) & (mo <= T_GAS)
            earn[A] = f*RATE*MW_PUB; cost[A] = T.loc[x, 'fixed_diesel']/12.0 + VR_DSL*f*e_m
        B = mo > T_GAS if kind == 'off' else (mo > T_GAS) & (mo <= T_grid)
        earn[B] = RATE*MW_PUB; cost[B] = T.loc[x, 'total']/12.0
        if kind == 'hyb':
            C = mo > T_grid; earn[C] = RATE*MW_PUB; cost[C] = T.loc[x, 'c_hybrid_connected_retire']*e_m
    disc = (1 + r_m)**(-mo.astype(float))
    return earn*disc, cost*disc


OFFC = {x: cmap(0.15 + 0.75*x) for x in XS}
CASES3 = [('On-grid', 'on', None, GRID)] + [(f'Off-grid, x = {x:.1f}', 'off', x, SELF) for x in XS] + \
         [(f'Hybrid, x = {x:.1f}', 'hyb', x, HYB) for x in XS]
FLOWS = {lab: monthly_flows(k, x) for lab, k, x, _ in CASES3}


figh2 = 74*MM
FS2 = 6.0                                          # one text size for the whole of Fig 2
f2 = plt.figure(figsize=(W2, figh2))
g2 = f2.add_gridspec(1, 2, width_ratios=[1, 1.6], wspace=0.22, left=0.065, right=0.99, top=0.97, bottom=0.21)
XT6 = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# ---- a  cost of electricity against x
ax = f2.add_subplot(g2[0, 0]); ax_a2 = ax
C_OFF = T.loc[0.0, 'lcoe_islanded']
S1F = f'{O}/S1_free_siting_gasonly.json'
C_FREE = json.load(open(S1F))['offgrid_free_lcoe'] if os.path.exists(S1F) else None
ax.axhline(C_ON, color=GRID, lw=1.2, ls=(0, (4, 2)), zorder=4)
ax.text(-0.04, C_ON - 0.8, f'On-grid ${C_ON:.2f}', ha='left', va='top', fontsize=FS2, color=GRID, fontweight='bold')
if C_FREE is not None:
    ax.axhline(C_FREE, color=INK3, lw=0.8, ls=(0, (4, 1.5, 1, 1.5)), zorder=3)
    ax.text(1.06, C_FREE - 0.8, f'Off-grid gas only, free siting ${C_FREE:.2f}', ha='right', va='top', fontsize=FS2, color=INK2)
ax.plot(XS, T['lcoe_islanded'], '-o', color=SELF, lw=1.0, ms=2.6, mec='white', mew=0.4, zorder=5, label='Off-grid')
ax.plot(XS, T['c_hybrid_connected_retire'], '-o', color=HYB, lw=1.0, ms=2.6, mec='white', mew=0.4, zorder=5, label='Hybrid')
for x, vals, col, dx, dy, ha, va in ((0.0, T['lcoe_islanded'], SELF, 0.0, 1.6, 'left', 'bottom'),
                                     (XF, T['lcoe_islanded'], SELF, -0.02, 2.2, 'right', 'bottom'),
                                     (XF, T['c_hybrid_connected_retire'], HYB, 0.0, 1.4, 'center', 'bottom')):
    ax.plot([x], [vals[x]], 'o', ms=6.5, color=col, mec='white', mew=1.0, zorder=7)
    ax.text(x + dx, vals[x] + dy, f'{vals[x]:.1f}', ha=ha, va=va, fontsize=FS2, fontweight='bold', color=col, zorder=8)
UTIL = {x: E/(T.loc[x, 'installed_MW']*8760.0) for x in XS}
GNB = json.load(open(f'{O}/../../figure_5/01_raw/grid_newbuild.json'))['ercot']['new_gen_cf'] if os.path.exists(f'{O}/../../figure_5/01_raw/grid_newbuild.json') else 0.959
ax.set_xlim(-0.06, 1.08); ax.set_ylim(38, 80)
ax.set_xticks(XT6); ax.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=FS2)
ax.tick_params(axis='y', labelsize=FS2)
ax.set_xlabel('Initial demand share served by diesel', fontsize=FS2, labelpad=2)
ax.set_ylabel('Cost of electricity (\\$ per MWh)', fontsize=FS2)
ax.legend(loc='lower left', bbox_to_anchor=(0.0, 0.20), fontsize=FS2, handlelength=1.6, borderpad=0.0, labelspacing=0.35)
clean(ax)

# ---- b  average annual cost from first power to grid connection, every approach and x; utilization below
ax = f2.add_subplot(g2[0, 1]); ax_b2 = ax
rows_b = []
for x in XS:
    f = T.loc[x, 'load_fraction_before_gas']
    m_dsl = (T_GAS - T_DSL) if x > 0 else 0; m_isl = T_CON - T_GAS; yrs = (m_dsl + m_isl)/12.0
    mwh = f*E*(m_dsl/12.0) + E*(m_isl/12.0)
    capg = (T.loc[x, 'fixed_gas'] + T.loc[x, 'land'])*(m_isl/12.0)/yrs
    capd = T.loc[x, 'fixed_diesel']*((m_dsl + m_isl)/12.0)/yrs
    fg = T.loc[x, 'fuel_gas']*(m_isl/12.0)/yrs
    fd = (VR_DSL*f*E*(m_dsl/12.0) + T.loc[x, 'fuel_diesel']*(m_isl/12.0))/yrs
    tot = capg + capd + fg + fd
    rows_b.append(dict(share=x, gas_capital_land_B=capg/1e9, genset_capital_B=capd/1e9, gas_fuel_B=fg/1e9, diesel_fuel_B=fd/1e9,
                       total_B=tot/1e9, window_months=m_dsl + m_isl, energy_window_TWh=mwh/1e6, lcoe_window=tot*yrs/mwh,
                       year1_cost_B=(VR_DSL*f*E + T.loc[x, 'fixed_diesel'])*(m_dsl/12.0)/1e9))
B_ = pd.DataFrame(rows_b).set_index('share')
DSLC = '#3d4f5c'
BLOCKS = (('gas_capital_land_B', SELF, 1.0, 'Gas capital and land'), ('genset_capital_B', DSLC, 1.0, 'Diesel capital'),
          ('gas_fuel_B', SELF, 0.30, 'Gas fuel and O&M'), ('diesel_fuel_B', DSLC, 0.35, 'Diesel fuel and O&M'))
SHOWC = list(XS)
wb = 0.66
ax.bar(0, ONG['inv']/1e9, wb, color=GRID, lw=0, zorder=3)
ax.bar(0, (ONG['cost'] - ONG['inv'])/1e9, wb, bottom=ONG['inv']/1e9, color=GRID, alpha=0.30, lw=0, zorder=3)
ax.text(0, ONG['cost']/1e9 + 0.4, f'{ONG["cost"]/1e9:.1f}', ha='center', va='bottom', fontsize=FS2, color=GRID, fontweight='bold')
for i, x in enumerate(SHOWC):
    xpos = 1 + i; bot = 0.0
    for key, col, al, lab in BLOCKS:
        v = B_.loc[x, key]
        if v > 0:
            ax.bar(xpos, v, wb, bottom=bot, color=col, alpha=al, lw=0, zorder=3)
        bot += v
    ax.text(xpos, bot + 0.4, f'{bot:.1f}', ha='center', va='bottom', fontsize=FS2, color=SELF, fontweight='bold')
ax.set_xticks(range(len(SHOWC) + 1)); ax.set_xticklabels(['On-\ngrid', '0'] + [f'{x:.1f}' for x in SHOWC[1:]], fontsize=FS2)
ax.tick_params(axis='y', labelsize=FS2); ax.tick_params(axis='x', pad=11)
_bt = ax.get_xaxis_transform()
ax.text(-0.62, -0.035, 'utilized', transform=_bt, ha='right', va='top', fontsize=FS2, color=INK2)
ax.text(0, -0.035, f'{100*GNB:.0f}%', transform=_bt, ha='center', va='top', fontsize=FS2, color=GRID, fontweight='bold')
for i, x in enumerate(SHOWC):
    ax.text(1 + i, -0.035, f'{100*UTIL[x]:.0f}%', transform=_bt, ha='center', va='top', fontsize=FS2, color=SELF, fontweight='bold')
ax.set_xlabel('Initial demand share served by diesel', fontsize=FS2, labelpad=2)
ax.set_xlim(-0.55, len(SHOWC) + 0.55); ax.set_ylim(0, 36)
ax.set_ylabel('Annual cost, first power to\nconnection (\\$ billion per year)', fontsize=FS2)
ax.legend(handles=[Patch(facecolor=c, alpha=a_, lw=0, label=l) for _, c, a_, l in BLOCKS]
          + [Patch(facecolor=GRID, lw=0, label='On-grid capital'), Patch(facecolor=GRID, alpha=0.30, lw=0, label='On-grid fuel and O&M')],
          loc='upper left', fontsize=FS2, ncol=3, handlelength=1.0, handleheight=0.85, handletextpad=0.35, labelspacing=0.3, columnspacing=1.2, borderpad=0.0)
clean(ax)


if LETTERS: letters(f2, [ax_a2, ax_b2], 'ab')
f2.savefig(f'{OUT}/fig2_cost.pdf'); f2.savefig(f'{OUT}/fig2_cost.png', dpi=400)

# ==================================================================== FIGURE 3  early access
figh3 = 76*MM
FS3 = 6.0                                          # one text size for the whole of Fig 3
f3 = plt.figure(figsize=(W2, figh3))
g3 = f3.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.24, left=0.085, right=0.99, top=0.96, bottom=0.2)

# ---- a  earnings before the grid arrives against the electricity cost above on-grid, by queue length
ax = f3.add_subplot(g3[0, 1]); ax_a3 = ax
CAPEX_DSL_YR = T['diesel_MW']*P['diesel']['capex']*1000.0*0.1019          # diesel capex annuity, $/yr (no FOM once retired)
NYRS = N_MONTHS/12.0
YLO_C, YTOP_C = 0.0, 36.0


def pv_route(kind, T_grid, x=None):
    """PV at day 0 of (earnings - electricity) for a route when the grid arrives at month T_grid.
    Off-grid never connects, so it does not depend on T_grid.  The hybrid data center knows the
    date and takes the better of (i) gas at T_GAS and connect at T_grid; (ii) diesel on x of
    the load until the grid comes, gas never built (diesel capital sunk)."""
    if kind in ('on', 'off'):
        e_, c_ = monthly_flows(kind, x, T_grid); return float((e_ - c_).sum())
    e_, c_ = monthly_flows('hyb', x, T_grid)
    v_gas = float((e_ - c_).sum()) if T_grid > T_GAS else -np.inf
    mo = np.arange(END + 1); disc = (1 + r_m)**(-mo.astype(float))
    earn = np.zeros(END + 1); cost = np.zeros(END + 1); f = T.loc[x, 'load_fraction_before_gas']
    A = (mo > T_DSL) & (mo <= T_grid); earn[A] = f*RATE*MW_PUB; cost[A] = T.loc[x, 'fixed_diesel']/12.0 + VR_DSL*f*e_m
    C = mo > T_grid; earn[C] = RATE*MW_PUB; cost[C] = C_ON*e_m + CAPEX_DSL_YR.loc[x]/12.0
    return max(v_gas, float(((earn - cost)*disc).sum()))


TG = np.arange(0, 73)
# Build-your-own versus waiting.  Off-grid and hybrid build the same plant (diesel on x of the load from
# month 12, gas at month 24) and are identical until the grid connection, so the decision to build, and
# its bound, depend on the diesel share x only.  Value = PV of (earnings - electricity) of the build-own
# route, connecting when the grid arrives, minus the on-grid route, per year of the horizon.
tot = lambda k, x, t: float(sum(e - c for e, c in zip(*monthly_flows(k, x, t))))
ON_C = np.array([tot('on', None, t) for t in TG])/1e9/NYRS
SHOW_C = [0.0, 0.3, 0.6, 1.0]
NPVC, BOUND = {'On-grid': ON_C}, {}
tf = np.linspace(0, 72, 3601)
for x in XS:
    lab = f'Build own, x = {x:.1f}'
    NPVC[lab] = np.array([tot('hyb', x, t) for t in TG])/1e9/NYRS
    nf = np.interp(tf, TG, NPVC[lab] - ON_C)
    BOUND[lab] = float(tf[np.argmax(nf >= 0)]) if (nf >= 0).any() and nf[0] < 0 else np.nan
# Decompose build-own minus waiting into the two sides of the trade-off:
#   earnings from early access (above 0)  = PV(earnings, build own) - PV(earnings, wait)
#   extra electricity cost    (below 0)  = PV(cost, build own)     - PV(cost, wait)
EARN_C, COST_C = {}, {}
for x in XS:
    lab = f'Build own, x = {x:.1f}'
    EARN_C[lab] = np.array([float((monthly_flows('hyb', x, t)[0] - monthly_flows('on', None, t)[0]).sum()) for t in TG])/1e9/NYRS
    COST_C[lab] = np.array([float((monthly_flows('hyb', x, t)[1] - monthly_flows('on', None, t)[1]).sum()) for t in TG])/1e9/NYRS
# Both sides as positive amounts, each a band over the diesel share x = 0 ... 1.0 with the
# x = 0.6 line; they cross at the bound, shown as a point (x = 0.6) with a bar over x = 0 ... 1.0.
XM = 0.6; lm = f'Build own, x = {XM:.1f}'
EARN_COL, COST_COL = SELF, DSL
EA = np.array([EARN_C[f'Build own, x = {x:.1f}'] for x in XS]); CA = np.array([COST_C[f'Build own, x = {x:.1f}'] for x in XS])
ax.fill_between(TG, EA.min(0), EA.max(0), color=EARN_COL, alpha=0.18, lw=0, zorder=1)
ax.fill_between(TG, CA.min(0), CA.max(0), color=COST_COL, alpha=0.25, lw=0, zorder=1)
ax.plot(TG, EARN_C[lm], color=EARN_COL, lw=1.6, zorder=4)
ax.plot(TG, COST_C[lm], color=COST_COL, lw=1.6, zorder=4)
bs = [BOUND[f'Build own, x = {x:.1f}'] for x in XS]; tb = BOUND[lm]; yb = float(np.interp(tb, TG, COST_C[lm]))
ax.errorbar([tb], [yb], xerr=[[tb - min(bs)], [max(bs) - tb]], fmt='o', ms=5.0, mfc='white', mec=INK, mew=1.2,
            ecolor=INK, elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=8)
ax.annotate(f'{tb:.0f} months\n({min(bs):.0f}–{max(bs):.0f})', xy=(tb, yb), xytext=(1.0, yb + 6.5), ha='left', va='bottom',
            fontsize=FS3, fontweight='bold', color=INK, linespacing=1.1, zorder=9, bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'),
            arrowprops=dict(arrowstyle='-', color=INK, lw=0.5, shrinkA=0, shrinkB=4))
# (the ERCOT-queue marker at T_CON was removed ; the queue length is discussed in the text)
VL = 0.50   # dashed event lines stop below the legend
for _t, _lab, _ha, _dx in ((T_DSL, 'Diesel\nstarts\nto run', 'right', -0.8), (T_GAS, 'Gas\nstarts\nto run', 'left', 0.8)):
    ax.axvline(_t, ymax=VL, color=INK3, lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax.text(_t + _dx, YLO_C + (YTOP_C - YLO_C)*VL, _lab, fontsize=FS3, color=INK2, ha=_ha, va='top', linespacing=1.05)
ax.set_xlim(0, 72); ax.set_ylim(YLO_C, YTOP_C)
ax.set_xticks(range(0, 73, 12)); ax.set_xticklabels([f'{m//12}' for m in range(0, 73, 12)])
ax.tick_params(labelsize=FS3)
ax.set_xlabel('Interconnection queue (years)', fontsize=FS3)
ax.set_ylabel('Present value of off-grid/hybrid above\non-grid (\\$ billion per year)', fontsize=FS3)
from matplotlib.legend_handler import HandlerTuple
_band = lambda c, a_: Patch(facecolor=c, alpha=a_, lw=0)
_line = lambda c: Line2D([], [], color=c, lw=1.6)
ax.legend(handles=[(_band(EARN_COL, 0.18), _line(EARN_COL)), (_band(COST_COL, 0.25), _line(COST_COL)),
                   Line2D([], [], color=INK2, lw=1.6), _band(INK2, 0.25),
                   Line2D([], [], marker='o', ms=4.6, mfc='white', mec=INK, mew=1.1, lw=0)],
          labels=['Earnings above on-grid', 'Electricity cost above on-grid', 'Line: x = 0.6', 'Shading: x = 0–1.0', 'Threshold'],
          handler_map={tuple: HandlerTuple(ndivide=1, pad=0)},
          loc='upper left', bbox_to_anchor=(0.0, 0.86), fontsize=FS3, handlelength=2.0, borderpad=0.0, labelspacing=0.45)
clean(ax)
ROUTES_C = []
C_ = pd.DataFrame({'month_grid_arrives': TG, **{k + ' net': v - ON_C for k, v in NPVC.items() if k != 'On-grid'},
                   **{k + ' earnings above on-grid': v for k, v in EARN_C.items()},
                   **{k + ' electricity cost above on-grid': v for k, v in COST_C.items()}, 'On-grid_pv_per_yr': ON_C})
print('bound (queue months) by diesel share:', {k: round(v, 1) for k, v in BOUND.items()})


# ---- b  present value of every approach, per year of the horizon
ax = f3.add_subplot(g3[0, 0]); ax_b3 = ax
labs = [c[0] for c in CASES3]
NYRS = N_MONTHS/12.0
earnT = np.array([FLOWS[l][0].sum() for l in labs])/1e9
costT = np.array([FLOWS[l][1].sum() for l in labs])/1e9
netT = earnT - costT
earnA, costA, netA = earnT/NYRS, costT/NYRS, netT/NYRS
IDX = {l: i for i, l in enumerate(labs)}
POS = {'On-grid': 0.0}
wd = 0.36
for i, x in enumerate(XS):
    POS[f'Off-grid, x = {x:.1f}'] = 1 + i - 0.19; POS[f'Hybrid, x = {x:.1f}'] = 1 + i + 0.19
for lab, k, x, col in CASES3:
    w_ = 0.64 if k == 'on' else wd
    ax.bar(POS[lab], earnA[IDX[lab]], w_, color=col, alpha=0.30, lw=0, zorder=3)
    ax.bar(POS[lab], -costA[IDX[lab]], w_, color=col, alpha=0.9, lw=0, zorder=3)
for k, col, dy, va in (('Off-grid', SELF, -2.0, 'top'), ('Hybrid', HYB, 2.0, 'bottom')):
    ls_ = [f'{k}, x = {x:.1f}' for x in XS]
    ax.plot([POS[l] for l in ls_], [netA[IDX[l]] for l in ls_], '-o', color=col, lw=1.0, ms=3.0, mec='white', mew=0.5, zorder=6)
    for x in (0.0, 0.6, 1.0):
        l = f'{k}, x = {x:.1f}'
        ax.text(POS[l], netA[IDX[l]] + dy, f'{netA[IDX[l]]:.1f}', ha='center', va=va, fontsize=FS3, fontweight='bold', color=col, zorder=7,
                bbox=dict(facecolor='white', edgecolor='none', pad=0.1, alpha=0.8))
for x in (0.0, 0.6, 1.0):
    for k, col, dy in (('Off-grid', SELF, 1.0), ('Hybrid', HYB, 8.0)):
        l = f'{k}, x = {x:.1f}'
        ax.text(POS[l], -costA[IDX[l]] - dy, f'\u2212{costA[IDX[l]]:.1f}', ha='center', va='top', fontsize=FS3, color=col, fontweight='bold')
ax.text(0, -costA[0] - 1.0, f'\u2212{costA[0]:.1f}', ha='center', va='top', fontsize=FS3, color=GRID, fontweight='bold')
ax.plot([0], [netA[0]], 'o', color=GRID, ms=4.0, mec='white', mew=0.6, zorder=6)
ax.text(0, netA[0] + 2.0, f'{netA[0]:.1f}', ha='center', va='bottom', fontsize=FS3, fontweight='bold', color=GRID)
ax.axhline(0, color=INK2, lw=0.6, zorder=2)
ax.set_xticks([0] + [1 + i for i in range(len(XS))]); ax.set_xticklabels(['On-\ngrid', '0'] + [f'{x:.1f}' for x in XS[1:]], fontsize=FS3)
ax.tick_params(axis='y', labelsize=FS3)
ax.set_xlabel('Initial demand share served by diesel', fontsize=FS3)
ax.set_xlim(-0.6, len(XS) + 0.65)
ax.set_ylabel('Present value, 20-year average\n(\\$ billion per year)', fontsize=FS3)
ax.set_ylim(-costA.max()*2.9, earnA.max()*1.25)
ax.legend(handles=[Patch(facecolor=INK2, alpha=0.30, lw=0, label='Earnings'), Patch(facecolor=INK2, alpha=0.9, lw=0, label='Electricity cost'),
                   Line2D([], [], marker='o', color=SELF, lw=1.0, ms=3.0, label='Off-grid, net'),
                   Line2D([], [], marker='o', color=HYB, lw=1.0, ms=3.0, label='Hybrid, net'),
                   Line2D([], [], marker='o', color=GRID, lw=0, ms=3.6, label='On-grid, net')],
          loc='upper left', fontsize=FS3, ncol=5, handlelength=1.2, handleheight=0.85, handletextpad=0.35, columnspacing=0.9, borderpad=0.0)
clean(ax)
ANNUAL = pd.DataFrame({lab: np.array([(FLOWS[lab][0] - FLOWS[lab][1])[12*y + 1:12*(y + 1) + 1].sum() for y in range(20)])/1e9 for lab, *_ in CASES3},
                      index=pd.Index(range(1, 21), name='year'))

if LETTERS: letters(f3, [ax_b3, ax_a3], 'ab')
f3.savefig(f'{OUT}/fig3_access.pdf'); f3.savefig(f'{OUT}/fig3_access.png', dpi=400)

# ---- panel data and numbers
pass
pd.DataFrame([dict(step=st[0].replace('\n', ' '), GW=st[1]/1e3) for st in steps]).to_csv(f'{DATA}/fig1_panel_b_overbuild.csv', index=False)
B_['utilization_before_connection'] = [UTIL[x] for x in B_.index]
pd.DataFrame(dict(share=XS, lcoe_islanded=T['lcoe_islanded'].values, lcoe_connected_retired=T['c_hybrid_connected_retire'].values,
                  lcoe_connected_backup=T['c_hybrid_connected_backup'].values, on_grid=C_ON)).to_csv(f'{DATA}/fig2_panel_a_cost_vs_x.csv', index=False)
B_.to_csv(f'{DATA}/fig2_panel_b_cost_to_connection.csv')
C_.to_csv(f'{DATA}/fig3_panel_b_value_vs_queue.csv', index=False)
cum = {lab: np.cumsum(FLOWS[lab][0] - FLOWS[lab][1])/1e9 for lab, *_ in CASES3}
pd.DataFrame({'month': np.arange(END + 1), **cum}).to_csv(f'{DATA}/figS_cumulative_pv.csv', index=False)
ANNUAL.to_csv(f'{DATA}/figS_net_pv_by_year.csv')
pd.DataFrame(dict(approach=labs, earnings_pv_B=earnT, electricity_pv_B=costT, net_pv_B=netT, earnings_per_year_B=earnA, electricity_per_year_B=costA, net_per_year_B=netA)).to_csv(f'{DATA}/fig3_panel_a_present_value.csv', index=False)
json.dump(dict(offgrid_gasonly=dict(MW=sep['installed_MW'], lcoe=sep['lcoe'], cost=sep['cost']),
               ongrid=dict(MW=ONG['new_MW'], lcoe=ONG['lcoe'], cost=ONG['cost']), overbuild=OB, opportunity=OC,
               timeline_months=dict(diesel=T_DSL, gas=T_GAS, grid=T_CON, horizon_end=END),
               hybrid=T.reset_index().to_dict('records'),
               bound_grid_month=BOUND, avg_cost_to_connection=B_.reset_index().to_dict('records'), utilization=UTIL,
               totals_20yr={l: dict(earnings_B=float(e), electricity_B=float(c), net_B=float(e - c)) for l, e, c in zip(labs, earnT, costT)}),
          open(f'{DATA}/fig123_numbers.json', 'w'), indent=1, default=float)
print(f'fig1 {W2/MM:.0f} x {figh/MM:.0f} mm; fig2 {W2/MM:.0f} x {figh2/MM:.0f} mm; fig3 {W2/MM:.0f} x {figh3/MM:.0f} mm')
print('20-yr net PV ($B):', {l: round(v) for l, v in zip(labs, netT)})
