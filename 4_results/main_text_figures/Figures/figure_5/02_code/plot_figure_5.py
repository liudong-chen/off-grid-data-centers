"""Figures: gas-only off-grid designs, pooled plant removed, three levers (no wind).

Figure 5 -- Georgia Power against ERCOT: the same physics on a different fleet.

  a  over-build against campus size: every campus in both fleets, one plant each,
     installed capacity over its own peak; the shared plant and the grid increment of
     each system at the fleet's coincident peak
  b  what a MWh costs and what it is made of: capital against fuel, three
     architectures, both systems
  c  why Georgian self-supply is dearer: four full re-solves of the 31 campuses,
     each with one Georgian input set to ERCOT's value, and the part they leave
  d  the time leg: value of connecting sooner over the electricity penalty, against
     the extra months the grid takes, both systems, with the break-even

SOURCES.  ERCOT on-grid: HOPE TX8z_GTEP_{dc,base}_vre_case Gurobi objectives (dc - base
over data-center energy).  ERCOT off-grid: offgrid_sites.json (17 campuses),
offgrid_pool.json (FarWest).  Georgia on-grid: HOPE GA1z_GTEP_{dc,base}_case.
Georgia off-grid: offgrid_georgia_sites.json (31 campuses), offgrid_georgia_pool.json.
Levers: georgia_decomp.json.  Time: calculate_opportunity_cost.py conventions.

    python 96_fig5.py
"""
import csv
import json
LETTERS = False   # panel letters off (added in layout)
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(f'{_PKG}/01_raw'):
    O, FIG, DATA = f'{_PKG}/01_raw', f'{_PKG}/04_figure', f'{_PKG}/03_panel_data'
else:
    OUT = os.environ.get('OFFGRID_OUT', 'build/fig5')
    O, FIG, DATA = f'{OUT}/out', OUT, OUT
MM = 1/25.4
W2 = 183*MM

GRID, POOL, SELF = '#2a78d6', '#e8663a', '#96331a'
INK, INK2, INK3 = '#161616', '#63625f', '#9a9995'
ERC, GEO = '#7d7a75', '#96331a'          # the two systems
STEP, RESID = '#7d6f5f', '#cfc9c0'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 6.6,
    'axes.linewidth': 0.5, 'axes.edgecolor': INK2,
    'axes.labelsize': 7.2, 'axes.labelcolor': INK,
    'xtick.labelsize': 6.4, 'ytick.labelsize': 6.4,
    'xtick.color': INK2, 'ytick.color': INK2,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.0, 'ytick.major.size': 2.0,
    'text.color': INK, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'legend.frameon': False,
    'hatch.linewidth': 0.5,
})


def clean(ax, keep=('left', 'bottom')):
    for sp in ('top', 'right', 'left', 'bottom'):
        ax.spines[sp].set_visible(sp in keep)
    for sp in keep:
        ax.spines[sp].set_position(('outward', 3))


def af_annual(n, r):
    return (1 - (1 + r)**-n)/r


def af_monthly(n, r):
    m = r/12.0
    return (1 - (1 + m)**-n)/m


def obj(case):
    for line in open(f'{O}/hope/{case}/metadata.yml'):
        if line.startswith('objective_value'):
            return float(line.split(':')[1])


def new_mw(case):
    r = csv.DictReader(open(f'{O}/hope/{case}/capacity.csv'))
    return sum(float(x['Capacity_FIN (MW)']) for x in r if x['EC_Category'] == 'Candidate')


def syscost(case):
    r = list(csv.DictReader(open(f'{O}/hope/{case}/system_cost.csv')))
    return sum(float(x['Inv_cost ($)']) for x in r), sum(float(x['Opr_cost ($)']) for x in r)


def capital(r):
    """Capital-like part of a solved campus: machines, land, and any solar it bought."""
    return r['fixed_cost'] + r['land_cost'] + r.get('vre_cost', 0.0)


# ======================================================================= ERCOT
E_SITES = json.load(open(f'{O}/offgrid_sites.json'))
E_ON = json.load(open(f'{O}/ongrid_final.json'))
E_MWH = E_ON['energy_MWh']
E = dict(
    sep=dict(cost=sum(r['cost'] for r in E_SITES), cap=sum(capital(r) for r in E_SITES),
             mw=sum(r['installed_MW'] for r in E_SITES), peak=sum(r['peak_MW'] for r in E_SITES)),
    grid=dict(cost=E_ON['obj_dc'] - E_ON['obj_base'], cap=E_ON['inv'], mw=E_ON['new_MW'],
              peak=30378.0),
    mwh=E_MWH, fleet_mw=40425.0)

# ===================================================================== Georgia
G_SITES = json.load(open(f'{O}/offgrid_georgia_sites.json'))
G_MWH = sum(r['energy_MWh'] for r in G_SITES)
OBJ_B, OBJ_D = obj('gabase'), obj('gadc')
INV_B, OPR_B = syscost('gabase')
INV_D, OPR_D = syscost('gadc')
G = dict(
    sep=dict(cost=sum(r['cost'] for r in G_SITES), cap=sum(capital(r) for r in G_SITES),
             mw=sum(r['installed_MW'] for r in G_SITES), peak=sum(r['peak_MW'] for r in G_SITES)),
    grid=dict(cost=OBJ_D - OBJ_B, cap=INV_D - INV_B, mw=new_mw('gadc') - new_mw('gabase'),
              peak=27668.7 - 17925.0),
    mwh=G_MWH, fleet_mw=12400.0)

# Utilization -- energy served over capacity x 8,760 -- is why the capital blocks differ
# so much more than the fuel blocks.  Both off-grid numbers come straight out of the
# solved designs; the grid's is the new-build capacity factor of the HOPE pair,
# 97_grid_newbuild.py.
GNB = json.load(open(f'{O}/grid_newbuild.json'))
E['grid_cf'], G['grid_cf'] = GNB['ercot']['new_gen_cf'], GNB['georgia']['new_gen_cf']

for S in (E, G):
    for k in ('sep', 'grid'):
        S[k]['lcoe'] = S[k]['cost']/S['mwh']
        S[k]['fuel'] = S[k]['cost'] - S[k]['cap']      # fuel, O&M and VRE net savings; closes exactly
    S['pen'] = S['sep']['lcoe'] - S['grid']['lcoe']
    S['sep']['util'] = S['mwh']/(S['sep']['mw']*8760.0)
    S['grid']['util'] = S['grid_cf']

# ------------------------------------------------------------------- levers
DEC = {c['tag'].split('  ')[1]: c for c in json.load(open(f'{O}/georgia_decomp.json'))['cases']}
G0 = DEC['Georgia as it is']['lcoe']
LEV = [('Gas price', G0 - DEC['+ ERCOT cheapest gas']['lcoe']),
       ('Campus size', G0 - DEC['+ ERCOT campus sizes']['lcoe']),
       ('Land', G0 - DEC['+ ERCOT land']['lcoe'])]      # gas-only: no wind lever
GAP = G0 - E['sep']['lcoe']
RESID_V = GAP - sum(v for _n, v in LEV)

# ---------------------------------------------------------------------- time
# forgone earnings while a built campus waits, $/MW of grid load/month.
# CoreWeave FY2024: $1.915 B revenue / 250 MW / 12 months x 62.7% adjusted-EBITDA
# margin = $400k.  Derivation and the three cross-checks: calculate_opportunity_cost.py.
RATE, R, N_ELEC = 400_000.0, 0.08, 20
E_PEN_PV = E['pen']*E['mwh']*af_annual(N_ELEC, R)
G_PEN_PV = G['pen']*G['mwh']*af_annual(N_ELEC, R)
MONTHS = np.arange(0, 49)
E_RATIO = RATE*E['fleet_mw']*af_monthly(MONTHS, R)/E_PEN_PV
G_RATIO = RATE*G['fleet_mw']*af_monthly(MONTHS, R)/G_PEN_PV
E_MARK, G_MARKS = 24, (6, 36)


def breakeven(fleet, pen_pv):
    m = np.arange(0.0, 60.0, 0.01)
    r = RATE*fleet*af_monthly(m, R)/pen_pv
    return float(m[np.argmax(r >= 1.0)])


E_BE, G_BE = breakeven(E['fleet_mw'], E_PEN_PV), breakeven(G['fleet_mw'], G_PEN_PV)

print('FIGURE 5')
for nm, S in (('ERCOT', E), ('Georgia', G)):
    print(f'  {nm:8s} sep {S["sep"]["lcoe"]:6.2f}  grid {S["grid"]["lcoe"]:6.2f}  penalty {S["pen"]:6.2f}   '
          f'x-peak {S["sep"]["mw"]/S["sep"]["peak"]:.2f}/{S["grid"]["mw"]/S["grid"]["peak"]:.2f}')
    for k in ('sep', 'grid'):
        print(f'           {k:5s} capital {S[k]["cap"]/S["mwh"]:6.2f}  fuel {S[k]["fuel"]/S["mwh"]:6.2f}')
print(f'  penalty ratio {G["pen"]/E["pen"]:.2f}   gap {GAP:.2f}  levers {[round(v,2) for _n,v in LEV]}  '
      f'resid {RESID_V:.2f}')
print(f'  break-even months  ERCOT {E_BE:.1f}  Georgia {G_BE:.1f};  ERCOT@24 {E_RATIO[24]:.2f}x  '
      f'Georgia@6 {G_RATIO[6]:.2f}x @36 {G_RATIO[36]:.2f}x')

# ------------------------------------------------ hybrid (diesel-first), x = 0.3, 0.6, 1.0
# Georgia: Code/OffGrid/summarize_hybrid_diesel_other.py georgia; ERCOT: summarize_hybrid_diesel.py.
# Drawn as a RANGE over the three shares (light = full range, darker = x 0.3..0.6, line = x 0.6).
HYB_X = [0.0, 0.3, 0.6, 1.0]
DSL = '#e8663a'                 # hybrid colour, as Fig 3a's connected line and Fig 4
T_DSL_M = 12          # gensets run for the twelve months before the gas machines arrive


def load_hyb(path, table_is_list):
    if not os.path.exists(path):
        return None
    h = json.load(open(path))
    tb = {f"{r['share']:.1f}": r for r in h['table']} if table_is_list else h['table']
    o = dict(isl={}, con={}, tot={}, y1_cost={}, gas_GW={}, dsl_GW={}, ratio={}, cap={})
    for x in HYB_X:
        k = f'{x:.1f}'
        if k not in tb:
            continue
        r = tb[k]
        o['isl'][x] = r['lcoe_islanded']; o['tot'][x] = r['total']
        o['con'][x] = r.get('c_connected_retire', r.get('c_hybrid_connected_retire'))
        o['gas_GW'][x] = r['gas_MW']/1e3; o['dsl_GW'][x] = r['diesel_MW']/1e3; o['ratio'][x] = r['ratio_to_sum_of_peaks']
        o['cap'][x] = r['fixed_gas'] + r['land'] + r['fixed_diesel']
        o.setdefault('fuel', {})[x] = r['fuel_gas'] + r['fuel_diesel']
        o.setdefault('fixed_diesel', {})[x] = r['fixed_diesel']
        o.setdefault('diesel_MW', {})[x] = r['diesel_MW']
        o.setdefault('installed_MW', {})[x] = r['installed_MW']
        # year one on diesel alone: diesel fixed cost for the year + diesel fuel on x of the load
        if 'year1_fuel_diesel' in r:
            o['y1_cost'][x] = r['year1_fixed_diesel'] + r['year1_fuel_diesel']
        else:
            vr = h['params']['diesel']['vom'] + h['params']['diesel']['hr']*h['params']['diesel_fuel'] if 'params' in h else None
            o['y1_cost'][x] = r['fixed_diesel'] + x*E_MWH*vr
    return o


HYB_E = load_hyb(f'{O}/hybrid_ercot_summary.json', True)
HYB_G = load_hyb(f'{O}/hybrid_georgia_summary.json', False)
for nm, Hh in (('ERCOT', HYB_E), ('Georgia', HYB_G)):
    if Hh:
        print(f'  hybrid {nm}: islanded ' + ', '.join(f'x={x:.1f} ${v:.1f}' for x, v in Hh['isl'].items())
              + '; connected ' + ', '.join('-' if v is None else f'${v:.1f}' for v in Hh['con'].values()))


import matplotlib as _mpl
CMAPH = _mpl.colormaps['copper_r']
XCOL = {0.3: CMAPH(0.375), 0.6: CMAPH(0.60), 1.0: CMAPH(0.90)}     # Fig 3's hybrid colours


def dumbbell(ax, x0, vals, top, filled=True, label_x=True, fmt='{:.1f}', dx_lab=0.05, hatch=False, w=0.30):
    """The hybrid as a bar WITH its statistic: a bar to the middle share (x = 0.6), a lighter
    box spanning the lowest to the highest value over x, and one dot per x (light -> dark =
    0.3 -> 0.6 -> 1.0).  Open dots and an outlined box = after connection."""
    v = {k: vv for k, vv in vals.items() if vv is not None}
    if len(v) < 2:
        return
    lo, hi = min(v.values()), max(v.values())
    mid = v.get(0.6, np.median(list(v.values())))
    if filled:
        ax.bar(x0, mid, w, color=XCOL[0.6], alpha=0.55, lw=0, zorder=3)
        ax.bar(x0, hi - lo, w, bottom=lo, color=XCOL[0.6], alpha=0.25, lw=0, zorder=4)
        if hatch:
            ax.bar(x0, mid, w, fill=False, hatch='////', edgecolor='white', lw=0, zorder=4)
    else:
        ax.bar(x0, hi - lo, w, bottom=lo, fill=False, edgecolor=XCOL[0.6], lw=0.7, zorder=4)
    for xv in sorted(v, key=lambda k: v[k]):
        vv = v[xv]
        mk = 's' if hatch else 'o'
        if filled:
            ax.plot([x0], [vv], mk, ms=4.6, color=XCOL[xv], mec='white', mew=0.6, zorder=6)
        else:
            ax.plot([x0], [vv], mk, ms=4.6, mfc='white', mec=XCOL[xv], mew=1.1, zorder=6)
    if label_x:
        prev = -1e9
        for xv in sorted(v, key=lambda k: v[k]):
            yy = max(v[xv], prev + top*0.05); prev = yy
            ax.text(x0 + w/2 + dx_lab, yy, f'{xv:.1f}: {fmt.format(v[xv])}', ha='left', va='center', fontsize=4.9, color=XCOL[xv])


def hybrid_net_vs_wait(Hh, S, x, M):
    """Net value of the hybrid (share x) against waiting for the grid, when the grid takes M
    extra months after the gas build.  PV at the gas-arrival date, as the off-grid curve in d:
    + earnings: x of the fleet for the 12 genset months (before month 0) + the whole fleet for M months
    - electricity: year one on gensets, islanded cost for M months, then the connected premium
      (c_connected - c_on) for the rest of the 20 years.  Where no HOPE run exists yet the
      connected cost falls back to the islanded cost (flagged in the console)."""
    r_m = (1 + R)**(1/12) - 1
    M = np.asarray(M, float)
    # gensets: months -11..0 relative to gas arrival, compounded forward (they are earlier)
    k = np.arange(1, T_DSL_M + 1)
    f_y1 = float(((1 + r_m)**(k - 1)).sum())                     # sum over the 12 months, PV at month 0
    earn = RATE*S['fleet_mw']*(x*f_y1 + af_monthly(M, R))
    c_con = Hh['con'][x] if Hh['con'].get(x) is not None else Hh['isl'][x]
    cost_y1 = Hh['y1_cost'][x]*f_y1/12.0
    cost_isl = (Hh['isl'][x] - S['grid']['lcoe'])*S['mwh']/12.0*af_monthly(M, R)
    # connected premium from month M+1 to month 240
    cost_con = (c_con - S['grid']['lcoe'])*S['mwh']/12.0*(af_monthly(240, R) - af_monthly(M, R))
    return earn - cost_y1 - cost_isl - cost_con


# ==================================================================== the figure
figh = 128*MM
fig = plt.figure(figsize=(W2, figh))
LEFT, RIGHT, WSP = 0.155, 0.985, 0.40
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=WSP,
                      left=LEFT, right=RIGHT, top=0.945, bottom=0.60)
gs2 = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=WSP,
                       left=LEFT, right=RIGHT, top=0.44, bottom=0.08)


# The left column carries its y-axis labels inside the shared margin, so its letters need
# a longer reach to clear them and sit flush at the figure's left edge; the right column
# only has to clear its own axis labels.  a stays over c, and b over d, either way.
LX, RX = -0.41, -0.20


def letter(ax, s, x=RX):
    ax.text(x, 1.04, s, transform=ax.transAxes, fontsize=8, fontweight='bold',
            va='bottom', ha='left', color=INK)


# ---------------------------------------------------- a  over-build against campus size
ax = fig.add_subplot(gs[0, 0])
ax.set_xscale('log')
for S, sites, col, lab in ((E, E_SITES, ERC, 'ERCOT, 17 data centers'),
                           (G, G_SITES, GEO, 'Georgia, 31 data centers')):
    ax.scatter([r['peak_MW'] for r in sites], [r['ratio'] for r in sites], s=11, color=col,
               lw=0, alpha=0.9, zorder=4, label=lab)
    ax.scatter([S['grid']['peak']], [S['grid']['mw']/S['grid']['peak']], s=30, marker='s',
               facecolor='white', edgecolor=col, lw=0.9, zorder=5)
ax.axhline(1.0, color=INK3, lw=0.5, ls=(0, (3, 2)), zorder=1)
# the diamonds and squares are named in the legend rather than beside the markers:
# each shape appears twice, once per system, so a floating label can only point at one
ax.scatter([], [], s=16, marker='s', facecolor='white', edgecolor=INK2, lw=0.8,
           label='On-grid')
ax.set_xlim(130, 60000); ax.set_ylim(0.75, 2.75)
ax.set_xticks([200, 500, 2000, 10000, 30000])
ax.set_xticklabels(['200', '500', '2,000', '10,000', '30,000'])
ax.minorticks_off()
ax.set_xlabel('Data center peak demand (MW)')
ax.set_ylabel('Capacity / peak')
clean(ax)
ax.legend(loc='upper right', bbox_to_anchor=(1.02, 1.0), fontsize=6.0, handletextpad=0.5,
          borderpad=0.0, labelspacing=0.55, markerscale=1.0, handlelength=1.0)
if LETTERS: letter(ax, 'a', LX)

# ---------------------------------------------------- b  what a MWh costs, and of what
ax = fig.add_subplot(gs[0, 1])
# one set of architecture names across Figures 2, 3, 4 and 5
ARCH = [('off', SELF, 'Off-grid\ngas only'), ('off6', SELF, 'Off-grid\nx = 0.6'), ('hyb', DSL, 'Hybrid\nx = 0.6'), ('grid', GRID, 'On-grid')]
xs, bw = [], 0.34


def errbar(ax, x0, vals, w=0.10, span=True):
    """Whiskers span the lowest to the highest value over x = 0 ... 1.0 (the bar is x = 0.6)."""
    v = [vv for vv in vals.values() if vv is not None]
    lo, hi = min(v), max(v)
    ax.plot([x0, x0], [lo, hi], color=INK, lw=0.8, zorder=7)
    for yy in (lo, hi):
        ax.plot([x0 - w/2, x0 + w/2], [yy, yy], color=INK, lw=0.8, zorder=7)


FOM_DSL = 35000.0
for i, (k, col, lab) in enumerate(ARCH):
    for j, (S, Hh, hat) in enumerate(((E, HYB_E, ''), (G, HYB_G, '////'))):
        x = i + (j - 0.5)*(bw + 0.04)
        if k == 'grid':
            cap, fuel, vals = S['grid']['cap']/S['mwh'], S['grid']['fuel']/S['mwh'], None
        elif k == 'off':
            cap, fuel, vals = S['sep']['cap']/S['mwh'], S['sep']['fuel']/S['mwh'], None
        elif k == 'off6':
            cap, fuel, vals = Hh['cap'][0.6]/S['mwh'], Hh['fuel'][0.6]/S['mwh'], Hh['isl']
        else:
            if Hh['con'].get(0.6) is None:          # no HOPE run: outline only
                ax.bar(x, 60, bw, fill=False, edgecolor=col, lw=0.6, ls=(0, (2, 2)), zorder=3)
                ax.text(x, 30, 'HOPE\npending', ha='center', va='center', fontsize=5.0, color=col, rotation=90, linespacing=1.0)
                continue
            cap = (Hh['cap'][0.6] - Hh['diesel_MW'][0.6]*FOM_DSL)/S['mwh']; fuel = Hh['con'][0.6] - cap; vals = Hh['con']
        ax.bar(x, cap, bw, color=col, lw=0, zorder=3)
        ax.bar(x, fuel, bw, bottom=cap, color=col, alpha=0.38, lw=0, zorder=3)
        if hat:
            ax.bar(x, cap + fuel, bw, fill=False, hatch=hat, edgecolor='white', lw=0, zorder=4)
        hi = cap + fuel
        if vals is not None and all(vals.get(q) is not None for q in (0.3, 1.0)):
            errbar(ax, x, vals); hi = max([hi] + [v for v in vals.values() if v is not None])
        ax.text(x, hi + 1.5, f'{cap+fuel:.0f}', ha='center', va='bottom', fontsize=6.2, fontweight='bold', color=col)
        xs.append(x)
ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels([a[2] for a in ARCH], fontsize=6.0, linespacing=1.2)
ax.tick_params(axis='x', pad=20)
_bt = ax.get_xaxis_transform()
HCU5 = {k: json.load(open(f'{O}/hybrid_connected_util_{k}.json'))['table']['0.6']['util'] for k in ('ercot', 'georgia')}
for j, (S, Hh, dy) in enumerate(((E, HYB_E, -0.035), (G, HYB_G, -0.105))):
    u_off = S['sep']['util']
    ax.text(2 + (j - 0.5)*(bw + 0.04), dy, f'{100*HCU5[("ercot", "georgia")[j]]:.0f}%', transform=_bt, ha='center', va='top', fontsize=6.0, fontweight='bold', color=DSL)
    ax.text(0 + (j - 0.5)*(bw + 0.04), dy, f'{100*u_off:.0f}% utilized' if j == 0 else f'{100*u_off:.0f}%',
            transform=_bt, ha='center', va='top', fontsize=6.0, fontweight='bold', color=SELF)
    ax.text(1 + (j - 0.5)*(bw + 0.04), dy, f'{100*S["mwh"]/(Hh["installed_MW"][0.6]*8760):.0f}%', transform=_bt, ha='center', va='top', fontsize=6.0, fontweight='bold', color=SELF)
    ax.text(3 + (j - 0.5)*(bw + 0.04), dy, f'{100*S["grid"]["util"]:.0f}%', transform=_bt, ha='center', va='top', fontsize=6.0, fontweight='bold', color=GRID)
ax.set_xlim(-0.6, 3.6); ax.set_ylim(0, 150)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel('Cost of electricity (\\$ per MWh)')
clean(ax)
ax.legend(handles=[Patch(facecolor=INK2, lw=0, label='capital and land'),
                   Patch(facecolor=INK2, alpha=0.38, lw=0, label='fuel and O&M'),
                   Patch(facecolor=INK2, lw=0, label='ERCOT'),
                   Patch(facecolor=INK2, hatch='////', edgecolor='white', lw=0, label='Georgia'),
                   Line2D([], [], color=INK, lw=0.8, marker='_', ms=5, label='whiskers: range over x')],
          loc='upper left', bbox_to_anchor=(0.0, 1.03), ncol=2, fontsize=5.2,
          handlelength=1.1, handleheight=0.9, handletextpad=0.4, columnspacing=1.0,
          borderpad=0.0, labelspacing=0.3)
if LETTERS: letter(ax, 'b')

# ---------------------------------------------------- c  why Georgian self-supply is dearer
# A waterfall from Georgian self-supply down to ERCOT's, one step per input switched to
# ERCOT's value (each a full re-solve of the 31 campuses), the part they do not explain last.
ax = fig.add_subplot(gs2[0, 0])
NAME = {'Gas price': 'Cheaper gas',
        'Campus size': 'Fewer, larger\ndata centers',
        'Land': 'Cheaper land'}
steps = [('Georgia off-grid', '', G0, GEO)]
for n, v in sorted(LEV, key=lambda t: -t[1]):
    steps.append((NAME[n], '', v, STEP))
steps.append(('Others', '', RESID_V, RESID))
steps.append(('ERCOT off-grid', '', E['sep']['lcoe'], ERC))
y = np.arange(len(steps))[::-1].astype(float)
run = G0
for k, (name, _sub, v, col) in enumerate(steps):
    if k == 0 or k == len(steps) - 1:
        ax.barh(y[k], v, 0.58, color=col, lw=0, zorder=3)
        ax.text(v - 1.0, y[k], f'{v:.2f}', va='center', ha='right', fontsize=6.6,
                fontweight='bold', color='white', zorder=6)
    else:
        ax.barh(y[k], v, 0.58, left=run - v, color=col, lw=0, zorder=3)
        ax.plot([run, run], [y[k] - 0.29, y[k] + 0.71], color=INK3, lw=0.5, zorder=2)
        ax.text(run - v - 1.0, y[k], f'−{v:.2f}', va='center', ha='right', fontsize=6.4,
                fontweight='bold', color=INK if col == STEP else INK2)
        run -= v
ax.plot([run, run], [y[-1] - 0.29, y[-1] + 0.71], color=INK3, lw=0.5, zorder=2)
ax.set_yticks(y)
ax.set_yticklabels([t[0] for t in steps], fontsize=6.4, linespacing=1.25)
for lbl in ax.get_yticklabels():
    lbl.set_color(INK)
ax.tick_params(axis='y', length=0)
ax.set_xlim(0, 108); ax.set_ylim(-0.7, len(steps) - 0.3)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel('Cost of electricity (\\$ per MWh)')
clean(ax, keep=('bottom',))
if LETTERS: letter(ax, 'c', LX)

# ---------------------------------------------------- d  the time leg, Georgia: net value of
# building against waiting, as a function of the MONTH the grid arrives (campus ready at 0),
# for the gas-only off-grid route and the hybrid (x = 0.3, 0.6, 1.0).  PV at day 0, one 20-yr
# horizon counted from the gas build (month 24), per year of it.  Zero crossings = bounds.
ax = fig.add_subplot(gs2[0, 1])
T_DSL, T_GAS, N_MONTHS = 12, 24, 240
END = T_GAS + N_MONTHS; r_m = (1 + R)**(1/12) - 1; NYRS = N_MONTHS/12.0
e_m = G['mwh']/12.0; MWP = G['fleet_mw']; C_ON_G = G['grid']['lcoe']; C_OFF_G = G['sep']['cost']
VR_DSL = 197.2 if not HYB_G else (HYB_G['y1_cost'][1.0] - HYB_G['fixed_diesel'][1.0])/G['mwh']


def flows_g(kind, x, T_grid):
    mo = np.arange(END + 1); disc = (1 + r_m)**(-mo.astype(float))
    earn = np.zeros(END + 1); cost = np.zeros(END + 1)
    if kind == 'on':
        m = mo > T_grid; earn[m] = RATE*MWP; cost[m] = C_ON_G*e_m
        return earn*disc, cost*disc
    if kind == 'off0':
        m = mo > T_GAS; earn[m] = RATE*MWP; cost[m] = C_OFF_G/12.0
        return earn*disc, cost*disc
    H = HYB_G
    A = (mo > T_DSL) & (mo <= T_GAS); earn[A] = x*RATE*MWP; cost[A] = H['fixed_diesel'][x]/12.0 + VR_DSL*x*e_m
    B = mo > T_GAS if kind == 'off' else (mo > T_GAS) & (mo <= T_grid)
    earn[B] = RATE*MWP; cost[B] = H['tot'][x]/12.0
    if kind == 'hyb':
        c_con = H['con'][x] if H['con'].get(x) is not None else H['isl'][x]
        C = mo > T_grid; earn[C] = RATE*MWP; cost[C] = c_con*e_m
    return earn*disc, cost*disc


def pv_route_g(kind, T_grid, x=None):
    """Off-grid never connects; the hybrid takes the better of gas-then-connect or diesel-until-grid."""
    if kind != 'hyb':
        e_, c_ = flows_g(kind, x, T_grid); return float((e_ - c_).sum())
    e_, c_ = flows_g('hyb', x, T_grid)
    v_gas = float((e_ - c_).sum()) if T_grid > T_GAS else -np.inf
    mo = np.arange(END + 1); disc = (1 + r_m)**(-mo.astype(float))
    earn = np.zeros(END + 1); cost = np.zeros(END + 1)
    A = (mo > T_DSL) & (mo <= T_grid); earn[A] = x*RATE*MWP; cost[A] = HYB_G['fixed_diesel'][x]/12.0 + VR_DSL*x*e_m
    C = mo > T_grid; earn[C] = RATE*MWP; cost[C] = C_ON_G*e_m + HYB_G['diesel_MW'][x]*800.0*1000.0*0.1019/12.0
    return max(v_gas, float(((earn - cost)*disc).sum()))


TG = np.arange(0, 73)
# Build-your-own versus waiting, by diesel share x (off-grid and hybrid are identical until connection).
totg = lambda k, x, t: float(sum(e - c for e, c in zip(*flows_g(k, x, t))))
ON_G = np.array([totg('on', None, t) for t in TG])/1e9/NYRS
if HYB_G and any(v is None for v in HYB_G['con'].values()):
    print('  [d] Georgia hybrid: no HOPE connected cost yet -> connected = before-connection cost (placeholder)')
NETG, BOUND_G, ROUTES, EARN_G, COST_G = {}, {}, [], {}, {}
tf = np.linspace(0, 72, 3601)
fon = {t: flows_g('on', None, t) for t in TG}
for x in HYB_X:
    lab = f'Build own, x = {x:.1f}'
    NETG[lab] = np.array([totg('hyb', x, t) for t in TG])/1e9/NYRS - ON_G
    fh = {t: flows_g('hyb', x, t) for t in TG}
    EARN_G[lab] = np.array([float((fh[t][0] - fon[t][0]).sum()) for t in TG])/1e9/NYRS
    COST_G[lab] = np.array([float((fh[t][1] - fon[t][1]).sum()) for t in TG])/1e9/NYRS
    nf = np.interp(tf, TG, NETG[lab])
    BOUND_G[lab] = float(tf[np.argmax(nf >= 0)]) if (nf >= 0).any() and nf[0] < 0 else np.nan
YLO, YHI = 0.0, float(max(v.max() for v in EARN_G.values()))*1.62
VTOP = 0.64                                  # dashed marks and the diesel band stop here; the legend sits above
# Both sides as positive amounts, each a band over x = 0 ... 1.0 with the x = 0.6 line;
# they cross at the bound, shown as a point (x = 0.6) with a bar over x = 0 ... 1.0.
XM = 0.6; lm = f'Build own, x = {XM:.1f}'; EARN_COL, COST_COL = SELF, '#3d4f5c'   # cost in slate, as Fig. 3c
EA = np.array([EARN_G[f'Build own, x = {x:.1f}'] for x in HYB_X]); CA = np.array([COST_G[f'Build own, x = {x:.1f}'] for x in HYB_X])
ax.fill_between(TG, EA.min(0), EA.max(0), color=EARN_COL, alpha=0.18, lw=0, zorder=1)
ax.fill_between(TG, CA.min(0), CA.max(0), color=COST_COL, alpha=0.25, lw=0, zorder=1)
ax.plot(TG, EARN_G[lm], color=EARN_COL, lw=1.6, zorder=4)
ax.plot(TG, COST_G[lm], color=COST_COL, lw=1.6, zorder=4)
FS5 = 6.0
bs = [BOUND_G[f'Build own, x = {x:.1f}'] for x in HYB_X]; tb = BOUND_G[lm]; yb = float(np.interp(tb, TG, COST_G[lm]))
ax.errorbar([tb], [yb], xerr=[[tb - min(bs)], [max(bs) - tb]], fmt='o', ms=5.0, mfc='white', mec=INK, mew=1.2,
            ecolor=INK, elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=8)
ax.annotate(f'{tb:.0f} months\n({min(bs):.0f}–{max(bs):.0f})', xy=(tb, yb), xytext=(1.0, yb + YHI*0.14), ha='left', va='bottom',
            fontsize=FS5, fontweight='bold', color=INK, linespacing=1.1, zorder=9, bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'), arrowprops=dict(arrowstyle='-', color=INK, lw=0.5, shrinkA=0, shrinkB=4))
# Georgia Power's own connection times: 6 months (existing generation) and 36 months (new plant) after the gas build
# (the Georgia Power 30- and 60-month queue markers were removed ; the queue lengths are discussed in the text)
for _t, _lab, _ha, _dx in ((T_DSL, 'Diesel\nstarts\nto run', 'right', -0.8), (T_GAS, 'Gas\nstarts\nto run', 'left', 0.8)):
    ax.axvline(_t, ymax=VTOP*0.86, color=INK3, lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax.text(_t + _dx, YLO + (YHI - YLO)*VTOP*0.86, _lab, fontsize=FS5, color=INK2, ha=_ha, va='top', linespacing=1.05)
ax.set_xlim(0, 72); ax.set_ylim(YLO, YHI)
ax.set_xticks(range(0, 73, 12)); ax.set_xticklabels([f'{m//12}' for m in range(0, 73, 12)])
ax.set_xlabel('Interconnection queue (years)')
ax.set_ylabel('Present value of off-grid/hybrid above\non-grid (\\$ billion per year)')
from matplotlib.legend_handler import HandlerTuple
_band = lambda c, a_: Patch(facecolor=c, alpha=a_, lw=0)
_line = lambda c: Line2D([], [], color=c, lw=1.6)
ax.legend(handles=[(_band(EARN_COL, 0.18), _line(EARN_COL)), (_band(COST_COL, 0.25), _line(COST_COL)),
                   Line2D([], [], color=INK2, lw=1.6), _band(INK2, 0.25),
                   Line2D([], [], marker='o', ms=4.6, mfc='white', mec=INK, mew=1.1, lw=0)],
          labels=['Earnings above on-grid', 'Electricity cost above on-grid', 'Line: x = 0.6', 'Shading: x = 0–1.0', 'Threshold'],
          handler_map={tuple: HandlerTuple(ndivide=1, pad=0)},
          loc='upper left', bbox_to_anchor=(0.0, 1.02), fontsize=FS5, handlelength=2.0, borderpad=0.0, labelspacing=0.35, ncol=2,
          columnspacing=1.2)
clean(ax)
if LETTERS: letter(ax, 'd')
print('Georgia bounds (grid month):', {k: round(v, 1) for k, v in BOUND_G.items()})

fig.savefig(f'{FIG}/fig5_georgia.pdf', bbox_inches='tight')
fig.savefig(f'{FIG}/fig5_georgia.png', dpi=400, bbox_inches='tight')
print(f'fig5 {W2/MM:.0f} x {figh/MM:.0f} mm')


# ============================================================ numbers and panel data
def pack(S):
    d = {k: dict(lcoe=S[k]['lcoe'], capital_per_MWh=S[k]['cap']/S['mwh'],
                 fuel_per_MWh=S[k]['fuel']/S['mwh'], installed_MW=S[k]['mw'],
                 peak_MW=S[k]['peak'], x_peak=S[k]['mw']/S[k]['peak'])
         for k in ('sep', 'grid')}
    d.update(penalty=S['pen'], energy_MWh=S['mwh'])
    return d


json.dump(dict(
    ercot=dict(pack(E), obj_dc=E_ON['obj_dc'], obj_base=E_ON['obj_base'],
               hope_cases='TX8z_GTEP_dc_vre_case, TX8z_GTEP_base_vre_case'),
    georgia=dict(pack(G), obj_dc=OBJ_D, obj_base=OBJ_B, prm=dict(base=GNB['georgia']['prm']['base'], dc=GNB['georgia']['prm']['dc']) if 'prm' in GNB['georgia'] else dict(base=0.151, dc=0.115),
                 hope_cases='GA1z_GTEP_dc_case, GA1z_GTEP_base_case'),
    penalty_ratio=G['pen']/E['pen'],
    levers=[dict(name=n, saving=v) for n, v in LEV] + [dict(name='interaction', saving=RESID_V)],
    gap=GAP,
    time=dict(rate=RATE, r=R, n_elec=N_ELEC, ercot_penalty_pv=E_PEN_PV,
              georgia_penalty_pv=G_PEN_PV, ercot_ratio_at_24=float(E_RATIO[24]),
              georgia_ratio_at_6=float(G_RATIO[6]), georgia_ratio_at_36=float(G_RATIO[36]),
              breakeven_months=dict(ercot=E_BE, georgia=G_BE))),
    open(f'{FIG}/fig5_numbers.json', 'w'), indent=1)

with open(f'{DATA}/fig5_panel_a_campuses.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['system', 'site', 'peak_MW', 'installed_MW', 'ratio', 'lcoe'])
    for S, nm, sites in ((E, 'ERCOT', E_SITES), (G, 'Georgia', G_SITES)):
        for r in sites:
            w.writerow([nm, r['site'], f'{r["peak_MW"]:.1f}', r['installed_MW'],
                        f'{r["ratio"]:.4f}', f'{r["lcoe"]:.3f}'])
        for k in ('grid',):
            w.writerow([nm, k, f'{S[k]["peak"]:.1f}', f'{S[k]["mw"]:.1f}',
                        f'{S[k]["mw"]/S[k]["peak"]:.4f}', f'{S[k]["lcoe"]:.3f}'])
with open(f'{DATA}/fig5_panel_b_cost.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['system', 'architecture', 'capital_land_USD_MWh',
                                   'fuel_om_USD_MWh', 'lcoe_USD_MWh'])
    for S, nm in ((E, 'ERCOT'), (G, 'Georgia')):
        for k in ('sep', 'grid'):
            w.writerow([nm, k, f'{S[k]["cap"]/S["mwh"]:.3f}', f'{S[k]["fuel"]/S["mwh"]:.3f}',
                        f'{S[k]["lcoe"]:.3f}'])
        w.writerow([nm, 'penalty', '', '', f'{S["pen"]:.3f}'])
with open(f'{DATA}/fig5_panel_c_waterfall.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['step', 'USD_MWh', 'running_total_after'])
    run = G0; w.writerow(['Georgia off-grid', f'{G0:.3f}', f'{run:.3f}'])
    for n, v in sorted(LEV, key=lambda t: -t[1]) + [('Others', RESID_V)]:
        run -= v; w.writerow([NAME.get(n, n).replace('\n', ' '), f'{v:.3f}', f'{run:.3f}'])
    w.writerow(['ERCOT off-grid', f'{E["sep"]["lcoe"]:.3f}', ''])
import pandas as pd
pd.DataFrame({'month_grid_arrives': TG, **{k + ' earnings above on-grid': v for k, v in EARN_G.items()},
              **{k + ' electricity cost above on-grid': v for k, v in COST_G.items()},
              **{k + ' net': v for k, v in NETG.items()}}).to_csv(f'{DATA}/fig5_panel_d_time.csv', index=False)   # $B per yr of horizon
print(f'written {FIG}/fig5_numbers.json and {DATA}/fig5_panel_*.csv')
