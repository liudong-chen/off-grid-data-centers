"""Figure 4 (Figures: gas-only off-grid designs, pooled plant removed; the pooled
file is read only for the coincident peak).

Figure 4 -- 2035: what changes when the data centers are 20 GW or 121 GW.

Two demand cases, three ways to power them, three questions:

  a  what is actually built                (sectors: how much, and of what)
  b  what the electricity costs           ($ per MWh)
  c  what it costs to build and run it    ($ billion a year, capital against fuel)

WHY SECTORS IN a.  The panel has to carry two things at once -- how much capacity an
architecture builds, and what kind.  A stacked bar carries the second well and the first
badly, because the 121 GW case is six times the 20 GW case and the smaller bars become
stubs.  A ring carries both: AREA is capacity built per MW of data-center demand, which
is the comparable quantity and lives in a narrow 0.9-2.0 range, and the wedges are the
technology split.  The absolute gigawatts are printed underneath, so nothing is lost to
the encoding.  It is the same visual vocabulary as Figure 2b.

WHY c IS NEW.  Cost per MWh is what a buyer pays; it is not what has to be financed.
Panel c separates the two, and the capital share is the number that decides whether an
architecture is buildable at all: at 121 GW the grid's answer needs $20.8 B/yr of
investment -- $20.0 B of generation plus $0.7 B of transmission -- against the shared
plant's $12.3 B, and it is still cheaper per MWh.

    python 79_fig4.py
    -> fig4_scale.pdf/.png, fig4_numbers.json
"""
import json, os
LETTERS = False   # panel letters off (added in layout)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Wedge
from matplotlib.lines import Line2D
import numpy as np

import os as _os
_PKG = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _os.path.isdir(f'{_PKG}/01_raw'):
    O, OUT = f'{_PKG}/01_raw', f'{_PKG}/04_figure'
else:
    OUT = 'build'
    O = f'{OUT}/out'
MM = 1/25.4
W2 = 183*MM

GRID, POOL, SELF = '#2a78d6', '#e8663a', '#96331a'
INK, INK2, INK3 = '#161616', '#63625f', '#9a9995'
GAS, WIND, SOL = '#5f5b55', '#2e8b78', '#efc14e'
LINE, LINE0 = '#8e6bb8', '#d6c8e6'      # transmission: added by the data centers,
                                        # and the corridor the grid builds anyway

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
})


def clean(ax, keep=('left', 'bottom')):
    for sp in ('top', 'right', 'left', 'bottom'):
        ax.spines[sp].set_visible(sp in keep)
    for sp in keep:
        ax.spines[sp].set_position(('outward', 3))


def panel(ax, s, x=-0.13, y=1.03):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight='bold',
            va='bottom', ha='left', color=INK)


# ======================================================== assemble, from solved output
DESIGN_MW = 40425.0
ON35 = json.load(open(f'{O}/ongrid2035.json'))
# The transmission figures reported per case are INCREMENTS, dc - base.  The base case
# builds a corridor of its own from the underlying demand growth, and saying "no new
# transmission" without that number invites the reader to think the grid builds none.
BASE_LINE_GW = sum(ON35['base']['lines'].values())/1000
# one set of architecture names across Figures 2, 3, 4 and 5
LAB = ['Off-grid', 'On-grid']          # Figures: gas-only off-grid, no pooled plant
COL = [SELF, GRID]

CASES = []
for k, tag, key in ((0.5, 'k0p5', 'dc20'), (3.0, 'k3', 'dc120')):
    st = json.load(open(f'{O}/offgrid2035_{tag}_sites.json'))
    p = json.load(open(f'{O}/offgrid2035_{tag}_pool.json'))
    on = ON35[key]
    best = min(p.values(), key=lambda r: r['cost'])

    # capital = fixed + land, fuel = the remainder of the annual cost.  The small net
    # gain from wind and solar sits in the fuel term, which is where it is earned.
    sep_cap = sum(r['fixed_cost'] + r['land_cost'] + r.get('vre_cost', 0.0) for r in st)
    sep_cost = sum(r['cost'] for r in st)
    sep_mw = sum(r['installed_MW'] for r in st)
    sep_pk = sum(r['peak_MW'] for r in st)
    sep_w = sum(r.get('wind_MW', 0.0) for r in st)
    sep_s = sum(r.get('solar_MW', 0.0) for r in st)
    sep_e = sum(r['energy_MWh'] for r in st)
    sep_b = sum(r.get('batt_MW', 0.0) for r in st)
    SZ = {'Recip': 20, 'Aero35': 35, 'Aero48': 48, 'FrameCT': 419, 'CC': 627}
    cnt = {}
    for r in st:
        for kk, nn in r['mix'].items():
            cnt[kk] = cnt.get(kk, 0) + nn
    sep_units = sorted([(SZ[kk], nn) for kk, nn in cnt.items()], key=lambda t: -t[0]*t[1])
    pool_units = sorted([(SZ[kk], nn) for kk, nn in best['mix'].items()], key=lambda t: -t[0]*t[1])

    pool_cap = best['fixed_cost'] + best['land_cost'] + best.get('vre_cost', 0.0)
    pw, ps = best.get('wind_MW', 0.0), best.get('solar_MW', 0.0)

    # dc minus base per technology; a small NEGATIVE entry (the data centers displace a little of the
    # base case's wind, -0.27 GW at 20 GW) is folded into the gas wedge rather than drawn as a wedge.
    on_w = max(on['by_tech'].get('WindOn', 0.0), 0.0)
    on_s = max(on['by_tech'].get('SolarPV', 0.0), 0.0)

    CASES.append(dict(
        k=k, gw=k*DESIGN_MW/1000, line_gw=on['new_line_MW']/1000,
        peak=best['peak_MW']/1000, sum_peak=sep_pk/1000,
        bars=[
            dict(lab='Off-grid, x = 0', col=SELF, lcoe=sep_cost/sep_e,
                 cap=sep_cap/1e9, fuel=(sep_cost - sep_cap)/1e9,
                 mw=sep_mw/1000, pk=sep_pk/1000,
                 tech=dict(Gas=(sep_mw - sep_w - sep_s - sep_b)/1000,
                           Wind=sep_w/1000, Solar=sep_s/1000), units=sep_units, batt=sep_b/1000),
            dict(lab=LAB[1], col=GRID, lcoe=on['lcoe'],
                 cap=(on['inv'] + on.get('inv_line', 0.0))/1e9, cap_line=on.get('inv_line', 0.0)/1e9,
                 fuel=on['opr_increment']/1e9,
                 mw=(on['new_MW'] + max(on['new_line_MW'], 0.0))/1000,
                 pk=best['peak_MW']/1000, base_line=BASE_LINE_GW,
                 tech=dict(Gas=(on['new_MW'] - on_w - on_s)/1000,
                           Wind=on_w/1000, Solar=on_s/1000,
                           Transmission=max(on['new_line_MW'], 0.0)/1000),
                 units=[(on.get('unit_MW', 627.0), (on['new_MW'] - on_w - on_s)/on.get('unit_MW', 627.0))], batt=0.0),
        ]))

for c in CASES:
    for b in c['bars']:
        b['ratio'] = b['mw']/b['pk']

print('FIGURE 4 inputs')
for c in CASES:
    print(f'  {c["gw"]:.0f} GW of data centers, {c["peak"]:.1f} GW coincident peak, '
          f'{c["sum_peak"]:.1f} GW sum of site peaks')
    for b in [c['bars'][0], c['bars'][-1]]:
        t = b['tech']
        print(f'     {b["lab"].replace(chr(10)," "):26s} ${b["lcoe"]:6.2f}/MWh  '
              f'cap ${b["cap"]:6.2f} B  fuel ${b["fuel"]:6.2f} B   '
              f'{b["mw"]:7.1f} GW ({b["ratio"]:.2f}/MW)  '
              f'gas {t["Gas"]:6.1f} wind {t["Wind"]:5.1f} solar {t["Solar"]:5.1f}')
    print(f'     new transmission {c["line_gw"]:.1f} GW')
    for b in [c['bars'][0], c['bars'][-1]]:
        chk = (b['cap'] + b['fuel'])*1e9/(c['gw']/121.275*666_153_000.0 if c['gw'] > 60
                                          else 111_025_500.0)
        print(f'     [check] {b["lab"].replace(chr(10)," "):38s} cap+fuel -> '
              f'${chk:.2f}/MWh against plotted ${b["lcoe"]:.2f}/MWh')

# ------------------------------------------------ hybrid (diesel-first), x = 0.3, 0.6, 1.0
# One band per demand case instead of three more bars: the range of the metric over the
# year-one diesel shares, darker between x = 0.3 and 0.6, lighter out to x = 1.0, with the
# three values ticked.  Summaries from Code/OffGrid/summarize_hybrid_diesel_other.py.
HYB_X = [0.0, 0.3, 0.6, 1.0]
HYBC = '#e8663a'                 # the hybrid's colour, as the connected line in Fig 3a
OFF6 = '#7e5033'                 # off-grid x = 0.6: copper brown, apart from gas-only red and hybrid orange
HYB = {}
for c, tag in zip(CASES, ('k0p5', 'k3')):
    f = f'{O}/hybrid_2035_{tag}_summary.json'
    if not os.path.exists(f):
        print(f'   (no hybrid summary for {tag} yet)'); continue
    h = json.load(open(f)); tb = h['table']
    HYB[c['k']] = dict(
        lcoe_isl={x: tb[f'{x:.1f}']['lcoe_islanded'] for x in HYB_X if f'{x:.1f}' in tb},
        lcoe_con={x: tb[f'{x:.1f}']['c_connected_retire'] for x in HYB_X if f'{x:.1f}' in tb},
        cost_B={x: tb[f'{x:.1f}']['total']/1e9 for x in HYB_X if f'{x:.1f}' in tb},
        cap_B={x: (tb[f'{x:.1f}']['fixed_gas'] + tb[f'{x:.1f}']['land'] + tb[f'{x:.1f}']['fixed_diesel'])/1e9 for x in HYB_X if f'{x:.1f}' in tb},
        fuel_B={x: (tb[f'{x:.1f}']['fuel_gas'] + tb[f'{x:.1f}']['fuel_diesel'])/1e9 for x in HYB_X if f'{x:.1f}' in tb},
        gas_GW={x: tb[f'{x:.1f}']['gas_MW']/1e3 for x in HYB_X if f'{x:.1f}' in tb},
        dsl_GW={x: tb[f'{x:.1f}']['diesel_MW']/1e3 for x in HYB_X if f'{x:.1f}' in tb},
        ratio={x: tb[f'{x:.1f}']['ratio_to_sum_of_peaks'] for x in HYB_X if f'{x:.1f}' in tb},
        c_on=h.get('c_on'),
        con_B={x: (tb[f'{x:.1f}']['c_connected_retire']*h['energy_MWh']/1e9 if tb[f'{x:.1f}'].get('c_connected_retire') is not None else None) for x in HYB_X if f'{x:.1f}' in tb},
        con_cap_B={x: (tb[f'{x:.1f}']['fixed_gas'] + tb[f'{x:.1f}']['land'] + tb[f'{x:.1f}']['fixed_diesel']
                       - tb[f'{x:.1f}']['diesel_MW']*h['FOM_DSL'])/1e9 for x in HYB_X if f'{x:.1f}' in tb},
        units_x1=[(627.0, tb['0.6']['n_CC']), (419.0, tb['0.6']['n_FrameCT']), (35.0, tb['0.6']['n_Aero35']),
                  (48.0, tb['0.6']['n_Aero48']), (20.0, tb['0.6']['n_Recip'])])
    r1 = tb['0.6']
    c['bars'].insert(1, dict(lab='Off-grid/hybrid, x = 0.6', col=HYBC, lcoe=r1['lcoe_islanded'], cap=None, fuel=None,
                          mw=r1['installed_MW']/1000, pk=c['bars'][0]['pk'], hybrid=True,
                          tech=dict(Gas=r1['gas_MW']/1000, Diesel=r1['diesel_MW']/1000),
                          units=[(sz, nn) for sz, nn in HYB[c['k']]['units_x1'] if nn > 0], dsl_units=r1['diesel_units'], batt=0.0))
    c['bars'][1]['ratio'] = c['bars'][1]['mw']/c['bars'][1]['pk']
    print(f'   hybrid {c["gw"]:.0f} GW: islanded ' + ', '.join(f'x={x:.1f} ${v:.1f}' for x, v in HYB[c["k"]]['lcoe_isl'].items())
          + '; connected ' + ', '.join(f'{"-" if v is None else f"${v:.1f}"}' for v in HYB[c['k']]['lcoe_con'].values()))


import matplotlib as _mpl
CMAPH = _mpl.colormaps['copper_r']
XCOL = {0.3: CMAPH(0.375), 0.6: CMAPH(0.60), 1.0: CMAPH(0.90)}     # Fig 3's hybrid colours


def errbar(ax, x0, vals, w=0.08, span=True):
    """Sensitivity to the year-one diesel share: whiskers span the lowest to the highest value
    over x = 0 ... 1.0 (the bar is x = 0.6)."""
    v = [vv for vv in vals.values() if vv is not None]
    if len(v) < 2:
        return
    lo, hi = min(v), max(v)
    ax.plot([x0, x0], [lo, hi], color=INK, lw=0.8, zorder=7)
    for yy in (lo, hi):
        ax.plot([x0 - w/2, x0 + w/2], [yy, yy], color=INK, lw=0.8, zorder=7)


def hybar_cost(ax, x0, H, w, top):
    """Hybrid at x = 0.6 as two half-width bars: before connection (left), after connection (right);
    whiskers = x 0.3 and 1.0."""
    isl, con = H['lcoe_isl'][0.6], H['lcoe_con'].get(0.6)
    hw = w/2
    ax.bar(x0 - hw/2, isl, hw*0.94, color=HYBC, alpha=0.45, lw=0, zorder=3)
    errbar(ax, x0 - hw/2, H['lcoe_isl'], w=0.05)
    ax.text(x0 - hw/2, max(H['lcoe_isl'].values()) + top*0.02, f'{isl:.1f}', ha='center', va='bottom', fontsize=6.0, fontweight='bold', color=HYBC)
    if con is not None:
        ax.bar(x0 + hw/2, con, hw*0.94, color=HYBC, lw=0, zorder=3)
        if all(v is not None for v in H['lcoe_con'].values()):
            errbar(ax, x0 + hw/2, H['lcoe_con'], w=0.05)
            ax.text(x0 + hw/2, max(H['lcoe_con'].values()) + top*0.02, f'{con:.1f}', ha='center', va='bottom', fontsize=6.0, fontweight='bold', color=HYBC)
        else:
            ax.text(x0 + hw/2, con + top*0.02, f'{con:.1f}', ha='center', va='bottom', fontsize=6.0, fontweight='bold', color=HYBC)


def hybar_annual(ax, x0, H, w, tot):
    """Hybrid at x = 0.6 before connection: capital block + fuel block; whiskers = x 0.3 and 1.0."""
    cap, fuel = H['cap_B'][0.6], H['fuel_B'][0.6]
    ax.bar(x0, cap, w, color=HYBC, lw=0, zorder=3)
    ax.bar(x0, fuel, w, bottom=cap, color=HYBC, alpha=0.34, lw=0, zorder=3)
    errbar(ax, x0, H['cost_B'])
    ax.text(x0, max(H['cost_B'][0.3], H['cost_B'][1.0], cap + fuel) + tot*0.022, f'{cap + fuel:.1f}', ha='center', va='bottom', fontsize=6.4, fontweight='bold', color=HYBC)
    if cap > tot*0.10:
        ax.text(x0, cap/2, f'{cap:.1f}', ha='center', va='center', fontsize=5.8, color='white', fontweight='bold', zorder=6)
    else:
        ax.text(x0, cap + tot*0.012, f'{cap:.1f}', ha='center', va='bottom', fontsize=5.6, color=HYBC, fontweight='bold', zorder=6)


# ==================================================================== the figure
figh = 128*MM
fig = plt.figure(figsize=(W2, figh))
gs = fig.add_gridspec(2, 2, height_ratios=[1.20, 1.0], width_ratios=[1.0, 1.0],
                      wspace=0.30, hspace=0.40,
                      left=0.065, right=0.985, top=0.955, bottom=0.075)

X = np.array([0.0, 1.15])
W = 0.26
OFF = np.array([-0.30, 0.30])      # off-grid (left), on-grid (right); the hybrid bar in the middle
XH = 0.0
XT = [f'{c["gw"]:.0f} GW of data centers' for c in CASES]

# ------------------------------------------------- a  what is built, as sectors
# Ring AREA is capacity built per MW of data-center peak; wedges are the technology
# split; the absolute gigawatts are printed underneath.
ax = fig.add_subplot(gs[0, :])
ax.set_axis_off()
_p = ax.get_position()
ax.set_position([0.015, _p.y0, 0.985 - 0.015, _p.height])
_LETTER_X = 0.065 - 0.135*gs[1, 0].get_position(fig).width
DSLC = '#3d4f5c'
TECH = [('Gas', GAS), ('Diesel', DSLC), ('Wind', WIND), ('Solar', SOL), ('Transmission', LINE)]

# Circles must LOOK circular, so the data box is given the axes box's own aspect ratio
# rather than a guessed one: fix the x span, read the realised axes geometry, and let
# the y span follow.  Without this the panel either letterboxes or draws ellipses.
fig.canvas.draw()
bb = ax.get_position()
XSPAN = 7.60
YSPAN = XSPAN*(bb.height*figh)/(bb.width*W2)
ax.set_xlim(0, XSPAN); ax.set_ylim(0, YSPAN); ax.set_aspect('equal')

# Every ring the same size, and the BUILD written in its hole: what the solution is made
# of. Gas arrives in whole machines, so it is size x count; wind, solar and storage are
# continuous decision variables with no unit size and are given as GW in their wedge's
# colour. The grid's gas build is continuous in HOPE and is expressed in units of the
# reference NGCC of the candidate file (627 MW, the EIA AEO2026 single-shaft CC), as Fig. 1c does.
# Six rings across 183 mm leave ~26 mm each, so the area encoding had to go (an
# area-scaled 121 GW off-grid ring with seven rows had a hole for 4 pt text); capacity per
# MW of peak is printed under every ring instead.
RMAX = 0.47
BAND = 0.17
CX = [0.72, 1.87, 3.02, 4.58, 5.73, 6.88]   # six rings, three per case, centred on the FIGURE
CY = YSPAN - 0.80
PTS_PER_UNIT = (bb.width*W2/XSPAN)*72.0    # data units -> points, for the fit below


def fitfs(rows, r_hole, cap=5.4, lsp=1.25):
    """Largest point size at which every row fits its own chord of the hole."""
    n = len(rows)
    for fs in np.arange(cap, 3.0, -0.1):
        h = n*fs*lsp/72.0*72.0                        # block height in points
        if h > 2*r_hole*PTS_PER_UNIT*0.84:
            continue
        ok = True
        for i, (txt, _c) in enumerate(rows):
            yc = (n/2 - i - 0.5)*fs*lsp                # row centre, points from the middle
            chord = 2*np.sqrt(max((r_hole*PTS_PER_UNIT)**2 - yc**2, 0.0))
            if 0.56*fs*len(txt) > chord*0.90:
                ok = False; break
        if ok:
            return fs
    return 3.0


UNITS_OUT = []
n = 0
for c in CASES:
    for b in c['bars']:
        cx = CX[n]
        R = RMAX
        th = 90.0
        for name, col in TECH:
            v = b['tech'].get(name, 0.0)
            if v <= 0:
                continue
            d = 360.0*v/b['mw']
            ax.add_patch(Wedge((cx, CY), R, th - d, th, width=BAND*R,
                               facecolor=col, lw=0.5, edgecolor='white', zorder=4))
            th -= d
        rows = [(f'{sz:.0f} MW × {nn:.0f}', INK2) for sz, nn in b['units'] if nn >= 0.5]
        if b.get('dsl_units'):
            rows.append((f'diesel 3 MW × {b["dsl_units"]:,}', DSLC))
        for name, col, key in (('wind', WIND, 'Wind'), ('solar', SOL, 'Solar')):
            v = b['tech'].get(key, 0.0)
            if v >= 0.05:
                rows.append((f'{name} {v:.1f} GW', col))
        if b.get('batt', 0.0) >= 0.1:
            rows.append((f'battery {b["batt"]:.1f} GW', INK2))
        v = b['tech'].get('Transmission', 0.0)
        if v >= 0.05:
            rows.append((f'lines {v:.1f} GW', LINE))
        r_hole = R*(1 - BAND)
        fs = fitfs(rows, r_hole)
        lh = fs*1.25/PTS_PER_UNIT                     # line height in data units
        UNITS_OUT.append(dict(data_centers_GW=round(c['gw'], 1), architecture=b['lab'],
                              built_GW=round(b['mw'], 2), per_MW_of_peak=round(b['ratio'], 3),
                              in_the_hole=' | '.join(t for t, _ in rows)))
        for i, (txt, col) in enumerate(rows):
            ax.text(cx, CY + (len(rows)/2 - i - 0.5)*lh, txt, ha='center', va='center',
                    fontsize=fs, color=col, zorder=6)
        if b.get('base_line'):
            # the corridor ERCOT builds with no data centres at all (2.8 GW, base 2035 case),
            # at each ring's own GW-per-degree scale -- a 68 deg arc on the 14.8 GW ring and a
            # 5 deg tick on the 199.5 GW one -- so both carry the number
            dd = 360.0*b['base_line']/b['mw']
            ax.add_patch(Wedge((cx, CY), R*1.17, 90.0 - dd, 90.0, width=0.10*R,
                               facecolor=LINE0, lw=0, zorder=4))
            a_ = np.radians(90.0 - dd/2)
            ax.text(cx + R*1.30*np.cos(a_), CY + R*1.30*np.sin(a_), f'{b["base_line"]:.1f} GW',
                    fontsize=5.6, fontweight='bold', color='#b9a2de', ha='center', va='bottom', zorder=6)
        ax.text(cx, CY - RMAX - 0.16, b['lab'], ha='center', va='top',
                fontsize=6.0, color=INK2, linespacing=1.25)
        ax.text(cx, CY - RMAX - 0.40, f'{b["mw"]:.1f} GW', ha='center', va='top',
                fontsize=6.6, fontweight='bold', color=b['col'])
        ax.text(cx, CY - RMAX - 0.62, f'{b["ratio"]:.2f} per MW of peak', ha='center', va='top',
                fontsize=5.2, color=INK3)
        n += 1

import csv as _csv
with open(f'{_os.path.dirname(OUT)}/03_panel_data/fig4_panel_a_units.csv', 'w', newline='') as _fh:
    _w = _csv.DictWriter(_fh, fieldnames=list(UNITS_OUT[0])); _w.writeheader(); _w.writerows(UNITS_OUT)
for i, c in enumerate(CASES):
    ax.text(np.mean(CX[3*i:3*i + 3]), CY + RMAX + 0.20, XT[i], ha='center',
            va='bottom', fontsize=6.8, fontweight='bold', color=INK)

ax.legend(handles=[Patch(facecolor=col, lw=0,
                         label=('Transmission added by the data centers'
                                if nm == 'Transmission' else nm))
                   for nm, col in TECH]
                  + [Patch(facecolor=LINE0, lw=0,
                           label='Transmission built because of demand growth')],
          loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=6, fontsize=6.0,
          handlelength=1.0, handleheight=0.85, handletextpad=0.35, borderpad=0.0,
          columnspacing=1.4)
if LETTERS: panel(ax, 'a', x=(_LETTER_X - bb.x0)/bb.width, y=1.005)

# ---------------------------------------------------------- b  cost of electricity
# Three approaches, x = 0.6 for off-grid (never connects) and hybrid (after connection), whiskers at
# x = 0.3 and x = 1.0; the gas-only off-grid design (x = 0, the paper's reference) as a white tick.
POS3 = np.array([-0.26, None, 0.0, 0.26], dtype=object)   # off-grid x=0, (unused), hybrid x=0.6, on-grid
WB = 0.24; XF = 0.6
ax = fig.add_subplot(gs[1, 0])
top = max([c['bars'][-1]['lcoe'] for c in CASES] + [max(h['lcoe_isl'].values()) for h in HYB.values()] + [c['bars'][0]['lcoe'] for c in CASES])
for i, c in enumerate(CASES):
    H = HYB.get(c['k']); off0, ong = c['bars'][0], c['bars'][-1]
    ax.bar(X[i] + POS3[3], ong['lcoe'], WB, color=GRID, lw=0, zorder=3)
    ax.text(X[i] + POS3[3], ong['lcoe'] + top*0.02, f'{ong["lcoe"]:.0f}', ha='center', fontsize=6.2, fontweight='bold', color=GRID)
    if H is None:
        continue
    for j, (vals, col, xc, wv) in ((0, (H['lcoe_isl'], SELF, 0.0, H['lcoe_isl'])), (2, (H['lcoe_con'], HYBC, XF, H['lcoe_con']))):
        if vals.get(xc) is None:
            continue
        xx = X[i] + POS3[j]
        ax.bar(xx, vals[xc], WB, color=col, lw=0, zorder=3)
        hi = vals[xc]
        if wv is not None:
            errbar(ax, xx, wv, w=0.05); hi = max([hi] + [v for v in wv.values() if v is not None])
        ax.text(xx, hi + top*0.02, f'{vals[xc]:.0f}', ha='center', fontsize=6.2, fontweight='bold', color=col)
ax.set_xticks(X); ax.set_xticklabels(XT, fontsize=6.6)
ax.set_xlim(-0.55, 1.75); ax.set_ylim(0, top*1.12)
ax.set_ylabel('Cost of electricity (\\$ per MWh)')
clean(ax)
ax.legend(handles=[Patch(facecolor=SELF, lw=0, label='Off-grid, x = 0'), Patch(facecolor=HYBC, lw=0, label='Hybrid, x = 0.6'),
                   Patch(facecolor=GRID, lw=0, label='On-grid'), Line2D([], [], color=INK, lw=0.8, marker='_', ms=5, label='Range over x')],
          loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=5.5,
          handlelength=1.0, handleheight=0.85, handletextpad=0.35, columnspacing=1.0, borderpad=0.0, labelspacing=0.3)
if LETTERS: panel(ax, 'b', x=-0.135, y=1.05)

# ---------------------------------------------------- c  what has to be financed
ax = fig.add_subplot(gs[1, 1])
tot = max([c['bars'][-1]['cap'] + c['bars'][-1]['fuel'] for c in CASES] + [max(h['cost_B'].values()) for h in HYB.values()])


def stack(ax, xx, cap, fuel, col, tot, vals=None, lab=True, span=False, cap_line=0.0):
    ax.bar(xx, cap - cap_line, WB, color=col, lw=0, zorder=3)
    if cap_line > 0:                      # transmission added by the data centers, split out of the capital
        ax.bar(xx, cap_line, WB, bottom=cap - cap_line, color=LINE, lw=0, zorder=3)
        ax.annotate(f'lines {cap_line:.1f}', xy=(xx + WB/2, cap - cap_line/2), xytext=(xx + WB/2 + 0.03, cap - cap_line/2),
                    fontsize=5.6, color=LINE, ha='left', va='center', fontweight='bold', zorder=6)
    ax.bar(xx, fuel, WB, bottom=cap, color=col, alpha=0.34, lw=0, zorder=3)
    hi = cap + fuel
    if vals is not None:
        errbar(ax, xx, vals, w=0.05); hi = max([hi] + [v for v in vals.values() if v is not None])
    ax.text(xx, hi + tot*0.022, f'{cap + fuel:.1f}', ha='center', fontsize=6.4, fontweight='bold', color=col)
    if cap > tot*0.10:
        ax.text(xx, (cap - cap_line)/2, f'{cap - cap_line:.1f}', ha='center', va='center', fontsize=5.8, color='white', fontweight='bold', zorder=6)
    else:
        ax.text(xx, cap + tot*0.012, f'{cap:.1f}', ha='center', va='bottom', fontsize=5.6, color=col, fontweight='bold', zorder=6)


for i, c in enumerate(CASES):
    H = HYB.get(c['k']); off0, ong = c['bars'][0], c['bars'][-1]
    stack(ax, X[i] + POS3[3], ong['cap'], ong['fuel'], GRID, tot, cap_line=ong.get('cap_line', 0.0))
    if H is None:
        continue
    stack(ax, X[i] + POS3[0], H['cap_B'][0.0], H['fuel_B'][0.0], SELF, tot, H['cost_B'])
    if H['con_B'].get(XF) is not None:
        stack(ax, X[i] + POS3[2], H['con_cap_B'][XF], H['con_B'][XF] - H['con_cap_B'][XF], HYBC, tot, H['con_B'])
ax.set_xticks(X); ax.set_xticklabels(XT, fontsize=6.6)
ax.set_xlim(-0.55, 1.75); ax.set_ylim(0, tot*1.16)
ax.set_ylabel('Annual cost (\\$ billion)')
clean(ax)
ax.legend(handles=[Patch(facecolor=INK2, lw=0, label='Generation capital and land'),
                   Patch(facecolor=LINE, lw=0, label='Transmission capital'),
                   Patch(facecolor=INK2, alpha=0.34, lw=0, label='Fuel and O&M'),
                   Line2D([], [], color=INK, lw=0.8, marker='_', ms=5, label='Range over x')],
          loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=5.5,
          handlelength=1.0, handleheight=0.85, handletextpad=0.35, borderpad=0.0, labelspacing=0.3)
if LETTERS: panel(ax, 'c', x=-0.135, y=1.05)

fig.savefig(f'{OUT}/fig4_scale.pdf', bbox_inches='tight')
fig.savefig(f'{OUT}/fig4_scale.png', dpi=400, bbox_inches='tight')
print(f'\nfig4 {W2/MM:.0f} x {figh/MM:.0f} mm')

json.dump([dict(data_centers_GW=c['gw'], coincident_peak_GW=c['peak'],
                sum_of_site_peaks_GW=c['sum_peak'],
                new_transmission_GW=c['line_gw'],
                architectures=[dict(name=b['lab'].replace('\n', ' '),
                                    lcoe_USD_MWh=b['lcoe'],
                                    capital_Byr=b['cap'], fuel_Byr=b['fuel'],
                                    built_GW=b['mw'],
                                    built_per_MW_of_peak=b['ratio'],
                                    technology_GW=b['tech']) for b in [c['bars'][0], c['bars'][-1]]])
           for c in CASES] + [dict(hybrid_diesel_first={f'{c["gw"]:.0f}GW': HYB.get(c['k']) for c in CASES})],
          open(f'{OUT}/fig4_numbers.json', 'w'), indent=1, default=float)
print(f'written {OUT}/fig4_numbers.json')
