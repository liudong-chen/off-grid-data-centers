"""Figure 1 -- the 17 data centers: where they are, how their load behaves, when each one
peaks, and what that is worth.

  a  the sites at their real coordinates, on the eight ERCOT weather zones
  b  one week, each workload family as a band spanning its sites, with the combined total
  c  the day and hour at which each site reaches its own annual peak
  d  the peak that has to be served, at three levels of aggregation

LAYOUT.  A plain 2 x 2 grid: both rows share the same two column widths, so the panels
line up down the page instead of alternating short-long-long-short. The left column is
sized from the map's own aspect ratio, which fixes the figure height and leaves no dead
space anywhere.

DATA (full record in fig1_manifest.md)
  1_source_data/DC_Load/load_profiles/dc_load_hourly_MW.csv            8,760 x 17, MW, weather year 2025
  1_source_data/DC_Load/load_profiles/site_parameters_and_results.csv  lat, lon, city, train_share
  out/tx_zones.json                                      the eight weather-zone polygons
  out/ongrid_build.json                                  system peak increment, 30,793 MW (122,729 - 91,936)

STYLE  Nature Energy: 183 mm double column; bold lower-case panel letters flush left;
left and bottom spines only, 0.5 pt, offset from the data; 6.4 pt tick labels against
7.2 pt axis labels; no gridlines; no legend boxes -- every series is named at its own
right-hand end. Colour is ordered by training share, so hue carries a quantity.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import shape

import os as _os
_PKG = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _os.path.isdir(f'{_PKG}/01_raw'):
    OUT, RAW = f'{_PKG}/04_figure', f'{_PKG}/01_raw'
else:
    OUT, RAW = 'build', '1_source_data/DC_Load/load_profiles'

MM = 1/25.4
W2 = 183*MM

TRAIN, MIXED, COLO = '#96331a', '#e8663a', '#f6b79b'
BLUE = '#2a78d6'
INK, INK2, INK3 = '#161616', '#63625f', '#9a9995'
LAND, EDGE, OUTS = '#f5f4f1', '#dcdad5', '#eae9e5'
RAMP = {0.8: TRAIN, 0.5: MIXED, 0.2: COLO}
GNAME = {0.8: 'Training-led', 0.5: 'Mixed', 0.2: 'Colocation'}

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

# ------------------------------------------------------------------------- data
L = pd.read_csv(f'{RAW}/dc_load_hourly_MW.csv', index_col=0, parse_dates=True)
L = L[~((L.index.month == 2) & (L.index.day == 29))].drop(columns=['TOTAL_MW'])
M = pd.read_csv(f'{RAW}/site_parameters_and_results.csv').set_index('site_id')
Z = {k: shape(v) for k, v in json.load(open(f'{RAW}/tx_zones.json')).items()}
SITES = list(L.columns)

TOT = L.sum(axis=1)
PK = L.max()
SUM_PK, COI_PK, SYS_INC = PK.sum(), TOT.max(), 30793.0   # ERCOT system peak 122,729 - 91,936 MW in TX8z_GTEP_dc_vre_case vs base_vre_case (was 30,378 from an older case)
ip = int(np.argmax(TOT.values))
ZORDER = ['North', 'NorthCentral', 'East', 'FarWest', 'West', 'SouthCentral',
          'Coast', 'South']

X0, X1, Y0, Y1 = -106.8, -93.3, 25.7, 36.7
ASPECT = (Y1 - Y0)/((X1 - X0)*np.cos(np.radians(31.0)))


def rings(g):
    for p in (g.geoms if g.geom_type == 'MultiPolygon' else [g]):
        x, y = p.exterior.xy
        yield np.asarray(x), np.asarray(y)


def clean(ax, keep=('left', 'bottom')):
    for sp in ('top', 'right', 'left', 'bottom'):
        ax.spines[sp].set_visible(sp in keep)
    for sp in keep:
        ax.spines[sp].set_position(('outward', 3))


def panel(ax, s, x=-0.15, y=1.03):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight='bold',
            va='bottom', ha='left', color=INK)


def area(mw):
    return 0.024*mw


# ---- a plain 2 x 2; the left column is sized from the map, which fixes the height
LEFT, RIGHT, TOP, BOT = 0.050, 0.985, 0.945, 0.090
WR, WSP, HSP, R2 = [0.80, 1.20], 0.30, 0.32, 0.70
plotw = W2*(RIGHT - LEFT)
col0 = plotw*WR[0]/(sum(WR) + WSP*np.mean(WR))
row1 = col0*ASPECT
figh = row1*(1 + R2 + HSP*(1 + R2)/2)/(TOP - BOT)

fig = plt.figure(figsize=(W2, figh))
gs = fig.add_gridspec(2, 2, width_ratios=WR, height_ratios=[1.0, R2],
                      wspace=WSP, hspace=HSP,
                      left=LEFT, right=RIGHT, top=TOP, bottom=BOT)

# ===================================================================== a  the map
ax = fig.add_subplot(gs[0, 0])
for z in ZORDER:
    for x, y in rings(Z[z]):
        ax.fill(x, y, color=LAND, lw=0.4, ec=EDGE, zorder=2)
for x, y in rings(Z['outside']):
    ax.fill(x, y, color=OUTS, lw=0.4, ec=EDGE, zorder=2)
for s in sorted(SITES, key=lambda k: -PK[k]):
    ax.scatter([M.loc[s, 'lon']], [M.loc[s, 'lat']], s=area(PK[s]),
               color=RAMP[M.loc[s, 'train_share']], alpha=0.92, lw=0.45,
               edgecolor='white', zorder=10)
for s, (dx, dy, ha) in {'S08': (0.50, 0.0, 'left'), 'S09': (0.45, 0.75, 'left'),
                        'S05': (0.55, 0.05, 'left'), 'S03': (0.0, -1.30, 'center'),
                        'S01': (-0.50, -0.60, 'right')}.items():
    ax.text(M.loc[s, 'lon'] + dx, M.loc[s, 'lat'] + dy,
            f"{M.loc[s,'city']} {PK[s]/1000:.1f}", fontsize=5.5, color=INK2,
            ha=ha, va='center', zorder=12)
ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1)
ax.set_aspect(1/np.cos(np.radians(31.0)))
ax.axis('off')

T = ax.transAxes
ax.text(0.0, 0.985, 'Peak, GW', fontsize=5.9, color=INK, ha='left', va='center',
        transform=T)
for m, lab, xx in ((5300, '5.3', 0.055), (2000, '2', 0.158), (200, '0.2', 0.222)):
    ax.scatter([xx], [0.885], s=area(m), facecolor='none', edgecolor=INK3, lw=0.5,
               transform=T)
    ax.text(xx, 0.800, lab, fontsize=5.5, color=INK2, ha='center', transform=T)
for i, t in enumerate((0.8, 0.5, 0.2)):
    ax.text(0.0, 0.215 - 0.068*i, GNAME[t], fontsize=6.1, color=RAMP[t],
            fontweight='bold', va='center', transform=T)
panel(ax, 'a', x=-0.20, y=1.03)

# ============================================================ b  one week of load
ax = fig.add_subplot(gs[0, 1])
lo = ip - 84
w = slice(lo, lo + 168)
h = np.arange(168)/24
N = 100*L[SITES].values[w]/PK[SITES].values
tot = 100*TOT.values[w]/COI_PK

MEAN = {}
for t in (0.2, 0.5, 0.8):                     # light families first, dark on top
    k = [n for n, q in enumerate(SITES) if M.loc[q, 'train_share'] == t]
    g = N[:, k]
    ax.fill_between(h, g.min(axis=1), g.max(axis=1), color=RAMP[t], alpha=0.16, lw=0,
                    zorder=2 + (0.8 - t))
    MEAN[t] = g.mean(axis=1)
for t in (0.2, 0.5, 0.8):
    ax.plot(h, MEAN[t], color=RAMP[t], lw=1.0, zorder=5, solid_capstyle='round')
ax.plot(h, tot, color=INK, lw=1.5, zorder=7, solid_capstyle='round')
ax.plot([(ip - lo)/24], [100], 'o', ms=3.8, color=INK, mec='white', mew=0.8, zorder=9)

ENDS = [(tot[-1], 'All 17 combined', INK, 'bold')]
ENDS += [(MEAN[t][-1], GNAME[t], RAMP[t], 'normal') for t in (0.8, 0.5, 0.2)]
ENDS.sort(key=lambda e: -e[0])
ys = [e[0] for e in ENDS]
for i in range(1, len(ys)):
    ys[i] = min(ys[i], ys[i - 1] - 7.0)
for (yv, lab, c, wt), yy in zip(ENDS, ys):
    ax.plot([7.03, 7.18], [yv, yy], color=c, lw=0.4, clip_on=False, zorder=6)
    ax.text(7.24, yy, lab, fontsize=6.1, color=c, va='center', fontweight=wt,
            clip_on=False)
ax.set_xlim(0, 7); ax.set_xticks(range(8))
ax.set_ylim(32, 103); ax.set_yticks([40, 60, 80, 100])
ax.set_xlabel('Days')
ax.set_ylabel('Load (% of annual peak)')
clean(ax)
panel(ax, 'b', x=-0.095)

# ======================================================= c  when each site peaks
ax = fig.add_subplot(gs[1, 0])
doy = {s: L.index[int(np.argmax(L[s].values))].dayofyear for s in SITES}
hod = {s: L.index[int(np.argmax(L[s].values))].hour for s in SITES}
for y in (6, 12, 18):
    ax.axhline(y, color='#efeeeb', lw=0.5, zorder=0)
for s in sorted(SITES, key=lambda k: -PK[k]):
    ax.scatter([doy[s]], [hod[s]], s=0.0085*PK[s] + 6,
               color=RAMP[M.loc[s, 'train_share']], alpha=0.9, lw=0.4,
               edgecolor='white', zorder=5)
ax.scatter([L.index[ip].dayofyear], [L.index[ip].hour], s=62, facecolor='none',
           edgecolor=INK, lw=1.1, zorder=8)
ax.annotate('All 17\ncombined', xy=(L.index[ip].dayofyear, L.index[ip].hour - 2.0),
            xytext=(268, 5.4), fontsize=6.0, color=INK, ha='center', va='center',
            fontweight='bold', linespacing=1.3,
            arrowprops=dict(arrowstyle='-', color=INK3, lw=0.45, shrinkA=1, shrinkB=3))
ax.set_xlim(72, 296)
ax.set_xticks([91, 152, 213, 274])
ax.set_xticklabels(['Apr', 'Jun', 'Aug', 'Oct'])
ax.set_ylim(0.5, 22.5); ax.set_yticks([3, 9, 15, 21])
ax.set_xlabel('Month of the annual peak')
ax.set_ylabel('Hour of the day')
clean(ax)
panel(ax, 'c', x=-0.20)

# ==================================================================== d  the peak
ax = fig.add_subplot(gs[1, 1])
v = [SUM_PK/1000, COI_PK/1000, SYS_INC/1000]
ax.bar(range(3), v, 0.46, color=[TRAIN, MIXED, BLUE], lw=0, zorder=3)
for i, q in enumerate(v):
    ax.text(i, q + 1.0, f'{q:.1f}', ha='center', fontsize=7.2, fontweight='bold',
            color=[TRAIN, MIXED, BLUE][i])
for i in (0, 1):
    ax.annotate('', xy=(i + 0.76, 43.0), xytext=(i + 0.24, 43.0),
                arrowprops=dict(arrowstyle='-|>', color=INK3, lw=0.6, mutation_scale=6))
    ax.text(i + 0.5, 43.9, f'−{v[i]-v[i+1]:.1f}', ha='center', fontsize=6.2,
            color=INK2)
ax.set_xticks(range(3))
# the same three names as Figs 2-5: each level is the peak one architecture has to serve
ax.set_xticklabels(['Off-grid', 'Off-grid,\npooling', 'On-grid'], fontsize=6.3)
ax.set_xlim(-0.62, 2.62)
ax.set_ylim(0, 48); ax.set_yticks([0, 10, 20, 30, 40])
ax.set_ylabel('Peak demand (GW)')
clean(ax)
panel(ax, 'd', x=-0.095)

fig.savefig(f'{OUT}/fig1_fleet.pdf', bbox_inches='tight')
fig.savefig(f'{OUT}/fig1_fleet.png', dpi=400, bbox_inches='tight')
print(f'figure {W2/MM:.0f} x {figh/MM:.0f} mm;  columns {col0/MM:.1f} mm and '
      f'{plotw*WR[1]/(sum(WR) + WSP*np.mean(WR))/MM:.1f} mm')
print(f'{SUM_PK:,.0f} -> {COI_PK:,.0f} -> {SYS_INC:,.0f} MW')
print(f'combined peak {L.index[ip]}')
print('written fig1_fleet')
