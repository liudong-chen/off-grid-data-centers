"""SI table and figure for the speed-to-power value sensitivity.
Inputs: speed_value_sensitivity.json (run_speed_value_sensitivity.py) and renewable_offgrid_value.json
(renewable_offgrid_value.py).  Outputs: ../SI_new/tables/tab_speed_value.tex, ../SI_new/FigS/figS_speed_value.pdf"""
import json, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); SI = os.path.join(os.path.dirname(H), 'SI_new')
D = json.load(open(f'{H}/speed_value_sensitivity.json'))
RN = json.load(open(f'{H}/renewable_offgrid_value.json'))
RATES = sorted(int(k) for k in D)
XS = ['0.0', '0.3', '0.6', '1.0']
f1 = lambda v: '--' if v is None else f'{v:.1f}'
rows = []
for r in RATES:
    d = D[str(r)]; pv = d['ercot_pv_per_year_B']; on = pv['On-grid']
    og, hy = pv['Off-grid, x = 0.0'], pv['Hybrid, x = 0.6']
    q = d['ercot_at_queue']
    rows.append(' & '.join([f'{r:,}'] + [f1(d['ercot_bound_months'][x]) for x in XS] + [f1(d['georgia_bound_months'][x]) for x in XS]
                           + [f"{og['earnings_per_year_B']-on['earnings_per_year_B']:.1f} / {og['electricity_per_year_B']-on['electricity_per_year_B']:.1f}",
                              f"{hy['earnings_per_year_B']-on['earnings_per_year_B']:.1f} / {hy['electricity_per_year_B']-on['electricity_per_year_B']:.1f}",
                              f"{q['36']['0.6']['net']:.1f}", f"{q['72']['0.6']['net']:.1f}",
                              f"{RN[str(r)]['renewable_4h_hybrid']['bound_months']}"]) + r' \\')
tex = r"""\begin{sidewaystable}
\centering
\caption{Sensitivity of the comparison to the value of speed to power, \$200,000--600,000 per MW-month.}
\label{tab:speedvalue}
\fontsize{6.6pt}{7.8pt}\selectfont\setlength{\tabcolsep}{3.0pt}
\begin{tabular}{@{}lrrrrrrrrcccccr@{}}
\toprule
 & \multicolumn{4}{c}{ERCOT threshold (months), $x =$} & \multicolumn{4}{c}{Georgia threshold (months), $x =$} & \multicolumn{2}{c}{Extra earnings / extra electricity} & \multicolumn{2}{c}{Net value of building, $x = 0.6$} & Renewable hybrid \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}
Value (\$/MW-month) & 0 & 0.3 & 0.6 & 1.0 & 0 & 0.3 & 0.6 & 1.0 & Off-grid, $x = 0$ & Hybrid, $x = 0.6$ & 3-yr queue & 6-yr queue & bound (months) \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}\footnotesize
\item The bound is the month of grid arrival below which waiting for the grid gives the larger net present value (earnings minus electricity cost, 8\%, 20 years after the gas units arrive). Extra earnings and extra electricity cost are present values per year of the horizon relative to the grid at ERCOT's 50-month queue (\$ billion per year); the net value of building is the same quantity at a 36- and 72-month queue. The renewable-hybrid column is the queue above which solar, wind and 4-hour batteries, built in 24 months and connected to the grid when the queue clears (HOPE re-solve), beat waiting (Supplementary Note~8). Designs and costs of electricity do not depend on the value, so every row uses the same solved designs; the \$400,000 row reproduces Fig.~3 and Fig.~5d. Code: \texttt{speed\_value\_sensitivity/run\_speed\_value\_sensitivity.py}.
\end{tablenotes}
\end{sidewaystable}
"""
open(f'{SI}/tables/tab_speed_value.tex', 'w').write(tex)

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 7,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.5})
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw=dict(wspace=0.38, left=0.07, right=0.99, bottom=0.2, top=0.9))
R = np.array(RATES)/1e3
cm = matplotlib.colormaps['copper_r']
for k, (key, title) in enumerate((('ercot_bound_months', 'ERCOT 2030'), ('georgia_bound_months', 'Georgia Power'))):
    a = ax[k]
    a.axhspan(36, 72, color='0.9', lw=0); a.text(205, 70, "3–6-year queue", va='top', fontsize=6, color='0.4')
    for x in XS:
        a.plot(R, [D[str(r)][key][x] for r in RATES], '-o', ms=2.5, lw=1, color=cm(0.15 + 0.75*float(x)), label=f'x = {x}')
    if k == 0:
        a.plot(R, [RN[str(r)]['renewable_4h_hybrid']['bound_months'] for r in RATES], '--s', ms=2.5, lw=1, color='#3a8d4f', label='Renewable hybrid')
    a.set_ylim(0, 110 if k == 0 else 72); a.set_xlabel('Value of speed to power ($000 per MW-month)')
    a.set_ylabel('Bound on the wait (months)'); a.set_title(title, fontsize=7)
    a.set_xticks(R)
ax[0].legend(fontsize=5.8, frameon=False, loc='upper right')
a = ax[2]
on = [D[str(r)]['ercot_pv_per_year_B']['On-grid'] for r in RATES]
ee = [D[str(r)]['ercot_pv_per_year_B']['Hybrid, x = 0.6']['earnings_per_year_B'] - on[i]['earnings_per_year_B'] for i, r in enumerate(RATES)]
ce = [D[str(r)]['ercot_pv_per_year_B']['Hybrid, x = 0.6']['electricity_per_year_B'] - on[i]['electricity_per_year_B'] for i, r in enumerate(RATES)]
w = 22
a.bar(R - w/2, ee, w, color='#96331a', label='Extra earnings')
a.bar(R + w/2, ce, w, color='#3d4f5c', label='Extra electricity cost')
a.set_xticks(R); a.set_xlabel('Value ($000 per MW-month)')
a.set_ylabel('Hybrid x = 0.6 above on-grid\n($ billion per year)'); a.set_title('ERCOT 2030, 50-month queue', fontsize=7)
a.legend(fontsize=5.8, frameon=False, loc='upper left')
for i, l in enumerate('abc'):
    ax[i].text(-0.22, 1.04, l, transform=ax[i].transAxes, fontweight='bold', fontsize=8)
fig.savefig(f'{SI}/FigS/figS_speed_value.pdf'); fig.savefig(f'{H}/figS_speed_value.png', dpi=200)
print(tex[:200]); print('\n'.join(rows))
