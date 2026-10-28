"""Speed-to-power value sensitivity: $200k, 300k, 400k, 500k, 600k per MW-month.

Re-runs the unmodified figure code (fig123_package/02_code/73_fig123.py for ERCOT 2030 and
fig5_package/02_code/96_fig5_new.py for Georgia) on a temporary copy of each package with only the
rate changed, then collects the bound on the interconnection wait by diesel share and the present
values per year of the 20-year horizon.  Designs and electricity costs do not depend on the rate,
so nothing is re-solved.  python3 run_speed_value_sensitivity.py  ->  speed_value_sensitivity.json/.csv
"""
import json, os, re, shutil, subprocess, sys, tempfile
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WN = os.path.dirname(HERE)
RATES = [200_000, 300_000, 400_000, 500_000, 600_000]
QUEUE_M = [36, 50, 72]          # 3 years, ERCOT central 50 months, 6 years


def run(pkg, script, pattern, repl, rate, tmp):
    dst = os.path.join(tmp, pkg)
    shutil.copytree(os.path.join(WN, pkg), dst, ignore=shutil.ignore_patterns('_before*', '_old'))
    p = os.path.join(dst, '02_code', script)
    s = open(p).read()
    assert len(re.findall(pattern, s)) == 1, (pkg, pattern)
    s = re.sub(pattern, repl.format(rate=float(rate)), s)
    open(p, 'w').write(s)
    r = subprocess.run([sys.executable, p], cwd=os.path.dirname(p), capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-3000:]); raise SystemExit(pkg)
    return dst, r.stdout


rows, out = [], {}
for rate in RATES:
    with tempfile.TemporaryDirectory() as tmp:
        # fig5 code reads ../../fig123... ? keep both packages side by side as in Figures
        e_dir, e_txt = run('fig123_package', '73_fig123.py', r"RATE, R_ = P\['RATE'\], P\['R'\]",
                           "RATE, R_ = {rate}, P['R']", rate, tmp)
        g_dir, g_txt = run('fig5_package', '96_fig5_new.py', r"RATE, R, N_ELEC = 400_000\.0, 0\.08, 20",
                           "RATE, R, N_ELEC = {rate}, 0.08, 20", rate, tmp)
        num = json.load(open(f'{e_dir}/03_panel_data/fig123_numbers.json'))
        pv = pd.read_csv(f'{e_dir}/03_panel_data/fig3_panel_a_present_value.csv').set_index('approach')
        q = pd.read_csv(f'{e_dir}/03_panel_data/fig3_panel_b_value_vs_queue.csv').set_index('month_grid_arrives')
        gq = pd.read_csv(f'{g_dir}/03_panel_data/fig5_panel_d_time.csv')
        gnum = json.load(open(f'{g_dir}/04_figure/fig5_numbers.json'))
    eb = {k.split('= ')[1]: v for k, v in num['bound_grid_month'].items()}
    gb = re.search(r"Georgia bounds \(grid month\): (\{.*\})", g_txt).group(1)
    gb = {k.split('= ')[1]: v for k, v in eval(gb.replace('nan', 'None')).items()}
    rec = dict(rate=rate, ercot_bound_months=eb, georgia_bound_months=gb,
               ercot_pv_per_year_B=pv[['earnings_per_year_B', 'electricity_per_year_B', 'net_per_year_B']].to_dict('index'),
               ercot_at_queue={})
    for m in QUEUE_M:
        rec['ercot_at_queue'][m] = {x: dict(earn=float(q.loc[m, f'Build own, x = {x} earnings above on-grid']),
                                            cost=float(q.loc[m, f'Build own, x = {x} electricity cost above on-grid']),
                                            net=float(q.loc[m, f'Build own, x = {x} net']))
                                    for x in ('0.0', '0.3', '0.6', '1.0')}
    rec['georgia_time_columns'] = list(gq.columns)
    rec['georgia_at_queue'] = {}
    gq = gq.set_index(gq.columns[0])
    for m in QUEUE_M:
        if m in gq.index:
            rec['georgia_at_queue'][m] = {c: float(gq.loc[m, c]) for c in gq.columns}
    out[rate] = rec
    on = pv.loc['On-grid']
    for appr in pv.index:
        r_ = pv.loc[appr]
        rows.append(dict(rate=rate, approach=appr, earnings_B_yr=r_.earnings_per_year_B,
                         electricity_B_yr=r_.electricity_per_year_B, net_B_yr=r_.net_per_year_B,
                         extra_earnings_vs_grid=r_.earnings_per_year_B - on.earnings_per_year_B,
                         extra_electricity_vs_grid=r_.electricity_per_year_B - on.electricity_per_year_B))
    print(rate, 'ERCOT bounds', {k: round(v, 1) for k, v in eb.items()}, '| Georgia', {k: (round(v, 1) if v else v) for k, v in gb.items()})

json.dump(out, open(f'{HERE}/speed_value_sensitivity.json', 'w'), indent=1, default=float)
pd.DataFrame(rows).to_csv(f'{HERE}/speed_value_sensitivity_pv.csv', index=False)
b = pd.DataFrame({r: out[r]['ercot_bound_months'] for r in RATES}).T
b.index.name = 'rate'; b.to_csv(f'{HERE}/speed_value_sensitivity_bounds_ercot.csv')
g = pd.DataFrame({r: out[r]['georgia_bound_months'] for r in RATES}).T
g.index.name = 'rate'; g.to_csv(f'{HERE}/speed_value_sensitivity_bounds_georgia.csv')
