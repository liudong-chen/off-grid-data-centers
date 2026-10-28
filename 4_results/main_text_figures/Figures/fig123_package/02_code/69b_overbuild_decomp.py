"""The overbuild decomposition of 69_ongrid_overbuild.py, recomputed from the off-grid
files in out/ and the already-reconciled ongrid_final.json (the on-grid half of 69 needs
the HOPE power.csv, which the packages do not carry, and it does not change when the
off-grid designs do). Writes out/overbuild.json.  python3 69b_overbuild_decomp.py [out_dir]
"""
import json, sys
OUT = sys.argv[1] if len(sys.argv) > 1 else 'build/out'
ONG = json.load(open(f'{OUT}/ongrid_final.json'))
SITES = json.load(open(f'{OUT}/offgrid_v4_sites.json'))
SUMM = json.load(open(f'{OUT}/offgrid_v4_summary.json'))
POOLR = json.load(open(f'{OUT}/offgrid_v4_pool.json'))
UNIT_MW = {'Recip': 20.0, 'Aero35': 35.0, 'Aero48': 48.0, 'FrameCT': 419.0, 'CC': 627.0}
A = SUMM['separate']['installed_MW']
bestpool = min(POOLR.values(), key=lambda r: r['cost'])
B = bestpool['installed_MW']; C = ONG['new_MW']
sum_peaks = float(sum(r['peak_MW'] for r in SITES)); comb_peak = float(bestpool['peak_MW'])
ratio_pool = B/comb_peak
D = sum_peaks*ratio_pool
lump = sum(0.5*max(UNIT_MW[k] for k in r['mix']) for r in SITES)
outage = (A - D) - lump; demand = D - B; leaving = B - C; total = A - C
print('OVERBUILD DECOMPOSITION')
for lab, v in (('A  separately', A), ('D  separate peaks, pooled reserve', D), ('B  combined', B), ('C  on-grid', C)):
    print(f'   {lab:<42s} {v:11,.0f} MW')
for lab, v in (('smaller fleets need more spare capacity', outage + lump), ('   outage backup', outage),
               ('   whole-machine rounding', lump), ('peaks falling at different hours', demand),
               ('use of existing ERCOT plants', leaving)):
    print(f'   {lab:<42s} {v:11,.0f} MW  {100*v/total:5.1f}%')
assert abs(outage + lump + demand + leaving - total) < 1.0
json.dump(dict(sum_peaks=sum_peaks, comb_peak=comb_peak, A=A, D=D, B=B, C=C, ratio_pool=ratio_pool,
               lumpiness=lump, outage_pooling=outage, demand=demand, leaving_grid=leaving, total=total),
          open(f'{OUT}/overbuild.json', 'w'), indent=1)
print(f'written {OUT}/overbuild.json')
