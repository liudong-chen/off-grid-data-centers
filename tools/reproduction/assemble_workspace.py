"""Assemble model inputs, source code and results into a project directory.
    python3 tools/reproduction/assemble_workspace.py <target_dir> [--force]
Every file keeps the path it had in the project, so all scripts run from <target_dir> as documented in README.md."""
import os, shutil, sys
SUBGROUPED = {'4_results': ('main_text_figures', 'SI_figures')}     # folders with one more level
PKG = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(PKG, '1_source_data')) and os.path.dirname(PKG) != PKG:   # package root
    PKG = os.path.dirname(PKG)
args = [a for a in sys.argv[1:] if a != '--force']
dst = os.path.abspath(args[0]) if args else os.path.join(os.path.dirname(PKG), 'workspace')
if os.path.commonpath([os.path.realpath(dst), os.path.realpath(PKG)]) in (os.path.realpath(dst), os.path.realpath(PKG)):
    sys.exit('the working tree must be outside the package: ' + dst)
if os.path.isdir(dst) and os.listdir(dst) and '--force' not in sys.argv:
    sys.exit(dst + ' exists and is not empty; choose another folder or pass --force to overwrite')
n = 0
roots = []
for cat in sorted(os.listdir(PKG)):
    src = os.path.join(PKG, cat)
    if not os.path.isdir(src) or cat == os.path.basename(dst) or not (cat[0].isdigit() or cat == 'tools'):
        continue
    if cat in SUBGROUPED:                               # one more level: 4_results/<part>/<original path>
        roots += [os.path.join(src, x) for x in SUBGROUPED[cat] if os.path.isdir(os.path.join(src, x))]
    else:
        roots.append(src)
for src in roots:
    for r, ds, fs in os.walk(src):
        ds[:] = [d for d in ds if d not in ('__pycache__', '.git', '.venv')]
        for f in fs:
            if f.endswith(('.pyc', '.pyo')):
                continue
            if f in ('CATEGORY.md', 'README.md', 'SCENARIOS.md') and r == src:
                continue
            s = os.path.join(r, f); d = os.path.join(dst, os.path.relpath(s, src))
            os.makedirs(os.path.dirname(d), exist_ok=True); shutil.copy2(s, d); n += 1
print(f'{n} files assembled into {dst}')
