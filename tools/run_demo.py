"""Rebuild main figures from supplied solved inputs in a new output directory.

Compare generated CSV/JSON values with packaged reference outputs. No solver,
API credentials, or changes to packaged scientific files are required.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / '4_results' / 'main_text_figures' / 'Figures'


def equal(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='new directory for figures, data, logs and report')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output == ROOT or ROOT in output.parents or output in ROOT.parents:
        parser.error('Choose a new directory outside the repository.')
    output.mkdir(parents=True)
    env = dict(os.environ, MPLBACKEND='Agg', MPLCONFIGDIR=str(output / '.mplconfig'))
    results = []
    started = time.monotonic()
    for package, scripts in [('fig123_package', ['73_fig123.py']),
                             ('fig4_package', ['79_fig4_new.py']),
                             ('fig5_package', ['96_fig5_new.py'])]:
        source = FIGURES / package
        target = output / package
        shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        references = {}
        for folder in ('03_panel_data', '04_figure'):
            for p in (source / folder).glob('*'):
                if p.suffix in ('.csv', '.json'):
                    references[p.relative_to(source)] = p
            # Remove copied outputs so a stale file cannot count as a successful regeneration.
            if (target / folder).exists():
                shutil.rmtree(target / folder)
            (target / folder).mkdir(parents=True)
        returncode = 0
        for script in scripts:
            run = subprocess.run([sys.executable, str(target / '02_code' / script)],
                                 cwd=target / '02_code', env=env, capture_output=True, text=True)
            (target / (script + '.log')).write_text(run.stdout + run.stderr)
            returncode = returncode or run.returncode
        checks = []
        for relative, reference in sorted(references.items()):
            generated = target / relative
            try:
                if reference.suffix == '.json':
                    matches = equal(json.loads(reference.read_text()), json.loads(generated.read_text()))
                else:
                    import pandas as pd
                    pd.testing.assert_frame_equal(pd.read_csv(reference), pd.read_csv(generated),
                                                  check_exact=False, rtol=1e-10, atol=1e-10)
                    matches = True
                checks.append(dict(path=str(relative), matches=matches))
            except Exception as exc:
                checks.append(dict(path=str(relative), matches=False, error=str(exc)))
        results.append(dict(package=package, returncode=returncode, checks=checks))
    report = dict(python=sys.version, seconds=round(time.monotonic() - started, 2), results=results)
    (output / 'demo_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(any(r['returncode'] or any(not c['matches'] for c in r['checks']) for r in results))


if __name__ == '__main__':
    main()
