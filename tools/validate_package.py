"""Check package integrity and syntax; optionally refresh MANIFEST.csv after edits.

Uses the Python standard library. This does not validate scientific conclusions.
"""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.venv', '.git', '__pycache__', 'workspace', 'demo_output', '.pytest_cache', 'dist'}
PRIVATE_FILES = set()


def files():
    for p in sorted(ROOT.rglob('*')):
        rel = p.relative_to(ROOT)
        if (p.is_file() and not any(x in EXCLUDED for x in rel.parts)
                and rel.as_posix() not in PRIVATE_FILES
                and p.suffix not in ('.pyc', '.pyo') and p.name != '.DS_Store'
                and not p.name.startswith('.env')):
            yield p


def record(p):
    rel = p.relative_to(ROOT)
    parts = rel.parts
    category = parts[0] if parts[0][0].isdigit() or parts[0] == 'tools' else ''
    depth = 1
    if category == '4_results' and len(parts) > 2 and parts[1] in ('main_text_figures', 'SI_figures'):
        depth = 2                                   # 4_results/<part>/<original path>
    return dict(category=category, path=rel.as_posix(),
                original_project_path='/'.join(parts[depth:]) if category else rel.as_posix(),
                bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest', action='store_true')
    parser.add_argument('--deep', action='store_true', help='parse JSON/CSV and test ZIP/XLSX/NPZ CRCs')
    args = parser.parse_args()
    records = [record(p) for p in files() if p.name != 'MANIFEST.csv']
    manifest = ROOT / 'MANIFEST.csv'
    if args.write_manifest:
        with manifest.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(f'Wrote {len(records)} manifest entries.')
    expected = {r['path']: r for r in csv.DictReader(manifest.open())}
    actual = {r['path']: r for r in records}
    errors = []
    for path in sorted(expected.keys() | actual.keys()):
        if path not in expected:
            errors.append(f'Unlisted file: {path}')
        elif path not in actual:
            errors.append(f'Missing file: {path}')
        elif (expected[path]['sha256'] != actual[path]['sha256']
              or int(expected[path]['bytes']) != actual[path]['bytes']):
            errors.append(f'Changed file: {path}')
    counts = {}
    for p in files():
        if p == manifest:
            continue
        counts[p.suffix] = counts.get(p.suffix, 0) + 1
        try:
            if p.suffix == '.py':
                compile(ast.parse(p.read_bytes(), filename=str(p)), str(p), 'exec')
            elif args.deep and p.suffix == '.json':
                json.loads(p.read_bytes())
            elif args.deep and p.suffix == '.csv':
                # Sources include metadata rows and different encodings; do not
                # assume rectangularity or convert scientific values.
                with p.open(encoding='utf-8-sig', errors='strict', newline='') as f:
                    for _ in csv.reader(f):
                        pass
            elif args.deep and p.suffix in ('.zip', '.xlsx', '.npz', '.docx'):
                with zipfile.ZipFile(p) as archive:
                    bad = archive.testzip()
                    if bad:
                        raise ValueError(f'CRC failure: {bad}')
        except Exception as exc:
            errors.append(f'{p.relative_to(ROOT)}: {type(exc).__name__}: {exc}')
    print(json.dumps(dict(files=len(records), bytes=sum(r['bytes'] for r in records),
                          extensions=counts, errors=errors), indent=2))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
