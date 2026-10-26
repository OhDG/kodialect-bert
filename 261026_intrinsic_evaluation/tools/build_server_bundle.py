#!/usr/bin/env python3
"""Build a small code-only server bundle, excluding local environments/data."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = ['.gitignore', '.gitattributes', 'README.md', 'requirements.txt',
             'run_server.sh', 'evaluate.py', 'metrics.py', 'validation_report.json']


def build(output):
    output = Path(output)
    files = [ROOT / name for name in TOP_LEVEL]
    files.extend(sorted((ROOT / 'tools').glob('*.py')))
    files.extend(sorted((ROOT / 'tests').glob('*.py')))
    if any(not path.is_file() for path in files):
        raise ValueError('A required source file is missing')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            data = path.read_bytes()
            if path.suffix == '.sh':
                data = data.replace(b'\r\n', b'\n')
            bundle.writestr(ROOT.name + '/' + path.relative_to(ROOT).as_posix(), data)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    Path(str(output) + '.sha256').write_bytes(f'{digest}  {output.name}\n'.encode('ascii'))
    return {'archive': str(output.resolve()), 'bytes': output.stat().st_size,
            'source_files': len(files), 'sha256': digest,
            'includes_data_or_environment': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/intrinsic_server_bundle.zip')
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
