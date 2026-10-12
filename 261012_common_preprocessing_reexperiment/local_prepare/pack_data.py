"""Package locally verified data for transfer alongside a Git checkout."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent.parent


def main():
    server = ROOT / 'server'
    ready = json.loads((server / 'bundle_ready.json').read_text(encoding='utf-8'))
    if ready.get('completed') is not True:
        raise RuntimeError('Complete verify_bundle.py --full before packaging')
    manifest_path = server / 'transfer_manifest.json'
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != ready['transfer_manifest_sha256']:
        raise RuntimeError('Transfer manifest does not match readiness record')
    records = json.loads(manifest_path.read_text(encoding='utf-8'))['files']
    selected = sorted(name for name in records if name.startswith((
        'region_classification/data/', 'translation/data/')) or
        name == 'region_classification/results/changed_flags_cache.npz')
    selected += ['bundle_ready.json', 'transfer_manifest.json', 'verification_report.json']
    destination = ROOT / 'artifacts'
    destination.mkdir(exist_ok=True)
    archive = destination / 'prepared_data.tar.gz'
    temporary = destination / 'prepared_data.tar.gz.partial'
    if archive.exists() or temporary.exists():
        raise FileExistsError('Prepared-data archive already exists; do not overwrite')
    with tarfile.open(temporary, 'w:gz', compresslevel=1) as output:
        for name in selected:
            path = (server / name).resolve()
            if not path.is_relative_to(server.resolve()) or not path.is_file():
                raise ValueError('Invalid archive input: ' + name)
            before = path.stat()
            if name in records and before.st_size != records[name]['size_bytes']:
                raise RuntimeError('Input size changed: ' + name)
            print('[PACK] ' + name, flush=True)
            output.add(path, arcname=name, recursive=False)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError('Input changed while packing: ' + name)
    temporary.rename(archive)
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    checksum = archive.with_name(archive.name + '.sha256')
    checksum.write_text(digest.hexdigest() + '  ' + archive.name + '\n', encoding='ascii')
    print(f'[OK] {archive}: {archive.stat().st_size:,} bytes; {len(selected)} files', flush=True)


if __name__ == '__main__':
    main()
