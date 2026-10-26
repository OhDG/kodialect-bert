#!/usr/bin/env python3
"""Prepare exact saved classification tokenizers for an offline CPU evaluation.

Only tokenizer files and already-existing local custom code are copied. Model
weights and source directories are never modified or loaded. Remote auto_map
routing is localized without changing tokenization settings or vocabulary.
"""

import argparse
import ast
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

try:
    from .pack_tokenizers import BASE_FILES, MANIFEST, TOKENIZERS, sha256_file, source_directory, verify_directory
except ImportError:
    from pack_tokenizers import BASE_FILES, MANIFEST, TOKENIZERS, sha256_file, source_directory, verify_directory


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT.parent / '261012_common_preprocessing_reexperiment/server/region_classification/outputs/mlm'


def normalized_references(config, name):
    mapping = config.get('auto_map', {})
    references = mapping.get('AutoTokenizer', []) if isinstance(mapping, dict) else []
    if isinstance(references, str):
        references = [references]
    if not isinstance(references, (list, tuple)):
        raise ValueError(f'{name}: invalid AutoTokenizer auto_map')
    references = [value.split('--', 1)[-1] if isinstance(value, str) else value for value in references]
    if name == 'kobert' and not references and config.get('tokenizer_class') == 'KoBertTokenizer':
        references = ['tokenization_kobert.KoBertTokenizer', None]
    for reference in references:
        if reference is None:
            continue
        if not isinstance(reference, str) or not re.fullmatch(r'[A-Za-z_]\w*\.[A-Za-z_]\w*', reference):
            raise ValueError(f'{name}: unsupported local tokenizer reference {reference!r}')
    return references


def default_code_roots(source_root, name):
    repository = ROOT.parent
    home_cache = Path.home() / '.cache/huggingface'
    roots = [repository / '261021_translation_4gpu/shared_inputs/models/mlm' / name / 'final_model',
             Path('/data/ohdg/translation_4090_run/models/mlm') / name / 'final_model']
    module_cache = Path(os.environ.get('HF_MODULES_CACHE', home_cache / 'modules'))
    hub_cache = Path(os.environ.get('HF_HUB_CACHE', home_cache / 'hub'))
    roots.extend([module_cache / 'transformers_modules/monologg/kobert',
                  hub_cache / 'models--monologg--kobert/snapshots',
                  home_cache / 'modules/transformers_modules/monologg/kobert',
                  home_cache / 'hub/models--monologg--kobert/snapshots'])
    return roots


def find_code(filename, directory, roots):
    direct = directory / filename
    if direct.is_file() and not direct.is_symlink():
        return direct
    # Prefer a caller-selected export or the already-verified portable export.
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        direct = root / filename
        if direct.is_file() and not direct.is_symlink():
            return direct
        candidates = [path for path in root.rglob(filename) if path.is_file() and not path.is_symlink()]
        by_hash = {}
        for path in candidates:
            by_hash.setdefault(sha256_file(path), path)
        if len(by_hash) > 1:
            raise ValueError(f'Multiple different cached versions of {filename} under {root}; '
                             'select the original code directory with --code-search-root')
        if by_hash:
            return next(iter(by_hash.values()))
    raise ValueError(f'Missing local custom tokenizer code {filename} for {directory}. '
                     'Keep the verified 261021_translation_4gpu/shared_inputs export accessible, '
                     'or specify the original cached code with --code-search-root. No code will be downloaded.')


def prepare_inputs(source_root=DEFAULT_SOURCE, output_dir=ROOT / 'inputs/tokenizers', code_search_roots=()):
    source_root, output_dir = Path(source_root).resolve(), Path(output_dir).absolute()
    if output_dir.exists():
        manifest = verify_directory(output_dir)
        if manifest.get('source_model_root') != str(source_root):
            raise ValueError(f'Existing inputs were prepared from a different source: {output_dir}')
        for record in manifest.get('original_sources', {}).values():
            path = Path(record['path'])
            if not path.is_file() or sha256_file(path) != record['sha256']:
                raise ValueError(f'Original tokenizer source changed or disappeared: {path}')
        return {'status': 'already_verified', 'output_dir': str(output_dir), 'files': len(manifest['files'])}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    originals, localized = {}, {}
    with tempfile.TemporaryDirectory(prefix='.intrinsic-inputs-', dir=output_dir.parent) as temporary:
        staging = Path(temporary) / 'prepared'
        staging.mkdir()
        for name in TOKENIZERS:
            directory = source_directory(source_root, name)
            config_path = directory / 'tokenizer_config.json'
            config = json.loads(config_path.read_text(encoding='utf-8-sig'))
            references = normalized_references(config, name)
            destination = staging / name
            destination.mkdir()
            for filename in sorted(BASE_FILES):
                path = directory / filename
                if not path.exists():
                    continue
                if not path.is_file() or path.is_symlink():
                    raise ValueError(f'Expected a regular tokenizer file: {path}')
                digest = sha256_file(path)
                shutil.copyfile(path, destination / filename)
                if sha256_file(destination / filename) != digest:
                    raise ValueError(f'Tokenizer source changed while copying: {path}')
                originals[f'{name}/{filename}'] = {'path': str(path.resolve()), 'sha256': digest}
            roots = [Path(path) for path in code_search_roots] + default_code_roots(source_root, name)
            pending = [(reference.rpartition('.')[0] + '.py', directory)
                       for reference in references if reference]
            included = set()
            while pending:
                filename, parent = pending.pop()
                if filename in included:
                    continue
                if not re.fullmatch(r'[A-Za-z_]\w*\.py', filename):
                    raise ValueError(f'Unsupported tokenizer code filename: {filename}')
                path = find_code(filename, parent, roots)
                shutil.copyfile(path, destination / filename)
                digest = sha256_file(path)
                if sha256_file(destination / filename) != digest:
                    raise ValueError(f'Tokenizer code changed while copying: {path}')
                originals[f'{name}/{filename}'] = {'path': str(path.resolve()), 'sha256': digest}
                included.add(filename)
                tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.level:
                        if node.level != 1 or not node.module or '.' in node.module:
                            raise ValueError(f'Unsupported relative tokenizer import in {path}')
                        pending.append((node.module + '.py', path.parent))
            if references:
                rewritten = dict(config)
                rewritten['auto_map'] = {**config.get('auto_map', {}), 'AutoTokenizer': references}
                (destination / 'tokenizer_config.json').write_text(
                    json.dumps(rewritten, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                localized[name] = {'auto_map': references,
                                   'change': 'local custom-code routing only; vocabulary/settings unchanged'}
        records = {path.relative_to(staging).as_posix():
                   {'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
                   for path in staging.rglob('*') if path.is_file()}
        manifest = {'format_version': 1, 'tokenizers': list(TOKENIZERS), 'files': records,
                    'source_model_root': str(source_root), 'original_sources': originals,
                    'localized_custom_code': localized, 'weights_loaded': False,
                    'network_downloads': False, 'source_files_modified': False}
        (staging / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        verify_directory(staging)
        os.replace(staging, output_dir)
    return {'status': 'prepared', 'output_dir': str(output_dir), 'files': len(records),
            'custom_code_localized': list(localized), 'model_weights_included': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'inputs/tokenizers')
    parser.add_argument('--code-search-root', type=Path, action='append', default=[])
    args = parser.parse_args(argv)
    try:
        report = prepare_inputs(args.source_root, args.output_dir, args.code_search_root)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'SERVER INPUT ERROR: {exc}\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
