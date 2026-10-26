#!/usr/bin/env python3
"""Export/import four saved tokenizers without model weights; stdlib only."""

import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile


TOKENIZERS = ("dialect", "klue", "kobert", "mbert")
MANIFEST = "tokenizers_manifest.json"
BASE_FILES = frozenset({"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                        "added_tokens.json", "vocab.txt", "vocab.json", "merges.txt",
                        "spiece.model", "sentencepiece.bpe.model", "tokenizer.model",
                        "tokenizer_78b3253a26.model", "config.json"})
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive path: {name!r}")
    pure = PurePosixPath(name)
    if pure.is_absolute() or any(part in (".", "..") for part in name.split("/")):
        raise ValueError(f"Unsafe archive path: {name!r}")
    if any(not part for part in name.split("/")):
        raise ValueError(f"Unsafe archive path: {name!r}")
    return pure


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def tokenizer_references(config):
    mapping = config.get("auto_map", {})
    references = mapping.get("AutoTokenizer", []) if isinstance(mapping, dict) else []
    if isinstance(references, str):
        references = [references]
    return [value for value in references if value] if isinstance(references, (list, tuple)) else []


def validate_tokenizer_directory(directory, name):
    directory = Path(directory)
    config_path = directory / "tokenizer_config.json"
    if not config_path.is_file():
        raise ValueError(f"{name}: missing saved tokenizer_config.json: {directory}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    references = tokenizer_references(config)
    for reference in references:
        if "--" in reference:
            raise ValueError(f"{name}: auto_map references remote code {reference!r}. "
                             "Use the existing portable shared_inputs export with local custom code.")
        module, separator, _ = reference.rpartition(".")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", module):
            raise ValueError(f"{name}: unsupported custom tokenizer module: {reference!r}")
        if not (directory / (module + ".py")).is_file():
            raise ValueError(f"{name}: missing custom tokenizer code {module}.py")
    if name == "kobert" and not references and "kobert" in str(config.get("tokenizer_class", "")).lower():
        raise ValueError("kobert: saved custom tokenizer needs a local AutoTokenizer auto_map and "
                         "tokenization_kobert.py; use the existing portable shared_inputs export")
    if not any(path.is_file() for path in directory.iterdir()
               if path.name in BASE_FILES - {"config.json", "tokenizer_config.json",
                                             "special_tokens_map.json", "added_tokens.json"}):
        raise ValueError(f"{name}: no tokenizer vocabulary/model files found")
    return config


def source_directory(root, name):
    root = Path(root)
    for candidate in (root / name / "final_model", root / name):
        if candidate.is_dir():
            return candidate
    raise ValueError(f"Missing tokenizer {name}: expected {root / name / 'final_model'}")


def collect_code_files(directory, config):
    seeds = {path.name for path in directory.glob("tokenization_*.py")}
    seeds.update(reference.rpartition(".")[0] + ".py" for reference in tokenizer_references(config))
    included = set()
    pending = list(seeds)
    while pending:
        name = pending.pop()
        if name in included:
            continue
        safe_name(name)
        if "/" in name or not name.endswith(".py"):
            raise ValueError(f"Unsupported custom-code path: {name}")
        path = directory / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing/linked custom tokenizer code: {path}")
        included.add(name)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                if node.level != 1 or not node.module or "." in node.module:
                    raise ValueError(f"Unsupported relative import in {path}: {ast.unparse(node)}")
                pending.append(node.module + ".py")
    return included


def collect_files(source_root):
    files = {}
    source_dirs = {}
    for name in TOKENIZERS:
        directory = source_directory(source_root, name)
        if directory.is_symlink():
            raise ValueError(f"Linked source directory is not allowed: {directory}")
        config = validate_tokenizer_directory(directory, name)
        code_files = collect_code_files(directory, config)
        candidates = BASE_FILES | code_files
        for filename in sorted(candidates):
            path = directory / filename
            if path.exists():
                if not path.is_file() or path.is_symlink():
                    raise ValueError(f"Expected a regular tokenizer file: {path}")
                files[f"{name}/{filename}"] = path
        source_dirs[name] = str(directory.resolve())
    return files, source_dirs


def export_archive(source_root, output):
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError(f"Archive already exists: {output}")
    files, source_dirs = collect_files(source_root)
    records = {name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
               for name, path in files.items()}
    manifest = {"format_version": 1, "tokenizers": list(TOKENIZERS), "source_directories": source_dirs,
                "purpose": "exact saved tokenizers only; model weights excluded", "files": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    try:
        with tarfile.open(temporary, "w:gz", compresslevel=1) as archive:
            for name, path in files.items():
                archive.add(path, arcname=name, recursive=False)
            payload = json_bytes(manifest)
            item = tarfile.TarInfo(MANIFEST)
            item.size = len(payload)
            item.mode = 0o644
            archive.addfile(item, io.BytesIO(payload))
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = sha256_file(output)
    Path(str(output) + ".sha256").write_bytes(f"{digest}  {output.name}\n".encode("ascii"))
    return {"archive": str(output), "sha256": digest, "files": len(files),
            "uncompressed_bytes": sum(record["bytes"] for record in records.values()),
            "model_weights_included": False}


def verify_directory(root):
    root = Path(root)
    manifest_path = root / MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1 or manifest.get("tokenizers") != list(TOKENIZERS):
        raise ValueError("Unsupported tokenizer manifest")
    records = manifest.get("files")
    if not isinstance(records, dict) or not records:
        raise ValueError("Manifest has no files")
    for name, record in records.items():
        pure = safe_name(name)
        if len(pure.parts) != 2 or pure.parts[0] not in TOKENIZERS:
            raise ValueError(f"Unexpected tokenizer file: {name}")
        filename = pure.parts[1]
        if filename not in BASE_FILES and not filename.endswith(".py"):
            raise ValueError(f"File is outside the tokenizer whitelist: {name}")
        path = root.joinpath(*pure.parts)
        if not path.is_file() or path.is_symlink() or path.stat().st_size != record.get("bytes"):
            raise ValueError(f"Missing/changed tokenizer file: {name}")
        if sha256_file(path) != record.get("sha256"):
            raise ValueError(f"Tokenizer checksum mismatch: {name}")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != set(records) | {MANIFEST}:
        raise ValueError("Tokenizer directory contains unlisted files")
    for name in TOKENIZERS:
        validate_tokenizer_directory(root / name, name)
    return manifest


def import_archive(archive_path, output_dir):
    archive_path, output_dir = Path(archive_path), Path(output_dir).absolute()
    if output_dir.exists():
        raise FileExistsError(f"Import destination already exists: {output_dir}")
    checksum_path = Path(str(archive_path) + ".sha256")
    if checksum_path.is_file():
        expected = checksum_path.read_text(encoding="ascii").split()[0]
        if sha256_file(archive_path) != expected:
            raise ValueError("Archive SHA256 mismatch")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tokenizers-import-", dir=output_dir.parent) as temporary:
        prepared = Path(temporary) / "prepared"
        prepared.mkdir()
        seen = set()
        total_size = 0
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                pure = safe_name(member.name)
                if not member.isfile() or member.name in seen:
                    raise ValueError(f"Duplicate/nonregular archive member: {member.name}")
                seen.add(member.name)
                total_size += member.size
                if total_size > MAX_ARCHIVE_BYTES or len(seen) > 200:
                    raise ValueError("Tokenizer archive exceeds the transfer size/file limit")
                if member.name != MANIFEST and (len(pure.parts) != 2 or pure.parts[0] not in TOKENIZERS):
                    raise ValueError(f"Unexpected archive member: {member.name}")
                target = prepared.joinpath(*pure.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Cannot read archive member: {member.name}")
                with source, target.open("wb") as destination:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        destination.write(chunk)
        manifest = verify_directory(prepared)
        os.replace(prepared, output_dir)
    return {"imported": str(output_dir), "files_verified": len(manifest["files"]),
            "tokenizers": list(TOKENIZERS)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source-root", type=Path, help="Directory containing {dialect,klue,kobert,mbert}/final_model")
    group.add_argument("--import", dest="import_path", type=Path, help="Import the exported tar.gz locally")
    group.add_argument("--verify", type=Path, help="Verify an already imported tokenizer directory")
    parser.add_argument("--output", type=Path, help="Export tar.gz path")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "inputs/tokenizers")
    args = parser.parse_args(argv)
    try:
        if args.source_root:
            if not args.output:
                parser.error("--source-root requires --output")
            report = export_archive(args.source_root, args.output)
        elif args.import_path:
            report = import_archive(args.import_path, args.output_dir)
        else:
            manifest = verify_directory(args.verify)
            report = {"verified": str(args.verify), "files": len(manifest["files"])}
    except (OSError, ValueError, tarfile.TarError) as exc:
        parser.exit(1, f"INPUT ERROR: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

