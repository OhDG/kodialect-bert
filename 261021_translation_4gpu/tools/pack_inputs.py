#!/usr/bin/env python3
"""Export only prepared translation inputs and four final MLM encoders.

Run --data-only locally and --models-only on the old A6000 server when the
inputs live on different machines. The default exports both. No training,
preprocessing, downloads, results, checkpoints, or external raw JSON are used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile

from verify_inputs import (DATA_FILES, DATA_PROVENANCE, InputError, MANIFESTS,
                           TOKENIZERS, capture_metadata, regular_file,
                           require, safe_name, validate_records)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def collect_inputs(root, scope):
    root = Path(root).absolute()
    require(root.is_dir(), f"Source server root does not exist: {root}")
    files = {}
    if scope in ("data", "full"):
        for name in (*DATA_FILES, "data/preparation_metadata.json"):
            files[name] = regular_file(root / "translation" / name, root)
        optional = "data/corpus/standard_test_corpus.txt"
        if (root / "translation" / optional).exists():
            files[optional] = regular_file(root / "translation" / optional, root)
        for name, original in DATA_PROVENANCE.items():
            files[name] = regular_file(root / original, root)
    if scope in ("models", "full"):
        for tokenizer in TOKENIZERS:
            source = root / "region_classification/outputs/mlm" / tokenizer / "final_model"
            require(source.is_dir(), f"Missing MLM encoder {tokenizer}: {source}. Export --models-only on the old A6000 server; local source data does not contain trained weights.")
            for directory, dirs, filenames in os.walk(source, followlinks=False):
                for sub in dirs:
                    node = Path(directory) / sub
                    require(not node.is_symlink() and not (hasattr(node, "is_junction") and node.is_junction()), f"Linked model directory: {node}")
                    require(not sub.startswith("checkpoint-") and sub not in ("outputs", "results", "logs", "__pycache__"), f"Unexpected training output under final_model: {node}")
                for filename in filenames:
                    path = regular_file(Path(directory) / filename, root)
                    require(filename not in ("optimizer.pt", "scheduler.pt", "trainer_state.json") and not filename.startswith("rng_state"),
                            f"Training checkpoint state is not a portable input: {path}")
                    name = f"models/mlm/{tokenizer}/final_model/" + path.relative_to(source).as_posix()
                    files[safe_name(name)] = path
    require(files, "No input files found")
    folded = set()
    for name, path in files.items():
        safe_name(name)
        require(name.casefold() not in folded, f"Case-colliding input path: {name}")
        folded.add(name.casefold())
        require(path.stat().st_size > 0, f"Empty input file: {path}")
    return files


class HashReader:
    def __init__(self, source, capture=False):
        self.source, self.digest, self.count, self.blocks = source, hashlib.sha256(), 0, []
        self.capture = capture

    def read(self, size=-1):
        block = self.source.read(size)
        self.digest.update(block)
        self.count += len(block)
        if self.capture:
            require(self.count <= 64 * 1024 * 1024, "Metadata exceeds 64 MiB")
            self.blocks.append(block)
        return block

    def record(self):
        return {"size_bytes": self.count, "sha256": self.digest.hexdigest()}


class HashWriter:
    """Hash compressed bytes during the write, avoiding a second archive read."""
    def __init__(self, output):
        self.output, self.digest = output, hashlib.sha256()

    def write(self, data):
        written = self.output.write(data)
        require(written == len(data), "Short archive write")
        self.digest.update(data)
        return written


def add_bytes(tar, name, data):
    member = tarfile.TarInfo(name)
    member.size, member.mode, member.mtime = len(data), 0o644, 0
    tar.addfile(member, io.BytesIO(data))
    return {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def pack_inputs(source_root, output, scope="full", compresslevel=1):
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink(), f"Refusing to overwrite existing output: {output}")
    checksum_path = Path(str(output) + ".sha256")
    require(not checksum_path.exists(), f"Refusing to overwrite existing checksum: {checksum_path}")
    files = collect_inputs(source_root, scope)
    # Validate small artifacts and model layouts before reading/compressing multi-GB data.
    initial_records = {name: {"size_bytes": path.stat().st_size, "sha256": "0" * 64} for name, path in files.items()}
    metadata = {name: path.read_bytes() for name, path in files.items() if capture_metadata(name)}
    if scope in ("models", "full"):
        from verify_inputs import validate_model_files
        validate_model_files(initial_records, metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    archive_created = checksum_created = False
    try:
        with output.open("xb") as raw:
            archive_created = True
            writer = HashWriter(raw)
            # Separate gzip wrapper supports streaming compression on Python 3.10 too.
            with gzip.GzipFile(fileobj=writer, mode="wb", compresslevel=compresslevel, mtime=0) as compressed, \
                    tarfile.open(fileobj=compressed, mode="w|") as tar:
                records = {}
                for name, path in sorted(files.items()):
                    before = path.stat()
                    member = tarfile.TarInfo(name)
                    member.size, member.mode, member.mtime = before.st_size, 0o644, 0
                    print(f"[PACK] {name} ({before.st_size:,} bytes)", flush=True)
                    with path.open("rb") as source:
                        reader = HashReader(source, capture_metadata(name))
                        tar.addfile(member, reader)
                    after = path.stat()
                    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), f"Input changed during export: {path}")
                    require(reader.count == before.st_size, f"Input truncated during export: {path}")
                    records[name] = reader.record()
                    if capture_metadata(name):
                        metadata[name] = b"".join(reader.blocks)
                manifest_name = MANIFESTS[scope]
                manifest = {"format_version": 1, "scope": scope,
                            "created_utc": datetime.now(timezone.utc).isoformat(),
                            "source_server_root": str(Path(source_root).absolute()),
                            "purpose": "fresh decoder CLM and 20 translations; reuse four existing MLM encoders",
                            "files": records.copy()}
                data = json_bytes(manifest)
                records[manifest_name] = add_bytes(tar, manifest_name, data)
                metadata[manifest_name] = data
                checksum_name = manifest_name + ".sha256"
                checksum = f"{records[manifest_name]['sha256']}  {Path(manifest_name).name}\n".encode("ascii")
                records[checksum_name] = add_bytes(tar, checksum_name, checksum)
                metadata[checksum_name] = checksum
                report = validate_records(records, metadata)
        digest = writer.digest
        with checksum_path.open("x", encoding="utf-8", newline="\n") as stream:
            checksum_created = True
            stream.write(f"{digest.hexdigest()}  {output.name}\n")
        report.update(archive=str(output), archive_bytes=output.stat().st_size, archive_sha256=digest.hexdigest(), checksum_file=str(checksum_path))
        return report
    except BaseException:
        if archive_created:
            output.unlink(missing_ok=True)
        if checksum_created:
            checksum_path.unlink(missing_ok=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-server-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--data-only", action="store_true")
    scope.add_argument("--models-only", action="store_true")
    parser.add_argument("--compresslevel", type=int, choices=range(0, 10), default=1,
                        help="gzip level (default 1: fast transfer preparation)")
    args = parser.parse_args(argv)
    try:
        report = pack_inputs(args.source_server_root, args.output,
                             "data" if args.data_only else "models" if args.models_only else "full", args.compresslevel)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (InputError, OSError, tarfile.TarError, EOFError) as exc:
        print(f"INPUT ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
