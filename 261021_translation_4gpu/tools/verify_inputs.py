#!/usr/bin/env python3
"""Verify/import portable translation inputs with Python's standard library.

Archives and exported directories contain component-specific manifests, so local
data and server models can be imported independently, including over NFS.
Checksums establish transfer integrity, not the
identity of an untrusted sender. No raw corpus JSON or network access is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tarfile
import tempfile

TOKENIZERS = ("dialect", "klue", "kobert", "mbert")
SPLITS = ("train", "validation", "test")
CHUNK = 4 * 1024 * 1024
MAX_METADATA = 64 * 1024 * 1024
MANIFESTS = {scope: f"provenance/inputs_{scope}_manifest.json" for scope in ("data", "models", "full")}
DATA_FILES = tuple(f"data/translation/translation_{s}.tsv" for s in SPLITS) + tuple(
    f"data/corpus/standard_{s}_corpus.txt" for s in ("train", "validation"))
DATA_PROVENANCE = {
    "provenance/data/split_manifest.csv": "region_classification/data/corpus_split_manifest_80_10_10.csv",
    "provenance/data/source_transfer_manifest.json": "transfer_manifest.json",
    "provenance/data/source_bundle_ready.json": "bundle_ready.json",
    "provenance/data/source_verification_report.json": "verification_report.json",
}


class InputError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InputError(message)


def safe_name(name):
    require(isinstance(name, str) and name and not any(c in name for c in "\\:*?\"<>|"),
            f"Unsafe archive path: {name!r}")
    parts = name.split("/")
    require(not name.startswith("/") and all(p not in ("", ".", "..") for p in parts),
            f"Unsafe archive path: {name!r}")
    require(all(not p.endswith((".", " ")) and not any(ord(c) < 32 for c in p)
                and not re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", p)
                for p in parts), f"Nonportable archive path: {name!r}")
    return name


def regular_file(path, base):
    """Reject links, junctions, and paths outside a declared input root."""
    path, base = Path(path).absolute(), Path(base).absolute()
    require(path.is_relative_to(base), f"Input escapes root: {path}")
    for node in (base, *[base.joinpath(*path.relative_to(base).parts[:i])
                          for i in range(1, len(path.relative_to(base).parts) + 1)]):
        require(not node.is_symlink() and not (hasattr(node, "is_junction") and node.is_junction()),
                f"Links/junctions are not allowed: {node}")
    require(path.is_file() and stat.S_ISREG(path.stat().st_mode), f"Missing required file: {path}")
    return path


def read_json_bytes(data, name):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key in {name}: {key}")
            result[key] = value
        return result
    try:
        result = json.loads(data.decode("utf-8-sig"), object_pairs_hook=unique)
    except (ValueError, UnicodeError) as exc:
        raise InputError(f"Invalid JSON in {name}: {exc}") from exc
    require(isinstance(result, dict), f"Expected JSON object: {name}")
    return result


def capture_metadata(name):
    return (name.endswith((".json", ".sha256")) or name == "provenance/data/split_manifest.csv") and not name.endswith("tokenizer.json")


def hash_stream(stream, destination=None, capture=False):
    digest, count, fragments = hashlib.sha256(), 0, []
    while True:
        block = stream.read(CHUNK)
        if not block:
            break
        digest.update(block)
        count += len(block)
        if destination is not None:
            destination.write(block)
        if capture:
            require(count <= MAX_METADATA, "Metadata exceeds the 64 MiB safety limit")
            fragments.append(block)
    return {"size_bytes": count, "sha256": digest.hexdigest()}, b"".join(fragments)


def validate_model_files(files, metadata):
    for tokenizer in TOKENIZERS:
        prefix = f"models/mlm/{tokenizer}/final_model/"
        local = {name[len(prefix):]: rec for name, rec in files.items() if name.startswith(prefix)}
        require(local, f"Missing MLM encoder {tokenizer}: export all four final_model trees on the old A6000 server with --models-only")
        for required in ("config.json", "tokenizer_config.json"):
            require(required in local and local[required]["size_bytes"] > 0,
                    f"Missing {tokenizer} model/tokenizer artifact: {prefix}{required}")
        config = read_json_bytes(metadata[prefix + "config.json"], prefix + "config.json")
        require(config.get("model_type") == "bert", f"Expected BERT MLM config: {prefix}config.json")
        weights = [name for name in ("model.safetensors", "pytorch_model.bin") if name in local]
        indexes = [name for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json") if name in local]
        require(weights or indexes, f"Missing model weights: {prefix}")
        for filename in weights:
            require(local[filename]["size_bytes"] > 0, f"Empty model weights: {prefix}{filename}")
        for filename in indexes:
            mapping = read_json_bytes(metadata[prefix + filename], prefix + filename).get("weight_map")
            require(isinstance(mapping, dict) and mapping, f"Invalid shard index: {prefix}{filename}")
            for shard in set(mapping.values()):
                safe_name(shard)
                require(shard in local and local[shard]["size_bytes"] > 0, f"Missing weight shard: {prefix}{shard}")
        artifacts = [name for name in local if "/" not in name and
                     (name in ("tokenizer.json", "vocab.txt", "spiece.model", "tokenizer.model") or name.endswith(".model"))]
        require(any(local[name]["size_bytes"] > 0 for name in artifacts),
                f"Missing saved tokenizer vocabulary/model for {tokenizer}; tokenizer_config.json alone is insufficient")
        tok = read_json_bytes(metadata[prefix + "tokenizer_config.json"], prefix + "tokenizer_config.json")
        auto_map = tok.get("auto_map", {})
        require(isinstance(auto_map, dict), f"Invalid tokenizer auto_map: {prefix}")
        for value in auto_map.values():
            values = value if isinstance(value, list) else [value]
            for entry in values:
                if entry is None:
                    continue
                require(isinstance(entry, str) and "." in entry, f"Invalid tokenizer auto_map entry: {entry!r}")
                module = entry.split("--")[-1].rsplit(".", 1)[0].replace(".", "/") + ".py"
                safe_name(module)
                require(module in local and local[module]["size_bytes"] > 0,
                        f"Saved tokenizer requires local code {prefix}{module}; export it with final_model for offline loading")
                require("--" not in entry,
                        f"Tokenizer auto_map still references a remote repository: {entry!r}; export it with local module references for offline loading")


def validate_data_files(files, metadata):
    for name in (*DATA_FILES, "data/preparation_metadata.json", *DATA_PROVENANCE):
        require(name in files and files[name]["size_bytes"] > 0, f"Missing required data/provenance file: {name}")
    prep = read_json_bytes(metadata["data/preparation_metadata.json"], "preparation_metadata.json")
    require(prep.get("completed") is True, "Data preparation did not complete")
    counts = prep.get("pair_counts", {})
    require(isinstance(counts, dict) and all(type(counts.get(s)) is int and counts[s] > 0 for s in SPLITS),
            "preparation_metadata.json must contain positive train/validation/test pair_counts")
    changed = prep.get("changed_counts", {})
    require(isinstance(changed, dict) and all(type(changed.get(s)) is int and 0 <= changed[s] <= counts[s] for s in SPLITS),
            "Invalid changed_counts in preparation metadata")
    require(prep.get("skipped_file_count", 0) == 0, "Preparation metadata reports skipped files")
    source = read_json_bytes(metadata["provenance/data/source_transfer_manifest.json"], "source_transfer_manifest.json")
    ready = read_json_bytes(metadata["provenance/data/source_bundle_ready.json"], "source_bundle_ready.json")
    require(ready.get("completed") is True, "Source bundle was not verified")
    require(ready.get("transfer_manifest_sha256") == files["provenance/data/source_transfer_manifest.json"]["sha256"],
            "Original transfer manifest differs from source ready marker")
    require(ready.get("translation_counts") == counts, "Source ready counts differ from preparation metadata")
    original_files = source.get("files", {})
    for name, record in files.items():
        if name.startswith("data/"):
            original = "translation/" + name
        elif name == "provenance/data/split_manifest.csv":
            original = DATA_PROVENANCE[name]
        else:
            continue
        require(original_files.get(original) == record,
                f"Input differs from the original verified transfer manifest: {original}")
    report = read_json_bytes(metadata["provenance/data/source_verification_report.json"], "source_verification_report.json")
    require(report.get("completed") is True, "Source verification report is incomplete")
    for split in SPLITS:
        observed = report.get("translation", {}).get(split, {})
        require(observed.get("pairs") == counts[split] and observed.get("changed_pairs") == changed[split],
                f"Source verified counts differ for {split}")
    text = metadata["provenance/data/split_manifest.csv"].decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    require({"source_id", "split", "num_parallel_pairs"}.issubset(reader.fieldnames or []),
            "Invalid split manifest header")
    totals, seen = {s: 0 for s in SPLITS}, set()
    for row in reader:
        source_id, split = row["source_id"], row["split"]
        require(source_id and source_id not in seen and split in SPLITS, "Duplicate source or invalid split in source manifest")
        seen.add(source_id)
        try:
            count = int(row["num_parallel_pairs"])
        except ValueError as exc:
            raise InputError("Invalid num_parallel_pairs in source manifest") from exc
        require(count >= 0, "Negative pair count in source manifest")
        totals[split] += count
    require(totals == counts, f"Split manifest pair totals differ from metadata: {totals} != {counts}")
    return counts


def validate_records(records, metadata, wanted=None):
    manifests = [name for name in MANIFESTS.values() if name in records]
    require(manifests, "Missing portable input manifest; import an archive made by pack_inputs.py")
    expected, components = {}, set()
    for name in manifests:
        sidecar = name + ".sha256"
        require(sidecar in metadata, f"Missing manifest checksum: {sidecar}")
        expected_checksum = f"{records[name]['sha256']}  {PurePosixPath(name).name}\n".encode("ascii")
        require(metadata[sidecar] == expected_checksum, f"Manifest checksum mismatch (expected LF): {sidecar}")
        manifest = read_json_bytes(metadata[name], name)
        require(manifest.get("format_version") == 1, f"Unsupported manifest version: {name}")
        scope = manifest.get("scope")
        require(scope in MANIFESTS and MANIFESTS[scope] == name, f"Manifest scope/path mismatch: {name}")
        components.update(("data", "models") if scope == "full" else (scope,))
        entries = manifest.get("files")
        require(isinstance(entries, dict) and entries, f"Empty/invalid manifest: {name}")
        for entry, record in entries.items():
            safe_name(entry)
            require(entry not in expected, f"Overlapping manifests: {entry}")
            require(isinstance(record, dict) and set(record) == {"size_bytes", "sha256"}
                    and type(record["size_bytes"]) is int and record["size_bytes"] > 0
                    and isinstance(record["sha256"], str) and re.fullmatch("[0-9a-f]{64}", record["sha256"]),
                    f"Invalid checksum record: {entry}")
            data_allowed = entry in (*DATA_FILES, "data/preparation_metadata.json",
                                     "data/corpus/standard_test_corpus.txt", *DATA_PROVENANCE)
            model_allowed = any(entry.startswith(f"models/mlm/{t}/final_model/") for t in TOKENIZERS)
            if model_allowed:
                parts = entry.split("/")[4:]
                model_allowed = not any(p.startswith("checkpoint-") or p in ("outputs", "results", "logs", "__pycache__") for p in parts)
                model_allowed = model_allowed and parts[-1] not in ("optimizer.pt", "scheduler.pt", "trainer_state.json") and not parts[-1].startswith("rng_state")
            allowed = ((scope in ("data", "full") and data_allowed) or
                       (scope in ("models", "full") and model_allowed))
            require(allowed, f"Unexpected input path: {entry}")
            expected[entry] = record
    manifest_paths = {p for name in manifests for p in (name, name + ".sha256")}
    require(set(records) == set(expected) | manifest_paths,
            f"Archive/input inventory mismatch: missing={sorted(set(expected) - set(records))[:6]}, extra={sorted(set(records) - set(expected) - manifest_paths)[:6]}")
    for name, record in expected.items():
        require(records.get(name) == record, f"SHA256/size mismatch: {name}")
    if wanted:
        require(set(wanted).issubset(components), f"Missing input component(s): {', '.join(sorted(set(wanted) - components))}; import the matching archive first")
    counts = validate_data_files(expected, metadata) if "data" in components else None
    if "models" in components:
        validate_model_files(expected, metadata)
    return {"completed": True, "components": sorted(components), "files_verified": len(expected),
            "bytes_verified": sum(r["size_bytes"] for r in expected.values()), "pair_counts": counts,
            "count_check": "preparation metadata + original verification report + split manifest; full TSV rows are not parsed"}


def inspect_archive(archive, staging=None, wanted=None):
    records, metadata, folded = {}, {}, set()
    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tar:
            name = safe_name(member.name)
            require(member.isfile() and not member.issparse(), f"Archive contains a non-regular file: {name}")
            require(name.casefold() not in folded, f"Duplicate/case-colliding archive path: {name}")
            folded.add(name.casefold())
            require(member.size > 0, f"Empty input file: {name}")
            capture = capture_metadata(name)
            require(not capture or member.size <= MAX_METADATA, f"Oversized metadata: {name}")
            stream = tar.extractfile(member)
            require(stream is not None, f"Cannot read archive member: {name}")
            if staging is None:
                record, data = hash_stream(stream, capture=capture)
            else:
                destination = staging.joinpath(*name.split("/"))
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as output:
                    record, data = hash_stream(stream, output, capture)
            require(record["size_bytes"] == member.size, f"Truncated member: {name}")
            records[name] = record
            if capture:
                metadata[name] = data
    return validate_records(records, metadata, wanted), records


def directory_files(root):
    """Inventory only portable namespaces, rejecting links and name collisions."""
    root = Path(root).absolute()
    require(root.is_dir(), f"Input root does not exist: {root}")
    require(not root.is_symlink() and not (hasattr(root, "is_junction") and root.is_junction()),
            f"Linked input root: {root}")
    files, folded = {}, set()
    # Only portable input namespaces are scanned; code, fresh results, and caches are independent.
    for folder in ("data", "models", "provenance"):
        base = root / folder
        if not base.exists() and not base.is_symlink():
            continue
        require(not base.is_symlink() and not (hasattr(base, "is_junction") and base.is_junction()), f"Linked input directory: {base}")
        require(base.is_dir(), f"Input namespace is not a directory: {base}")
        for directory, dirs, filenames in os.walk(base, followlinks=False):
            for sub in dirs:
                child = Path(directory) / sub
                require(not child.is_symlink() and not (hasattr(child, "is_junction") and child.is_junction()), f"Linked input directory: {child}")
            for filename in filenames:
                path = regular_file(Path(directory) / filename, root)
                name = safe_name(path.relative_to(root).as_posix())
                require(name.casefold() not in folded, f"Duplicate/case-colliding input path: {name}")
                folded.add(name.casefold())
                files[name] = path
    return files


def inspect_directory(root, staging=None, wanted=None, files=None):
    """Hash (and optionally copy) an export without altering its source files."""
    root = Path(root).absolute()
    files = directory_files(root) if files is None else files
    records, metadata = {}, {}
    for name, path in files.items():
        path = regular_file(path, root)
        capture = capture_metadata(name)
        with path.open("rb") as stream:
            if staging is None:
                record, data = hash_stream(stream, capture=capture)
            else:
                destination = staging.joinpath(*name.split("/"))
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as output:
                    record, data = hash_stream(stream, output, capture)
        records[name] = record
        if capture:
            metadata[name] = data
    return validate_records(records, metadata, wanted), records


def verify_root(root, wanted=("data", "models")):
    return inspect_directory(root, wanted=wanted)[0]


def prepare_import_root(root):
    root = Path(root).absolute()
    root.mkdir(parents=True, exist_ok=True)
    require(root.is_dir() and not root.is_symlink() and not (hasattr(root, "is_junction") and root.is_junction()), f"Unsafe import root: {root}")
    return root


def check_import_destinations(root, names):
    for name in names:
        destination = root.joinpath(*name.split("/"))
        require(not os.path.lexists(destination), f"Refusing to overwrite existing input: {destination}")
        for parent in destination.parents:
            if parent == root:
                break
            require(not parent.is_symlink() and not (hasattr(parent, "is_junction") and parent.is_junction()), f"Linked import parent: {parent}")
            require(not parent.exists() or parent.is_dir(), f"Import parent is not a directory: {parent}")


def commit_staged_inputs(staging, root, records):
    check_import_destinations(root, records)
    committed = []
    try:
        # Manifests arrive last, so an interrupted import cannot appear complete.
        ordered = sorted(records, key=lambda n: (n in MANIFESTS.values() or n.endswith("_manifest.json.sha256"), n))
        for name in ordered:
            source, destination = staging / name, root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Both paths are on the destination filesystem. Link only our staged
            # copy, never the NFS/source file; commit atomically refuses collisions.
            os.link(source, destination)
            committed.append(destination)
    except BaseException:
        for destination in reversed(committed):
            destination.unlink()
        raise


def extract_archive(archive, root, wanted=None):
    root = prepare_import_root(root)
    # Stage, verify everything, then move; never let tarfile extract paths directly.
    with tempfile.TemporaryDirectory(prefix=".input-import-", dir=root) as temporary:
        staging = Path(temporary)
        report, records = inspect_archive(archive, staging, wanted)
        commit_staged_inputs(staging, root, records)
    return report


def import_directory(source, root, wanted=None):
    """Copy a prepared portable export from NFS/local storage into a bundle.

    Scope flags require the named component, matching archive imports; every
    declared component in the source export is imported. Use separate exports
    for data and models when they are to be installed independently.
    """
    source, root = Path(source).absolute(), Path(root).absolute()
    require(source.is_dir(), f"Input root does not exist: {source}")
    source_real, root_real = source.resolve(), root.resolve()
    require(not source_real.is_relative_to(root_real) and not root_real.is_relative_to(source_real),
            "Source and destination must be separate, non-nested directories")
    files = directory_files(source)
    root = prepare_import_root(root)
    # Detect existing files before copying potentially several gigabytes over NFS.
    check_import_destinations(root, files)
    with tempfile.TemporaryDirectory(prefix=".input-import-", dir=root) as temporary:
        staging = Path(temporary)
        report, records = inspect_directory(source, staging, wanted, files)
        # Also fail if the source inventory changed during the copy.
        require(set(directory_files(source)) == set(files), "Source input inventory changed during import")
        commit_staged_inputs(staging, root, records)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--archive", type=Path, help="Verify an archive without extracting")
    mode.add_argument("--extract", type=Path, help="Verify and import an archive; refuses existing files")
    mode.add_argument("--import-directory", type=Path,
                      help="Verify and copy a portable export from NFS/local storage; source stays unchanged")
    parser.add_argument("--root", type=Path, help="Bundle root to verify/import; required without --archive")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--data-only", action="store_true")
    scope.add_argument("--models-only", action="store_true")
    args = parser.parse_args(argv)
    wanted = ("data",) if args.data_only else ("models",) if args.models_only else None
    try:
        if args.archive:
            report, _ = inspect_archive(args.archive, wanted=wanted)
        elif args.extract:
            require(args.root is not None, "--extract requires --root")
            report = extract_archive(args.extract, args.root, wanted)
        elif args.import_directory:
            require(args.root is not None, "--import-directory requires --root")
            report = import_directory(args.import_directory, args.root, wanted)
        else:
            require(args.root is not None, "Supply --archive, or --root to verify installed inputs")
            report = verify_root(args.root, wanted or ("data", "models"))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (InputError, OSError, tarfile.TarError, EOFError) as exc:
        print(f"INPUT ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
