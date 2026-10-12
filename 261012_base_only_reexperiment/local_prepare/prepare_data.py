"""Prepare the new base-only corpus locally; never train a tokenizer or model.

The original text cleaners and file splitter are loaded from their function ASTs
without importing their GPU/ML dependencies. No deduplication is performed.
"""

import argparse
import ast
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Dict, List, Tuple
import unicodedata


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
REGION_LABELS = ["강원도", "경상도", "전라도", "제주도", "충청도"]
LABEL2ID = {region: index for index, region in enumerate(REGION_LABELS)}
PREFIXES = {"강원도": "013.", "경상도": "014.", "전라도": "015.", "제주도": "016.", "충청도": "017."}
SPLITS = ("train", "validation", "test")
RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}
CLASS_HEADER = ["text", "label", "region", "file_id", "source_group"]
PAIR_HEADER = ["source_text", "target_text", "region", "file_id", "source_group", "is_changed"]
MANIFEST_FIELDS = [
    "region", "source_group", "source_type", "path", "num_sentences", "source_id",
    "origin", "official_split", "sha256",
]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def read_json(path):
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def function_node(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise ValueError(f"Original function not found: {name}")


def load_function_asts(path, names, namespace):
    """Load selected untouched function bodies, avoiding module-level ML imports."""
    source = path.read_text(encoding="utf-8-sig")
    nodes = [function_node(source, name) for name in names]
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return source


def load_original_helpers(server_root):
    records = read_json(server_root / "original_code_sha256.json")
    helper_files = [
        "region_classification/step1_prepare_data_80_10_10.py",
        "translation/experiment_common.py",
    ]
    for relative in helper_files:
        expected = records.get(relative, {}).get("sha256")
        if not expected or file_sha256(server_root / relative) != expected:
            raise RuntimeError(f"Original helper hash mismatch: {relative}")
    namespace = {
        "Path": Path, "Dict": Dict, "List": List, "Tuple": Tuple,
        "hashlib": hashlib, "random": random, "re": re, "json": json,
        "defaultdict": defaultdict, "unicodedata": unicodedata,
        "LABEL2ID": LABEL2ID, "REGION_LABELS": REGION_LABELS,
    }
    source = load_function_asts(
        server_root / helper_files[0],
        ["clean_base_text", "stable_group_seed", "assign_file_splits"], namespace,
    )
    load_function_asts(server_root / helper_files[1], ["normalize_text"], namespace)
    node = function_node(source, "clean_base_text")
    namespace["clean_base_text_source"] = ast.get_source_segment(source, node)
    namespace["helper_hashes"] = {name: records[name]["sha256"] for name in helper_files}
    return namespace


def json_files(directory):
    for current, directories, files in os.walk(directory):
        directories.sort()
        for name in sorted(files):
            if name.lower().endswith(".json"):
                yield (Path(current) / name).resolve()


def official_split(path):
    training = any(part in ("Training", "1.Training") for part in path.parts)
    validation = any(part in ("Validation", "2.Validation") for part in path.parts)
    if training == validation:
        raise ValueError(f"Cannot determine official Training/Validation origin: {path}")
    return "Training" if training else "Validation"


def select_sources(old_root, new_root, include_new_jeolla_validation=False):
    """Select JSON only; official split is provenance, not the experiment split."""
    old_root, new_root = Path(old_root).resolve(), Path(new_root).resolve()
    if not old_root.is_dir() or not new_root.is_dir():
        raise FileNotFoundError("Both old and new dataset roots must exist.")
    pools = defaultdict(list)
    for region in REGION_LABELS:
        if region == "전라도":
            old_directory = old_root / "JSON만_모은폴더_전라도"
            if not old_directory.is_dir():
                raise FileNotFoundError(old_directory)
            pools[region].extend((p, "old_project1_dataset", "legacy_flattened") for p in json_files(old_directory))
            if not include_new_jeolla_validation:
                continue
        candidates = sorted(p for p in new_root.iterdir() if p.is_dir() and p.name.startswith(PREFIXES[region]))
        if len(candidates) != 1:
            raise ValueError(f"Expected one {PREFIXES[region]} dataset directory, found {len(candidates)}.")
        for path in json_files(candidates[0]):
            provided_split = official_split(path)
            if region == "전라도" and provided_split != "Validation":
                continue
            pools[region].append((path, "new_download", provided_split))

    rows = []
    seen = set()
    for region in REGION_LABELS:
        if not pools[region]:
            raise ValueError(f"No selected JSON files for region: {region}")
        # Do not stratify by the downloaded Train/Validation folder: all base
        # files for one region retain the original single source_group.
        for path, origin, provided_split in sorted(pools[region], key=lambda item: (item[0].stem, str(item[0]))):
            source_id = f"{region}/{path.stem}"
            identity = source_id.casefold()
            if identity in seen:
                raise ValueError(f"Duplicate source ID (not silently deduplicated): {source_id}")
            seen.add(identity)
            rows.append({
                "region": region,
                "source_group": f"JSON만_모은폴더_{region}",
                "source_type": "base",
                "path": str(path),
                "num_sentences": "",
                "source_id": source_id,
                "origin": origin,
                "official_split": provided_split,
                "sha256": "",
            })
    return rows


def path_key(path):
    return os.path.normcase(str(Path(path).resolve()))


def load_audit_counts(audit_path, helpers):
    if audit_path is None:
        return {}, {"requested": False, "compatible": False}
    audit_path = Path(audit_path).resolve()
    metadata_path = audit_path.parent / "comparison_summary.json"
    details = {"requested": True, "path": str(audit_path), "compatible": False}
    if not audit_path.is_file() or not metadata_path.is_file():
        details["reason"] = "Audit JSONL or companion comparison_summary.json is missing."
        return {}, details
    metadata = read_json(metadata_path)
    recorded_source = metadata.get("normalizer_function", "")
    try:
        matches = ast.dump(ast.parse(recorded_source), include_attributes=False) == ast.dump(
            ast.parse(helpers["clean_base_text_source"]), include_attributes=False
        )
    except (SyntaxError, TypeError):
        matches = False
    if not matches:
        details["reason"] = "Audit normalizer is not the unchanged experiment clean_base_text function."
        return {}, details
    counts = {}
    with audit_path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            for key in ("old", "new_json"):
                record = entry.get(key)
                if not isinstance(record, dict):
                    continue
                path = record.get("path")
                count = record.get("counts", {}).get("cleaned_rows")
                digest = record.get("sha256")
                if not isinstance(path, str) or not isinstance(count, int) or count < 0:
                    continue
                if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                    continue
                normalized = path_key(path)
                if normalized in counts and counts[normalized] != (count, digest):
                    raise ValueError(f"Conflicting audit count for {path}, line {line_number}")
                counts[normalized] = (count, digest)
    details.update({"compatible": True, "usable_path_count": len(counts), "sha256": file_sha256(audit_path)})
    return counts, details


def ordered_parallel_map(function, items, workers):
    """Bound queued decoded JSONs while retaining the exact manifest row order."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = deque()
        for _ in range(workers * 2):
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                pass


def parse_source(path):
    raw = Path(path).read_bytes()
    data = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(data, dict) or not isinstance(data.get("utterance", []), list):
        raise ValueError(f"Expected an utterance list: {path}")
    return data, hashlib.sha256(raw).hexdigest()


def count_source(row, cleaner):
    data, digest = parse_source(row["path"])
    count = 0
    for utterance in data.get("utterance", []):
        text = utterance.get("dialect_form", "")
        if isinstance(text, str) and text.strip() and cleaner(text):
            count += 1
    return row["source_id"], count, digest


def attach_counts(rows, helpers, audit_counts, workers):
    missing = []
    reused = 0
    for row in rows:
        cached = audit_counts.get(path_key(row["path"]))
        if cached is None:
            missing.append(row)
        else:
            row["num_sentences"], row["sha256"] = str(cached[0]), cached[1]
            reused += 1
    row_by_id = {row["source_id"]: row for row in missing}
    print(f"[COUNT] Reused audited counts: {reused:,}; fresh JSON count pass: {len(missing):,}", flush=True)
    for index, (source_id, count, digest) in enumerate(
        ordered_parallel_map(lambda row: count_source(row, helpers["clean_base_text"]), missing, workers), 1
    ):
        row_by_id[source_id]["num_sentences"] = str(count)
        row_by_id[source_id]["sha256"] = digest
        if index % 500 == 0 or index == len(missing):
            print(f"[COUNT] {index:,}/{len(missing):,} JSON files", flush=True)
    return {"audit_count_files": reused, "fresh_count_files": len(missing)}


def process_source(row, helpers):
    data, digest = parse_source(row["path"])
    if digest != row["sha256"]:
        raise RuntimeError(f"Source JSON changed since count/audit: {row['path']}")
    cleaner, normalizer = helpers["clean_base_text"], helpers["normalize_text"]
    classifier_rows, pairs = [], []
    for utterance in data.get("utterance", []):
        dialect = utterance.get("dialect_form", "")
        standard = utterance.get("standard_form", "")
        if isinstance(dialect, str) and dialect.strip():
            cleaned = cleaner(dialect)
            if cleaned:
                # Mirrors original step6, including strip() for known-standard.
                known = isinstance(standard, str) and bool(standard.strip())
                changed_clean = known and cleaned != cleaner(standard)
                changed_raw = known and normalizer(dialect) != normalizer(standard)
                classifier_rows.append((cleaned, bool(known), bool(changed_clean), bool(changed_raw)))
        # Translation filtering is independent: do not discard pairs merely
        # because their source was removed by the classifier character whitelist.
        if isinstance(dialect, str) and isinstance(standard, str):
            source, target = normalizer(dialect), normalizer(standard)
            if source and target:
                pairs.append((source, target))
    if len(classifier_rows) != int(row["num_sentences"]):
        raise RuntimeError(f"Cleaned count differs from count/audit for {row['path']}")
    return row, classifier_rows, pairs, len(data.get("utterance", []))


def save_manifest(path, rows, include_split=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS + (["split"] if include_split else []))
        writer.writeheader()
        writer.writerows(rows)


def publish_path(source, destination, package_root):
    """Move one completed artifact only within the checked project directory."""
    root = Path(package_root).resolve()
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_relative_to(root) or not destination.is_relative_to(root):
        raise ValueError(f"Publication path escapes the project: {source} -> {destination}")
    if source == root or destination == root or not source.exists():
        raise ValueError(f"Invalid publication path: {source} -> {destination}")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite publication target: {destination}")
    source.rename(destination)


def write_data(assigned, helpers, staging_root, final_server_root, workers):
    import numpy as np

    region_data = staging_root / "region_data"
    translation_data = staging_root / "translation_data"
    cache_dir = staging_root / "region_results"
    for path in (
        region_data / "corpus", region_data / "region_classification",
        translation_data / "corpus", translation_data / "translation", cache_dir,
    ):
        path.mkdir(parents=True, exist_ok=False)
    save_manifest(region_data / "corpus_split_manifest_80_10_10.csv", assigned, include_split=True)
    class_counts = {split: 0 for split in SPLITS}
    class_files = {split: 0 for split in SPLITS}
    pair_counts = {split: 0 for split in SPLITS}
    pair_files = {split: 0 for split in SPLITS}
    changed_counts = {split: 0 for split in SPLITS}
    region_class_counts = {region: {split: 0 for split in SPLITS} for region in REGION_LABELS}
    region_pair_counts = {region: {split: 0 for split in SPLITS} for region in REGION_LABELS}
    raw_counts = {region: 0 for region in REGION_LABELS}
    no_pair_files = 0
    cache = {key: [] for key in ("texts", "regions", "changed_clean", "changed_raw", "has_standard")}

    with ExitStack() as stack:
        class_corpora, standard_corpora, class_writers, pair_writers = {}, {}, {}, {}
        for split in SPLITS:
            class_corpora[split] = stack.enter_context((region_data / "corpus" / f"dialect_{split}_corpus.txt").open("w", encoding="utf-8"))
            standard_corpora[split] = stack.enter_context((translation_data / "corpus" / f"standard_{split}_corpus.txt").open("w", encoding="utf-8"))
            cf = stack.enter_context((region_data / "region_classification" / f"dialect_region_{split}.tsv").open("w", encoding="utf-8-sig", newline=""))
            pf = stack.enter_context((translation_data / "translation" / f"translation_{split}.tsv").open("w", encoding="utf-8-sig", newline=""))
            class_writers[split] = csv.writer(cf, delimiter="\t")
            pair_writers[split] = csv.writer(pf, delimiter="\t")
            class_writers[split].writerow(CLASS_HEADER)
            pair_writers[split].writerow(PAIR_HEADER)

        for index, (row, classifier_rows, pairs, raw_count) in enumerate(
            ordered_parallel_map(lambda row: process_source(row, helpers), assigned, workers), 1
        ):
            region, split = row["region"], row["split"]
            file_id = hashlib.sha1(str(Path(row["path"])).encode("utf-8")).hexdigest()[:20]
            source_group = row["source_group"]
            class_files[split] += 1
            raw_counts[region] += raw_count
            for text, known, changed_clean, changed_raw in classifier_rows:
                class_corpora[split].write(text + "\n")
                class_writers[split].writerow([text, LABEL2ID[region], region, file_id, source_group])
                if split == "test":
                    cache["texts"].append(text)
                    cache["regions"].append(region)
                    cache["has_standard"].append(known)
                    cache["changed_clean"].append(changed_clean)
                    cache["changed_raw"].append(changed_raw)
            class_counts[split] += len(classifier_rows)
            region_class_counts[region][split] += len(classifier_rows)
            if pairs:
                pair_files[split] += 1
                for source, target in pairs:
                    changed = helpers["normalize_text"](source) != helpers["normalize_text"](target)
                    pair_writers[split].writerow([source, target, region, file_id, source_group, int(changed)])
                    standard_corpora[split].write(target + "\n")
                    changed_counts[split] += int(changed)
                pair_counts[split] += len(pairs)
                region_pair_counts[region][split] += len(pairs)
            else:
                no_pair_files += 1
            if index % 250 == 0 or index == len(assigned):
                print(f"[WRITE] {index:,}/{len(assigned):,} JSON; classification={sum(class_counts.values()):,}; parallel={sum(pair_counts.values()):,}", flush=True)

    if not all(class_counts.values()) or not all(pair_counts.values()):
        raise RuntimeError(f"All splits must contain classification rows and parallel pairs: {class_counts}; {pair_counts}")
    for key in cache:
        cache[key] = np.array(cache[key], dtype=object if key in ("texts", "regions") else bool)
    np.savez_compressed(cache_dir / "changed_flags_cache.npz", **cache)
    expected_count = class_counts["test"]
    if any(len(array) != expected_count for array in cache.values()):
        raise RuntimeError("Changed-subset cache row count mismatch.")

    final_region = final_server_root / "region_classification" / "data"
    final_translation = final_server_root / "translation" / "data"
    manifest_path = final_region / "corpus_split_manifest_80_10_10.csv"
    class_metadata = {
        "completed": True,
        "split_sentence_counts": class_counts,
        "split_file_counts": class_files,
        "region_sentence_counts": region_class_counts,
        "skipped_file_count": 0, "skipped_files": [],
        "corpus_paths": {split: str(final_region / "corpus" / f"dialect_{split}_corpus.txt") for split in SPLITS},
        "tsv_paths": {split: str(final_region / "region_classification" / f"dialect_region_{split}.tsv") for split in SPLITS},
        "source_manifest": str(final_server_root.parent / "local_prepare" / "source_manifest.csv"),
        "split_manifest": str(manifest_path), "ratios": RATIOS, "seed": 42,
    }
    translation_metadata = {
        "completed": True, "split_manifest": str(manifest_path),
        "pair_counts": pair_counts, "changed_counts": changed_counts,
        "changed_fraction": {split: changed_counts[split] / pair_counts[split] for split in SPLITS},
        "paired_file_counts": pair_files, "region_pair_counts": region_pair_counts,
        "files_without_parallel_pairs": no_pair_files, "skipped_file_count": 0,
    }
    save_json(region_data / "preparation_metadata.json", class_metadata)
    save_json(translation_data / "preparation_metadata.json", translation_metadata)
    return class_metadata, translation_metadata, raw_counts


def build_dataset(old_root, new_root, package_root=PACKAGE_ROOT, workers=8,
                  audit_counts=None, include_new_jeolla_validation=False):
    started = time.monotonic()
    package_root = Path(package_root).resolve()
    local_root, server_root = package_root / "local_prepare", package_root / "server"
    helpers = load_original_helpers(server_root)
    stage = local_root / "build_staging"
    outputs = [
        stage, server_root / "region_classification" / "data",
        server_root / "translation" / "data", server_root / "bundle_ready.json",
        server_root / "region_classification" / "results" / "changed_flags_cache.npz",
        local_root / "source_manifest.csv", local_root / "source_counts.jsonl",
        local_root / "build_report.json", local_root / "count_provenance.json",
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite an existing build. Existing paths: " + ", ".join(existing))
    rows = select_sources(old_root, new_root, include_new_jeolla_validation)
    cached, audit_details = load_audit_counts(audit_counts, helpers)
    if audit_details.get("requested") and not audit_details.get("compatible"):
        print(f"[COUNT] Audit not reused: {audit_details.get('reason')}", flush=True)
    count_details = attach_counts(rows, helpers, cached, workers)
    local_root.mkdir(parents=True, exist_ok=True)
    save_manifest(local_root / "source_manifest.csv", rows)
    with (local_root / "source_counts.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps({"path": row["path"], "source_id": row["source_id"], "cleaned_rows": int(row["num_sentences"]), "sha256": row["sha256"]}, ensure_ascii=False) + "\n")
    save_json(local_root / "count_provenance.json", {
        "normalizer_function": helpers["clean_base_text_source"], "helper_hashes": helpers["helper_hashes"],
        "audit": audit_details, **count_details,
    })
    assigned = helpers["assign_file_splits"](rows, 0.8, 0.1, 0.1, 42)
    class_metadata, translation_metadata, raw_counts = write_data(assigned, helpers, stage, server_root, workers)

    # Publish completed outputs only; a separate full validator owns bundle_ready.json.
    final_region_data = server_root / "region_classification" / "data"
    final_translation_data = server_root / "translation" / "data"
    publish_path(stage / "region_data", final_region_data, package_root)
    publish_path(stage / "translation_data", final_translation_data, package_root)
    results_dir = server_root / "region_classification" / "results"
    results_dir.mkdir(exist_ok=True)
    publish_path(stage / "region_results" / "changed_flags_cache.npz", results_dir / "changed_flags_cache.npz", package_root)
    (stage / "region_results").rmdir()
    stage.rmdir()
    selected_files = {region: sum(row["region"] == region for row in rows) for region in REGION_LABELS}
    official_counts = defaultdict(int)
    for row in rows:
        official_counts[f"{row['region']} | {row['origin']} | {row['official_split']}"] += 1
    report = {
        "completed": True, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Five-region base JSON only; no senior datasets, ZIPs, TXTs, or content deduplication.",
        "old_root": str(Path(old_root).resolve()), "new_root": str(Path(new_root).resolve()),
        "include_new_jeolla_validation": bool(include_new_jeolla_validation),
        "selected_file_counts": selected_files, "official_origin_counts": dict(official_counts),
        "total_selected_files": len(rows), "raw_utterance_counts": raw_counts,
        "classification": class_metadata, "translation": translation_metadata,
        "audit_count_reuse": {**audit_details, **count_details},
        "all_source_hashes_verified_while_writing": True,
        "helper_hashes": helpers["helper_hashes"], "seconds": time.monotonic() - started,
        "server_ready_marker_written": False,
    }
    save_json(local_root / "build_report.json", report)
    print(f"[OK] Local data build complete. Report: {local_root / 'build_report.json'}", flush=True)
    print("[NEXT] Run the separate full-bundle validator before server training.", flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Prepare base-only local data with unchanged 260819/260811 text and split rules.")
    parser.add_argument("--old-root", type=Path, default=Path(r"D:\documents\project_1\git\project1_dataset"))
    parser.add_argument("--new-root", type=Path, default=Path(r"D:\방언 데이터 총정리"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--audit-counts", type=Path)
    parser.add_argument("--include-new-jeolla-validation", action="store_true", help="Opt in to new Jeolla Validation JSONs in addition to old Jeolla. Default: excluded.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    build_dataset(args.old_root, args.new_root, workers=args.workers, audit_counts=args.audit_counts,
                  include_new_jeolla_validation=args.include_new_jeolla_validation)


if __name__ == "__main__":
    main()
