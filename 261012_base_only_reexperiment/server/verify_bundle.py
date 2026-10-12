"""Validate the prepared experiment bundle without reopening its source JSONs.

The default check is a fast transfer inventory/size check. --full additionally
streams every TSV/corpus row, checks the cached classification flags, and hashes
all transfer files. Only a successful full check marks the bundle ready.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
import sys
import unicodedata


SPLITS = ("train", "validation", "test")
REGIONS = ("강원도", "경상도", "전라도", "제주도", "충청도")
LABELS = {region: str(index) for index, region in enumerate(REGIONS)}
CLASS_HEADER = ["text", "label", "region", "file_id", "source_group"]
PARALLEL_HEADER = ["source_text", "target_text", "region", "file_id", "source_group", "is_changed"]
SPACE = re.compile(r"\s+")
HEX256 = re.compile(r"[0-9a-f]{64}\Z")
CLASS_ORIGINALS = {
    "experiment_common.py", "gpu_monitor.py", "requirements_260819.txt",
    "run_full_experiment.py", "step1_prepare_data_80_10_10.py",
    "step2_train_dialect_tokenizer.py", "step3_pretrain_small_bert_mlm.py",
    "step4_finetune_region_classifier.py", "step5_summarize_results.py",
    "step6_changed_subset_analysis.py",
}
TRANSLATION_ORIGINALS = {
    "experiment_common.py", "process_gpu_monitor.py", "requirements_260811.txt",
    "run_full_experiment.py", "step1_prepare_parallel_data.py",
    "step2_pretrain_standard_decoder_clm.py", "step3_train_dialect_to_standard.py",
    "step4_summarize_results.py",
}
EXPECTED_ORIGINALS = (
    {f"region_classification/{name}" for name in CLASS_ORIGINALS}
    | {f"translation/{name}" for name in TRANSLATION_ORIGINALS}
)
GENERATED_JSON = {"bundle_ready.json", "transfer_manifest.json", "verification_report.json"}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def nonempty(path):
    require(path.is_file() and path.stat().st_size > 0, f"Missing or empty file: {path}")


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def safe_path(root, relative):
    require(isinstance(relative, str) and relative, "Invalid transfer path")
    candidate = (root / relative).resolve()
    require(candidate.is_relative_to(root.resolve()), f"Path escapes bundle: {relative}")
    return candidate


def verify_originals(root):
    records = read_json(root / "original_code_sha256.json")
    require(isinstance(records, dict) and set(records) == EXPECTED_ORIGINALS,
            "Original-code hash index must contain the expected 18 copied files")
    for relative, record in records.items():
        require(isinstance(record, dict) and isinstance(record.get("sha256"), str)
                and HEX256.fullmatch(record["sha256"]), f"Invalid original-code hash: {relative}")
        path = safe_path(root, relative)
        nonempty(path)
        require(digest(path) == record["sha256"], f"Original-code SHA256 mismatch: {relative}")
    print("[OK] All 18 copied original files match their recorded SHA256", flush=True)
    return len(records)


def required_data_paths(root):
    paths = [
        root / "region_classification/data/preparation_metadata.json",
        root / "region_classification/data/corpus_split_manifest_80_10_10.csv",
        root / "translation/data/preparation_metadata.json",
        root / "region_classification/results/changed_flags_cache.npz",
    ]
    for split in SPLITS:
        paths.extend([
            root / f"region_classification/data/region_classification/dialect_region_{split}.tsv",
            root / f"region_classification/data/corpus/dialect_{split}_corpus.txt",
            root / f"translation/data/translation/translation_{split}.tsv",
            root / f"translation/data/corpus/standard_{split}_corpus.txt",
        ])
    return paths


def load_manifest(root):
    path = root / "region_classification/data/corpus_split_manifest_80_10_10.csv"
    files = {}
    source_paths = set()
    split_counts = Counter()
    expected_rows = Counter()
    region_rows = {region: Counter() for region in REGIONS}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"region", "source_group", "source_type", "path", "num_sentences", "split"}
        require(required.issubset(reader.fieldnames or []), "Invalid shared split-manifest header")
        for row_number, row in enumerate(reader, 2):
            region, split, raw = row["region"], row["split"], row["path"]
            require(region in REGIONS and split in SPLITS, f"Invalid manifest region/split at row {row_number}")
            require(row["source_type"] == "base", f"Non-base source in manifest at row {row_number}")
            require(raw and PureWindowsPath(raw).suffix.lower() == ".json",
                    f"Non-JSON source in manifest at row {row_number}")
            normalized_path = str(PureWindowsPath(raw)).casefold()
            require(normalized_path not in source_paths,
                    f"Source file occurs more than once (possibly across splits): {raw}")
            source_paths.add(normalized_path)
            count = int(row["num_sentences"])
            require(count >= 0, f"Negative sentence count in manifest: {raw}")
            file_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]
            require(file_id not in files, f"Duplicate source file identifier: {file_id}")
            files[file_id] = {
                "region": region, "split": split, "source_group": row["source_group"],
                "num_sentences": count,
            }
            split_counts[split] += 1
            expected_rows[split] += count
            region_rows[region][split] += count
    require(all(split_counts[split] > 0 for split in SPLITS), "An experiment split has no source files")
    require(all(sum(region_rows[region].values()) > 0 for region in REGIONS), "A region has no input rows")
    return files, dict(split_counts), dict(expected_rows), {
        region: {split: region_rows[region][split] for split in SPLITS} for region in REGIONS
    }


def load_flags(root, expected_count):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Full validation requires NumPy to inspect changed_flags_cache.npz") from exc
    path = root / "region_classification/results/changed_flags_cache.npz"
    keys = {"texts", "regions", "changed_clean", "changed_raw", "has_standard"}
    # This is the local builder's trusted cache. Object arrays are the format
    # used by the unchanged original changed-subset analysis code.
    with np.load(path, allow_pickle=True) as archive:
        require(set(archive.files) == keys, "Changed-flags cache has unexpected arrays")
        flags = {key: archive[key] for key in keys}
    for key, values in flags.items():
        require(values.ndim == 1 and len(values) == expected_count,
                f"Changed-flags cache shape/count mismatch: {key}")
    for key in ("changed_clean", "changed_raw", "has_standard"):
        require(flags[key].dtype.kind == "b", f"Changed-flags cache must use Boolean flags: {key}")
    unknown = ~flags["has_standard"]
    require(not np.any(unknown & (flags["changed_clean"] | flags["changed_raw"])),
            "Rows without standard forms must not be flagged as changed")
    return flags


def read_header(reader, expected, path):
    require(next(reader, None) == expected, f"Unexpected TSV header: {path}")


def check_reference(file_id, region, group, split, manifest, location):
    record = manifest.get(file_id)
    require(record is not None, f"Unknown file_id at {location}: {file_id}")
    require((record["split"], record["region"], record["source_group"]) == (split, region, group),
            f"Source file split/region/group mismatch at {location}: {file_id}")


def validate_classification(root, split, manifest, metadata, flags):
    tsv = root / f"region_classification/data/region_classification/dialect_region_{split}.tsv"
    corpus = root / f"region_classification/data/corpus/dialect_{split}_corpus.txt"
    per_file = Counter()
    per_region = Counter()
    count = 0
    with tsv.open("r", encoding="utf-8-sig", newline="") as stream, \
            corpus.open("r", encoding="utf-8", newline="") as lines:
        reader = csv.reader(stream, delimiter="\t")
        read_header(reader, CLASS_HEADER, tsv)
        for count, row in enumerate(reader, 1):
            require(len(row) == len(CLASS_HEADER), f"Classification {split} row {count}: invalid column count")
            text, label, region, file_id, group = row
            require(bool(text) and label == LABELS.get(region),
                    f"Classification {split} row {count}: invalid text, region, or label")
            line = lines.readline()
            require(line != "" and line.rstrip("\r\n") == text,
                    f"Classification {split} row {count}: corpus/TSV text mismatch")
            check_reference(file_id, region, group, split, manifest, f"classification {split} row {count}")
            if split == "test":
                index = count - 1
                require(index < len(flags["texts"]), "Test TSV exceeds changed-cache row count")
                require(flags["texts"][index] == text and flags["regions"][index] == region,
                        f"Changed-cache text/region alignment mismatch at test row {count}")
            per_file[file_id] += 1
            per_region[region] += 1
            if count % 1_000_000 == 0:
                print(f"[CHECK] Classification {split}: {count:,} rows", flush=True)
        require(lines.readline() == "", f"Classification {split}: extra corpus lines")
    expected = metadata["split_sentence_counts"][split]
    require(count == expected, f"Classification {split} count {count} != metadata {expected}")
    for file_id, record in manifest.items():
        if record["split"] == split:
            require(per_file[file_id] == record["num_sentences"],
                    f"Classification count differs from manifest for file {file_id}")
    for region in REGIONS:
        require(per_region[region] == metadata["region_sentence_counts"][region][split],
                f"Classification {region}/{split} count differs from metadata")
    print(f"[OK] Classification {split}: {count:,} rows; corpus, labels, file references verified", flush=True)
    return {"rows": count, "files_with_rows": len(per_file), "region_rows": dict(per_region)}


def normalize(text):
    return SPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def validate_translation(root, split, manifest, metadata):
    tsv = root / f"translation/data/translation/translation_{split}.tsv"
    corpus = root / f"translation/data/corpus/standard_{split}_corpus.txt"
    per_region = Counter()
    per_file = set()
    count = changed_count = 0
    with tsv.open("r", encoding="utf-8-sig", newline="") as stream, \
            corpus.open("r", encoding="utf-8", newline="") as lines:
        reader = csv.reader(stream, delimiter="\t")
        read_header(reader, PARALLEL_HEADER, tsv)
        for count, row in enumerate(reader, 1):
            require(len(row) == len(PARALLEL_HEADER), f"Translation {split} row {count}: invalid column count")
            source, target, region, file_id, group, changed = row
            source_normalized, target_normalized = normalize(source), normalize(target)
            require(bool(source) and bool(target) and source == source_normalized and target == target_normalized,
                    f"Translation {split} row {count}: text is empty or not normalized")
            expected_changed = str(int(source_normalized != target_normalized))
            require(changed == expected_changed,
                    f"Translation {split} row {count}: is_changed differs from normalized text comparison")
            line = lines.readline()
            require(line != "" and line.rstrip("\r\n") == target,
                    f"Translation {split} row {count}: corpus/TSV target mismatch")
            check_reference(file_id, region, group, split, manifest, f"translation {split} row {count}")
            changed_count += int(changed)
            per_region[region] += 1
            per_file.add(file_id)
            if count % 1_000_000 == 0:
                print(f"[CHECK] Translation {split}: {count:,} pairs", flush=True)
        require(lines.readline() == "", f"Translation {split}: extra corpus lines")
    require(count == metadata["pair_counts"][split], f"Translation {split}: pair count differs from metadata")
    require(changed_count == metadata["changed_counts"][split],
            f"Translation {split}: changed count differs from metadata")
    require(len(per_file) == metadata["paired_file_counts"][split],
            f"Translation {split}: paired-file count differs from metadata")
    for region in REGIONS:
        require(per_region[region] == metadata["region_pair_counts"][region][split],
                f"Translation {region}/{split} count differs from metadata")
    print(f"[OK] Translation {split}: {count:,} pairs; changed={changed_count:,}", flush=True)
    return {"pairs": count, "changed_pairs": changed_count, "paired_files": len(per_file),
            "region_pairs": dict(per_region)}


def collect_transfer_paths(root):
    paths = set()
    for directory in (root, root / "region_classification", root / "translation"):
        for path in directory.iterdir():
            if path.is_file() and path.suffix in {".py", ".txt", ".md", ".json"}:
                if directory == root and path.name in GENERATED_JSON:
                    continue
                paths.add(path)
    for directory in (root / "region_classification/data", root / "translation/data"):
        for path in directory.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                paths.add(path)
    paths.add(root / "region_classification/results/changed_flags_cache.npz")
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def transfer_records(root, paths):
    records = {}
    for index, path in enumerate(paths, 1):
        relative = path.relative_to(root).as_posix()
        print(f"[HASH {index}/{len(paths)}] {relative}", flush=True)
        before = path.stat()
        value = digest(path)
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                f"File changed while hashing: {relative}")
        records[relative] = {"size_bytes": after.st_size, "sha256": value}
    return records


def load_transfer(root):
    transfer = read_json(root / "transfer_manifest.json")
    require(transfer.get("format_version") == 1 and isinstance(transfer.get("files"), dict),
            "Invalid transfer manifest format")
    for relative, record in transfer["files"].items():
        safe_path(root, relative)
        require(isinstance(record, dict) and isinstance(record.get("size_bytes"), int)
                and record["size_bytes"] >= 0 and isinstance(record.get("sha256"), str)
                and HEX256.fullmatch(record["sha256"]), f"Invalid transfer hash/size: {relative}")
    return transfer


def fast_check(root):
    ready = read_json(root / "bundle_ready.json")
    require(ready.get("completed") is True, "Bundle is not ready; complete --full validation first")
    transfer = load_transfer(root)
    require(digest(root / "transfer_manifest.json") == ready.get("transfer_manifest_sha256"),
            "Transfer-manifest SHA256 differs from the ready marker")
    expected = transfer["files"]
    current = {path.relative_to(root).as_posix() for path in collect_transfer_paths(root)}
    require(current == set(expected), "Transfer file inventory differs from the verified bundle")
    for relative, record in expected.items():
        path = safe_path(root, relative)
        require(path.is_file() and path.stat().st_size == record["size_bytes"],
                f"Transfer file missing or size differs: {relative}")
    verify_originals(root)
    print(f"[OK] Fast check: {len(expected)} files exist with recorded sizes; original code hashes match.")
    print("[INFO] Data content hashes and all rows are checked only with --full.")
    return {"completed": True, "mode": "fast", "file_count": len(expected)}


def full_check(root):
    started = utc_now()
    nonempty(root / "original_code_sha256.json")
    previous = load_transfer(root) if (root / "transfer_manifest.json").is_file() else None
    write_json(root / "bundle_ready.json", {
        "completed": False, "status": "full_validation_in_progress", "started_utc": started,
    })
    original_count = verify_originals(root)
    for path in required_data_paths(root):
        nonempty(path)
    classification = read_json(root / "region_classification/data/preparation_metadata.json")
    translation = read_json(root / "translation/data/preparation_metadata.json")
    require(classification.get("skipped_file_count") == 0, "Classification preparation skipped source files")
    require(classification.get("seed") == 42, "Classification split seed differs from 42")
    require(classification.get("ratios") == {"train": 0.8, "validation": 0.1, "test": 0.1},
            "Classification split ratios differ from 80/10/10")
    require(translation.get("completed") is True and translation.get("skipped_file_count") == 0,
            "Translation preparation is incomplete or skipped source files")
    manifest, split_files, split_rows, region_rows = load_manifest(root)
    require(split_rows == classification["split_sentence_counts"], "Manifest/metadata sentence counts differ")
    require(split_files == classification["split_file_counts"], "Manifest/metadata file counts differ")
    require(region_rows == classification["region_sentence_counts"], "Manifest/metadata regional counts differ")
    flags = load_flags(root, split_rows["test"])
    classification_results = {}
    translation_results = {}
    for split in SPLITS:
        classification_results[split] = validate_classification(root, split, manifest, classification, flags)
        translation_results[split] = validate_translation(root, split, manifest, translation)
    changed_report = {
        "rows": len(flags["texts"]),
        "known_standard_rows": int(flags["has_standard"].sum()),
        "changed_clean_rows": int(flags["changed_clean"].sum()),
        "changed_raw_rows": int(flags["changed_raw"].sum()),
        "all_test_texts_and_regions_aligned": True,
        "raw_json_flags_recomputed": False,
    }
    del flags
    records = transfer_records(root, collect_transfer_paths(root))
    if previous is not None:
        require(records == previous["files"],
                "Bundle content differs from the prior verified transfer manifest; do not re-baseline silently")
    finished = utc_now()
    report = {
        "completed": True, "mode": "full", "started_utc": started, "finished_utc": finished,
        "source_json_reopened": False, "original_files_sha256_verified": original_count,
        "manifest_file_count": len(manifest), "manifest_files_single_split": True,
        "classification": classification_results, "translation": translation_results,
        "changed_flags_cache": changed_report,
        "transfer_file_count": len(records),
        "transfer_bytes": sum(record["size_bytes"] for record in records.values()),
    }
    write_json(root / "transfer_manifest.json", {"format_version": 1, "created_utc": finished, "files": records})
    write_json(root / "verification_report.json", report)
    write_json(root / "bundle_ready.json", {
        "completed": True, "verified_utc": finished,
        "verification_mode": "full",
        "transfer_manifest": "transfer_manifest.json",
        "transfer_manifest_sha256": digest(root / "transfer_manifest.json"),
        "verification_report": "verification_report.json",
        "file_count": len(records),
        "classification_counts": {split: value["rows"] for split, value in classification_results.items()},
        "translation_counts": {split: value["pairs"] for split, value in translation_results.items()},
    })
    print(f"[OK] Full bundle validation complete: {len(records)} transfer files; bundle_ready.json written.", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="Server bundle directory (defaults to this script's directory)")
    parser.add_argument("--full", action="store_true",
                        help="Stream every row, inspect cached flags, hash all files, and mark ready on success")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    try:
        if args.full:
            full_check(root)
        else:
            fast_check(root)
    except Exception as exc:
        if args.full and root.is_dir() and (root / "original_code_sha256.json").is_file():
            failure = {"completed": False, "mode": "full", "failed_utc": utc_now(), "error": str(exc)}
            write_json(root / "verification_report.json", failure)
            write_json(root / "bundle_ready.json", failure)
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
