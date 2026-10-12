"""Build local common-preprocessing data while freezing every file's split.

No tokenizers/models are trained. No file split is generated or randomized.
The original experiment modules are verified, not imported or edited.
"""

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import time

from common_preprocessing import PREPROCESSING_VERSION, process_utterance


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = PACKAGE_ROOT / "local_prepare" / "frozen_split_manifest.csv"
REGION_LABELS = ["강원도", "경상도", "전라도", "제주도", "충청도"]
LABEL2ID = {region: index for index, region in enumerate(REGION_LABELS)}
SPLITS = ("train", "validation", "test")
RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}
CLASS_HEADER = ["text", "label", "region", "file_id", "source_group", "utterance_id"]
PAIR_HEADER = ["source_text", "target_text", "region", "file_id", "source_group", "is_changed", "utterance_id"]
ORIGINAL_FILES = {
    "region_classification/experiment_common.py",
    "region_classification/gpu_monitor.py",
    "region_classification/requirements_260819.txt",
    "region_classification/run_full_experiment.py",
    "region_classification/step1_prepare_data_80_10_10.py",
    "region_classification/step2_train_dialect_tokenizer.py",
    "region_classification/step3_pretrain_small_bert_mlm.py",
    "region_classification/step4_finetune_region_classifier.py",
    "region_classification/step5_summarize_results.py",
    "region_classification/step6_changed_subset_analysis.py",
    "translation/experiment_common.py",
    "translation/process_gpu_monitor.py",
    "translation/requirements_260811.txt",
    "translation/run_full_experiment.py",
    "translation/step1_prepare_parallel_data.py",
    "translation/step2_pretrain_standard_decoder_clm.py",
    "translation/step3_train_dialect_to_standard.py",
    "translation/step4_summarize_results.py",
}


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


def verify_original_code(server_root):
    server_root = Path(server_root).resolve()
    records = read_json(server_root / "original_code_sha256.json")
    if set(records) != ORIGINAL_FILES:
        raise ValueError("The original-code registry must contain exactly the 18 frozen experiment files.")
    for relative, entry in records.items():
        path = (server_root / relative).resolve()
        if not path.is_relative_to(server_root) or not path.is_file():
            raise ValueError(f"Invalid original-code path: {relative}")
        if file_sha256(path) != entry.get("sha256"):
            raise ValueError(f"Original experiment file was modified: {relative}")
    return {relative: entry["sha256"] for relative, entry in records.items()}


def load_frozen_manifest(path):
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    required = {"region", "source_group", "source_type", "path", "num_sentences", "source_id", "origin", "official_split", "sha256", "split"}
    if not rows or required - set(fields):
        raise ValueError(f"Missing fields or empty frozen manifest: {sorted(required - set(fields))}")
    additions = {"original_num_sentences", "num_parallel_pairs", "raw_utterances"}
    if additions.intersection(fields):
        raise ValueError("Use the original frozen manifest, not a previously rebuilt manifest.")
    source_ids, paths = set(), set()
    for row in rows:
        if row["region"] not in LABEL2ID or row["split"] not in SPLITS or row["source_type"] != "base":
            raise ValueError(f"Invalid frozen source or split: {row['source_id']}")
        if not row["path"] or not Path(row["path"]).is_absolute():
            raise ValueError(f"Source paths must remain their original absolute local paths: {row['source_id']}")
        count = int(row["num_sentences"])
        if count < 0 or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError(f"Invalid original count/hash: {row['source_id']}")
        source_id = row["source_id"].casefold()
        path_identity = os.path.normcase(str(Path(row["path"]).resolve()))
        if source_id in source_ids or path_identity in paths:
            raise ValueError(f"Duplicate source in frozen manifest: {row['source_id']}")
        source_ids.add(source_id)
        paths.add(path_identity)
    if {row["region"] for row in rows} != set(REGION_LABELS):
        raise ValueError("Frozen manifest must contain all five regions.")
    return fields, rows


def file_id_for(row):
    # Exactly the original builders' calculation. Do not resolve or re-root.
    return hashlib.sha1(str(Path(row["path"])).encode("utf-8")).hexdigest()[:20]


def process_file(row):
    """One raw read/hash check; use the same canonical source in all tasks."""
    raw = Path(row["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != row["sha256"]:
        raise ValueError(f"Source JSON changed since frozen split: {row['path']}")
    data = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(data, dict) or not isinstance(data.get("utterance", []), list):
        raise ValueError(f"Expected utterance list: {row['path']}")
    class_rows, pair_rows = [], []
    stats, flags, exclusions = Counter(), Counter(), Counter()
    row_counts = Counter(
        raw_utterances=len(data.get("utterance", [])), classification_rows=0,
        parallel_pairs=0, excluded_from_all_tasks=0, source_only_rows=0,
    )
    file_id = file_id_for(row)
    for index, utterance in enumerate(data.get("utterance", [])):
        if not isinstance(utterance, dict):
            raise ValueError(f"Invalid utterance object: {row['path']} index {index}")
        result = process_utterance(utterance)
        stats.update(result["stats"])
        flags.update(set(result["flags"]))  # Flags count utterances, not occurrences.
        reason = result["drop_reason"]
        source, target = result["source_text"], result["target_text"]
        if reason is not None:
            if source or target is not None:
                raise ValueError("Dropped utterance returned usable text; common policy contract violated.")
            exclusions[reason] += 1
            row_counts["excluded_from_all_tasks"] += 1
            continue
        if not isinstance(source, str) or not source:
            exclusions["empty_source_after_common_preprocessing"] += 1
            row_counts["excluded_from_all_tasks"] += 1
            continue
        if "\n" in source or "\r" in source or "\t" in source:
            raise ValueError("Common source normalization left TSV/corpus control whitespace.")
        known = isinstance(target, str) and bool(target)
        if bool(result["has_standard"]) != known:
            raise ValueError("has_standard must mean a nonempty canonical target.")
        if known and any(char in target for char in "\n\r\t"):
            raise ValueError("Common target normalization left TSV/corpus control whitespace.")
        utterance_id = f"{file_id}:{index}"
        changed_clean = bool(known and source != target)
        changed_raw = bool(result["changed_raw"])
        class_rows.append((source, utterance_id, known, changed_clean, changed_raw))
        row_counts["classification_rows"] += 1
        if known:
            pair_rows.append((source, target, utterance_id, changed_clean))
            row_counts["parallel_pairs"] += 1
        else:
            exclusions["translation_no_valid_target"] += 1
            row_counts["source_only_rows"] += 1
    if row_counts["classification_rows"] + row_counts["excluded_from_all_tasks"] != row_counts["raw_utterances"]:
        raise RuntimeError("Raw/classification/drop counts do not reconcile.")
    return row, class_rows, pair_rows, dict(row_counts), dict(stats), dict(flags), dict(exclusions)


def ordered_process_map(rows, workers):
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        for row in rows:
            yield process_file(row)
        return
    iterator = iter(rows)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = deque()
        for _ in range(workers * 2):
            row = next(iterator, None)
            if row is not None:
                pending.append(executor.submit(process_file, row))
        while pending:
            yield pending.popleft().result()
            row = next(iterator, None)
            if row is not None:
                pending.append(executor.submit(process_file, row))


def write_manifest(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def publish_path(source, destination, package_root):
    root = Path(package_root).resolve()
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_relative_to(root) or not destination.is_relative_to(root):
        raise ValueError(f"Publication path escapes the new project: {source} -> {destination}")
    if source == root or destination == root or not source.exists():
        raise ValueError("Invalid publication path.")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    source.rename(destination)


def new_counters():
    return {"row_counts": Counter(), "policy_stats": Counter(), "flagged_utterances": Counter(), "exclusion_reasons": Counter()}


def update_counters(target, counts, stats, flags, exclusions):
    for key, value in zip(("row_counts", "policy_stats", "flagged_utterances", "exclusion_reasons"), (counts, stats, flags, exclusions)):
        target[key].update(value)


def write_data(frozen_rows, original_fields, stage, server_root, manifest_provenance, workers):
    import numpy as np

    region_data, translation_data = stage / "region_data", stage / "translation_data"
    for directory in (region_data / "corpus", region_data / "region_classification", translation_data / "corpus", translation_data / "translation", stage / "region_results", stage / "reports"):
        directory.mkdir(parents=True, exist_ok=False)
    class_counts = {split: 0 for split in SPLITS}
    class_files = {split: 0 for split in SPLITS}
    pair_counts = {split: 0 for split in SPLITS}
    pair_files = {split: 0 for split in SPLITS}
    changed_counts = {split: 0 for split in SPLITS}
    region_class_counts = {region: {split: 0 for split in SPLITS} for region in REGION_LABELS}
    region_pair_counts = {region: {split: 0 for split in SPLITS} for region in REGION_LABELS}
    raw_counts = {region: 0 for region in REGION_LABELS}
    zero_class_files, no_pair_files = [], 0
    totals = new_counters()
    by_region = {region: new_counters() for region in REGION_LABELS}
    by_split = {split: new_counters() for split in SPLITS}
    updated_rows = []
    cache = {key: [] for key in ("texts", "regions", "has_standard", "changed_clean", "changed_raw")}
    with ExitStack() as stack:
        ccorpora, tcorpora, cwriters, pwriters = {}, {}, {}, {}
        for split in SPLITS:
            ccorpora[split] = stack.enter_context((region_data / "corpus" / f"dialect_{split}_corpus.txt").open("w", encoding="utf-8"))
            tcorpora[split] = stack.enter_context((translation_data / "corpus" / f"standard_{split}_corpus.txt").open("w", encoding="utf-8"))
            cf = stack.enter_context((region_data / "region_classification" / f"dialect_region_{split}.tsv").open("w", encoding="utf-8-sig", newline=""))
            pf = stack.enter_context((translation_data / "translation" / f"translation_{split}.tsv").open("w", encoding="utf-8-sig", newline=""))
            cwriters[split], pwriters[split] = csv.writer(cf, delimiter="\t"), csv.writer(pf, delimiter="\t")
            cwriters[split].writerow(CLASS_HEADER)
            pwriters[split].writerow(PAIR_HEADER)
        per_file_stream = stack.enter_context((stage / "reports" / "preprocessing_per_file.jsonl").open("w", encoding="utf-8"))
        for index, (row, classifier_rows, pairs, counts, stats, flags, exclusions) in enumerate(ordered_process_map(frozen_rows, workers), 1):
            split, region = row["split"], row["region"]
            file_id = file_id_for(row)
            class_files[split] += 1  # Includes empty files to preserve split membership.
            raw_counts[region] += counts["raw_utterances"]
            if not classifier_rows:
                zero_class_files.append(row["source_id"])
            for source, utterance_id, known, changed_clean, changed_raw in classifier_rows:
                ccorpora[split].write(source + "\n")
                cwriters[split].writerow([source, LABEL2ID[region], region, file_id, row["source_group"], utterance_id])
                if split == "test":
                    cache["texts"].append(source)
                    cache["regions"].append(region)
                    cache["has_standard"].append(known)
                    cache["changed_clean"].append(changed_clean)
                    cache["changed_raw"].append(changed_raw)
            class_counts[split] += len(classifier_rows)
            region_class_counts[region][split] += len(classifier_rows)
            if pairs:
                pair_files[split] += 1
                for source, target, utterance_id, changed in pairs:
                    pwriters[split].writerow([source, target, region, file_id, row["source_group"], int(changed), utterance_id])
                    tcorpora[split].write(target + "\n")
                    changed_counts[split] += int(changed)
                pair_counts[split] += len(pairs)
                region_pair_counts[region][split] += len(pairs)
            else:
                no_pair_files += 1
            updated = dict(row)
            updated.update({"original_num_sentences": row["num_sentences"], "num_sentences": str(len(classifier_rows)), "num_parallel_pairs": str(len(pairs)), "raw_utterances": str(counts["raw_utterances"])})
            updated_rows.append(updated)
            for target in (totals, by_region[region], by_split[split]):
                update_counters(target, counts, stats, flags, exclusions)
            per_file_stream.write(json.dumps({
                "source_id": row["source_id"], "region": region, "split": split,
                "file_id": file_id, "sha256": row["sha256"],
                "original_num_sentences": int(row["num_sentences"]),
                "row_counts": counts, "policy_stats": stats,
                "flagged_utterances": flags, "exclusion_reasons": exclusions,
            }, ensure_ascii=False) + "\n")
            if index % 250 == 0 or index == len(frozen_rows):
                print(f"[WRITE] {index:,}/{len(frozen_rows):,} files; classification={sum(class_counts.values()):,}; pairs={sum(pair_counts.values()):,}", flush=True)

    if not all(class_counts.values()) or not all(pair_counts.values()):
        raise RuntimeError(f"Each split must retain classification and translation examples: {class_counts}; {pair_counts}")
    for key in cache:
        cache[key] = np.array(cache[key], dtype=object if key in ("texts", "regions") else bool)
    np.savez_compressed(stage / "region_results" / "changed_flags_cache.npz", **cache)
    if any(len(array) != class_counts["test"] for array in cache.values()):
        raise RuntimeError("Changed-subset cache count mismatch.")
    fields = original_fields + ["original_num_sentences", "num_parallel_pairs", "raw_utterances"]
    write_manifest(region_data / "corpus_split_manifest_80_10_10.csv", fields, updated_rows)
    write_manifest(stage / "reports" / "source_manifest.csv", fields, updated_rows)
    final_region = server_root / "region_classification" / "data"
    final_manifest = final_region / "corpus_split_manifest_80_10_10.csv"
    actual_ratios = {split: class_counts[split] / sum(class_counts.values()) for split in SPLITS}
    class_metadata = {
        "completed": True, "split_sentence_counts": class_counts, "split_file_counts": class_files,
        "region_sentence_counts": region_class_counts, "skipped_file_count": 0, "skipped_files": [],
        "source_manifest": str(server_root.parent / "local_prepare" / "source_manifest.csv"),
        "split_manifest": str(final_manifest), "seed": 42, "ratios": RATIOS,
        "actual_sentence_ratios": actual_ratios, "file_split_frozen": True,
        "split_provenance": manifest_provenance,
        "zero_cleaned_file_count": len(zero_class_files), "zero_cleaned_source_ids": zero_class_files,
        "preprocessing": "shared_common_preprocessing",
        "corpus_paths": {split: str(final_region / "corpus" / f"dialect_{split}_corpus.txt") for split in SPLITS},
        "tsv_paths": {split: str(final_region / "region_classification" / f"dialect_region_{split}.tsv") for split in SPLITS},
    }
    translation_metadata = {
        "completed": True, "split_manifest": str(final_manifest), "pair_counts": pair_counts,
        "changed_counts": changed_counts,
        "changed_fraction": {split: changed_counts[split] / pair_counts[split] for split in SPLITS},
        "paired_file_counts": pair_files, "region_pair_counts": region_pair_counts,
        "files_without_parallel_pairs": no_pair_files, "skipped_file_count": 0,
        "file_split_frozen": True, "preprocessing": "shared_common_preprocessing",
    }
    statistics = {
        "totals": totals, "by_region": by_region, "by_split": by_split,
        "counter_definitions": {
            "flagged_utterances": "Each named flag counts at most once per raw utterance; flags can overlap.",
            "policy_stats": "Counters returned by common_preprocessing.process_utterance; some count operations, not distinct utterances.",
            "exclusion_reasons": "All-task drop reasons and translation_no_valid_target are disjoint; the latter retains the source for MLM/classification.",
        },
    }
    save_json(region_data / "preparation_metadata.json", class_metadata)
    save_json(translation_data / "preparation_metadata.json", translation_metadata)
    save_json(stage / "reports" / "preprocessing_statistics.json", statistics)
    return class_metadata, translation_metadata, raw_counts, statistics


def build_dataset(split_manifest=DEFAULT_MANIFEST, package_root=PACKAGE_ROOT, workers=8):
    started = time.monotonic()
    package_root = Path(package_root).resolve()
    local_root, server_root = package_root / "local_prepare", package_root / "server"
    split_manifest = Path(split_manifest).resolve()
    if workers < 1:
        raise ValueError("workers must be positive")
    original_hashes = verify_original_code(server_root)
    policy_path = Path(__file__).resolve().with_name("common_preprocessing.py")
    policy_hash = file_sha256(policy_path)
    frozen_hash = file_sha256(split_manifest)
    fields, frozen_rows = load_frozen_manifest(split_manifest)
    stage = local_root / "build_staging"
    output_paths = [stage, server_root / "region_classification" / "data", server_root / "translation" / "data", server_root / "bundle_ready.json", server_root / "region_classification" / "results" / "changed_flags_cache.npz"]
    output_paths.extend(local_root / name for name in ("source_manifest.csv", "preprocessing_per_file.jsonl", "preprocessing_statistics.json", "build_report.json"))
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite an existing build: " + ", ".join(existing))
    provenance = {
        "path": str(split_manifest), "sha256": frozen_hash, "file_count": len(frozen_rows),
        "policy": "Reuse every original file assignment and manifest row order; no resplitting or shuffling.",
        "original_split_seed": 42, "original_target_ratios": RATIOS,
    }
    print(f"[FROZEN SPLIT] {len(frozen_rows):,} files; no new split will be generated.", flush=True)
    class_metadata, translation_metadata, raw_counts, statistics = write_data(frozen_rows, fields, stage, server_root, provenance, workers)
    if file_sha256(split_manifest) != frozen_hash or file_sha256(policy_path) != policy_hash:
        raise RuntimeError("Frozen manifest or common preprocessing code changed during the build.")
    verify_original_code(server_root)
    publish_path(stage / "region_data", server_root / "region_classification" / "data", package_root)
    publish_path(stage / "translation_data", server_root / "translation" / "data", package_root)
    result_dir = server_root / "region_classification" / "results"
    result_dir.mkdir(exist_ok=True)
    publish_path(stage / "region_results" / "changed_flags_cache.npz", result_dir / "changed_flags_cache.npz", package_root)
    for name in ("source_manifest.csv", "preprocessing_per_file.jsonl", "preprocessing_statistics.json"):
        publish_path(stage / "reports" / name, local_root / name, package_root)
    (stage / "region_results").rmdir()
    (stage / "reports").rmdir()
    stage.rmdir()
    file_counts = Counter(row["region"] for row in frozen_rows)
    origin_counts = Counter(f"{row['region']} | {row['origin']} | {row['official_split']}" for row in frozen_rows)
    report = {
        "completed": True, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Exactly the previous base-only selected source JSONs; fixed file splits, common preprocessing, no content deduplication.",
        "total_selected_files": len(frozen_rows), "selected_file_counts": dict(file_counts),
        "official_origin_counts": dict(origin_counts), "raw_utterance_counts": raw_counts,
        "classification": class_metadata, "translation": translation_metadata,
        "file_split_frozen": True, "split_provenance": provenance,
        "all_source_hashes_verified_while_writing": True,
        "common_preprocessing": {"path": str(policy_path), "sha256": policy_hash, "version": PREPROCESSING_VERSION},
        "preprocessing_statistics": statistics,
        "preprocessing_per_file_path": str(local_root / "preprocessing_per_file.jsonl"),
        "original_code_sha256": original_hashes, "original_training_code_unchanged": True,
        "utterance_id_definition": "Original file_id + ':' + zero-based original utterance-array index; identical in both TSVs, independent of missing/repeated source id fields.",
        "changed_clean_definition": "Shared canonical source_text differs from canonical target_text; only valid target rows included.",
        "changed_raw_definition": "Original dialect/standard fields after NFKC and whitespace normalization, as returned by the shared policy.",
        "seconds": time.monotonic() - started, "server_ready_marker_written": False,
    }
    save_json(local_root / "build_report.json", report)
    print(f"[OK] Local common-preprocessing build complete: {local_root / 'build_report.json'}", flush=True)
    print("[NEXT] Run the separate full-bundle validator before server training.", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    build_dataset(args.split_manifest, workers=args.workers)


if __name__ == "__main__":
    main()
