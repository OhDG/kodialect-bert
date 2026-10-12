import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from experiment_common import REGION_LABELS, normalize_text, save_json


DEFAULT_SPLIT_MANIFEST = (
    "../260807_4tokenizer_5seed_region_classification/"
    "data/corpus_split_manifest_80_10_10.csv"
)


def load_manifest(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"region", "source_group", "source_type", "path", "split"}
    missing = required - set(rows[0].keys()) if rows else required
    if missing:
        raise ValueError(f"Split manifest is missing columns: {sorted(missing)}")
    invalid = sorted({row["split"] for row in rows} - {"train", "validation", "test"})
    if invalid:
        raise ValueError(f"Invalid split values: {invalid}")
    return rows


def resolve_json_path(raw_path: str, manifest_path: Path) -> Path:
    path = Path(raw_path)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(manifest_path.parent / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"JSON file not found: {raw_path}")


def extract_parallel_pairs(path: Path) -> List[Tuple[str, str]]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    pairs = []
    for utterance in data.get("utterance", []):
        dialect = utterance.get("dialect_form", "")
        standard = utterance.get("standard_form", "")
        if not isinstance(dialect, str) or not isinstance(standard, str):
            continue
        dialect = normalize_text(dialect)
        standard = normalize_text(standard)
        if dialect and standard:
            pairs.append((dialect, standard))
    return pairs


def outputs_complete(output_dir: Path) -> bool:
    required = [output_dir / "preparation_metadata.json"]
    required.extend(
        output_dir / "translation" / f"translation_{split}.tsv"
        for split in ("train", "validation", "test")
    )
    required.extend(
        output_dir / "corpus" / f"standard_{split}_corpus.txt"
        for split in ("train", "validation", "test")
    )
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def prepare(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    if outputs_complete(output_dir) and not args.overwrite:
        print(f"[SKIP] Parallel data already exists: {output_dir}")
        return

    manifest_path = Path(args.split_manifest)
    rows = load_manifest(manifest_path)
    translation_dir = output_dir / "translation"
    corpus_dir = output_dir / "corpus"
    translation_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    tsv_files = {
        split: (translation_dir / f"translation_{split}.tsv").open(
            "w", encoding="utf-8-sig", newline=""
        )
        for split in ("train", "validation", "test")
    }
    corpus_files = {
        split: (corpus_dir / f"standard_{split}_corpus.txt").open(
            "w", encoding="utf-8"
        )
        for split in ("train", "validation", "test")
    }
    writers = {split: csv.writer(f, delimiter="\t") for split, f in tsv_files.items()}
    header = [
        "source_text",
        "target_text",
        "region",
        "file_id",
        "source_group",
        "is_changed",
    ]
    for writer in writers.values():
        writer.writerow(header)

    pair_counts = {split: 0 for split in writers}
    changed_counts = {split: 0 for split in writers}
    file_counts = {split: 0 for split in writers}
    region_counts = {
        region: {split: 0 for split in writers} for region in REGION_LABELS
    }
    no_pair_files = 0
    skipped_files = []

    try:
        for row in tqdm(rows, desc="Writing dialect-standard parallel data"):
            split = row["split"]
            region = row["region"]
            if region not in region_counts:
                raise ValueError(f"Unknown region: {region}")
            try:
                json_path = resolve_json_path(row["path"], manifest_path)
                pairs = extract_parallel_pairs(json_path)
            except Exception as exc:
                skipped_files.append({"path": row["path"], "error": str(exc)})
                continue
            if not pairs:
                no_pair_files += 1
                continue

            file_counts[split] += 1
            file_id = hashlib.sha1(str(json_path).encode("utf-8")).hexdigest()[:20]
            for source_text, target_text in pairs:
                changed = normalize_text(source_text) != normalize_text(target_text)
                writers[split].writerow(
                    [
                        source_text,
                        target_text,
                        region,
                        file_id,
                        row["source_group"],
                        int(changed),
                    ]
                )
                corpus_files[split].write(target_text + "\n")
                pair_counts[split] += 1
                changed_counts[split] += int(changed)
                region_counts[region][split] += 1
    finally:
        for f in tsv_files.values():
            f.close()
        for f in corpus_files.values():
            f.close()

    if skipped_files:
        raise RuntimeError(
            f"Failed to read {len(skipped_files)} files. First errors: {skipped_files[:5]}"
        )
    if not all(pair_counts.values()):
        raise RuntimeError(f"At least one split has no parallel pairs: {pair_counts}")

    metadata = {
        "completed": True,
        "split_manifest": str(manifest_path),
        "pair_counts": pair_counts,
        "changed_counts": changed_counts,
        "changed_fraction": {
            split: changed_counts[split] / pair_counts[split]
            for split in pair_counts
        },
        "paired_file_counts": file_counts,
        "region_pair_counts": region_counts,
        "files_without_parallel_pairs": no_pair_files,
        "skipped_file_count": len(skipped_files),
    }
    save_json(output_dir / "preparation_metadata.json", metadata)

    print("\n=== Parallel data preparation complete ===")
    for split in ("train", "validation", "test"):
        print(
            f"{split}: {pair_counts[split]:,} pairs, "
            f"changed={changed_counts[split]:,} "
            f"({metadata['changed_fraction'][split]:.2%})"
        )
    print(f"files without dialect-standard pairs: {no_pair_files:,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare file-level 80/10/10 dialect-standard parallel data."
    )
    parser.add_argument("--split_manifest", default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--output_dir", default="./data")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
