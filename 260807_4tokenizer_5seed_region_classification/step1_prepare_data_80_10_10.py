import argparse
import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from experiment_common import LABEL2ID, REGION_LABELS, save_json


DEFAULT_SOURCE_MANIFEST = "../260630_test_1/corpus_split_manifest.csv"
DEFAULT_OUTPUT_DIR = "./data"


def clean_base_text(text: str) -> str:
    cleaned = re.sub(r"\([^)]*\)|\[[^\]]*\]", "", text)
    return re.sub(r"[^가-힣a-zA-Z0-9.,?! ]", "", cleaned).strip()


def clean_extra_text(text: str) -> str:
    cleaned = re.sub(r"\([^)]*\)|\[[^\]]*\]", "", text)
    cleaned = re.sub(r"[^가-힣a-zA-Z0-9.,?! ]", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_texts(file_path: Path, source_type: str) -> List[str]:
    with file_path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if source_type == "base":
        texts = []
        for utterance in data.get("utterance", []):
            text = utterance.get("dialect_form", "")
            if isinstance(text, str) and text.strip():
                cleaned = clean_base_text(text)
                if cleaned:
                    texts.append(cleaned)
        return texts

    if source_type == "extra":
        text = data.get("transcription", {}).get("dialect", "")
        if isinstance(text, str) and text.strip():
            cleaned = clean_extra_text(text)
            return [cleaned] if cleaned else []
        return []

    raise ValueError(f"Unsupported source_type: {source_type}")


def load_source_manifest(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Source manifest not found: {path}\n"
            "Run from /data/ohdg/kodialect-bert/260807_4tokenizer_5seed_region_classification "
            "or pass --source_manifest."
        )
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"region", "source_group", "source_type", "path", "num_sentences"}
    missing = required - set(rows[0].keys()) if rows else required
    if missing:
        raise ValueError(f"Source manifest is missing columns: {sorted(missing)}")
    return rows


def stable_group_seed(seed: int, key: Tuple[str, str]) -> int:
    digest = hashlib.sha256(f"{seed}:{key[0]}:{key[1]}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def assign_file_splits(
    rows: List[Dict[str, str]],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> List[Dict[str, str]]:
    if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("train_ratio + validation_ratio + test_ratio must equal 1.0")

    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        region = row["region"]
        if region not in LABEL2ID:
            raise ValueError(f"Unknown region in manifest: {region}")
        grouped[(region, row["source_group"])].append(dict(row))

    assigned: List[Dict[str, str]] = []
    for key in sorted(grouped):
        group_rows = grouped[key]
        rng = random.Random(stable_group_seed(seed, key))
        rng.shuffle(group_rows)
        total_sentences = sum(int(row["num_sentences"]) for row in group_rows)
        target_test = round(total_sentences * test_ratio)
        target_validation = round(total_sentences * validation_ratio)
        test_count = 0
        validation_count = 0

        for row in group_rows:
            sentence_count = int(row["num_sentences"])
            if test_count < target_test:
                split = "test"
                test_count += sentence_count
            elif validation_count < target_validation:
                split = "validation"
                validation_count += sentence_count
            else:
                split = "train"
            row["split"] = split
            assigned.append(row)

        print(
            f"[{key[0]} | {key[1]}] files={len(group_rows):,}, sentences={total_sentences:,}, "
            f"validation={validation_count:,}, test={test_count:,}"
        )
    return assigned


def resolve_json_path(raw_path: str, manifest_path: Path) -> Path:
    path = Path(raw_path)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(manifest_path.parent / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"JSON file not found. Checked: {', '.join(str(p) for p in candidates)}")


def save_split_manifest(rows: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_experiment_data(
    rows: List[Dict[str, str]],
    source_manifest: Path,
    output_dir: Path,
) -> Dict[str, object]:
    corpus_dir = output_dir / "corpus"
    classification_dir = output_dir / "region_classification"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    classification_dir.mkdir(parents=True, exist_ok=True)

    corpus_paths = {split: corpus_dir / f"dialect_{split}_corpus.txt" for split in ("train", "validation", "test")}
    tsv_paths = {
        split: classification_dir / f"dialect_region_{split}.tsv"
        for split in ("train", "validation", "test")
    }
    corpus_files = {split: path.open("w", encoding="utf-8") for split, path in corpus_paths.items()}
    tsv_files = {split: path.open("w", encoding="utf-8-sig", newline="") for split, path in tsv_paths.items()}
    tsv_writers = {split: csv.writer(file, delimiter="\t") for split, file in tsv_files.items()}
    for writer in tsv_writers.values():
        writer.writerow(["text", "label", "region", "file_id", "source_group"])

    split_counts = {split: 0 for split in ("train", "validation", "test")}
    split_files = {split: 0 for split in split_counts}
    region_counts = {
        region: {split: 0 for split in split_counts}
        for region in REGION_LABELS
    }
    skipped_files = []

    try:
        for row in tqdm(rows, desc="Writing 80/10/10 corpus and TSV files"):
            split = row["split"]
            region = row["region"]
            try:
                json_path = resolve_json_path(row["path"], source_manifest)
                texts = extract_texts(json_path, row["source_type"])
            except Exception as exc:
                skipped_files.append({"path": row["path"], "error": str(exc)})
                print(f"\n[WARN] Skipping {row['path']}: {exc}")
                continue

            file_id = hashlib.sha1(str(json_path).encode("utf-8")).hexdigest()[:20]
            split_files[split] += 1
            for text in texts:
                corpus_files[split].write(text + "\n")
                tsv_writers[split].writerow(
                    [text, LABEL2ID[region], region, file_id, row["source_group"]]
                )
                split_counts[split] += 1
                region_counts[region][split] += 1
    finally:
        for file in corpus_files.values():
            file.close()
        for file in tsv_files.values():
            file.close()

    return {
        "split_sentence_counts": split_counts,
        "split_file_counts": split_files,
        "region_sentence_counts": region_counts,
        "skipped_file_count": len(skipped_files),
        "skipped_files": skipped_files[:100],
        "corpus_paths": {key: str(value) for key, value in corpus_paths.items()},
        "tsv_paths": {key: str(value) for key, value in tsv_paths.items()},
    }


def outputs_complete(output_dir: Path) -> bool:
    required = [
        output_dir / "preparation_metadata.json",
        output_dir / "corpus_split_manifest_80_10_10.csv",
        output_dir / "corpus" / "dialect_train_corpus.txt",
        output_dir / "corpus" / "dialect_validation_corpus.txt",
        output_dir / "corpus" / "dialect_test_corpus.txt",
        output_dir / "region_classification" / "dialect_region_train.tsv",
        output_dir / "region_classification" / "dialect_region_validation.tsv",
        output_dir / "region_classification" / "dialect_region_test.tsv",
    ]
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def prepare_data(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    if outputs_complete(output_dir) and not args.overwrite:
        print(f"[SKIP] Prepared 80/10/10 data already exists: {output_dir}")
        return

    source_manifest = Path(args.source_manifest)
    rows = load_source_manifest(source_manifest)
    assigned = assign_file_splits(
        rows,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = output_dir / "corpus_split_manifest_80_10_10.csv"
    save_split_manifest(assigned, split_manifest)
    metadata = write_experiment_data(assigned, source_manifest, output_dir)
    metadata.update(
        {
            "source_manifest": str(source_manifest),
            "split_manifest": str(split_manifest),
            "ratios": {
                "train": args.train_ratio,
                "validation": args.validation_ratio,
                "test": args.test_ratio,
            },
            "seed": args.seed,
        }
    )
    save_json(output_dir / "preparation_metadata.json", metadata)

    print("\n=== Data preparation complete ===")
    for split, count in metadata["split_sentence_counts"].items():
        print(f"{split}: {count:,} sentences")
    if metadata["skipped_file_count"]:
        raise RuntimeError(
            f"Data preparation skipped {metadata['skipped_file_count']} files. "
            "Review preparation_metadata.json before training."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a leakage-safe file-level 80/10/10 split.")
    parser.add_argument("--source_manifest", default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--validation_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    prepare_data(parse_args())

