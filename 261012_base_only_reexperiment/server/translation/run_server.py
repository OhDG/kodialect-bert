"""Run the unchanged translation experiment using locally prepared data only."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SERVER_ROOT = ROOT.parent
SPLITS = ("train", "validation", "test")
TOKENIZERS = ("dialect", "klue", "kobert", "mbert")
TSV_FIELDS = (
    "source_text", "target_text", "region", "file_id", "source_group", "is_changed"
)
ORIGINAL_TRANSLATION_FILES = {
    "experiment_common.py",
    "process_gpu_monitor.py",
    "requirements_260811.txt",
    "run_full_experiment.py",
    "step1_prepare_parallel_data.py",
    "step2_pretrain_standard_decoder_clm.py",
    "step3_train_dialect_to_standard.py",
    "step4_summarize_results.py",
}


def read_json(path):
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def require_nonempty_file(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Required file is missing or empty: {path}")


def verify_original_code():
    index_path = SERVER_ROOT / "original_code_sha256.json"
    require_nonempty_file(index_path)
    records = read_json(index_path)
    required = {f"translation/{name}" for name in ORIGINAL_TRANSLATION_FILES}
    if not isinstance(records, dict) or not required.issubset(records):
        raise RuntimeError("The original-code hash index is incomplete for translation.")
    for relative, record in records.items():
        path = (SERVER_ROOT / relative).resolve()
        if not path.is_relative_to(SERVER_ROOT.resolve()):
            raise RuntimeError(f"Invalid path in original-code hash index: {relative}")
        require_nonempty_file(path)
        expected = record.get("sha256") if isinstance(record, dict) else None
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Copied original code differs from the recorded source: {relative}")
    print(f"[OK] Original-code SHA256 verification: {len(records)} files")


def verify_prepared_data():
    ready_path = SERVER_ROOT / "bundle_ready.json"
    require_nonempty_file(ready_path)
    ready = read_json(ready_path)
    if not isinstance(ready, dict) or ready.get("completed") is not True:
        raise RuntimeError("The local data bundle has not completed preparation.")

    data_root = ROOT / "data"
    metadata_path = data_root / "preparation_metadata.json"
    require_nonempty_file(metadata_path)
    metadata = read_json(metadata_path)
    if not isinstance(metadata, dict) or metadata.get("completed") is not True:
        raise RuntimeError("Parallel-data preparation metadata is not complete.")
    for split in SPLITS:
        count = metadata.get("pair_counts", {}).get(split)
        changed = metadata.get("changed_counts", {}).get(split)
        if not isinstance(count, int) or count <= 0:
            raise RuntimeError(f"Missing or invalid prepared pair count: {split}")
        if not isinstance(changed, int) or not 0 <= changed <= count:
            raise RuntimeError(f"Missing or invalid changed-pair count: {split}")
        tsv_path = data_root / "translation" / f"translation_{split}.tsv"
        corpus_path = data_root / "corpus" / f"standard_{split}_corpus.txt"
        require_nonempty_file(tsv_path)
        require_nonempty_file(corpus_path)
        with tsv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, delimiter="\t")
            if next(reader, None) != list(TSV_FIELDS):
                raise RuntimeError(f"Unexpected prepared TSV header: {tsv_path}")
            first = next(reader, None)
            if first is None or len(first) != len(TSV_FIELDS):
                raise RuntimeError(f"Prepared TSV has no valid first data row: {tsv_path}")
        print(f"[OK] Prepared {split}: {count:,} pairs; changed={changed:,}")

    manifest = SERVER_ROOT / "region_classification" / "data" / "corpus_split_manifest_80_10_10.csv"
    require_nonempty_file(manifest)
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"region", "source_group", "source_type", "path", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError(f"Invalid split-manifest header: {manifest}")
        seen_splits = set()
        for row in reader:
            if row["source_type"] != "base" or row["split"] not in SPLITS:
                raise RuntimeError("The manifest must contain base data and train/validation/test only.")
            seen_splits.add(row["split"])
        if seen_splits != set(SPLITS):
            raise RuntimeError("At least one split is missing from the prepared manifest.")
    # The manifest and metadata retain local Windows source paths for provenance.
    # They are never opened as source JSON paths on the server.
    return manifest


def missing_encoders(source_root):
    pending = []
    for tokenizer in TOKENIZERS:
        model_dir = source_root / tokenizer / "final_model"
        config = model_dir / "config.json"
        weight_files = (
            model_dir / "model.safetensors",
            model_dir / "pytorch_model.bin",
        )
        index_files = (
            model_dir / "model.safetensors.index.json",
            model_dir / "pytorch_model.bin.index.json",
        )
        has_weights = any(path.is_file() and path.stat().st_size > 0 for path in weight_files)
        if not has_weights:
            for index_path in index_files:
                if not index_path.is_file():
                    continue
                shards = set(read_json(index_path).get("weight_map", {}).values())
                if shards and all(
                    (model_dir / shard).is_file() and (model_dir / shard).stat().st_size > 0
                    for shard in shards
                ):
                    has_weights = True
                    break
        if not config.is_file() or config.stat().st_size == 0 or not has_weights:
            pending.append(tokenizer)
    return pending


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the unchanged 260811 translation experiment on the prepared base-only corpus.",
        epilog=("Only --check-only is accepted. --overwrite, --smoke, data preparation, "
                "training overrides, and alternate MLM roots are intentionally unavailable."),
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Validate the prepared bundle and original code using only the Python standard library.",
    )
    args = parser.parse_args(argv)
    verify_original_code()
    manifest = verify_prepared_data()
    source_root = SERVER_ROOT / "region_classification" / "outputs" / "mlm"
    pending = missing_encoders(source_root)
    if pending:
        message = "New-corpus MLM encoders are pending: " + ", ".join(pending)
        if args.check_only:
            print(f"[PENDING] {message}; run region_classification/run_server.py first.")
        else:
            raise RuntimeError(message + "; run region_classification/run_server.py first.")
    else:
        print("[OK] All four new-corpus MLM encoders are present")
    if args.check_only:
        print("[OK] Prepared bundle and code checks passed; no training was started.")
        return 0

    command = [
        sys.executable,
        str(ROOT / "run_full_experiment.py"),
        "--split_manifest", str(manifest),
        "--source_mlm_root", str(source_root),
    ]
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    print("[RUN] Original runner; prepared data is reused and all training defaults are unchanged.", flush=True)
    return subprocess.call(command, cwd=ROOT, env=environment)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, csv.Error) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
