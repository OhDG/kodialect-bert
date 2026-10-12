"""Run the unchanged 260819 experiment using data prepared locally.

Only the original runner's data-preparation stage is skipped. Its tokenizer,
MLM, classifier, resume, GPU-monitoring and summary stages are called unchanged.
The --check-only path imports only Python's standard library.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parent
SERVER_ROOT = ROOT.parent
ORIGINAL_FILES = (
    "experiment_common.py",
    "gpu_monitor.py",
    "requirements_260819.txt",
    "run_full_experiment.py",
    "step1_prepare_data_80_10_10.py",
    "step2_train_dialect_tokenizer.py",
    "step3_pretrain_small_bert_mlm.py",
    "step4_finetune_region_classifier.py",
    "step5_summarize_results.py",
    "step6_changed_subset_analysis.py",
)
DATA_FILES = (
    "preparation_metadata.json",
    "corpus_split_manifest_80_10_10.csv",
    "corpus/dialect_train_corpus.txt",
    "corpus/dialect_validation_corpus.txt",
    "corpus/dialect_test_corpus.txt",
    "region_classification/dialect_region_train.tsv",
    "region_classification/dialect_region_validation.tsv",
    "region_classification/dialect_region_test.tsv",
)
CACHE_KEYS = {"texts", "regions", "changed_clean", "changed_raw", "has_standard"}


class BundleValidationError(RuntimeError):
    """The transferred data bundle is missing or not approved for execution."""


def require_nonempty(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise BundleValidationError(f"Required nonempty file is missing: {path}")


def read_json(path):
    require_nonempty(path)
    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            value = json.load(stream)
    except (OSError, ValueError) as exc:
        raise BundleValidationError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BundleValidationError(f"Expected a JSON object: {path}")
    return value


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_bundle():
    """Validate the local-build completion marker, data contract and originals.

    Full row-by-row data verification is performed by the local build before it
    writes bundle_ready.json. This fast server check does not claim to repeat it.
    """
    marker = read_json(SERVER_ROOT / "bundle_ready.json")
    if marker.get("completed") is not True:
        raise BundleValidationError("Local dataset build is not marked completed.")

    recorded = read_json(SERVER_ROOT / "original_code_sha256.json")
    for name in ORIGINAL_FILES:
        path = ROOT / name
        require_nonempty(path)
        entry = recorded.get(f"region_classification/{name}")
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
            raise BundleValidationError(f"Original-code SHA256 record missing: {name}")
        if sha256_file(path) != entry["sha256"]:
            raise BundleValidationError(f"Original code was changed: {path}")

    for name in DATA_FILES:
        require_nonempty(ROOT / "data" / name)
    metadata = read_json(ROOT / "data" / "preparation_metadata.json")
    if metadata.get("skipped_file_count") != 0:
        raise BundleValidationError("Prepared data must report skipped_file_count=0.")
    if metadata.get("seed") != 42:
        raise BundleValidationError("Prepared split must use seed 42.")
    ratios = metadata.get("ratios", {})
    if not isinstance(ratios, dict):
        raise BundleValidationError("Prepared data ratios are missing.")
    for split, expected in (("train", 0.8), ("validation", 0.1), ("test", 0.1)):
        value = ratios.get(split)
        if not isinstance(value, (int, float)) or not math.isclose(value, expected):
            raise BundleValidationError(f"Prepared {split} ratio must equal {expected}.")
    counts = metadata.get("split_sentence_counts", {})
    if not isinstance(counts, dict) or any(
        not isinstance(counts.get(split), int) or counts[split] <= 0
        for split in ("train", "validation", "test")
    ):
        raise BundleValidationError("Prepared split_sentence_counts must be positive integers.")

    cache_path = ROOT / "results" / "changed_flags_cache.npz"
    require_nonempty(cache_path)
    try:
        with zipfile.ZipFile(cache_path) as archive:
            names = set(archive.namelist())
            if not {f"{key}.npy" for key in CACHE_KEYS}.issubset(names):
                raise BundleValidationError("Changed-subset cache is missing required arrays.")
    except (OSError, zipfile.BadZipFile) as exc:
        raise BundleValidationError(f"Cannot read changed-subset cache: {exc}") from exc
    return metadata


def train():
    """Call the original runner, bypassing only its JSON-preparation subprocess."""
    original_path = ROOT / "run_full_experiment.py"
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("region_original_runner", original_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    original_run_stage = runner.run_stage

    def prepared_data_stage(stage_name, command, *args, **kwargs):
        if stage_name == "01_prepare_data":
            if len(command) < 2 or Path(command[1]).resolve() != runner.DATA_SCRIPT.resolve():
                raise BundleValidationError("Unexpected original data-preparation command.")
            validate_bundle()
            print("[LOCAL DATA] Verified prepared files; source JSON preparation is skipped.", flush=True)
            return None
        return original_run_stage(stage_name, command, *args, **kwargs)

    runner.run_stage = prepared_data_stage
    saved_argv = sys.argv
    sys.argv = [
        str(original_path),
        "--source_manifest",
        str(ROOT / "data" / "corpus_split_manifest_80_10_10.csv"),
    ]
    try:
        runner.main()
    finally:
        sys.argv = saved_argv
    return 0


def analyze():
    """Use the original clean-definition metric logic and the local-built cache."""
    return subprocess.run(
        [sys.executable, str(ROOT / "step6_changed_subset_analysis.py")],
        cwd=ROOT,
        check=False,
    ).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Use the prepared dataset with the unchanged 260819 training conditions."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Validate without ML packages or GPU.")
    mode.add_argument("--analyze-only", action="store_true", help="Analyze saved predictions using the local cache.")
    args = parser.parse_args(argv)
    metadata = validate_bundle()
    print("[OK] Local-build marker, prepared files, changed-subset cache and original-code hashes verified.")
    print(f"[DATA] Input rows: {metadata['split_sentence_counts']}")
    if args.check_only:
        return 0
    if args.analyze_only:
        return analyze()
    return train()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BundleValidationError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
