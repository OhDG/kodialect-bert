"""Post-hoc changed/identity subset analysis for region classification.

The translation experiment (260811) reports metrics separately for sentences
whose dialect form actually differs from the standard form ("changed") and for
those that are already identical. This script adds the same breakdown to the
region classification results.

No retraining is required. Every classification run already saved its full
per-row test predictions to `test_predictions.npz`, and `Trainer.predict` uses
a sequential sampler with `group_by_length=False`, so those arrays are in the
exact row order of `dialect_region_test.tsv`. This script rebuilds the
`is_changed` flag for each of those rows straight from the source JSON files
and re-slices the saved predictions.

Two definitions of "changed" are computed:

  * `clean`   - compares the two forms after the SAME cleaning the region
                classifier's input went through (bracket removal, character
                whitelist). This is what the model actually saw, so it is the
                primary definition used for the reported subsets.
  * `raw`     - compares the two forms after NFKC + whitespace normalization
                only, matching the definition 260811 used. Reported alongside
                for comparability with the translation experiment.

Unlike 260811, this covers the full test split: single-utterance transcript
files carry `transcription.standard` even though they have no `utterance`
array, so they are not excluded here.
"""

import argparse
import csv
import json
import re
import statistics
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from experiment_common import REGION_LABELS, classification_metrics, save_json


METRICS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
DEFAULT_TOKENIZERS = ["dialect", "klue", "kobert", "mbert"]
DEFAULT_SEEDS = [13, 21, 42, 87, 100]


# --- replicated verbatim from step1_prepare_data_80_10_10.py -----------------

def clean_base_text(text: str) -> str:
    cleaned = re.sub(r"\([^)]*\)|\[[^\]]*\]", "", text)
    return re.sub(r"[^가-힣a-zA-Z0-9.,?! ]", "", cleaned).strip()


def clean_extra_text(text: str) -> str:
    cleaned = re.sub(r"\([^)]*\)|\[[^\]]*\]", "", text)
    cleaned = re.sub(r"[^가-힣a-zA-Z0-9.,?! ]", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


# --- replicated from 260811's experiment_common.normalize_text ---------------

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    return re.sub(r"\s+", " ", text).strip()


def resolve_json_path(raw_path: str, manifest_path: Path, dataset_root: Optional[Path]) -> Path:
    path = Path(raw_path)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(manifest_path.parent / path)
    if dataset_root is not None:
        # Fallback for running against a dataset copy at a different location:
        # keep the trailing "<corpus folder>/<file>.json" and re-root it.
        candidates.append(dataset_root / Path(*path.parts[-2:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"JSON file not found. Checked: {', '.join(str(c) for c in candidates)}"
    )


def extract_rows(json_path: Path, source_type: str) -> List[Tuple[str, str, str]]:
    """Return (classifier_text, raw_dialect, raw_standard) in step1's row order.

    The filtering conditions mirror step1's `extract_texts` exactly so that the
    emitted rows line up one-for-one with `dialect_region_test.tsv`. Where a
    standard form is unavailable the third element is an empty string and the
    row is later marked `unknown`.
    """
    with json_path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    rows: List[Tuple[str, str, str]] = []

    if source_type == "base":
        for utterance in data.get("utterance", []):
            text = utterance.get("dialect_form", "")
            if isinstance(text, str) and text.strip():
                cleaned = clean_base_text(text)
                if cleaned:
                    standard = utterance.get("standard_form", "")
                    if not isinstance(standard, str):
                        standard = ""
                    rows.append((cleaned, text, standard))
        return rows

    if source_type == "extra":
        transcription = data.get("transcription", {})
        text = transcription.get("dialect", "")
        if isinstance(text, str) and text.strip():
            cleaned = clean_extra_text(text)
            if cleaned:
                standard = transcription.get("standard", "")
                if not isinstance(standard, str):
                    standard = ""
                rows.append((cleaned, text, standard))
        return rows

    raise ValueError(f"Unsupported source_type: {source_type}")


def rebuild_changed_flags(
    manifest_path: Path,
    dataset_root: Optional[Path],
) -> Dict[str, np.ndarray]:
    """Rebuild per-test-row changed flags in `dialect_region_test.tsv` order."""
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        manifest_rows = [row for row in csv.DictReader(f) if row["split"] == "test"]

    texts: List[str] = []
    changed_clean: List[int] = []
    changed_raw: List[int] = []
    has_standard: List[int] = []
    regions: List[str] = []

    total = len(manifest_rows)
    for index, row in enumerate(manifest_rows, start=1):
        json_path = resolve_json_path(row["path"], manifest_path, dataset_root)
        for cleaned, raw_dialect, raw_standard in extract_rows(json_path, row["source_type"]):
            texts.append(cleaned)
            regions.append(row["region"])
            if raw_standard.strip():
                has_standard.append(1)
                cleaner = clean_base_text if row["source_type"] == "base" else clean_extra_text
                changed_clean.append(int(cleaned != cleaner(raw_standard)))
                changed_raw.append(
                    int(normalize_text(raw_dialect) != normalize_text(raw_standard))
                )
            else:
                has_standard.append(0)
                changed_clean.append(0)
                changed_raw.append(0)
        if index % 5000 == 0 or index == total:
            print(f"  [{index:,}/{total:,}] files read, {len(texts):,} rows", flush=True)

    return {
        "texts": np.array(texts, dtype=object),
        "regions": np.array(regions, dtype=object),
        "changed_clean": np.array(changed_clean, dtype=bool),
        "changed_raw": np.array(changed_raw, dtype=bool),
        "has_standard": np.array(has_standard, dtype=bool),
    }


def load_test_tsv(tsv_path: Path) -> Tuple[List[str], np.ndarray]:
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    texts: List[str] = []
    labels: List[int] = []
    with tsv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        text_index = header.index("text")
        label_index = header.index("label")
        for row in reader:
            texts.append(row[text_index])
            labels.append(int(row[label_index]))
    return texts, np.array(labels, dtype=np.int64)


def verify_alignment(rebuilt_texts: np.ndarray, tsv_texts: List[str]) -> None:
    if len(rebuilt_texts) != len(tsv_texts):
        raise RuntimeError(
            f"Row count mismatch: rebuilt {len(rebuilt_texts):,} vs TSV {len(tsv_texts):,}. "
            "The manifest or source data does not match the one used to build the TSV."
        )
    mismatches = [
        index
        for index in range(len(tsv_texts))
        if rebuilt_texts[index] != tsv_texts[index]
    ]
    if mismatches:
        first = mismatches[0]
        raise RuntimeError(
            f"Text mismatch at {len(mismatches):,} of {len(tsv_texts):,} rows. "
            f"First at index {first}:\n"
            f"  rebuilt: {rebuilt_texts[first]!r}\n"
            f"  tsv    : {tsv_texts[first]!r}"
        )
    print(f"[OK] Alignment verified: {len(tsv_texts):,} rows match exactly.")


def subset_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    return classification_metrics(labels[mask], predictions[mask])


def mean_std(values: List[float]) -> Dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def analyze(args: argparse.Namespace) -> None:
    manifest_path = Path(args.split_manifest).resolve()
    tsv_path = Path(args.test_tsv).resolve()
    outputs_root = Path(args.outputs_root).resolve()
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else None

    cache_path = result_dir / "changed_flags_cache.npz"
    if cache_path.is_file() and not args.rebuild:
        print(f"[CACHE] Loading rebuilt changed flags: {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        flags = {key: cached[key] for key in cached.files}
    else:
        print(f"[REBUILD] Reading source JSON files listed in {manifest_path}")
        flags = rebuild_changed_flags(manifest_path, dataset_root)
        np.savez_compressed(cache_path, **flags)
        print(f"[CACHE] Saved: {cache_path}")

    tsv_texts, tsv_labels = load_test_tsv(tsv_path)
    verify_alignment(flags["texts"], tsv_texts)

    changed = flags["changed_clean"] if args.definition == "clean" else flags["changed_raw"]
    has_standard = flags["has_standard"]

    # Rows without any standard form cannot be classified as changed/identity.
    known = has_standard
    changed_mask = changed & known
    identity_mask = (~changed) & known

    composition = {
        "definition": args.definition,
        "total_rows": int(len(tsv_labels)),
        "rows_with_standard_form": int(known.sum()),
        "rows_without_standard_form": int((~known).sum()),
        "changed_rows": int(changed_mask.sum()),
        "identity_rows": int(identity_mask.sum()),
        "changed_fraction_of_known": float(changed_mask.sum() / known.sum()),
        "changed_fraction_clean": float(
            (flags["changed_clean"] & known).sum() / known.sum()
        ),
        "changed_fraction_raw": float((flags["changed_raw"] & known).sum() / known.sum()),
    }
    print("\n=== Test split composition ===")
    for key, value in composition.items():
        print(f"{key}: {value}")

    per_region_composition = {}
    for region in REGION_LABELS:
        region_mask = flags["regions"] == region
        region_known = region_mask & known
        if region_known.sum():
            per_region_composition[region] = {
                "rows": int(region_mask.sum()),
                "rows_with_standard_form": int(region_known.sum()),
                "changed_rows": int((changed & region_known).sum()),
                "changed_fraction": float(
                    (changed & region_known).sum() / region_known.sum()
                ),
            }

    results: Dict[str, object] = {
        "composition": composition,
        "per_region_composition": per_region_composition,
        "seeds": args.seeds,
        "tokenizers": {},
    }

    for tokenizer_name in args.tokenizers:
        per_seed = {"overall": [], "changed": [], "identity": []}
        per_seed_region_f1 = {"changed": [], "identity": []}
        for seed in args.seeds:
            npz_path = (
                outputs_root
                / "classifiers"
                / tokenizer_name
                / f"seed_{seed}"
                / "test_predictions.npz"
            )
            if not npz_path.is_file():
                raise FileNotFoundError(f"Missing saved predictions: {npz_path}")
            saved = np.load(npz_path)
            labels = saved["labels"].astype(np.int64)
            predictions = saved["predictions"].astype(np.int64)
            if len(labels) != len(tsv_labels):
                raise RuntimeError(
                    f"{npz_path}: prediction count {len(labels):,} != TSV rows {len(tsv_labels):,}"
                )
            if not np.array_equal(labels, tsv_labels):
                raise RuntimeError(
                    f"{npz_path}: saved labels do not match the TSV label column; "
                    "row order cannot be assumed."
                )

            all_mask = np.ones(len(labels), dtype=bool)
            for name, mask in (
                ("overall", all_mask),
                ("changed", changed_mask),
                ("identity", identity_mask),
            ):
                metrics, report = subset_metrics(labels, predictions, mask)
                per_seed[name].append(metrics)
                if name in per_seed_region_f1:
                    per_seed_region_f1[name].append(
                        {
                            region: float(report["per_label"][region]["f1"])
                            for region in REGION_LABELS
                        }
                    )

        summary = {
            name: {
                metric: mean_std([item[metric] for item in per_seed[name]])
                for metric in METRICS
            }
            for name in per_seed
        }
        region_summary = {
            name: {
                region: mean_std([item[region] for item in per_seed_region_f1[name]])
                for region in REGION_LABELS
            }
            for name in per_seed_region_f1
        }
        results["tokenizers"][tokenizer_name] = {
            "summary": summary,
            "per_region_f1": region_summary,
            "per_seed": per_seed,
        }
        print(
            f"[OK] {tokenizer_name}: "
            f"overall macro_f1 {summary['overall']['macro_f1']['mean']:.6f}, "
            f"changed {summary['changed']['macro_f1']['mean']:.6f}, "
            f"identity {summary['identity']['macro_f1']['mean']:.6f}"
        )

    if "dialect" in results["tokenizers"]:
        paired = {}
        for baseline in args.tokenizers:
            if baseline == "dialect":
                continue
            paired[baseline] = {}
            for name in ("overall", "changed", "identity"):
                paired[baseline][name] = {
                    metric: mean_std(
                        [
                            results["tokenizers"]["dialect"]["per_seed"][name][index][metric]
                            - results["tokenizers"][baseline]["per_seed"][name][index][metric]
                            for index in range(len(args.seeds))
                        ]
                    )
                    for metric in METRICS
                }
        results["dialect_minus_baseline"] = paired

    save_json(result_dir / "changed_subset_results.json", results)
    write_markdown(results, result_dir / "changed_subset_results.md", args)
    print(f"\n[OK] Written: {result_dir / 'changed_subset_results.md'}")


def write_markdown(results: Dict[str, object], path: Path, args: argparse.Namespace) -> None:
    composition = results["composition"]
    lines = [
        "# Region classification: changed vs identity subset analysis",
        "",
        "Post-hoc re-slicing of the saved per-row test predictions. No model was retrained.",
        "",
        "## Test split composition",
        "",
        f"- Definition used: `{composition['definition']}` "
        f"({'compares the cleaned text the classifier actually received' if composition['definition'] == 'clean' else 'NFKC + whitespace normalization only, matching 260811'})",
        f"- Total test rows: {composition['total_rows']:,}",
        f"- Rows with a standard form available: {composition['rows_with_standard_form']:,}",
        f"- Rows without a standard form (excluded from subsets): {composition['rows_without_standard_form']:,}",
        f"- Changed rows: {composition['changed_rows']:,} "
        f"({composition['changed_fraction_of_known']:.2%} of rows with a standard form)",
        f"- Identity rows: {composition['identity_rows']:,}",
        "",
        f"Changed fraction under each definition — clean: {composition['changed_fraction_clean']:.2%}, "
        f"raw: {composition['changed_fraction_raw']:.2%}",
        "",
        "### Per-region composition",
        "",
        "| Region | Test rows | With standard form | Changed rows | Changed fraction |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for region, stats in results["per_region_composition"].items():
        lines.append(
            f"| {region} | {stats['rows']:,} | {stats['rows_with_standard_form']:,} | "
            f"{stats['changed_rows']:,} | {stats['changed_fraction']:.2%} |"
        )

    for name, title in (
        ("overall", "All test sentences (sanity check — should match final_results.md)"),
        ("changed", "Changed sentences only"),
        ("identity", "Identity sentences only"),
    ):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Tokenizer | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for tokenizer_name, tokenizer_result in results["tokenizers"].items():
            cells = []
            for metric in METRICS:
                value = tokenizer_result["summary"][name][metric]
                cells.append(f"{value['mean']:.6f} +/- {value['std']:.6f}")
            lines.append(f"| {tokenizer_name} | " + " | ".join(cells) + " |")

    if "dialect_minus_baseline" in results:
        for name, title in (
            ("overall", "all sentences"),
            ("changed", "changed sentences only"),
            ("identity", "identity sentences only"),
        ):
            lines.extend(
                [
                    "",
                    f"## Paired improvement: dialect minus baseline ({title})",
                    "",
                    "| Baseline | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |",
                    "| --- | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for baseline, baseline_result in results["dialect_minus_baseline"].items():
                cells = []
                for metric in METRICS:
                    value = baseline_result[name][metric]
                    cells.append(f"{value['mean']:+.6f} +/- {value['std']:.6f}")
                lines.append(f"| {baseline} | " + " | ".join(cells) + " |")

    lines.extend(["", "## Per-region F1 (changed sentences only)", ""])
    lines.append("| Tokenizer | " + " | ".join(REGION_LABELS) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in REGION_LABELS) + " |")
    for tokenizer_name, tokenizer_result in results["tokenizers"].items():
        cells = [
            f"{tokenizer_result['per_region_f1']['changed'][region]['mean']:.6f}"
            for region in REGION_LABELS
        ]
        lines.append(f"| {tokenizer_name} | " + " | ".join(cells) + " |")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Changed/identity subset breakdown of saved region classification predictions."
    )
    parser.add_argument("--split_manifest", default="./data/corpus_split_manifest_80_10_10.csv")
    parser.add_argument("--test_tsv", default="./data/region_classification/dialect_region_test.tsv")
    parser.add_argument("--outputs_root", default="./outputs")
    parser.add_argument("--result_dir", default="./results")
    parser.add_argument(
        "--dataset_root",
        default=None,
        help="Optional fallback root for the source JSON corpus if the manifest's "
        "relative paths do not resolve (re-roots the trailing folder/file).",
    )
    parser.add_argument(
        "--definition",
        choices=["clean", "raw"],
        default="clean",
        help="clean: compare after the classifier's own text cleaning (default). "
        "raw: NFKC + whitespace only, matching the 260811 translation experiment.",
    )
    parser.add_argument("--tokenizers", nargs="+", default=DEFAULT_TOKENIZERS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Ignore the cached changed-flag file and re-read every source JSON.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
