"""Regression tests for frozen splits and common-input dataset construction."""

import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

import prepare_data as prep
from common_preprocessing import process_utterance


ORIGINAL_SERVER = prep.PACKAGE_ROOT / "server"
FIELDS = ["region", "source_group", "source_type", "path", "num_sentences", "source_id", "origin", "official_split", "sha256", "split"]


class FrozenCommonBuildTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="frozen_common_build_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.package = self.root / "new_package"
        self.server = self.package / "server"
        self.local = self.package / "local_prepare"
        self.local.mkdir(parents=True)
        for relative in prep.ORIGINAL_FILES:
            destination = self.server / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ORIGINAL_SERVER / relative, destination)
        shutil.copyfile(ORIGINAL_SERVER / "original_code_sha256.json", self.server / "original_code_sha256.json")
        self.rows = []
        for region in prep.REGION_LABELS:
            for index, split in enumerate(("test", "train", "validation", "train")):
                path = self.root / "untouched_source" / region / f"utterances_{index}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                utterances = self.utterances() if index < 3 else [
                    {"dialect_form": "((추정))/(())", "standard_form": "((추정))/(())"},
                    {"dialect_form": " ", "standard_form": "표준어만 있음"},
                ]
                path.write_text(json.dumps({"utterance": utterances}, ensure_ascii=False), encoding="utf-8-sig")
                self.rows.append({
                    "region": region, "source_group": f"JSON만_모은폴더_{region}",
                    "source_type": "base", "path": str(path), "num_sentences": str(50 + index),
                    "source_id": f"{region}/{path.stem}", "origin": "old_project1_dataset" if region == "전라도" else "new_download",
                    "official_split": "legacy_flattened" if region == "전라도" else ("Training" if index % 2 else "Validation"),
                    "sha256": prep.file_sha256(path), "split": split,
                })
        # A deliberately non-grouped order proves that the builder does not
        # call the old shuffler or reorder regions before writing.
        self.rows = list(reversed(self.rows))
        self.manifest = self.local / "frozen_split_manifest.csv"
        prep.write_manifest(self.manifest, FIELDS, self.rows)

    @staticmethod
    def utterances():
        return [
            {"id": "duplicate", "dialect_form": "(헌디)/(그런데) 그려", "standard_form": "(헌디)/(그런데) 그려"},
            {"id": "duplicate", "dialect_form": "ＡＢＣ\t１２３\n제주ᄒᆞᆫ글", "standard_form": "ABC 123 제주ᄒᆞᆫ글"},
            {"dialect_form": "말 (괄호 내용) [기호] !", "standard_form": "말 (괄호 내용) [기호] !"},
            {"dialect_form": "가\t나\n다\"라\\", "standard_form": "가 나 다\"라\\"},
            {"dialect_form": "((추정))/(())", "standard_form": "확실한 목표"},
            {"dialect_form": "확실한 입력", "standard_form": "((추정))/(())"},
            {"dialect_form": "입력은 남는다", "standard_form": None},
            {"dialect_form": "목표 없는 입력", "standard_form": "  "},
            {"dialect_form": None, "standard_form": "표준어만 있음"},
            {"dialect_form": "", "standard_form": "표준어만 있음"},
            {"dialect_form": "😀 ㆍ ㄱ", "standard_form": "😀 ㆍ ㄱ"},
            {"dialect_form": "동일 문장", "standard_form": "동일 문장"},
            {"dialect_form": "동일 문장", "standard_form": "동일 문장"},
        ]

    @staticmethod
    def read_csv(path, delimiter=","):
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream, delimiter=delimiter))

    def build(self, workers=1):
        return prep.build_dataset(self.manifest, package_root=self.package, workers=workers)

    def test_source_policy_and_raw_vs_canonical_changed(self):
        row = next(row for row in self.rows if row["path"].endswith("utterances_0.json"))
        _, classes, pairs, counts, stats, flags, exclusions = prep.process_file(row)
        self.assertEqual(classes[0][0], "헌디 그려")
        self.assertEqual(pairs[0][0:2], ("헌디 그려", "그런데 그려"))
        self.assertTrue(classes[0][3])  # Canonical changed.
        self.assertFalse(classes[0][4])  # Identical original paired-notation fields.
        self.assertTrue(classes[1][0].startswith("ABC 123 제주"))
        self.assertIn("ᄒᆞᆫ", classes[1][0])
        self.assertTrue(any("(괄호 내용) [기호]" in item[0] for item in classes))
        self.assertTrue(any("😀" in item[0] for item in classes))
        self.assertEqual(counts["raw_utterances"], len(self.utterances()))
        self.assertEqual(counts["excluded_from_all_tasks"], 4)
        self.assertEqual(counts["source_only_rows"], 2)
        self.assertEqual(counts["classification_rows"], counts["parallel_pairs"] + 2)
        self.assertEqual(sum(item[0] == "동일 문장" for item in classes), 2)
        self.assertNotEqual(classes[-1][1], classes[-2][1])

    def test_full_build_preserves_split_order_hashes_and_shared_sources(self):
        input_hash = prep.file_sha256(self.manifest)
        source_hashes = {row["path"]: prep.file_sha256(Path(row["path"])) for row in self.rows}
        report = self.build(workers=2)  # Exercises Windows spawn and ordered results.
        self.assertEqual(prep.file_sha256(self.manifest), input_hash)
        self.assertEqual(source_hashes, {path: prep.file_sha256(Path(path)) for path in source_hashes})
        self.assertTrue(report["file_split_frozen"])
        self.assertTrue(report["original_training_code_unchanged"])
        self.assertFalse((self.server / "bundle_ready.json").exists())
        output_manifest = self.read_csv(self.server / "region_classification/data/corpus_split_manifest_80_10_10.csv")
        self.assertEqual([row["source_id"] for row in output_manifest], [row["source_id"] for row in self.rows])
        for before, after in zip(self.rows, output_manifest):
            for key in FIELDS:
                self.assertEqual(after["original_num_sentences"] if key == "num_sentences" else after[key], before[key])
            expected = prep.process_file(before)
            self.assertEqual(int(after["num_sentences"]), len(expected[1]))
            self.assertEqual(int(after["num_parallel_pairs"]), len(expected[2]))
        zero_rows = [row for row in output_manifest if row["num_sentences"] == "0"]
        self.assertEqual(len(zero_rows), 5)
        self.assertTrue(all(row["split"] == "train" for row in zero_rows))
        self.assertEqual(report["classification"]["zero_cleaned_file_count"], 5)
        self.assertEqual(report["classification"]["split_file_counts"]["train"], 10)

        expected_cache = {key: [] for key in ("texts", "regions", "has_standard", "changed_clean", "changed_raw")}
        for split in prep.SPLITS:
            classes = self.read_csv(self.server / f"region_classification/data/region_classification/dialect_region_{split}.tsv", "\t")
            pairs = self.read_csv(self.server / f"translation/data/translation/translation_{split}.tsv", "\t")
            indexed_classes = {row["utterance_id"]: row for row in classes}
            self.assertEqual(len(classes), len(indexed_classes))
            for pair in pairs:
                self.assertEqual(pair["source_text"], indexed_classes[pair["utterance_id"]]["text"])
                self.assertEqual(pair["file_id"], indexed_classes[pair["utterance_id"]]["file_id"])
                self.assertEqual(pair["is_changed"], str(int(pair["source_text"] != pair["target_text"])))
            expected_classes, expected_pairs = [], []
            for row in self.rows:
                if row["split"] != split:
                    continue
                file_id = prep.file_id_for(row)
                data = prep.read_json(Path(row["path"]))
                for index, utterance in enumerate(data["utterance"]):
                    result = process_utterance(utterance)
                    if result["drop_reason"] is not None or not result["source_text"]:
                        continue
                    sid = f"{file_id}:{index}"
                    source, target = result["source_text"], result["target_text"]
                    expected_classes.append((source, sid))
                    if target:
                        expected_pairs.append((source, target, sid))
                    if split == "test":
                        expected_cache["texts"].append(source)
                        expected_cache["regions"].append(row["region"])
                        expected_cache["has_standard"].append(bool(target))
                        expected_cache["changed_clean"].append(bool(target and source != target))
                        expected_cache["changed_raw"].append(bool(result["changed_raw"]))
            self.assertEqual([(r["text"], r["utterance_id"]) for r in classes], expected_classes)
            self.assertEqual([(r["source_text"], r["target_text"], r["utterance_id"]) for r in pairs], expected_pairs)
            self.assertEqual((self.server / f"region_classification/data/corpus/dialect_{split}_corpus.txt").read_text(encoding="utf-8").splitlines(), [r["text"] for r in classes])
            self.assertEqual((self.server / f"translation/data/corpus/standard_{split}_corpus.txt").read_text(encoding="utf-8").splitlines(), [r["target_text"] for r in pairs])
        with np.load(self.server / "region_classification/results/changed_flags_cache.npz", allow_pickle=True) as cache:
            for key, expected in expected_cache.items():
                self.assertEqual(cache[key].tolist(), expected)
                self.assertEqual(cache[key].dtype, np.dtype(object if key in ("texts", "regions") else bool))
        self.assertEqual(prep.verify_original_code(self.server), report["original_code_sha256"])
        with self.assertRaises(FileExistsError):
            self.build()

    def test_source_hash_and_original_code_integrity_fail(self):
        row = self.rows[0]
        with Path(row["path"]).open("a", encoding="utf-8") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(ValueError, "Source JSON changed"):
            prep.process_file(row)
        code = self.server / "translation/step3_train_dialect_to_standard.py"
        with code.open("a", encoding="utf-8") as stream:
            stream.write("\n# changed\n")
        with self.assertRaisesRegex(ValueError, "Original experiment file was modified"):
            self.build()
        self.assertFalse((self.local / "build_staging").exists())

    def test_frozen_manifest_rejects_duplicate_and_rebuilt_input(self):
        prep.write_manifest(self.manifest, FIELDS, self.rows + [self.rows[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate source"):
            prep.load_frozen_manifest(self.manifest)
        prep.write_manifest(self.manifest, FIELDS + ["original_num_sentences"], [{**row, "original_num_sentences": "1"} for row in self.rows])
        with self.assertRaisesRegex(ValueError, "original frozen manifest"):
            prep.load_frozen_manifest(self.manifest)

    def test_publish_does_not_escape_or_overwrite(self):
        source = self.package / "stage"
        source.mkdir()
        with self.assertRaises(ValueError):
            prep.publish_path(source, self.root / "outside", self.package)
        target = self.package / "existing"
        target.mkdir()
        with self.assertRaises(FileExistsError):
            prep.publish_path(source, target, self.package)


if __name__ == "__main__":
    unittest.main()
