"""Differential checks against the original extraction/split functions."""

import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

import prepare_data as prep


ORIGINAL_SERVER = prep.PACKAGE_ROOT / "server"


class PrepareDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="base_only_prepare_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package = self.root / "package"
        self.server = self.package / "server"
        for experiment in ("region_classification", "translation"):
            target = self.server / experiment
            target.mkdir(parents=True)
            for source in (ORIGINAL_SERVER / experiment).glob("*.py"):
                if source.name != "run_server.py":
                    shutil.copyfile(source, target / source.name)
        shutil.copyfile(ORIGINAL_SERVER / "original_code_sha256.json", self.server / "original_code_sha256.json")
        self.helpers = prep.load_original_helpers(self.server)
        prep.load_function_asts(
            self.server / "region_classification" / "step1_prepare_data_80_10_10.py",
            ["clean_extra_text", "extract_texts"], self.helpers,
        )
        prep.load_function_asts(
            self.server / "translation" / "step1_prepare_parallel_data.py",
            ["extract_parallel_pairs"], self.helpers,
        )
        prep.load_function_asts(
            self.server / "region_classification" / "step6_changed_subset_analysis.py",
            ["extract_rows"], self.helpers,
        )
        self.old = self.root / "old"
        self.new = self.root / "new"
        self.old.mkdir()
        self.new.mkdir()
        for region in prep.REGION_LABELS:
            source_directory = self.new / f"{prep.PREFIXES[region]}한국어 방언 발화 데이터({region})"
            for index in range(12):
                if region == "전라도":
                    path = self.old / "JSON만_모은폴더_전라도" / f"legacy_{index:03}.json"
                else:
                    split = "1.Training" if index % 2 == 0 else "2.Validation"
                    path = source_directory / "01.데이터" / split / "labels" / f"base_{index:03}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                self.write_source(path, self.utterances(index))
            if region == "전라도":
                validation = source_directory / "01.데이터" / "2.Validation" / "new_validation.json"
                validation.parent.mkdir(parents=True)
                self.write_source(validation, self.utterances(99))
                training = source_directory / "01.데이터" / "1.Training" / "txt_only.txt"
                training.parent.mkdir(parents=True)
                training.write_text("must not be parsed", encoding="utf-8")
        senior = self.new / "139-1.중노년층" / "invalid.json"
        senior.parent.mkdir()
        senior.write_text("invalid JSON excluded by selection", encoding="utf-8")
        unused_old = self.old / "JSON만_모은폴더_강원도" / "unused.json"
        unused_old.parent.mkdir()
        unused_old.write_text("invalid JSON excluded by selection", encoding="utf-8")

    @staticmethod
    def utterances(index):
        return [
            {"dialect_form": f"안녕 (주석) [숨]  하세요! {index}", "standard_form": f"안녕  하세요! {index}"},
            {"dialect_form": "ＡＢＣ １２３", "standard_form": "ABC 123"},
            {"dialect_form": "제주ㆍ어\u200b", "standard_form": "제주어"},
            {"dialect_form": "가\t나\n다\"라\\", "standard_form": "가 나 다\"라\\"},
            {"dialect_form": "...", "standard_form": None},
            {"dialect_form": None, "standard_form": "표준어"},
            {"dialect_form": " ", "standard_form": "표준어"},
            {"dialect_form": "(소리)", "standard_form": "말"},
            {"dialect_form": "말", "standard_form": 42},
            {"dialect_form": "😀", "standard_form": "😀"},
            {"dialect_form": "가", "standard_form": "가"},
            {"dialect_form": "뭐여?", "standard_form": "뭐예요?"},
            {"dialect_form": "그대로", "standard_form": "그대로"},
            {"dialect_form": "값", "standard_form": " "},
            {"dialect_form": "가", "standard_form": "[]"},
            {"standard_form": "방언 표기 없음"},
        ]

    @staticmethod
    def write_source(path, utterances):
        path.write_text(json.dumps({"utterance": utterances}, ensure_ascii=False), encoding="utf-8-sig")

    @staticmethod
    def read_tsv(path):
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.reader(stream, delimiter="\t"))

    def test_selection_and_duplicate_id_fail(self):
        selected = prep.select_sources(self.old, self.new)
        self.assertEqual(len(selected), 60)
        self.assertTrue(all(row["source_type"] == "base" for row in selected))
        self.assertTrue(all(row["origin"] == "old_project1_dataset" for row in selected if row["region"] == "전라도"))
        self.assertEqual(len(prep.select_sources(self.old, self.new, True)), 61)
        duplicate = Path(selected[0]["path"]).parent / "nested" / Path(selected[0]["path"]).name
        duplicate.parent.mkdir()
        shutil.copyfile(selected[0]["path"], duplicate)
        with self.assertRaisesRegex(ValueError, "Duplicate source ID"):
            prep.select_sources(self.old, self.new)

    def test_combined_extraction_matches_original_functions(self):
        rows = prep.select_sources(self.old, self.new)
        prep.attach_counts(rows, self.helpers, {}, workers=2)
        row, classifier, pairs, raw_count = prep.process_source(rows[0], self.helpers)
        path = Path(row["path"])
        self.assertEqual(raw_count, len(self.utterances(0)))
        self.assertEqual([item[0] for item in classifier], self.helpers["extract_texts"](path, "base"))
        self.assertEqual(pairs, self.helpers["extract_parallel_pairs"](path))
        self.assertIn(("ABC 123", "ABC 123"), pairs)
        self.assertNotIn("ABC 123", [item[0] for item in classifier])
        expected = []
        for text, raw_dialect, raw_standard in self.helpers["extract_rows"](path, "base"):
            known = bool(raw_standard.strip())
            expected.append((text, known, known and text != self.helpers["clean_base_text"](raw_standard), known and self.helpers["normalize_text"](raw_dialect) != self.helpers["normalize_text"](raw_standard)))
        self.assertEqual(classifier, expected)

    def test_split_boundary_keeps_whole_crossing_file(self):
        rows = [{"region": "강원도", "source_group": "JSON만_모은폴더_강원도", "num_sentences": "7", "source_id": str(i)} for i in range(10)]
        # 70 rows => target 7 exactly; then deliberately choose 11 files so
        # target round(77*.1)=8, which must be crossed by a second whole file.
        rows.append({**rows[0], "source_id": "10"})
        assigned = self.helpers["assign_file_splits"](rows, .8, .1, .1, 42)
        self.assertEqual([row["split"] for row in assigned], ["test"] * 2 + ["validation"] * 2 + ["train"] * 7)
        self.assertEqual(assigned, self.helpers["assign_file_splits"](rows, .8, .1, .1, 42))

    def test_full_output_matches_original_and_cache_alignment(self):
        report = prep.build_dataset(self.old, self.new, package_root=self.package, workers=3)
        self.assertFalse((self.server / "bundle_ready.json").exists())
        self.assertTrue(report["completed"])
        self.assertEqual(report["total_selected_files"], 60)
        with (self.server / "region_classification/data/corpus_split_manifest_80_10_10.csv").open("r", encoding="utf-8-sig", newline="") as stream:
            manifest = list(csv.DictReader(stream))
        expected_classes, expected_pairs = {split: [] for split in prep.SPLITS}, {split: [] for split in prep.SPLITS}
        expected_cache = {key: [] for key in ("texts", "regions", "has_standard", "changed_clean", "changed_raw")}
        for row in manifest:
            path = Path(row["path"])
            split, region, source_group = row["split"], row["region"], row["source_group"]
            fid = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:20]
            for text in self.helpers["extract_texts"](path, "base"):
                expected_classes[split].append([text, str(prep.LABEL2ID[region]), region, fid, source_group])
            for source, target in self.helpers["extract_parallel_pairs"](path):
                expected_pairs[split].append([source, target, region, fid, source_group, str(int(source != target))])
            if split == "test":
                for text, dialect, standard in self.helpers["extract_rows"](path, "base"):
                    known = bool(standard.strip())
                    expected_cache["texts"].append(text)
                    expected_cache["regions"].append(region)
                    expected_cache["has_standard"].append(known)
                    expected_cache["changed_clean"].append(known and text != self.helpers["clean_base_text"](standard))
                    expected_cache["changed_raw"].append(known and self.helpers["normalize_text"](dialect) != self.helpers["normalize_text"](standard))
        for split in prep.SPLITS:
            cpath = self.server / f"region_classification/data/region_classification/dialect_region_{split}.tsv"
            ppath = self.server / f"translation/data/translation/translation_{split}.tsv"
            self.assertEqual(self.read_tsv(cpath), [prep.CLASS_HEADER] + expected_classes[split])
            self.assertEqual(self.read_tsv(ppath), [prep.PAIR_HEADER] + expected_pairs[split])
            self.assertEqual((self.server / f"region_classification/data/corpus/dialect_{split}_corpus.txt").read_text(encoding="utf-8").splitlines(), [row[0] for row in expected_classes[split]])
            self.assertEqual((self.server / f"translation/data/corpus/standard_{split}_corpus.txt").read_text(encoding="utf-8").splitlines(), [row[1] for row in expected_pairs[split]])
        with np.load(self.server / "region_classification/results/changed_flags_cache.npz", allow_pickle=True) as cache:
            for key, expected in expected_cache.items():
                self.assertEqual(cache[key].tolist(), expected)
                self.assertEqual(cache[key].dtype, np.dtype(object if key in ("texts", "regions") else bool))
        with self.assertRaises(FileExistsError):
            prep.build_dataset(self.old, self.new, package_root=self.package, workers=2)

    def test_audit_reuse_and_hash_or_count_mismatch_fail(self):
        rows = prep.select_sources(self.old, self.new)
        one = rows[0]
        count = len(self.helpers["extract_texts"](Path(one["path"]), "base"))
        audit_dir = self.root / "audit"
        audit_dir.mkdir()
        prep.save_json(audit_dir / "comparison_summary.json", {"normalizer_function": self.helpers["clean_base_text_source"]})
        record = {"new_json": {"path": one["path"], "counts": {"cleaned_rows": count}, "sha256": prep.file_sha256(Path(one["path"]))}}
        audit_path = audit_dir / "per_file_counts.jsonl"
        audit_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        cached, details = prep.load_audit_counts(audit_path, self.helpers)
        self.assertTrue(details["compatible"])
        stats = prep.attach_counts(rows, self.helpers, cached, workers=2)
        self.assertEqual(stats, {"audit_count_files": 1, "fresh_count_files": 59})
        prep.process_source(one, self.helpers)
        one["num_sentences"] = str(count + 1)
        with self.assertRaisesRegex(RuntimeError, "Cleaned count differs"):
            prep.process_source(one, self.helpers)
        one["num_sentences"] = str(count)
        with Path(one["path"]).open("a", encoding="utf-8") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(RuntimeError, "Source JSON changed"):
            prep.process_source(one, self.helpers)
        prep.save_json(audit_dir / "comparison_summary.json", {"normalizer_function": "def clean_base_text(text: str) -> str:\n    return text"})
        cached, details = prep.load_audit_counts(audit_path, self.helpers)
        self.assertEqual(cached, {})
        self.assertFalse(details["compatible"])

    def test_publish_refuses_outside_root_or_existing_target(self):
        source = self.package / "safe_source"
        source.mkdir()
        with self.assertRaises(ValueError):
            prep.publish_path(source, self.root / "outside", self.package)
        target = self.package / "already_exists"
        target.mkdir()
        with self.assertRaises(FileExistsError):
            prep.publish_path(source, target, self.package)


if __name__ == "__main__":
    unittest.main()
