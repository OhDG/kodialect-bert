"""Transfer integrity/security tests; tiny fixtures do not emulate trained weights."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from pack_inputs import pack_inputs
from verify_inputs import (DATA_PROVENANCE, InputError, TOKENIZERS, extract_archive,
                           inspect_archive, verify_root)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8", newline="\n")


def record(path):
    value = path.read_bytes()
    return {"size_bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}


def source_fixture(root):
    counts, changed = {s: 1 for s in ("train", "validation", "test")}, {s: 0 for s in ("train", "validation", "test")}
    for split in counts:
        path = root / f"translation/data/translation/translation_{split}.tsv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source\ttarget\nhello\thello\n", encoding="utf-8")
        path = root / f"translation/data/corpus/standard_{split}_corpus.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("hello\n", encoding="utf-8")
    write_json(root / "translation/data/preparation_metadata.json",
               {"completed": True, "pair_counts": counts, "changed_counts": changed, "skipped_file_count": 0})
    split_path = root / DATA_PROVENANCE["provenance/data/split_manifest.csv"]
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text("source_id,split,num_parallel_pairs\na,train,1\nb,validation,1\nc,test,1\n", encoding="utf-8")
    files = {p.relative_to(root).as_posix(): record(p) for p in (root / "translation/data").rglob("*") if p.is_file()}
    files[split_path.relative_to(root).as_posix()] = record(split_path)
    write_json(root / "transfer_manifest.json", {"format_version": 1, "files": files})
    write_json(root / "bundle_ready.json", {"completed": True, "transfer_manifest_sha256": record(root / "transfer_manifest.json")["sha256"], "translation_counts": counts})
    write_json(root / "verification_report.json", {"completed": True, "translation": {s: {"pairs": 1, "changed_pairs": 0} for s in counts}})
    for name in TOKENIZERS:
        model = root / "region_classification/outputs/mlm" / name / "final_model"
        write_json(model / "config.json", {"model_type": "bert", "vocab_size": 5})
        write_json(model / "tokenizer_config.json", {"tokenizer_class": "BertTokenizer"})
        (model / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n", encoding="utf-8")
        (model / "model.safetensors").write_bytes(b"fixture weights")
    return root


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.source = source_fixture(self.base / "old")

    def tearDown(self):
        self.temporary.cleanup()

    def pack(self, scope):
        path = self.base / f"{scope}.tar.gz"
        with redirect_stdout(io.StringIO()):
            result = pack_inputs(self.source, path, scope)
        self.assertEqual(result["archive_sha256"], record(path)["sha256"])
        self.assertNotIn(b"\r", Path(str(path) + ".sha256").read_bytes())
        return path

    def test_separate_imports_then_full_verification(self):
        root = self.base / "new"
        root.mkdir()
        (root / "README.md").write_text("code bundle")
        data, models = self.pack("data"), self.pack("models")
        self.assertEqual(inspect_archive(data)[0]["components"], ["data"])
        extract_archive(data, root)
        self.assertEqual(verify_root(root, ("data",))["pair_counts"], {"train": 1, "validation": 1, "test": 1})
        with self.assertRaisesRegex(InputError, "Missing input component"):
            verify_root(root)
        extract_archive(models, root)
        self.assertEqual(verify_root(root)["components"], ["data", "models"])
        self.assertEqual((root / "README.md").read_text(), "code bundle")

    def test_full_archive(self):
        archive = self.pack("full")
        self.assertEqual(inspect_archive(archive)[0]["components"], ["data", "models"])

    def test_refuses_existing_export_and_import(self):
        archive = self.pack("data")
        before = archive.read_bytes()
        with self.assertRaisesRegex(InputError, "overwrite"):
            pack_inputs(self.source, archive, "data")
        self.assertEqual(archive.read_bytes(), before)
        root = self.base / "new"
        extract_archive(archive, root)
        with self.assertRaisesRegex(InputError, "overwrite"):
            extract_archive(archive, root)
        verify_root(root, ("data",))

    def test_modified_data_cannot_be_rebaselined(self):
        (self.source / "translation/data/translation/translation_train.tsv").write_text("changed\n")
        with self.assertRaisesRegex(InputError, "original verified transfer manifest"):
            self.pack("data")
        self.assertFalse((self.base / "data.tar.gz").exists())

    def test_missing_model_and_tokenizer_fail_closed(self):
        directory = self.source / "region_classification/outputs/mlm/kobert/final_model"
        (directory / "vocab.txt").unlink()
        with self.assertRaisesRegex(InputError, "Missing saved tokenizer"):
            self.pack("models")
        self.assertFalse((self.base / "models.tar.gz").exists())
        (directory / "model.safetensors").unlink()
        with self.assertRaisesRegex(InputError, "Missing model weights"):
            self.pack("models")

    def test_remote_tokenizer_code_must_be_saved(self):
        model = self.source / "region_classification/outputs/mlm/kobert/final_model"
        write_json(model / "tokenizer_config.json", {"auto_map": {"AutoTokenizer": ["org/repo--tokenization_kobert.KoBertTokenizer", None]}})
        with self.assertRaisesRegex(InputError, "requires local code"):
            self.pack("models")
        (model / "tokenization_kobert.py").write_text("# saved source code\n")
        self.pack("models")

    def test_missing_shard(self):
        model = self.source / "region_classification/outputs/mlm/dialect/final_model"
        (model / "model.safetensors").unlink()
        write_json(model / "model.safetensors.index.json", {"weight_map": {"a": "model-00001-of-00002.safetensors"}})
        with self.assertRaisesRegex(InputError, "Missing weight shard"):
            self.pack("models")

    def test_installed_corruption(self):
        root = self.base / "new"
        extract_archive(self.pack("data"), root)
        (root / "data/corpus/standard_train_corpus.txt").write_text("corrupt\n")
        with self.assertRaisesRegex(InputError, "SHA256/size mismatch"):
            verify_root(root, ("data",))

    def test_traversal_and_links_and_duplicates(self):
        for name, kind in (("../escape", tarfile.REGTYPE), ("/absolute", tarfile.REGTYPE),
                           ("C:/drive", tarfile.REGTYPE), ("data/link", tarfile.SYMTYPE),
                           ("data/hardlink", tarfile.LNKTYPE)):
            with self.subTest(name=name):
                archive = self.base / "hostile.tar.gz"
                with tarfile.open(archive, "w:gz") as tar:
                    member = tarfile.TarInfo(name)
                    member.type, member.size, member.linkname = kind, 1, "../../escape"
                    tar.addfile(member, io.BytesIO(b"x"))
                with self.assertRaises(InputError):
                    extract_archive(archive, self.base / "hostile-out")
                self.assertFalse((self.base / "escape").exists())
        archive = self.base / "duplicates.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for _ in range(2):
                member = tarfile.TarInfo("data/a")
                member.size = 1
                tar.addfile(member, io.BytesIO(b"x"))
        with self.assertRaisesRegex(InputError, "Duplicate"):
            inspect_archive(archive)

    def test_metadata_counts_mismatch(self):
        write_json(self.source / "verification_report.json", {"completed": True, "translation": {"train": {"pairs": 42}}})
        with self.assertRaisesRegex(InputError, "Source verified counts differ"):
            self.pack("data")


if __name__ == "__main__":
    unittest.main()
