import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.pack_tokenizers import (TOKENIZERS, export_archive, import_archive,
                                   safe_name, verify_directory)


def fixture_source(root):
    for name in TOKENIZERS:
        target = Path(root) / name / "final_model"
        target.mkdir(parents=True)
        (target / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "BertTokenizerFast"}), encoding="utf-8")
        (target / "vocab.txt").write_text("[PAD]\n[UNK]\n하나\n둘\n", encoding="utf-8")
        (target / "model.safetensors").write_bytes(b"weights-must-not-transfer")
        (target / "optimizer.pt").write_bytes(b"optimizer-must-not-transfer")


class TransferTests(unittest.TestCase):
    def test_export_import_hashes_and_weights_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_source(root / "source")
            report = export_archive(root / "source", root / "tokens.tar.gz")
            self.assertFalse(report["model_weights_included"])
            import_archive(root / "tokens.tar.gz", root / "imported")
            manifest = verify_directory(root / "imported")
            self.assertEqual(len(manifest["files"]), 8)
            self.assertFalse(any("safetensors" in name or ".pt" in name for name in manifest["files"]))
            (root / "imported/dialect/vocab.txt").write_text("corrupt", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_directory(root / "imported")

    def test_custom_tokenizer_code_is_required_and_bundled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_source(root / "source")
            kobert = root / "source/kobert/final_model"
            (kobert / "tokenizer_config.json").write_text(json.dumps({
                "tokenizer_class": "KoBertTokenizer", "auto_map": {"AutoTokenizer": ["tokenization_kobert.KoBertTokenizer", None]}
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing custom tokenizer code"):
                export_archive(root / "source", root / "tokens.tar.gz")
            (kobert / "tokenization_kobert.py").write_text("from .helper import name\n", encoding="utf-8")
            (kobert / "helper.py").write_text("name = 'local'\n", encoding="utf-8")
            export_archive(root / "source", root / "tokens.tar.gz")
            import_archive(root / "tokens.tar.gz", root / "imported")
            self.assertTrue((root / "imported/kobert/tokenization_kobert.py").is_file())
            self.assertTrue((root / "imported/kobert/helper.py").is_file())

    def test_remote_custom_code_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_source(root / "source")
            target = root / "source/kobert/final_model/tokenizer_config.json"
            target.write_text(json.dumps({"auto_map": {"AutoTokenizer": ["remote/repo--module.Tokenizer", None]}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "remote code"):
                export_archive(root / "source", root / "tokens.tar.gz")

    def test_unsafe_paths_rejected(self):
        for name in ("../escape", "/absolute", "dialect/../../escape", "dialect\\evil", "C:/escape", "dialect//vocab.txt"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_name(name)

    def test_archive_traversal_and_links_never_extract(self):
        for name, type_code in (("../escaped", tarfile.REGTYPE), ("dialect/link", tarfile.SYMTYPE)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                archive = root / "bad.tar.gz"
                with tarfile.open(archive, "w:gz") as handle:
                    member = tarfile.TarInfo(name)
                    member.type = type_code
                    if type_code == tarfile.REGTYPE:
                        member.size = 1
                        handle.addfile(member, io.BytesIO(b"x"))
                    else:
                        member.linkname = "/tmp/escape"
                        handle.addfile(member)
                with self.assertRaises(ValueError):
                    import_archive(archive, root / "imported")
                self.assertFalse((root / "imported").exists())
                self.assertFalse((root / "escaped").exists())

    def test_duplicate_archive_members_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "duplicate.tar.gz"
            with tarfile.open(archive, "w:gz") as handle:
                for _ in range(2):
                    member = tarfile.TarInfo("dialect/vocab.txt")
                    member.size = 1
                    handle.addfile(member, io.BytesIO(b"x"))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                import_archive(archive, root / "imported")

    def test_archive_checksum_corruption_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_source(root / "source")
            archive = root / "tokens.tar.gz"
            export_archive(root / "source", archive)
            archive.write_bytes(archive.read_bytes() + b"corruption")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                import_archive(archive, root / "imported")
            self.assertFalse((root / "imported").exists())


if __name__ == "__main__":
    unittest.main()

