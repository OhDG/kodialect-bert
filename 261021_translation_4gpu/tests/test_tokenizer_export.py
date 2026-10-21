"""Portable tokenizer export tests; normal discovery uses only the stdlib.

The optional --integration-report PATH check uses an existing cached KoBERT
tokenizer, never downloads files, and does not read or modify trained weights.
"""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))
import pack_inputs
import tokenizer_export
from test_inputs import source_fixture, write_json
from verify_inputs import InputError, MANIFESTS, extract_archive, verify_root


def snapshot(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


class LegacyTokenizer:
    """Small fake with the real old save_vocabulary call signature."""
    model_max_length = 128
    padding_side = "right"
    truncation_side = "right"
    all_special_ids = [0, 1, 2, 3, 4]
    special_tokens_map = {"pad_token": "[PAD]", "unk_token": "[UNK]"}

    def __init__(self, path, changed=False, keep_remote=False):
        self.path = Path(path)
        self.changed = changed
        self.keep_remote = keep_remote

    def get_vocab(self):
        return {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3,
                "[MASK]": 8 if self.changed else 4}

    def __call__(self, texts, add_special_tokens=True):
        return {"input_ids": [[2, 1, len(t) % 5, 3] for t in texts]}

    def register_for_auto_class(self, auto_class):
        if auto_class != "AutoTokenizer":
            raise AssertionError(auto_class)

    def save_vocabulary(self, save_directory):
        destination = Path(save_directory) / "vocab.txt"
        destination.write_bytes((self.path / "vocab.txt").read_bytes())
        return (str(destination),)

    def save_pretrained(self, destination):
        destination = Path(destination)
        self.save_vocabulary(destination, filename_prefix=None)
        ref = "tokenization_kobert.KoBertTokenizer"
        if self.keep_remote:
            ref = "monologg/kobert--" + ref
        write_json(destination / "tokenizer_config.json",
                   {"auto_map": {"AutoTokenizer": [ref, None]}})
        (destination / "tokenization_kobert.py").write_text(
            "# Mock custom tokenizer code; not a trained tokenizer.\n", encoding="utf-8")


class TokenizerExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.source = source_fixture(self.base / "old")
        self.kobert = self.source / "region_classification/outputs/mlm/kobert/final_model"
        write_json(self.kobert / "tokenizer_config.json", {"auto_map": {"AutoTokenizer": [
            "monologg/kobert--tokenization_kobert.KoBertTokenizer", None]}})

    def tearDown(self):
        self.temporary.cleanup()

    def fake_transformers(self, changed=False, keep_remote=False, failure=None):
        calls = []

        def load(path, **kwargs):
            calls.append((Path(path), kwargs))
            if failure:
                raise OSError(failure)
            return LegacyTokenizer(path, changed=changed and Path(path) != self.kobert,
                                   keep_remote=keep_remote)

        module = types.ModuleType("transformers")
        module.AutoTokenizer = types.SimpleNamespace(from_pretrained=load)
        return module, calls

    def test_without_opt_in_missing_code_still_fails(self):
        before = snapshot(self.source)
        module, calls = self.fake_transformers()
        with patch.dict(sys.modules, {"transformers": module}), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(InputError, "requires local code"):
                pack_inputs.pack_inputs(self.source, self.base / "models.tar.gz", "models")
            with self.assertRaisesRegex(InputError, "requires local code"):
                pack_inputs.export_directory(self.source, self.base / "shared", "models")
        self.assertFalse((self.base / "shared").exists())
        self.assertFalse((self.base / "models.tar.gz").exists())
        self.assertEqual(calls, [])
        self.assertEqual(snapshot(self.source), before)

    def test_overlay_localizes_references_without_editing_sources_or_weights(self):
        before = snapshot(self.source)
        files = pack_inputs.collect_inputs(self.source, "models")
        module, calls = self.fake_transformers()
        with patch.dict(sys.modules, {"transformers": module}), redirect_stdout(io.StringIO()):
            overlay, report = tokenizer_export.portable_tokenizers(files, self.base / "stage")
        prefix = "models/mlm/kobert/final_model/"
        config = json.loads(overlay[prefix + "tokenizer_config.json"].read_text())
        self.assertEqual(config["auto_map"]["AutoTokenizer"],
                         ["tokenization_kobert.KoBertTokenizer", None])
        self.assertIn(prefix + "tokenization_kobert.py", overlay)
        for name, path in files.items():
            if name.endswith("model.safetensors"):
                self.assertEqual(overlay[name], path)
                self.assertEqual(overlay[name].read_bytes(), b"fixture weights")
        self.assertEqual(snapshot(self.source), before)
        self.assertTrue(report["kobert"]["identity_verified"])
        self.assertFalse(report["kobert"]["source_files_modified"])
        self.assertFalse(report["kobert"]["network_downloads"])
        self.assertEqual(len(calls), 2)
        for _, kwargs in calls:
            self.assertEqual(kwargs, {"use_fast": True, "trust_remote_code": True,
                                      "local_files_only": True})

    def test_changed_token_ids_or_vocabulary_are_rejected(self):
        files = pack_inputs.collect_inputs(self.source, "models")
        module, _ = self.fake_transformers(changed=True)
        with patch.dict(sys.modules, {"transformers": module}), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(InputError, "token IDs changed"):
                tokenizer_export.portable_tokenizers(files, self.base / "stage")

    def test_missing_cache_fails_without_download_fallback(self):
        files = pack_inputs.collect_inputs(self.source, "models")
        module, calls = self.fake_transformers(failure="cached module absent")
        with patch.dict(sys.modules, {"transformers": module}), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(InputError, "local cache.*cached module absent"):
                tokenizer_export.portable_tokenizers(files, self.base / "stage")
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][1]["local_files_only"])

    def test_resaved_remote_reference_is_rejected(self):
        files = pack_inputs.collect_inputs(self.source, "models")
        module, _ = self.fake_transformers(keep_remote=True)
        with patch.dict(sys.modules, {"transformers": module}), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(InputError, "still references remote"):
                tokenizer_export.portable_tokenizers(files, self.base / "stage")

    def test_directory_and_archive_have_identical_verified_input_files(self):
        before = snapshot(self.source)
        module, _ = self.fake_transformers()
        directory, archive = self.base / "shared", self.base / "models.tar.gz"
        with patch.dict(sys.modules, {"transformers": module}), redirect_stdout(io.StringIO()):
            plain_report = pack_inputs.export_directory(self.source, directory, "models", True)
            pack_inputs.pack_inputs(self.source, archive, "models", resolve_tokenizer_code=True)
            extract_archive(archive, self.base / "extracted")
        plain = json.loads((directory / MANIFESTS["models"]).read_text())
        packed = json.loads((self.base / "extracted" / MANIFESTS["models"]).read_text())
        self.assertEqual(plain["files"], packed["files"])
        self.assertEqual(plain["tokenizer_portability"], packed["tokenizer_portability"])
        self.assertEqual(plain_report["components"], ["models"])
        self.assertTrue(verify_root(directory, ("models",))["completed"])
        for name in plain["files"]:
            self.assertEqual((directory / name).read_bytes(),
                             (self.base / "extracted" / name).read_bytes())
        self.assertEqual(snapshot(self.source), before)

    def test_directory_refuses_existing_destination_and_preserves_it(self):
        output = self.base / "shared"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("already present")
        with self.assertRaisesRegex(InputError, "overwrite"):
            pack_inputs.export_directory(self.source, output, "models", True)
        self.assertEqual(marker.read_text(), "already present")

    def test_cli_dispatches_opt_in_and_directory_flag(self):
        with patch.object(pack_inputs, "export_directory", return_value={"completed": True}) as export, \
                redirect_stdout(io.StringIO()):
            code = pack_inputs.main(["--models-only", "--source-server-root", str(self.source),
                "--output-dir", str(self.base / "shared"), "--resolve-tokenizer-code"])
        self.assertEqual(code, 0)
        export.assert_called_once_with(self.source, self.base / "shared", "models", True)

    def test_cli_requires_exactly_one_destination(self):
        common = ["--source-server-root", str(self.source), "--models-only"]
        with redirect_stderr(io.StringIO()):
            for extra in ([], ["--output", "one.tar.gz", "--output-dir", "two"]):
                with self.subTest(extra=extra), self.assertRaises(SystemExit) as raised:
                    pack_inputs.main(common + extra)
                self.assertEqual(raised.exception.code, 2)


def run_cached_kobert_integration(evidence_path):
    """Exercise real cached custom code and fresh-cache offline reload on CPU."""
    import inspect
    import transformers
    from transformers import AutoTokenizer

    with tempfile.TemporaryDirectory(prefix="kobert-export-integration-") as temporary:
        base = Path(temporary)
        source = source_fixture(base / "old")
        directory = source / "region_classification/outputs/mlm/kobert/final_model"
        original = AutoTokenizer.from_pretrained("monologg/kobert", use_fast=True,
            local_files_only=True, trust_remote_code=True)
        # Reproduce the original remote-load/save path, which lacks auto registration.
        original.__class__._auto_class = None
        if "filename_prefix" not in inspect.signature(original.save_vocabulary).parameters:
            save_vocabulary = original.save_vocabulary

            def compatible(_self, save_directory, filename_prefix=None):
                return save_vocabulary(save_directory)

            original.save_vocabulary = types.MethodType(compatible, original)
        original.model_max_length = 128
        original.save_pretrained(directory)
        write_json(directory / "config.json", {"model_type": "bert", "vocab_size": len(original)})
        assert not (directory / "tokenization_kobert.py").exists()
        before = snapshot(source)
        files = pack_inputs.collect_inputs(source, "models")
        overlay, report = tokenizer_export.portable_tokenizers(files, base / "stage")
        prefix = "models/mlm/kobert/final_model/"
        portable = overlay[prefix + "tokenizer_config.json"].parent
        assert snapshot(source) == before
        assert all(overlay[k] == p and p.read_bytes() == b"fixture weights"
                   for k, p in files.items() if k.endswith("model.safetensors"))
        expected = report["kobert"]["identity"]
        env = os.environ.copy()
        env.update(HF_HOME=str(base / "isolated_hf"),
                   HF_HUB_CACHE=str(base / "isolated_hub"),
                   HF_MODULES_CACHE=str(base / "isolated_modules"),
                   TRANSFORMERS_CACHE=str(base / "isolated_transformers"),
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   PYTHONIOENCODING="utf-8")
        child = ("import json,sys; sys.path.insert(0,sys.argv[1]); "
                 "from transformers import AutoTokenizer; "
                 "from tokenizer_export import tokenizer_identity; "
                 "t=AutoTokenizer.from_pretrained(sys.argv[2],use_fast=True,"
                 "local_files_only=True,trust_remote_code=True); "
                 "print(json.dumps({'identity':tokenizer_identity(t),"
                 "'module':t.__class__.__module__,'vocab_paths':[t.vocab_file,t.vocab_txt]},"
                 "ensure_ascii=True))")
        result = subprocess.run([sys.executable, "-c", child, str(HERE.parent / "tools"),
                                 str(portable)], env=env, capture_output=True,
                                text=True, encoding="utf-8", check=True)
        actual = json.loads(result.stdout)
        assert actual["identity"] == expected
        assert all(Path(p).parent == portable for p in actual["vocab_paths"])
        evidence = {"completed": True, "python": sys.version.split()[0],
                    "transformers": transformers.__version__,
                    "network_downloads": False, "source_files_unchanged": True,
                    "dummy_weight_bytes_unchanged": True,
                    "isolated_empty_cache_reload": True,
                    "vocabulary_and_probe_ids_identical": True,
                    "isolated_module": actual["module"],
                    "tokenizer_portability": report,
                    "scope": "real cached tokenizer; dummy fixture model weights; no trained weights loaded"}
        evidence_path = Path(evidence_path)
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8", newline="\n")
        print(f"[PASS] Cached KoBERT portability integration: {evidence_path}")
        return evidence


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--integration-report":
        run_cached_kobert_integration(sys.argv[2])
    else:
        unittest.main()
