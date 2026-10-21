"""NFS directory imports copy immutable exports without touching their sources."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from pack_inputs import pack_inputs
from verify_inputs import (InputError, extract_archive, import_directory, main,
                           verify_root)
from test_inputs import source_fixture


def snapshot(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


class DirectoryImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.original = source_fixture(self.base / "original")

    def tearDown(self):
        self.temp.cleanup()

    def export(self, scope):
        archive, directory = self.base / f"{scope}.tar.gz", self.base / f"export-{scope}"
        with redirect_stdout(io.StringIO()):
            pack_inputs(self.original, archive, scope)
        extract_archive(archive, directory)
        return directory

    def test_separate_components_preserve_source_and_destination_code(self):
        data, models = self.export("data"), self.export("models")
        source_before = snapshot(data), snapshot(models)
        destination = self.base / "bundle"
        destination.mkdir()
        (destination / "README.md").write_text("keep local code")
        # Source root may contain code, but only the declared input namespaces move.
        (data / "source-only.md").write_text("not an input")
        report = import_directory(data, destination, ("data",))
        self.assertEqual(report["components"], ["data"])
        import_directory(models, destination, ("models",))
        self.assertEqual(verify_root(destination)["components"], ["data", "models"])
        self.assertEqual((destination / "README.md").read_text(), "keep local code")
        self.assertFalse((destination / "source-only.md").exists())
        self.assertEqual({k: v for k, v in snapshot(data).items() if k != "source-only.md"}, source_before[0])
        self.assertEqual(snapshot(models), source_before[1])
        # The imported data are copies: modifying the destination cannot alter NFS source.
        relative = "data/corpus/standard_train_corpus.txt"
        self.assertFalse(os.path.samefile(data / relative, destination / relative))
        (destination / relative).write_text("new content")
        self.assertEqual((data / relative).read_text(), "hello\n")

    def test_full_export(self):
        source = self.export("full")
        result = import_directory(source, self.base / "bundle")
        self.assertEqual(result["components"], ["data", "models"])

    def test_corrupt_source_leaves_no_imported_files(self):
        source = self.export("data")
        (source / "data/corpus/standard_train_corpus.txt").write_text("corrupt")
        destination = self.base / "bundle"
        with self.assertRaisesRegex(InputError, "SHA256/size mismatch"):
            import_directory(source, destination)
        self.assertEqual(snapshot(destination), {})
        self.assertFalse(list(destination.glob(".input-import-*")))

    def test_collision_detected_before_copy(self):
        source, destination = self.export("data"), self.base / "bundle"
        import_directory(source, destination)
        before = snapshot(destination)
        with patch("verify_inputs.hash_stream", side_effect=AssertionError("must not start copying")):
            with self.assertRaisesRegex(InputError, "overwrite"):
                import_directory(source, destination)
        self.assertEqual(snapshot(destination), before)

    def test_same_and_nested_directories_fail_before_copy(self):
        source = self.export("data")
        for destination in (source, source / "child", self.base):
            with self.subTest(destination=destination):
                with patch("verify_inputs.hash_stream", side_effect=AssertionError("must not start copying")):
                    with self.assertRaisesRegex(InputError, "separate, non-nested"):
                        import_directory(source, destination)
        self.assertFalse((source / "child").exists())

    def test_missing_required_component_fails_without_commit(self):
        source, destination = self.export("data"), self.base / "bundle"
        with self.assertRaisesRegex(InputError, "Missing input component"):
            import_directory(source, destination, ("models",))
        self.assertEqual(snapshot(destination), {})

    def test_unmanifested_file_is_rejected(self):
        source, destination = self.export("data"), self.base / "bundle"
        (source / "data/unexpected.txt").write_text("do not ignore")
        with self.assertRaisesRegex(InputError, "inventory mismatch"):
            import_directory(source, destination)
        self.assertEqual(snapshot(destination), {})

    def test_failed_commit_rolls_back_own_files(self):
        source, destination = self.export("data"), self.base / "bundle"
        destination.mkdir()
        (destination / "README.md").write_text("keep")
        original_link, calls = os.link, []

        def fail_after_one(src, dst):
            calls.append(dst)
            if len(calls) == 2:
                raise OSError("simulated link failure")
            return original_link(src, dst)

        with patch("verify_inputs.os.link", side_effect=fail_after_one):
            with self.assertRaisesRegex(OSError, "simulated link failure"):
                import_directory(source, destination)
        self.assertEqual(set(snapshot(destination)), {"README.md"})

    def test_source_symlink_rejected(self):
        source = self.export("data")
        target = self.base / "target.txt"
        target.write_text("outside")
        link = source / "data/linked.txt"
        try:
            link.symlink_to(target)
        except OSError as exc:
            self.skipTest(f"Symlink creation unavailable: {exc}")
        with self.assertRaisesRegex(InputError, "Links/junctions"):
            import_directory(source, self.base / "bundle")

    def test_cli_import_and_required_root(self):
        source, destination = self.export("data"), self.base / "bundle"
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--import-directory", str(source), "--data-only"]), 2)
            self.assertEqual(main(["--import-directory", str(source), "--root", str(destination), "--data-only"]), 0)
        self.assertEqual(verify_root(destination, ("data",))["components"], ["data"])


if __name__ == "__main__":
    unittest.main()
