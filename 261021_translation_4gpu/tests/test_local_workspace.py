"""A shared checkout exports only code into a fresh, independent local workspace."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from create_local_workspace import (CODE_PATTERNS, ROOT_FILES, WorkspaceError,
                                    create_workspace)


def write(path, text="fixture\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class LocalWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.source, self.destination = self.base / "shared", self.base / "local"
        for name in ROOT_FILES:
            write(self.source / name)
        for folder, patterns in CODE_PATTERNS.items():
            for pattern in patterns:
                write(self.source / folder / pattern.replace("*", "example"))

    def tearDown(self):
        self.temp.cleanup()

    def test_only_whitelisted_code_copied(self):
        excluded = ("data/train.tsv", "models/model.safetensors", "provenance/manifest.json",
                    "artifacts/inputs.tar.gz", "runs/state.json", "cache/cached.arrow",
                    "smoke/report.json", ".env/bin/python", ".venv/bin/python",
                    "engine/model.bin", "tools/__pycache__/module.pyc",
                    "configs/not_config.txt", "random.py", "tools/nested/hidden.py")
        for name in excluded:
            write(self.source / name)
        report = create_workspace(self.source, self.destination)
        self.assertTrue(report["completed"])
        for name in excluded:
            self.assertFalse((self.destination / name).exists(), name)
            self.assertTrue((self.source / name).exists(), name)
        origin = json.loads((self.destination / "workspace_origin.json").read_text())
        self.assertEqual(origin["source_directory"], str(self.source))
        for name, record in origin["files"].items():
            self.assertEqual((self.destination / name).read_bytes(), (self.source / name).read_bytes())
            self.assertEqual(hashlib.sha256((self.destination / name).read_bytes()).hexdigest(), record["sha256"])
            self.assertFalse(os.path.samefile(self.destination / name, self.source / name))
        self.assertFalse((self.source / "workspace_origin.json").exists())

    def test_existing_empty_destination_refused(self):
        self.destination.mkdir()
        with self.assertRaisesRegex(WorkspaceError, "overwrite existing"):
            create_workspace(self.source, self.destination)
        self.assertEqual(list(self.destination.iterdir()), [])

    def test_inside_source_refused(self):
        target = self.source / "local"
        with self.assertRaisesRegex(WorkspaceError, "outside the source"):
            create_workspace(self.source, target)
        self.assertFalse(target.exists())

    def test_nfs_destination_refused(self):
        with patch("create_local_workspace.filesystem_type", return_value="nfs4"):
            with self.assertRaisesRegex(WorkspaceError, "choose local server storage"):
                create_workspace(self.source, self.destination)
        self.assertFalse(self.destination.exists())

    def test_failure_does_not_publish_or_leave_staging(self):
        with patch("create_local_workspace.shutil.copystat", side_effect=OSError("copy failed")):
            with self.assertRaisesRegex(OSError, "copy failed"):
                create_workspace(self.source, self.destination)
        self.assertFalse(self.destination.exists())
        self.assertFalse(list(self.base.glob(".workspace-import-*")))

    def test_publish_collision_keeps_concurrent_destination(self):
        from create_local_workspace import publish_directory

        def raced(staging, destination):
            destination.mkdir()
            write(destination / "keep.txt", "other creator")
            publish_directory(staging, destination)

        with patch("create_local_workspace.publish_directory", side_effect=raced):
            with self.assertRaises(OSError):
                create_workspace(self.source, self.destination)
        self.assertEqual((self.destination / "keep.txt").read_text(), "other creator")
        self.assertFalse(list(self.base.glob(".workspace-import-*")))

    def test_missing_required_file_does_not_create_destination(self):
        (self.source / "run_parallel.py").unlink()
        with self.assertRaisesRegex(WorkspaceError, "Missing required code file"):
            create_workspace(self.source, self.destination)
        self.assertFalse(self.destination.exists())

    def test_linked_code_refused(self):
        link = self.source / "tools/link.py"
        try:
            link.symlink_to(self.source / "run_parallel.py")
        except OSError as exc:
            self.skipTest(f"Symlink creation unavailable: {exc}")
        with self.assertRaisesRegex(WorkspaceError, "linked code file"):
            create_workspace(self.source, self.destination)
        self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
