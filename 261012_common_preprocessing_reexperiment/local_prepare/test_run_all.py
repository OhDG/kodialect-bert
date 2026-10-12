"""Workflow routing, provenance and failure tests without any training."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parent.parent / "server" / "run_all.py"
spec = importlib.util.spec_from_file_location("workflow_under_test", MODULE_PATH)
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class RunAllTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="two_python_workflow_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "server"
        self.root.mkdir()
        self.patch = mock.patch.object(workflow, "ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def record(self):
        files = list((self.root / "run_records").glob("*/run_provenance.json"))
        self.assertEqual(len(files), 1)
        return json.loads(files[0].read_text(encoding="utf-8"))

    @staticmethod
    def fake_resolve(value):
        return {"requested": value, "launch_path": f"/envs/{value}/bin/python", "real_path": "/usr/bin/python3"}

    @staticmethod
    def success(command, **kwargs):
        if command[1] == "-c":
            return subprocess.CompletedProcess(command, 0, json.dumps({"executable": command[0], "version": "3.10", "packages": {"sacrebleu": "2.4.2" if "dialect_translate" in command[0] else "2.3.1"}}), "")
        return subprocess.CompletedProcess(command, 0)

    def test_two_environment_routing_and_provenance(self):
        with mock.patch.object(workflow, "resolve_interpreter", side_effect=self.fake_resolve), mock.patch.object(workflow.subprocess, "run", side_effect=self.success) as run:
            self.assertEqual(workflow.main(["--region-python", "dialect1", "--translation-python", "dialect_translate"]), 0)
        record = self.record()
        self.assertTrue(record["completed"])
        self.assertEqual(len(record["interpreter_checks"]), 2)
        self.assertEqual([step["environment"] for step in record["steps"]], ["region", "region", "translation", "region", "region", "translation"])
        self.assertEqual([step["stage"] for step in record["steps"]], ["verify_bundle", "check_region", "check_translation", "train_region", "analyze_region", "train_translation"])
        for step in record["steps"]:
            expected = "/envs/dialect1/bin/python" if step["environment"] == "region" else "/envs/dialect_translate/bin/python"
            self.assertEqual(step["command"][0], expected)
            self.assertEqual(step["status"], "completed")
        self.assertEqual(record["interpreters"]["translation"]["runtime"]["packages"]["sacrebleu"], "2.4.2")
        self.assertEqual(record["source_mlm_root"], str(self.root / "region_classification/outputs/mlm"))
        self.assertEqual(len(run.call_args_list), 8)
        for call in run.call_args_list:
            self.assertEqual(call.kwargs["env"]["PYTHONUTF8"], "1")
            self.assertEqual(call.kwargs["env"]["PYTHONIOENCODING"], "utf-8")

    def test_failure_stops_training_and_is_recorded(self):
        def fail_translation_check(command, **kwargs):
            if Path(command[1]).parts[-2:] == ("translation", "run_server.py") and "--check-only" in command:
                return subprocess.CompletedProcess(command, 7)
            return self.success(command, **kwargs)
        with mock.patch.object(workflow, "resolve_interpreter", side_effect=self.fake_resolve), mock.patch.object(workflow.subprocess, "run", side_effect=fail_translation_check):
            with self.assertRaisesRegex(RuntimeError, "exit code 7"):
                workflow.main(["--region-python", "dialect1", "--translation-python", "dialect_translate"])
        record = self.record()
        self.assertFalse(record["completed"])
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["steps"][2]["status"], "failed")
        self.assertEqual(record["steps"][2]["returncode"], 7)
        self.assertTrue(all(step["status"] == "not_started" for step in record["steps"][3:]))

    def test_invalid_interpreter_stops_before_artifacts(self):
        with self.assertRaises(FileNotFoundError):
            workflow.main(["--region-python", str(self.root / "missing-python")])
        self.assertFalse((self.root / "run_records").exists())
        with self.assertRaises(FileNotFoundError):
            workflow.resolve_interpreter("conda run -n dialect1 python")

    def test_actual_check_only_runs_python_without_training(self):
        script = "import os, sys\nassert os.environ['PYTHONUTF8'] == '1'\nassert os.environ['PYTHONIOENCODING'] == 'utf-8'\n"
        (self.root / "verify_bundle.py").write_text(script, encoding="utf-8")
        for directory in ("region_classification", "translation"):
            path = self.root / directory
            path.mkdir()
            (path / "run_server.py").write_text(script + "assert sys.argv[1:] == ['--check-only']\n", encoding="utf-8")
        self.assertEqual(workflow.main(["--check-only"]), 0)
        record = self.record()
        self.assertEqual(len(record["steps"]), 3)
        self.assertTrue(record["completed"])
        self.assertEqual(record["interpreters"]["region"]["runtime"]["executable"], sys.executable)

    def test_virtualenv_symlink_launch_path_is_preserved(self):
        link = self.root / ("venv-python.exe" if os.name == "nt" else "venv-python")
        try:
            link.symlink_to(sys.executable)
        except OSError:
            self.skipTest("Creating executable symlinks is unavailable on this host.")
        info = workflow.resolve_interpreter(str(link))
        self.assertEqual(info["launch_path"], str(link))
        self.assertEqual(info["real_path"], str(Path(sys.executable).resolve()))


if __name__ == "__main__":
    unittest.main()
