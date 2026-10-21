"""Smoke-run safety guarantees, runnable without installing torch."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "smoke_test.py"
spec = importlib.util.spec_from_file_location("bundle_smoke", MODULE_PATH)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SmokeSafetyTests(unittest.TestCase):
    def fake_torch(self, available=False):
        return types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: available),
                                     version=types.SimpleNamespace(cuda="11.8"))

    def test_gpu_check_never_falls_back_to_cpu(self):
        with patch.dict(sys.modules, {"torch": self.fake_torch()}), \
             patch.object(smoke.importlib.metadata, "version", side_effect=smoke.EXPECTED.__getitem__), \
             patch.object(smoke.sys, "version_info", (3, 10)), \
             patch.object(smoke.subprocess, "run", side_effect=FileNotFoundError("no driver")):
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                smoke.environment_report(cpu=False)

    def test_cpu_check_does_not_certify_gpu(self):
        with patch.dict(sys.modules, {"torch": self.fake_torch()}), \
             patch.object(smoke.importlib.metadata, "version", return_value="different"), \
             patch.object(smoke.subprocess, "run", side_effect=FileNotFoundError("no driver")):
            report = smoke.environment_report(cpu=True)
        self.assertTrue(report["cpu_only"])
        self.assertFalse(report["gpu_readiness_certified"])
        self.assertTrue(report["version_mismatches"])

    def test_gpu_check_rejects_version_drift(self):
        with patch.dict(sys.modules, {"torch": self.fake_torch(True)}), \
             patch.object(smoke.importlib.metadata, "version", return_value="different"), \
             patch.object(smoke.subprocess, "run", side_effect=FileNotFoundError("no driver")):
            with self.assertRaisesRegex(RuntimeError, "pinned Python 3.10"):
                smoke.environment_report(cpu=False)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "previous-result.json"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                smoke.orchestrate(types.SimpleNamespace(output_dir=Path(directory)))
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_duplicate_gpu_argument_fails_before_launch(self):
        with self.assertRaises(SystemExit) as raised, patch.object(smoke.subprocess, "Popen") as launch:
            smoke.main(["--gpus", "0", "0"])
        self.assertEqual(raised.exception.code, 2)
        launch.assert_not_called()

    def test_cpu_cannot_claim_production_memory_validation(self):
        with self.assertRaises(SystemExit) as raised, patch.object(smoke.subprocess, "Popen") as launch:
            smoke.main(["--cpu", "--production-batches"])
        self.assertEqual(raised.exception.code, 2)
        launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
