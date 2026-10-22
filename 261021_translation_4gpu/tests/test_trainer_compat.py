"""Regression checks for single-GPU fused-AdamW/FP16 checkpoint restoration.

CUDA placement policy is tested with a CPU-backed Tensor subclass; the optional
CUDA test exercises a real resumed optimizer update when CUDA is available.
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
import trainer_compat as compat


class PlacementTensor(torch.Tensor):
    """A real CPU tensor with observable, simulated CUDA placement."""

    @staticmethod
    def __new__(cls, values, *, device="cpu", dtype=torch.float32):
        data = torch.as_tensor(values, dtype=dtype, device="cpu").clone()
        result = torch.Tensor._make_subclass(cls, data, require_grad=False)
        result.reported_device = torch.device(device)
        result.moves = []
        return result

    @property
    def device(self):
        return self.reported_device

    def to(self, device=None, dtype=None, **kwargs):
        self.moves.append((torch.device(device), dtype))
        return PlacementTensor(self.as_subclass(torch.Tensor),
                               device=device, dtype=self.dtype if dtype is None else dtype)


class ParameterPlacement:
    def __init__(self, device):
        self.device = torch.device(device)
        self.is_cuda = self.device.type == "cuda"


def placement_optimizer(*, fused=True, device="cuda:2", step_device="cpu"):
    # Construct a real AdamW to exercise the implementation's optimizer type
    # check, then replace its uninitialized parameter with a placement fixture.
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))])
    parameter = ParameterPlacement(device)
    optimizer.param_groups[0]["params"] = [parameter]
    optimizer.param_groups[0]["fused"] = fused
    step = PlacementTensor([7.0], device=step_device)
    first = torch.tensor([0.2, 0.3])
    second = torch.tensor([0.4, 0.5])
    optimizer.state[parameter] = {"step": step, "exp_avg": first, "exp_avg_sq": second}
    return optimizer, parameter, step, first, second


class FusedAdamWPlacementTests(unittest.TestCase):
    def test_restores_only_fused_step_device_preserving_saved_state(self):
        optimizer, parameter, old_step, first, second = placement_optimizer()
        groups_before = {key: value for key, value in optimizer.param_groups[0].items() if key != "params"}

        self.assertEqual(compat.repair_fused_adamw_step_devices(optimizer), 1)

        state = optimizer.state[parameter]
        self.assertEqual(state["step"].device, parameter.device)
        self.assertEqual(state["step"].dtype, old_step.dtype)
        self.assertEqual(state["step"].shape, old_step.shape)
        self.assertTrue(torch.equal(state["step"].as_subclass(torch.Tensor), torch.tensor([7.0])))
        self.assertEqual(old_step.device, torch.device("cpu"))
        self.assertIs(state["exp_avg"], first)
        self.assertIs(state["exp_avg_sq"], second)
        self.assertEqual({key: value for key, value in optimizer.param_groups[0].items() if key != "params"}, groups_before)
        self.assertEqual(compat.repair_fused_adamw_step_devices(optimizer), 0)

    def test_accelerate_style_optimizer_wrapper_is_supported(self):
        optimizer, parameter, _, _, _ = placement_optimizer()
        wrapped = SimpleNamespace(optimizer=optimizer)
        self.assertEqual(compat.repair_fused_adamw_step_devices(wrapped), 1)
        self.assertEqual(optimizer.state[parameter]["step"].device, parameter.device)

    def test_ordinary_adamw_step_is_untouched(self):
        optimizer, parameter, step, _, _ = placement_optimizer(fused=False)
        self.assertEqual(compat.repair_fused_adamw_step_devices(optimizer), 0)
        self.assertIs(optimizer.state[parameter]["step"], step)
        self.assertEqual(step.moves, [])

    def test_cpu_parameter_and_already_placed_step_are_untouched(self):
        for device, step_device in (("cpu", "cpu"), ("cuda:2", "cuda:2")):
            with self.subTest(device=device, step_device=step_device):
                optimizer, parameter, step, _, _ = placement_optimizer(device=device, step_device=step_device)
                self.assertEqual(compat.repair_fused_adamw_step_devices(optimizer), 0)
                self.assertIs(optimizer.state[parameter]["step"], step)
                self.assertEqual(step.moves, [])

    def test_missing_and_non_tensor_steps_are_untouched(self):
        optimizer, parameter, _, _, _ = placement_optimizer()
        optimizer.state[parameter]["step"] = 7
        self.assertEqual(compat.repair_fused_adamw_step_devices(optimizer), 0)
        self.assertEqual(optimizer.state[parameter]["step"], 7)
        del optimizer.state[parameter]["step"]
        self.assertEqual(compat.repair_fused_adamw_step_devices(optimizer), 0)
        self.assertNotIn("step", optimizer.state[parameter])

    def test_other_optimizer_types_are_untouched(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(2))], lr=0.1)
        optimizer.param_groups[0]["fused"] = True
        self.assertEqual(compat.repair_fused_adamw_step_devices(optimizer), 0)


class RecordingScaler:
    def __init__(self, events, *, enabled=True, value=None):
        self.events = events
        self.enabled = enabled
        self.value = copy.deepcopy(value if value is not None else {
            "scale": 8192.0, "growth_factor": 2.0, "backoff_factor": 0.5,
            "growth_interval": 2000, "_growth_tracker": 17,
        })

    def is_enabled(self):
        return self.enabled

    def state_dict(self):
        self.events.append("scaler_saved")
        return copy.deepcopy(self.value)

    def load_state_dict(self, value):
        self.events.append("scaler_loaded")
        self.value = copy.deepcopy(value)


class RecordingTrainerBase:
    def _save_optimizer_and_scheduler(self, output_dir):
        self.events.append("base_saved")
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "base.marker").write_text("saved", encoding="utf-8")

    def _load_optimizer_and_scheduler(self, checkpoint):
        self.events.append("base_loaded")


class CompatibilityHarness(compat._ResumeCompatibilityMixin, RecordingTrainerBase):
    def __init__(self, *, enabled=True, should_save=True, scaler_present=True):
        self.events = []
        self.args = SimpleNamespace(should_save=should_save)
        self.optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))])
        scaler = RecordingScaler(self.events, enabled=enabled) if scaler_present else None
        self.accelerator = SimpleNamespace(scaler=scaler)


class ScalerLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name) / "checkpoint-2"

    def tearDown(self):
        self.temporary.cleanup()

    def test_save_then_load_preserves_scaler_and_runs_after_parent_hooks(self):
        initial = CompatibilityHarness()
        expected = copy.deepcopy(initial.accelerator.scaler.value)
        initial._save_optimizer_and_scheduler(str(self.directory))
        self.assertEqual(initial.events, ["base_saved", "scaler_saved"])
        self.assertEqual(torch.load(self.directory / "scaler.pt", map_location="cpu", weights_only=True), expected)
        self.assertEqual({path.name for path in self.directory.iterdir()}, {"base.marker", "scaler.pt"})

        restored = CompatibilityHarness()
        restored.accelerator.scaler.value["scale"] = 65536.0
        restored.accelerator.scaler.value["_growth_tracker"] = 0
        def repair(optimizer):
            self.assertIs(optimizer, restored.optimizer)
            restored.events.append("optimizer_repaired")
            return 0
        with patch.object(compat, "repair_fused_adamw_step_devices", side_effect=repair):
            restored._load_optimizer_and_scheduler(str(self.directory))
        self.assertEqual(restored.events, ["base_loaded", "optimizer_repaired", "scaler_loaded"])
        self.assertEqual(restored.accelerator.scaler.value, expected)

    def test_resume_with_active_scaler_rejects_missing_scaler_state(self):
        self.directory.mkdir()
        trainer = CompatibilityHarness()
        with self.assertRaises(FileNotFoundError):
            trainer._load_optimizer_and_scheduler(str(self.directory))
        self.assertIn("base_loaded", trainer.events)
        self.assertNotIn("scaler_loaded", trainer.events)

    def test_new_training_does_not_require_a_scaler_checkpoint(self):
        trainer = CompatibilityHarness()
        trainer._load_optimizer_and_scheduler(None)
        self.assertNotIn("scaler_loaded", trainer.events)

    def test_disabled_or_absent_scaler_does_not_save_or_require_state(self):
        for scaler_present, enabled in ((True, False), (False, True)):
            with self.subTest(scaler_present=scaler_present, enabled=enabled):
                trainer = CompatibilityHarness(enabled=enabled, scaler_present=scaler_present)
                trainer._save_optimizer_and_scheduler(str(self.directory))
                trainer._load_optimizer_and_scheduler(str(self.directory))
                self.assertFalse((self.directory / "scaler.pt").exists())
                self.assertNotIn("scaler_saved", trainer.events)
                self.assertNotIn("scaler_loaded", trainer.events)

    def test_non_saving_rank_does_not_write_scaler(self):
        trainer = CompatibilityHarness(should_save=False)
        trainer._save_optimizer_and_scheduler(str(self.directory))
        self.assertFalse((self.directory / "scaler.pt").exists())
        self.assertNotIn("scaler_saved", trainer.events)


@unittest.skipUnless(torch.cuda.is_available(), "requires real CUDA for a fused optimizer update")
class CudaResumeRegressionTests(unittest.TestCase):
    def test_cpu_loaded_fused_state_matches_uninterrupted_next_update(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, -0.5], device="cuda"))
        optimizer = torch.optim.AdamW([parameter], lr=0.01, fused=True)
        for _ in range(2):
            parameter.grad = torch.tensor([0.3, -0.2], device="cuda")
            optimizer.step()
        restored_parameter = torch.nn.Parameter(parameter.detach().clone())
        restored = torch.optim.AdamW([restored_parameter], lr=0.01, fused=True)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "optimizer.pt"
            torch.save(optimizer.state_dict(), checkpoint)
            restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        # Explicitly recreate the torch 2.0.1 bug even under newer test runtimes.
        restored.state[restored_parameter]["step"] = restored.state[restored_parameter]["step"].cpu()
        self.assertEqual(compat.repair_fused_adamw_step_devices(restored), 1)
        for item, current in ((parameter, optimizer), (restored_parameter, restored)):
            item.grad = torch.tensor([0.4, -0.1], device="cuda")
            current.step()
        torch.cuda.synchronize()
        torch.testing.assert_close(restored_parameter, parameter, rtol=0, atol=0)
        self.assertEqual(int(restored.state[restored_parameter]["step"]), 3)


if __name__ == "__main__":
    unittest.main()
