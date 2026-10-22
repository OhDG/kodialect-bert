"""FP16 smoke progress accounting, independent of CUDA and optional packages."""
import importlib.util
from pathlib import Path
import types
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "smoke_test.py"
spec = importlib.util.spec_from_file_location("bundle_smoke_updates", MODULE_PATH)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def scaler_state(scale=65536.0, tracker=0):
    return {"scale": scale, "growth_factor": 2.0, "backoff_factor": 0.5,
            "growth_interval": 2000, "_growth_tracker": tracker}


class UpdateProgressTests(unittest.TestCase):
    def test_resume_requires_exact_optimizer_counters(self):
        progress = smoke.UpdateProgress({"0:0": 5, "0:1": 4}, scaler_state())
        with self.assertRaisesRegex(AssertionError, "counters were not restored exactly"):
            progress.begin({"0:0": 5, "0:1": 0}, scaler_state())

    def test_resume_requires_all_saved_optimizer_state(self):
        progress = smoke.UpdateProgress({"0:0": 5, "0:1": 4}, scaler_state())
        with self.assertRaisesRegex(AssertionError, "counters were not restored exactly"):
            progress.begin({"0:0": 5}, scaler_state())

    def test_resume_requires_exact_scaler_including_growth_tracker(self):
        progress = smoke.UpdateProgress({"0:0": 5}, scaler_state(tracker=5))
        with self.assertRaisesRegex(AssertionError, "scaler state was not restored exactly"):
            progress.begin({"0:0": 5}, scaler_state(tracker=0))

    def test_resume_rejects_missing_fp16_scaler(self):
        progress = smoke.UpdateProgress({"0:0": 5}, scaler_state())
        with self.assertRaisesRegex(AssertionError, "scaler state was not restored exactly"):
            progress.begin({"0:0": 5}, None)

    def test_overflow_backoff_then_real_updates_are_counted(self):
        progress = smoke.UpdateProgress({"0:0": 2, "0:1": 2}, scaler_state())
        progress.begin({"0:0": 2, "0:1": 2}, scaler_state())
        self.assertFalse(progress.record({"0:0": 2, "0:1": 2}, 32768.0))
        self.assertTrue(progress.record({"0:0": 3, "0:1": 3}, 32768.0))
        self.assertTrue(progress.record({"0:0": 4, "0:1": 4}, 32768.0))
        self.assertEqual(progress.require_successes(), 2)
        self.assertEqual(len(progress.records), 3)
        self.assertEqual(progress.records[0], {
            "updated": False, "scale_before": 65536.0, "scale_after": 32768.0})
        self.assertEqual(progress.initial_steps, {"0:0": 2, "0:1": 2})

    def test_initial_overflow_may_allocate_zero_step_state(self):
        progress = smoke.UpdateProgress()
        progress.begin({}, scaler_state())
        self.assertFalse(progress.record({"0:0": 0, "0:1": 0}, 32768.0))
        self.assertTrue(progress.record({"0:0": 1, "0:1": 1}, 32768.0))
        self.assertEqual(progress.require_successes(minimum=1), 1)

    def test_no_progress_without_backoff_fails(self):
        for next_scale in (65536.0, 131072.0, None):
            with self.subTest(next_scale=next_scale):
                progress = smoke.UpdateProgress()
                progress.begin({"0:0": 1}, scaler_state())
                with self.assertRaisesRegex(AssertionError, "loss scaling did not back off"):
                    progress.record({"0:0": 1}, next_scale)

    def test_persistent_eight_overflows_do_not_pass(self):
        progress = smoke.UpdateProgress()
        progress.begin({}, scaler_state())
        scale = 65536.0
        for _ in range(smoke.SMOKE_ATTEMPTS):
            scale /= 2.0
            self.assertFalse(progress.record({"0:0": 0}, scale))
        with self.assertRaisesRegex(AssertionError, "Only 0 successful optimizer updates in 8 attempts; need 2"):
            progress.require_successes()

    def test_single_success_does_not_pass_two_update_requirement(self):
        progress = smoke.UpdateProgress()
        progress.begin({}, scaler_state())
        scale = 65536.0
        for _ in range(smoke.SMOKE_ATTEMPTS - 1):
            scale /= 2.0
            progress.record({"0:0": 0}, scale)
        progress.record({"0:0": 1}, scale)
        with self.assertRaisesRegex(AssertionError, "Only 1 successful optimizer updates"):
            progress.require_successes()

    def test_counter_reset_jump_or_disappearance_fails(self):
        cases = (
            ({"0:0": 0, "0:1": 2}, "reset or jumped"),
            ({"0:0": 4, "0:1": 2}, "reset or jumped"),
            ({"0:0": 3}, "state disappeared"),
        )
        for next_steps, message in cases:
            with self.subTest(next_steps=next_steps):
                progress = smoke.UpdateProgress()
                progress.begin({"0:0": 2, "0:1": 2}, scaler_state())
                with self.assertRaisesRegex(AssertionError, message):
                    progress.record(next_steps, 32768.0)

    def test_new_parameter_state_cannot_start_with_multiple_updates(self):
        progress = smoke.UpdateProgress()
        progress.begin({"0:0": 2}, None)
        with self.assertRaisesRegex(AssertionError, "reset or jumped"):
            progress.record({"0:0": 3, "0:1": 2}, None)

    def test_cpu_updates_need_no_scaler(self):
        progress = smoke.UpdateProgress()
        progress.begin({}, None)
        self.assertTrue(progress.record({"0:0": 1, "0:1": 1}, None))
        self.assertTrue(progress.record({"0:0": 2, "0:1": 2}, None))
        self.assertEqual(progress.require_successes(), 2)

    def test_cpu_no_update_is_an_error(self):
        progress = smoke.UpdateProgress()
        progress.begin({"0:0": 2}, None)
        with self.assertRaisesRegex(AssertionError, "loss scaling did not back off"):
            progress.record({"0:0": 2}, None)

    def test_delayed_parameter_state_and_inactive_parameters_are_allowed(self):
        progress = smoke.UpdateProgress()
        progress.begin({"0:0": 5, "0:1": 3}, None)
        self.assertTrue(progress.record({"0:0": 6, "0:1": 3, "1:0": 1}, None))
        self.assertTrue(progress.record({"0:0": 7, "0:1": 4, "1:0": 2}, None))
        self.assertEqual(progress.require_successes(), 2)

    def test_step_count_extraction_unwraps_optimizer_and_skips_uninitialized_state(self):
        first, second, third = object(), object(), object()
        raw = types.SimpleNamespace(
            param_groups=[{"params": [first, second]}, {"params": [third]}],
            state={first: {"step": 3.0}, second: {}, third: {"step": 1}},
        )
        wrapped = types.SimpleNamespace(optimizer=types.SimpleNamespace(optimizer=raw))
        self.assertEqual(smoke.optimizer_step_counts(wrapped), {"0:0": 3, "1:0": 1})


if __name__ == "__main__":
    unittest.main()
