import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "estimate_runtime.py"
SPEC = importlib.util.spec_from_file_location("estimate_runtime", SCRIPT)
estimator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(estimator)
NOW = datetime(2026, 10, 21, 12, tzinfo=timezone.utc)


def stage_name(tokenizer, seed):
    return f"03_translate_{tokenizer}_seed_{seed}"


def completed(hours):
    return {"status": "completed", "elapsed_seconds": hours * 3600}


class RuntimeEstimateTests(unittest.TestCase):
    def test_unmeasured_four_lane_baseline_excludes_decoder_and_mlm(self):
        result = estimator.build_estimate(now=NOW)
        self.assertEqual((result["remaining_hours_low"], result["remaining_hours_high"]), (52.5, 60.0))
        self.assertEqual(result["total_translation_runs"], 20)
        self.assertTrue(result["decoder_remaining_time_unknown"])
        self.assertTrue(all(len(lane["jobs"]) == 5 for lane in result["lanes"]))

    def test_decoder_remaining_budget_is_added_once(self):
        result = estimator.build_estimate(decoder_hours=7.5, now=NOW)
        self.assertEqual((result["remaining_hours_low"], result["remaining_hours_high"]), (60.0, 67.5))
        self.assertFalse(result["decoder_remaining_time_unknown"])

    def test_tokenizer_mean_and_elapsed_are_applied_to_its_lane(self):
        config = {"gpus": ["2"], "tokenizers": ["klue"], "seeds": [13, 21, 42, 87, 100],
                  "assignments": {"2": ["klue"]}}
        stages = {stage_name("klue", 13): completed(2), stage_name("klue", 21): completed(4),
                  stage_name("klue", 42): {"status": "running", "elapsed_seconds": 0,
                                         "started_at": (NOW - timedelta(hours=1)).isoformat(), "gpu": "2"}}
        result = estimator.build_estimate(config, {"stages": stages}, now=NOW)
        self.assertEqual(result["translation_hours_low"], 8)
        self.assertEqual(result["translation_hours_high"], 8)
        self.assertEqual(result["completed_translation_runs"], 2)
        self.assertEqual(result["calibration"]["klue"]["sample_count"], 2)

    def test_unobserved_tokenizer_uses_pooled_mean_and_slowest_lane(self):
        config = {"gpus": ["0", "1"], "tokenizers": ["dialect", "klue"], "seeds": [13, 21],
                  "assignments": {"0": ["dialect"], "1": ["klue"]}}
        result = estimator.build_estimate(config, {"stages": {stage_name("dialect", 13): completed(6)}}, now=NOW)
        self.assertEqual(result["translation_hours_low"], 12)
        self.assertEqual(result["calibration"]["klue"]["source"], "pooled_completed_mean")
        self.assertEqual([lane["hours_high"] for lane in result["lanes"]], [6, 12])

    def test_running_overrun_never_becomes_negative_or_claims_completion(self):
        config = {"gpus": ["0"], "tokenizers": ["dialect"], "seeds": [13, 21],
                  "assignments": {"0": ["dialect"]}}
        stages = {stage_name("dialect", 13): {"status": "running", "elapsed_seconds": 13 * 3600}}
        result = estimator.build_estimate(config, {"stages": stages}, now=NOW)
        self.assertEqual((result["translation_hours_low"], result["translation_hours_high"]), (10.5, 12))
        self.assertTrue(result["additional_running_time_unknown"])
        self.assertEqual(result["completed_translation_runs"], 0)
        self.assertEqual(result["lanes"][0]["remaining_runs"], 2)
        self.assertIn("additional running time is unknown", " ".join(result["warnings"]))

    def test_failed_job_keeps_full_budget_and_warns_about_resume(self):
        config = {"gpus": ["0"], "tokenizers": ["dialect"], "seeds": [13],
                  "assignments": {"0": ["dialect"]}}
        result = estimator.build_estimate(config, {"stages": {
            stage_name("dialect", 13): {"status": "failed", "elapsed_seconds": 15 * 3600}}}, now=NOW)
        self.assertEqual(result["translation_hours_high"], 12)
        self.assertIn("Resume may repeat evaluation", " ".join(result["warnings"]))

    def test_completed_all_jobs_returns_zero_even_with_historical_mlm_stage(self):
        stages = {stage_name(tokenizer, seed): completed(10)
                  for tokenizer in estimator.TOKENIZERS for seed in estimator.SEEDS}
        stages[estimator.DECODER_STAGE] = completed(8)
        stages["old_mlm"] = {"status": "running", "elapsed_seconds": 1}
        result = estimator.build_estimate(state={"stages": stages}, decoder_hours=9, now=NOW)
        self.assertEqual(result["remaining_hours_high"], 0)
        self.assertEqual(result["completed_translation_runs"], 20)
        self.assertFalse(result["decoder_remaining_time_unknown"])

    def test_completed_duration_can_be_reconstructed_from_timezone_stamps(self):
        stages = {stage_name("dialect", 13): {"status": "completed",
                  "started_at": "2026-10-21T12:00:00+09:00", "ended_at": "2026-10-21T05:00:00Z"}}
        result = estimator.build_estimate(state={"stages": stages}, now=NOW)
        self.assertEqual(result["calibration"]["dialect"]["hours_low"], 2)

    def test_zero_or_invalid_completed_duration_does_not_poison_calibration(self):
        stages = {stage_name("dialect", 13): completed(0), stage_name("klue", 13): completed(float("nan"))}
        result = estimator.build_estimate(state={"stages": stages}, now=NOW)
        self.assertEqual(result["completed_translation_runs"], 2)
        self.assertEqual(result["calibration"]["dialect"]["source"], "old_server_planning_baseline")
        self.assertTrue(result["warnings"])

    def test_incomplete_completed_timing_is_not_used_in_tokenizer_or_pooled_means(self):
        incomplete = {**completed(0.5), "timing_incomplete": True}
        stages = {stage_name("dialect", 13): incomplete,
                  stage_name("klue", 13): completed(9)}
        result = estimator.build_estimate(state={"stages": stages}, now=NOW)
        self.assertEqual(result["completed_translation_runs"], 2)
        self.assertEqual(result["calibration"]["dialect"]["source"], "pooled_completed_mean")
        for calibration in result["calibration"].values():
            self.assertEqual(calibration["hours_low"], 9)
            self.assertEqual(calibration["sample_count"], 1)
        self.assertIn("wall time is unknown", " ".join(result["warnings"]))

    def test_only_incomplete_completed_timings_fall_back_to_baseline(self):
        stages = {stage_name("dialect", 13): {**completed(1), "timing_incomplete": True}}
        result = estimator.build_estimate(state={"stages": stages}, now=NOW)
        self.assertEqual(result["completed_translation_runs"], 1)
        self.assertTrue(all(value["source"] == "old_server_planning_baseline"
                            for value in result["calibration"].values()))

    def test_running_resume_does_not_deduct_prior_attempt_time(self):
        config = {"gpus": ["0"], "tokenizers": ["dialect"], "seeds": [13],
                  "assignments": {"0": ["dialect"]}}
        stages = {stage_name("dialect", 13): {
            "status": "running", "prior_elapsed_seconds": 9 * 3600,
            "started_at": (NOW - timedelta(hours=1)).isoformat()}}
        result = estimator.build_estimate(config, {"stages": stages}, now=NOW)
        self.assertEqual((result["translation_hours_low"], result["translation_hours_high"]), (9.5, 11))
        self.assertIn("prior attempt time is not deducted", " ".join(result["warnings"]))

    def test_invalid_assignment_cannot_silently_drop_or_duplicate_jobs(self):
        config = estimator.default_config()
        config["assignments"]["1"] = ["dialect"]
        with self.assertRaisesRegex(ValueError, "every tokenizer exactly once"):
            estimator.build_estimate(config, now=NOW)

    def test_negative_or_nonfinite_decoder_budget_is_rejected(self):
        for value in (-1, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                estimator.build_estimate(decoder_hours=value, now=NOW)

    def test_cli_reads_contract_files_and_emits_json_without_training_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_config.json").write_text(json.dumps(estimator.default_config()), encoding="utf-8")
            (root / "state.json").write_text(json.dumps({"stages": {}}), encoding="utf-8")
            process = subprocess.run([sys.executable, "-S", str(SCRIPT), "--run-dir", str(root),
                                      "--decoder-hours", "3", "--json"], text=True, capture_output=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(process.stdout)
            self.assertEqual(result["remaining_hours_low"], 55.5)
            self.assertEqual(result["remaining_hours_high"], 63.0)


if __name__ == "__main__":
    unittest.main()
