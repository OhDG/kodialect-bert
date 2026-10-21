"""Controller tests use stdlib and tiny artifact fixtures; no GPU/ML imports."""
import argparse
import ast
from contextlib import redirect_stdout
import copy
import csv
import gzip
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

BUNDLE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("parallel_runner_under_test", BUNDLE / "run_parallel.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def model_fixture(path):
    write_json(path / "config.json", {"model_type": "bert"})
    (path / "model.safetensors").write_bytes(b"fixture weights")


def tokenizer_fixture(path):
    write_json(path / "tokenizer_config.json", {"tokenizer_class": "BertTokenizer"})
    (path / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n", encoding="utf-8")


def checkpoint_fixture(output, step):
    path = output / f"checkpoint-{step}"
    model_fixture(path)
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "scaler.pt"):
        (path / name).write_bytes(b"fixture state")
    write_json(path / "trainer_state.json", {"global_step": step})
    return path


def complete_fixture(output, tokenizer=None, seed=None, count=7):
    model_fixture(output / "final_model")
    if tokenizer is None:
        tokenizer_fixture(output / "final_model")
        write_json(output / "decoder_pretraining_metadata.json", {"completed": True})
        write_json(output / "validation_results.json", {"eval_loss": 1.0})
        return
    tokenizer_fixture(output / "final_model/source_tokenizer")
    tokenizer_fixture(output / "final_model/target_tokenizer")
    write_json(output / "experiment_metadata.json", {"completed": True, "tokenizer_name": tokenizer,
               "seed": seed, "train_process_metrics": {"wall_seconds": 1.0}})
    for split, rows in (("validation", 50000), ("test", count)):
        write_json(output / f"{split}_generation_report.json", {"overall": {"count": rows}})
        with gzip.open(output / f"{split}_predictions.tsv.gz", "wt", encoding="utf-8") as stream:
            stream.write("region\tis_changed\tsource_text\treference\tprediction\nfixture\t0\tx\tx\tx\n")


def options(command):
    return dict(zip(command[2::2], command[3::2]))


def engine_parser(script, command):
    """Execute just the real argparse function, without importing torch/transformers."""
    tree = ast.parse(script.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parse_args")
    namespace = {"argparse": argparse}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(script), "exec"), namespace)
    with patch.object(sys, "argv", command[1:]):
        return namespace["parse_args"]()


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "engine").mkdir()
        (self.root / "run_parallel.py").write_text("# fixture controller\n")
        (self.root / "requirements.txt").write_text("# fixture requirements\n")
        (self.root / "engine/train.py").write_text("# fixture engine\n")
        write_json(self.root / "provenance/inputs_data_manifest.json", {"fixture": "data"})
        self.profile = runner.read_json(BUNDLE / "configs/rtx4090.json")
        self.preflight = {"inputs": {"pair_counts": {"test": 7}}}

    def tearDown(self):
        self.temporary.cleanup()

    def config(self, run_id="test_run", gpus=None):
        return runner.make_config(self.root, run_id, gpus or ["0", "1", "2", "3"], copy.deepcopy(self.profile))

    def test_four_gpu_assignments_cover_twenty_unique_jobs(self):
        lanes = runner.assignment(["0", "1", "2", "3"])
        self.assertEqual(lanes, {"0": ["dialect"], "1": ["klue"], "2": ["kobert"], "3": ["mbert"]})
        jobs = [(gpu, tokenizer, seed) for gpu, names in lanes.items() for tokenizer in names for seed in runner.SEEDS]
        self.assertEqual(len(jobs), 20)
        self.assertEqual(len({(tokenizer, seed) for _, tokenizer, seed in jobs}), 20)
        self.assertEqual({gpu: sum(job[0] == gpu for job in jobs) for gpu in lanes}, {str(i): 5 for i in range(4)})

    def test_one_to_four_gpu_assignments_keep_each_tokenizer_in_one_lane(self):
        for count in range(1, 5):
            lanes = runner.assignment([str(i) for i in range(count)])
            flattened = [tokenizer for names in lanes.values() for tokenizer in names]
            self.assertCountEqual(flattened, runner.TOKENIZERS)
        for gpus in ([], ["0", "0"], ["0", "00"], ["01"], ["-1"], ["a"], ["0", "1", "2", "3", "4"]):
            with self.subTest(gpus=gpus), self.assertRaises(ValueError):
                runner.assignment(gpus)

    def test_lane_runs_each_tokenizers_seeds_in_order(self):
        instance = runner.Runner.__new__(runner.Runner)
        instance.stop = threading.Event()
        calls = []
        instance.run_stage = lambda gpu, tokenizer, seed: calls.append((gpu, tokenizer, seed))
        instance.lane("2", ["dialect", "kobert"])
        self.assertEqual(calls, [("2", tokenizer, seed) for tokenizer in ("dialect", "kobert") for seed in runner.SEEDS])
        instance.stop.set()
        instance.lane("2", ["klue"])
        self.assertEqual(len(calls), 10)

    def test_per_tokenizer_cache_shared_across_seeds_and_isolated_across_tokenizers_runs(self):
        run_dir = self.root / "runs/one"
        cache_paths = {}
        for tokenizer in runner.TOKENIZERS:
            values = [options(runner.build_command(self.root, run_dir, self.profile, tokenizer, seed)) for seed in runner.SEEDS]
            for key in ("--dataset_cache_dir", "--tokenized_cache_dir"):
                self.assertEqual(len({value[key] for value in values}), 1)
            cache_paths[tokenizer] = values[0]["--tokenized_cache_dir"]
            self.assertEqual(len({value["--output_dir"] for value in values}), 5)
        self.assertEqual(len(set(cache_paths.values())), 4)
        other = options(runner.build_command(self.root, self.root / "runs/two", self.profile, "dialect", 13))
        self.assertNotEqual(other["--tokenized_cache_dir"], cache_paths["dialect"])

    def test_commands_parse_in_actual_engines_and_preserve_effective_batch_256(self):
        run_dir = self.root / "runs/one"
        decoder = runner.build_command(BUNDLE, run_dir, self.profile)
        decoder_args = engine_parser(Path(decoder[1]), decoder)
        self.assertEqual(decoder_args.train_batch_size * decoder_args.gradient_accumulation_steps, 256)
        self.assertEqual(Path(decoder_args.target_tokenizer_name), BUNDLE / "models/mlm/klue/final_model")
        self.assertEqual(decoder_args.num_train_epochs, 3.0)
        for tokenizer in runner.TOKENIZERS:
            command = runner.build_command(BUNDLE, run_dir, self.profile, tokenizer, 13)
            args = engine_parser(Path(command[1]), command)
            self.assertEqual(args.train_batch_size * args.gradient_accumulation_steps, 256)
            self.assertEqual(args.validation_generation_samples, 50000)
            self.assertEqual(args.generation_num_beams, 4)
            self.assertEqual(args.seed, 13)
            self.assertIsNone(args.max_test_samples)
            self.assertIsNone(args.resume_from_checkpoint)
            self.assertFalse(args.overwrite_output_dir)

    def test_profile_rejects_changed_effective_batch(self):
        for section in ("decoder", "translation"):
            profile = copy.deepcopy(self.profile)
            profile[section]["gradient_accumulation_steps"] = 1
            with self.subTest(section=section), self.assertRaisesRegex(ValueError, "256"):
                runner.make_config(self.root, "bad_batch", ["0"], profile)

    def test_plan_works_with_site_packages_disabled_and_creates_no_run(self):
        run_id = "unittest_plan_stdlib_only"
        destination = BUNDLE / "runs" / run_id
        self.assertFalse(destination.exists())
        result = subprocess.run([sys.executable, "-S", str(BUNDLE / "run_parallel.py"), "--plan", "--run-id", run_id],
                                text=True, encoding="utf-8", capture_output=True, timeout=30, cwd=self.root)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertTrue(plan["fresh_decoder"])
        self.assertTrue(plan["reuse_imported_mlm_encoders"])
        self.assertEqual(len(plan["assignments"]), 4)
        self.assertFalse(destination.exists())

    def test_child_environment_pins_one_gpu_and_clears_distributed_settings(self):
        with patch.dict(os.environ, {"RANK": "9", "LOCAL_RANK": "3", "WORLD_SIZE": "4", "MASTER_ADDR": "old", "CUDA_VISIBLE_DEVICES": "0,1,2,3"}):
            env = runner.child_environment("2", self.profile)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2")
        for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR"):
            self.assertNotIn(name, env)
        for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONUNBUFFERED"):
            self.assertEqual(env[name], "1")

    def test_final_model_alone_never_marks_either_stage_complete(self):
        output = self.root / "output"
        model_fixture(output / "final_model")
        tokenizer_fixture(output / "final_model")
        self.assertFalse(runner.output_complete(output))
        self.assertFalse(runner.output_complete(output, "dialect", 13, 7))

    def test_decoder_requires_completion_metadata_validation_and_tokenizer(self):
        output = self.root / "decoder"
        complete_fixture(output)
        self.assertTrue(runner.output_complete(output))
        for relative in ("decoder_pretraining_metadata.json", "validation_results.json", "final_model/tokenizer_config.json", "final_model/vocab.txt"):
            path = output / relative
            original = path.read_bytes()
            path.unlink()
            self.assertFalse(runner.output_complete(output), relative)
            path.write_bytes(original)

    def test_translation_requires_identity_reports_predictions_and_expected_counts(self):
        output = self.root / "translation"
        complete_fixture(output, "dialect", 13)
        self.assertTrue(runner.output_complete(output, "dialect", 13, 7))
        self.assertFalse(runner.output_complete(output, "klue", 13, 7))
        self.assertFalse(runner.output_complete(output, "dialect", 21, 7))
        self.assertFalse(runner.output_complete(output, "dialect", 13, 8))
        for relative in ("experiment_metadata.json", "validation_generation_report.json", "test_generation_report.json", "validation_predictions.tsv.gz", "test_predictions.tsv.gz"):
            path = output / relative
            original = path.read_bytes()
            path.write_bytes(b"")
            self.assertFalse(runner.output_complete(output, "dialect", 13, 7), relative)
            path.write_bytes(original)

    def test_translation_completion_requires_saved_source_and_target_tokenizer_artifacts(self):
        output = self.root / "translation"
        complete_fixture(output, "dialect", 13)
        for side in ("source_tokenizer", "target_tokenizer"):
            for filename in ("tokenizer_config.json", "vocab.txt"):
                path = output / "final_model" / side / filename
                original = path.read_bytes()
                path.unlink()
                self.assertFalse(runner.output_complete(output, "dialect", 13, 7), str(path))
                path.write_bytes(original)

    def test_checkpoint_requires_optimizer_scheduler_rng_and_matching_trainer_step(self):
        output = self.root / "output"
        checkpoint = checkpoint_fixture(output, 100)
        self.assertEqual(runner.latest_checkpoint(output), checkpoint)
        for filename in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"):
            path = checkpoint / filename
            original = path.read_bytes()
            path.write_bytes(b"")
            self.assertIsNone(runner.latest_checkpoint(output), filename)
            path.write_bytes(original)
        write_json(checkpoint / "trainer_state.json", {"global_step": 99})
        self.assertIsNone(runner.latest_checkpoint(output))

    def test_checkpoint_selection_ignores_newer_incomplete_and_uses_numeric_step(self):
        output = self.root / "output"
        checkpoint_fixture(output, 9)
        expected = checkpoint_fixture(output, 100)
        newest = checkpoint_fixture(output, 200)
        (newest / "rng_state.pth").unlink()
        self.assertEqual(runner.latest_checkpoint(output), expected)

    def test_fp16_resume_requires_saved_gradient_scaler(self):
        output = self.root / "fp16_output"
        checkpoint = checkpoint_fixture(output, 100)
        self.assertEqual(runner.latest_checkpoint(output, require_scaler=True), checkpoint)
        (checkpoint / "scaler.pt").unlink()
        self.assertIsNone(runner.latest_checkpoint(output, require_scaler=True))
        self.assertEqual(runner.latest_checkpoint(output), checkpoint)

    def test_sharded_weights_require_all_nonempty_local_shards(self):
        directory = self.root / "sharded"
        write_json(directory / "config.json", {"model_type": "bert"})
        write_json(directory / "model.safetensors.index.json", {"weight_map": {"x": "part-1.safetensors", "y": "part-2.safetensors"}})
        (directory / "part-1.safetensors").write_bytes(b"one")
        self.assertFalse(runner.weights_present(directory))
        (directory / "part-2.safetensors").write_bytes(b"two")
        self.assertTrue(runner.weights_present(directory))
        write_json(directory / "model.safetensors.index.json", {"weight_map": {"x": "../part-1.safetensors"}})
        self.assertFalse(runner.weights_present(directory))

    def test_resume_requires_existing_id_and_identical_configuration(self):
        config = self.config()
        first = runner.Runner(self.root, config, False, self.preflight)
        resumed = runner.Runner(self.root, copy.deepcopy(config), True, self.preflight)
        self.assertEqual(first.signature, resumed.signature)
        with self.assertRaisesRegex(RuntimeError, "exists"):
            runner.Runner(self.root, config, False, self.preflight)
        with self.assertRaisesRegex(RuntimeError, "does not exist"):
            runner.Runner(self.root, self.config("missing_run"), True, self.preflight)
        changed = copy.deepcopy(config)
        changed["profile"]["translation"]["generation_num_beams"] = 2
        with self.assertRaisesRegex(RuntimeError, "changed"):
            runner.Runner(self.root, changed, True, self.preflight)
        fresh = runner.Runner(self.root, self.config("fresh_second_run"), False, self.preflight)
        self.assertNotEqual(first.directory, fresh.directory)

    def test_config_identity_detects_engine_input_and_gpu_changes(self):
        config = self.config()
        runner.Runner(self.root, config, False, self.preflight)
        for path in (self.root / "engine/train.py", self.root / "provenance/inputs_data_manifest.json"):
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            changed = self.config()
            self.assertNotEqual(runner.fingerprint(config), runner.fingerprint(changed))
            with self.assertRaisesRegex(RuntimeError, "changed"):
                runner.Runner(self.root, changed, True, self.preflight)
            path.write_bytes(original)
        with self.assertRaisesRegex(RuntimeError, "changed"):
            runner.Runner(self.root, self.config(gpus=["3", "2", "1", "0"]), True, self.preflight)

    def test_gpu_summary_combines_attempts_without_other_stage_measurements(self):
        logs = self.root / "logs"
        logs.mkdir()
        for name, values in (("stage_attempt1_gpu.csv", [(100, 0), (500, 50)]),
                             ("stage_attempt2_gpu.csv", [(300, 80), ("", "N/A")]),
                             ("other_attempt1_gpu.csv", [(9999, 99)])):
            with (logs / name).open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["process_gpu_memory_mib", "process_sm_utilization_percent"])
                writer.writerows(values)
        result = runner.gpu_summary(logs, "stage")
        self.assertEqual(result["peak_process_gpu_memory_mib"], 500)
        self.assertEqual(result["average_active_process_sm_percent"], 65)
        self.assertEqual(result["samples"], 3)

    def test_monitor_start_failure_terminates_and_reaps_started_training_process(self):
        instance = runner.Runner(self.root, self.config(), False, self.preflight)
        process = Mock(pid=12345, returncode=None)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("fixture", 30), 0]
        with patch.object(runner.subprocess, "Popen", side_effect=[process, OSError("monitor unavailable")]), \
                patch.object(runner.os, "killpg", create=True) as killpg, \
                patch.object(runner.signal, "SIGKILL", 9, create=True), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(OSError, "monitor unavailable"):
                instance.run_stage("0")
        self.assertTrue(instance.stop.is_set())
        self.assertEqual(process.wait.call_count, 2)
        self.assertIn(unittest.mock.call(process.pid, 9), killpg.call_args_list)
        self.assertEqual(instance.children, {})
        self.assertEqual(instance.state["stages"][runner.DECODER_STAGE]["status"], "failed")

    def test_summary_cancellation_is_tracked_reaped_and_never_successful(self):
        instance = runner.Runner(self.root, self.config(), False, self.preflight)
        process = Mock(pid=12346, returncode=-15)
        process.poll.side_effect = [None, None, -15]
        def stop_while_waiting(_timeout):
            self.assertIs(instance.children["summary"], process)
            instance.stop.set()
            return True
        with patch.object(runner.subprocess, "Popen", return_value=process), \
                patch.object(runner.os, "killpg", create=True) as killpg, \
                patch.object(instance.stop, "wait", side_effect=stop_while_waiting):
            with self.assertRaisesRegex(RuntimeError, "Summary interrupted"):
                instance.summarize()
        process.wait.assert_called_once_with(timeout=30)
        killpg.assert_called()
        self.assertEqual(instance.children, {})
        self.assertFalse((instance.directory / "results/pipeline_stage_times.json").exists())

    def test_abrupt_attempt_is_marked_as_incomplete_timing_on_resume(self):
        config = self.config()
        instance = runner.Runner(self.root, config, False, self.preflight)
        instance.state["stages"][runner.DECODER_STAGE] = {"status": "running", "attempts": [{"status": "running", "started_at": runner.now()}]}
        instance.save()
        resumed = runner.Runner(self.root, config, True, self.preflight)
        record = resumed.state["stages"][runner.DECODER_STAGE]
        self.assertTrue(record["timing_incomplete"])
        self.assertEqual(record["attempts"][0]["status"], "interrupted_without_end_record")

    def test_real_subprocess_orchestration_runs_decoder_then_twenty_jobs_and_resumes(self):
        helpers = "\n\n".join(inspect.getsource(function) for function in
                               (write_json, model_fixture, tokenizer_fixture, complete_fixture))
        engine = "import sys,os,json,gzip,time\nfrom pathlib import Path\n" + helpers + '''
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
output = Path(args['--output_dir'])
tokenizer = args.get('--tokenizer_name')
seed = int(args['--seed']) if tokenizer else None
if tokenizer:
    assert Path(args['--shared_decoder_model_dir']).is_dir(), 'decoder must finish first'
    cache = Path(args['--tokenized_cache_dir'])
    cache.mkdir(parents=True, exist_ok=True)
    lock = cache / 'active-seed.lock'
    with lock.open('x') as handle:
        handle.write(str(seed))
    time.sleep(0.02)
complete_fixture(output, tokenizer, seed)
write_json(output / 'invocation.json', {'gpu': os.environ['CUDA_VISIBLE_DEVICES'],
           'seed': seed, 'tokenizer': tokenizer, 'rank': os.environ.get('RANK')})
if tokenizer:
    lock.unlink()
'''
        for name in ("step2_pretrain_standard_decoder_clm.py", "step3_train_dialect_to_standard.py"):
            (self.root / "engine" / name).write_text(engine, encoding="utf-8")
        (self.root / "engine/process_gpu_monitor.py").write_text('''
import sys
from pathlib import Path
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
Path(args['--output']).write_text('process_gpu_memory_mib,process_sm_utilization_percent\\n123,50\\n')
''', encoding="utf-8")
        (self.root / "engine/step4_summarize_results.py").write_text('''
import sys,json
from pathlib import Path
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
reports = list(Path(args['--outputs_root']).glob('translation/*/seed_*/test_generation_report.json'))
assert len(reports) == 20, len(reports)
result = Path(args['--result_dir'])
result.mkdir(parents=True, exist_ok=True)
(result / 'final_results.json').write_text(json.dumps({'completed_jobs': len(reports)}))
(result / 'final_results.md').write_text('# Twenty fixture translations complete\\n')
''', encoding="utf-8")
        config = self.config()
        instance = runner.Runner(self.root, config, False, self.preflight)
        original_wait = instance.stop.wait
        instance.stop.wait = lambda timeout=None: original_wait(min(timeout or 0.01, 0.01))
        with patch.object(runner.signal, "signal"), redirect_stdout(io.StringIO()):
            instance.run()
        self.assertEqual(instance.state["status"], "completed")
        self.assertEqual(len(instance.state["stages"]), 21)
        self.assertTrue(all(record["status"] == "completed" and len(record["attempts"]) == 1 for record in instance.state["stages"].values()))
        for gpu, names in config["assignments"].items():
            for tokenizer in names:
                for seed in runner.SEEDS:
                    _, output = runner.stage_paths(instance.directory, tokenizer, seed)
                    self.assertEqual(runner.read_json(output / "invocation.json"),
                                     {"gpu": gpu, "seed": seed, "tokenizer": tokenizer, "rank": None})
        self.assertEqual(runner.read_json(instance.directory / "results/final_results.json")["completed_jobs"], 20)
        self.assertEqual(len(runner.read_json(instance.directory / "results/pipeline_stage_times.json")), 21)
        # Complete resume invokes only the summary subprocess, retaining all stage attempts.
        resumed = runner.Runner(self.root, config, True, self.preflight)
        original_wait = resumed.stop.wait
        resumed.stop.wait = lambda timeout=None: original_wait(min(timeout or 0.01, 0.01))
        actual_popen = subprocess.Popen
        with patch.object(runner.subprocess, "Popen", wraps=actual_popen) as popen, \
                patch.object(runner.signal, "signal"), redirect_stdout(io.StringIO()):
            resumed.run()
        self.assertEqual(popen.call_count, 1)
        self.assertIn("step4_summarize_results.py", popen.call_args.args[0][1])
        self.assertEqual(resumed.state["status"], "completed")
        self.assertTrue(all(len(record["attempts"]) == 1 for record in resumed.state["stages"].values()))


if __name__ == "__main__":
    unittest.main()
